"""Reusable inference for the Brain MRI image-classification model.

This module deliberately returns model classifications only. A confidence score is
the selected softmax output, not a medical probability or diagnosis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.preprocessing.mri_preprocessing import DEFAULT_TARGET_SIZE, SUPPORTED_IMAGE_TYPES, preprocess_mri_image
else:
    from ..preprocessing.mri_preprocessing import DEFAULT_TARGET_SIZE, SUPPORTED_IMAGE_TYPES, preprocess_mri_image


logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PACKAGE_ROOT / "models"
DEFAULT_CHECKPOINT_NAME = "mri_xception_selective_finetuning_best.keras"
DEFAULT_CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}


def find_default_model_path() -> Optional[Path]:
    """Locate weights from an environment setting or the local ``models/`` directory."""
    configured_path = os.getenv("CLINIQ_MRI_MODEL_PATH") or os.getenv("MRI_MODEL_PATH")
    if configured_path:
        candidate = Path(configured_path)
        if candidate.is_file():
            return candidate

    expected_path = MODELS_DIR / DEFAULT_CHECKPOINT_NAME
    if expected_path.is_file():
        return expected_path

    candidates = sorted([*MODELS_DIR.glob("*.keras"), *MODELS_DIR.glob("*.h5")])
    return candidates[0] if candidates else None


class MRIPredictor:
    """Load the saved MRI model once and classify individual images or batches."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        model_instance: Any = None,
        **_deprecated_options: Any,
    ) -> None:
        self.class_names = self._load_class_names(class_names_path)
        self.model: Any = model_instance
        self.model_path = Path("in_memory_instance") if model_instance is not None else (
            Path(model_path) if model_path else find_default_model_path()
        )
        if model_instance is None and self.model_path and self.model_path.is_file():
            self._load_model()

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {"success": False, "prediction": None, "class_index": None, "confidence": 0.0, "probabilities": {}, "error": message}

    def _load_class_names(self, class_names_path: Optional[Union[str, Path]]) -> Dict[int, str]:
        path = Path(class_names_path) if class_names_path else MODELS_DIR / "class_names.json"
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as file:
                    parsed = {int(index): str(name) for index, name in json.load(file).items()}
                if set(parsed) == set(DEFAULT_CLASS_NAMES):
                    return parsed
            except (OSError, ValueError, TypeError) as error:
                logger.warning("Unable to read class mapping at %s: %s", path, error)
        return DEFAULT_CLASS_NAMES.copy()

    def _load_model(self) -> None:
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as error:
            raise RuntimeError(f"Unable to load MRI model weights from {self.model_path}: {error}") from error

    def predict(self, image_input: SUPPORTED_IMAGE_TYPES) -> Dict[str, Any]:
        """Return the model class, confidence, and softmax distribution for one image."""
        if self.model is None:
            return self._error(
                "MRI model weights are unavailable. Set CLINIQ_MRI_MODEL_PATH or place "
                f"'{DEFAULT_CHECKPOINT_NAME}' in '{MODELS_DIR}'."
            )
        try:
            tensor = preprocess_mri_image(image_input, target_size=DEFAULT_TARGET_SIZE)
            output = np.asarray(self.model.predict(tensor, verbose=0), dtype=np.float32)
            if output.shape != (1, len(self.class_names)):
                raise ValueError(f"Unexpected model output shape {output.shape}; expected (1, {len(self.class_names)}).")
            values = output[0]
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Model returned invalid probability values.")
            total = float(values.sum())
            if not np.isclose(total, 1.0, atol=1e-3):
                raise ValueError(f"Model output does not sum to 1 (received {total:.6f}).")
            class_index = int(np.argmax(values))
            return {
                "success": True,
                "prediction": self.class_names[class_index],
                "class_index": class_index,
                "confidence": float(values[class_index]),
                "probabilities": {self.class_names[index]: float(value) for index, value in enumerate(values)},
                "error": None,
            }
        except Exception as error:
            logger.warning("MRI inference failed: %s", error)
            return self._error(f"Image processing or inference failed: {error}")

    def predict_batch(self, images: Sequence[SUPPORTED_IMAGE_TYPES]) -> List[Dict[str, Any]]:
        return [self.predict(image) for image in images]

    def predict_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        return self.predict(file_path)

    def predict_bytes(self, raw_bytes: Union[bytes, bytearray]) -> Dict[str, Any]:
        return self.predict(raw_bytes)


_GLOBAL_PREDICTOR: Optional[MRIPredictor] = None


def get_mri_predictor(model_path: Optional[Union[str, Path]] = None, reload_model: bool = False, **deprecated_options: Any) -> MRIPredictor:
    """Return a cached predictor so application reruns do not reload model weights."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or reload_model:
        _GLOBAL_PREDICTOR = MRIPredictor(model_path=model_path, **deprecated_options)
    return _GLOBAL_PREDICTOR


def predict_mri(image_input: SUPPORTED_IMAGE_TYPES, model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Convenience function for one MRI image classification."""
    return get_mri_predictor(model_path=model_path).predict(image_input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain MRI image-classification inference")
    parser.add_argument("--image", "-i", required=True, help="Path to a JPG, JPEG, or PNG image.")
    parser.add_argument("--model", "-m", default=None, help="Optional .keras or .h5 weights path.")
    args = parser.parse_args()
    print(json.dumps(predict_mri(args.image, model_path=args.model), indent=2))


if __name__ == "__main__":
    main()
