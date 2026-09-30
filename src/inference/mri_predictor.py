"""
ClinIQ Computer Vision - MRI Tumor Inference Pipeline
=====================================================
Production-grade standalone predictor for Brain MRI tumor classification.

Features:
- Encapsulated OOP predictor with cached model loading (`MRIPredictor`).
- Functional convenience API (`predict_mri`) with singleton instance caching.
- Standardized dictionary output format with probability distribution and confidence scores.
- Safety thresholding: Flags low-confidence predictions (< 0.85) for human clinical review.
- High fault-tolerance: Catches malformed images, missing files, or corrupt bytes gracefully.
- CLI support for quick ad-hoc testing from the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mri_predictor")

# Import preprocessing utilities
if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.preprocessing.mri_preprocessing import (
        SUPPORTED_IMAGE_TYPES,
        preprocess_mri_image,
        preprocess_mri_batch,
        DEFAULT_TARGET_SIZE,
    )
else:
    from ..preprocessing.mri_preprocessing import (
        SUPPORTED_IMAGE_TYPES,
        preprocess_mri_image,
        preprocess_mri_batch,
        DEFAULT_TARGET_SIZE,
    )

# Default paths
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PACKAGE_ROOT / "models"
DEFAULT_CHECKPOINT_NAME = "mri_xception_selective_finetuning_best.keras"
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",
}


def find_default_model_path() -> Optional[Path]:
    """
    Attempts to locate the trained model checkpoint in standard workspace locations
    or environment variables.
    """
    # 1. Environment variable override
    env_path = os.getenv("CLINIQ_MRI_MODEL_PATH") or os.getenv("MRI_MODEL_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    # 2. Standard models directory
    candidate = MODELS_DIR / DEFAULT_CHECKPOINT_NAME
    if candidate.exists():
        return candidate

    # 3. Any .keras file in models directory
    if MODELS_DIR.exists():
        keras_files = list(MODELS_DIR.glob("*.keras")) + list(MODELS_DIR.glob("*.h5"))
        if keras_files:
            return keras_files[0]

    return None


class MRIPredictor:
    """
    Production predictor for Brain MRI Tumor Classification.

    Attributes
    ----------
    model_path : Path | None
        Path to the serialized Keras model (.keras).
    confidence_threshold : float
        Probability threshold below which predictions are flagged as low confidence.
    class_names : Dict[int, str]
        Index-to-class-label dictionary.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_instance: Any = None,
    ) -> None:
        """
        Initializes the MRI Predictor.

        Parameters
        ----------
        model_path : str | Path, optional
            Path to the saved `.keras` checkpoint file. If None, auto-discovery is performed.
        class_names_path : str | Path, optional
            Path to `class_names.json`. If None, standard locations are checked.
        confidence_threshold : float
            Confidence threshold (0.0 to 1.0) below which predictions trigger review warning.
        model_instance : Any, optional
            Pre-loaded Keras model or mock model instance (useful for testing or dependency injection).
        """
        self.confidence_threshold = float(confidence_threshold)
        self.class_names = self._load_class_names(class_names_path)
        self.model = None

        if model_instance is not None:
            self.model = model_instance
            self.model_path = Path("in_memory_instance")
            logger.info("Initialized MRIPredictor with pre-loaded model instance.")
        else:
            resolved_path = Path(model_path) if model_path else find_default_model_path()
            self.model_path = resolved_path
            if self.model_path and self.model_path.exists():
                self._load_model()
            else:
                logger.warning(
                    f"Model checkpoint not found at: {self.model_path}. "
                    f"Predictor initialized in unweighted state. "
                    f"Place '{DEFAULT_CHECKPOINT_NAME}' into '{MODELS_DIR}' or pass explicit path to predict."
                )

    def _load_class_names(self, class_names_path: Optional[Union[str, Path]]) -> Dict[int, str]:
        """Loads class mapping from json or falls back to default."""
        target_path = Path(class_names_path) if class_names_path else MODELS_DIR / "class_names.json"
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    raw_mapping = json.load(f)
                return {int(k): str(v) for k, v in raw_mapping.items()}
            except Exception as e:
                logger.warning(f"Failed to parse class_names.json from {target_path} ({e}). Using default.")
        return DEFAULT_CLASS_NAMES

    def _load_model(self) -> None:
        """Loads and compiles Keras model for inference."""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info(f"Loading Keras MRI model from: {self.model_path}")
        try:
            # We import tensorflow / keras lazily on load
            import tensorflow as tf
            # compile=False optimizes loading time for pure inference
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info("MRI Model successfully loaded and ready for inference.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load Keras model from {self.model_path}: {exc}") from exc

    def predict(self, image_input: SUPPORTED_IMAGE_TYPES) -> Dict[str, Any]:
        """
        Executes classification on a single MRI image input.

        Parameters
        ----------
        image_input : str | Path | bytes | bytearray | io.BytesIO | Image.Image | np.ndarray
            The image to classify.

        Returns
        -------
        Dict[str, Any]
            Prediction result dictionary containing class, confidence, probabilities,
            and safety threshold warnings.
        """
        if self.model is None:
            return {
                "success": False,
                "prediction": None,
                "class_index": None,
                "confidence": 0.0,
                "low_confidence": True,
                "probabilities": {},
                "confidence_threshold": self.confidence_threshold,
                "is_pathology": None,
                "warning": None,
                "error": (
                    f"Model checkpoint is not loaded. Please ensure '{DEFAULT_CHECKPOINT_NAME}' "
                    f"exists in '{MODELS_DIR}' or provide a valid path to MRIPredictor."
                ),
            }

        try:
            # Preprocess to shape (1, 299, 299, 3)
            tensor = preprocess_mri_image(image_input, target_size=DEFAULT_TARGET_SIZE)

            # Model forward pass
            preds = self.model.predict(tensor, verbose=0)
            probs = preds[0]

            class_idx = int(np.argmax(probs))
            predicted_class = self.class_names.get(class_idx, f"unknown_class_{class_idx}")
            confidence = float(probs[class_idx])

            # Class probability mapping
            prob_dict = {
                self.class_names.get(i, f"class_{i}"): round(float(p), 4)
                for i, p in enumerate(probs)
            }

            # Safety flag
            is_low_conf = confidence < self.confidence_threshold
            warning = None
            if is_low_conf:
                warning = (
                    f"Confidence score ({confidence * 100:.2f}%) is below the clinical "
                    f"verification threshold ({self.confidence_threshold * 100:.2f}%). "
                    "Manual specialist review is recommended."
                )

            # Clinical triage flag (notumor is considered normal scan)
            is_pathology = predicted_class.lower() != "notumor"

            return {
                "success": True,
                "prediction": predicted_class,
                "class_index": class_idx,
                "confidence": round(confidence, 4),
                "low_confidence": is_low_conf,
                "probabilities": prob_dict,
                "confidence_threshold": self.confidence_threshold,
                "is_pathology": is_pathology,
                "warning": warning,
                "error": None,
            }

        except Exception as exc:
            logger.error(f"Inference error encountered: {exc}", exc_info=True)
            return {
                "success": False,
                "prediction": None,
                "class_index": None,
                "confidence": 0.0,
                "low_confidence": True,
                "probabilities": {},
                "confidence_threshold": self.confidence_threshold,
                "is_pathology": None,
                "warning": None,
                "error": f"Image processing or inference failed: {str(exc)}",
            }

    def predict_batch(self, images: Sequence[SUPPORTED_IMAGE_TYPES]) -> List[Dict[str, Any]]:
        """
        Executes classification on a batch of MRI images.

        Parameters
        ----------
        images : Sequence[SUPPORTED_IMAGE_TYPES]
            Sequence of image inputs.

        Returns
        -------
        List[Dict[str, Any]]
            List of prediction result dictionaries.
        """
        if not images:
            return []

        # If any single image might fail preprocessing, process individually or batch
        # Individual execution provides resilient per-sample error isolation:
        return [self.predict(img) for img in images]

    def predict_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Convenience method to predict an image from file path."""
        return self.predict(file_path)

    def predict_bytes(self, raw_bytes: Union[bytes, bytearray]) -> Dict[str, Any]:
        """Convenience method to predict an image from in-memory bytes."""
        return self.predict(raw_bytes)


# ------------------------------------------------------------------------------
# Module-level Cached Singleton Instance
# ------------------------------------------------------------------------------
_GLOBAL_PREDICTOR: Optional[MRIPredictor] = None


def get_mri_predictor(
    model_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    reload_model: bool = False,
) -> MRIPredictor:
    """
    Returns a cached singleton instance of MRIPredictor to avoid re-loading
    heavy model weights on each prediction.
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or reload_model:
        _GLOBAL_PREDICTOR = MRIPredictor(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
    return _GLOBAL_PREDICTOR


def predict_mri(
    image_input: SUPPORTED_IMAGE_TYPES,
    model_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Functional convenience entrypoint for Brain MRI prediction.

    Parameters
    ----------
    image_input : str | Path | bytes | Image.Image | np.ndarray
        Input image.
    model_path : str | Path, optional
        Explicit path to model file.
    confidence_threshold : float
        Threshold for flagging low confidence predictions.

    Returns
    -------
    Dict[str, Any]
        Standardized inference result dictionary.
    """
    predictor = get_mri_predictor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
    return predictor.predict(image_input)


# ------------------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClinIQ Brain MRI Tumor Classification Inference Tool"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the MRI image file (JPEG, PNG, etc.) to analyze.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.keras file).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for warning flag (default: {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )

    args = parser.parse_args()

    result = predict_mri(
        image_input=args.image,
        model_path=args.model,
        confidence_threshold=args.threshold,
    )

    print("\n" + "=" * 50)
    print("CLINIQ MRI INFERENCE RESULT")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
