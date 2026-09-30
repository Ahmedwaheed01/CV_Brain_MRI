"""Tests for MRI preprocessing and non-clinical classification inference."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.mri_predictor import MRIPredictor
from src.preprocessing.mri_preprocessing import DEFAULT_TARGET_SIZE, load_image_to_pil, normalize_xception, pad_and_resize, preprocess_mri_batch, preprocess_mri_image


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
def test_load_image_to_pil_converts_common_modes_to_rgb(mode: str) -> None:
    loaded = load_image_to_pil(Image.new(mode, (80, 50)))
    assert loaded.mode == "RGB"
    assert loaded.size == (80, 50)


def test_load_image_from_bytes_and_rejects_invalid_data() -> None:
    stream = io.BytesIO()
    Image.new("RGB", (64, 64)).save(stream, format="PNG")
    assert load_image_to_pil(stream.getvalue()).mode == "RGB"
    with pytest.raises(ValueError, match="Failed to decode"):
        load_image_to_pil(b"not an image")


def test_pad_and_resize_preserves_aspect_ratio_with_black_padding() -> None:
    padded = pad_and_resize(Image.new("RGB", (400, 200), (200, 200, 200)))
    assert padded.size == DEFAULT_TARGET_SIZE
    assert padded.getpixel((149, 5)) == (0, 0, 0)
    assert padded.getpixel((149, 149)) == (200, 200, 200)


def test_xception_normalization_and_end_to_end_shapes() -> None:
    np.testing.assert_allclose(normalize_xception(np.array([0.0, 127.5, 255.0])), [-1.0, 0.0, 1.0])
    image = Image.new("RGBA", (200, 150), (12, 34, 56, 128))
    tensor = preprocess_mri_image(image)
    batch = preprocess_mri_batch([image, image])
    assert tensor.shape == (1, 299, 299, 3)
    assert batch.shape == (2, 299, 299, 3)
    assert tensor.dtype == np.float32
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


class MockKerasModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.tile(self.probabilities, (batch.shape[0], 1))


def test_predictor_returns_classification_outputs_and_probabilities_sum_to_one() -> None:
    predictor = MRIPredictor(model_instance=MockKerasModel([0.965, 0.015, 0.005, 0.015]))
    result = predictor.predict(Image.new("RGB", (256, 256)))
    assert result["success"] is True
    assert result["prediction"] == "glioma"
    assert result["confidence"] == pytest.approx(0.965)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert not {"is_pathology", "warning", "low_confidence"} & result.keys()


def test_predictor_accepts_grayscale_and_rgba_images() -> None:
    predictor = MRIPredictor(model_instance=MockKerasModel([0.1, 0.1, 0.1, 0.7]))
    for image in (Image.new("L", (30, 50)), Image.new("RGBA", (50, 30))):
        result = predictor.predict(image)
        assert result["success"] is True
        assert result["prediction"] == "pituitary"


def test_predictor_returns_readable_error_for_corrupt_image_or_missing_model() -> None:
    predictor = MRIPredictor(model_instance=MockKerasModel([0.25] * 4))
    assert predictor.predict(b"corrupt bytes")["success"] is False
    unavailable = MRIPredictor(model_path="missing_model.keras")
    result = unavailable.predict(Image.new("RGB", (10, 10)))
    assert result["success"] is False
    assert "weights are unavailable" in result["error"]
