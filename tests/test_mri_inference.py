"""
ClinIQ Computer Vision - MRI Inference Pipeline Test Suite
==========================================================
Comprehensive tests for preprocessing, aspect-ratio padding, Xception normalization,
OOP predictor, thresholding logic, clinical triage flags, and fault-tolerance.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure CV feature root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.mri_preprocessing import (
    load_image_to_pil,
    pad_and_resize,
    normalize_xception,
    preprocess_mri_image,
    preprocess_mri_batch,
    DEFAULT_TARGET_SIZE,
)
from src.inference.mri_predictor import (
    MRIPredictor,
    predict_mri,
    DEFAULT_CONFIDENCE_THRESHOLD,
)


# ==============================================================================
# 1. Preprocessing Unit Tests
# ==============================================================================

def test_load_image_to_pil_rgb():
    """Verify RGB PIL images are loaded directly and remain in RGB mode."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = load_image_to_pil(img)
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_load_image_to_pil_grayscale():
    """Verify Grayscale ('L') images are converted safely to 3-channel RGB."""
    img = Image.new("L", (80, 80), color=128)
    result = load_image_to_pil(img)
    assert result.mode == "RGB"
    assert result.size == (80, 80)
    # R, G, B channels should all be equal to 128
    assert result.getpixel((10, 10)) == (128, 128, 128)


def test_load_image_to_pil_rgba():
    """Verify RGBA images are converted safely to RGB."""
    img = Image.new("RGBA", (50, 50), color=(255, 128, 0, 200))
    result = load_image_to_pil(img)
    assert result.mode == "RGB"


def test_load_image_to_pil_raw_bytes():
    """Verify raw image bytes (e.g. from an API request) decode accurately."""
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    result = load_image_to_pil(raw_bytes)
    assert result.mode == "RGB"
    assert result.size == (64, 64)


def test_load_image_to_pil_numpy_array():
    """Verify numpy arrays (H, W, 3) or (H, W) decode into RGB PIL images."""
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[:, :] = [50, 100, 150]
    result = load_image_to_pil(arr)
    assert result.mode == "RGB"
    assert result.size == (120, 120)


def test_load_image_invalid_inputs():
    """Verify invalid inputs raise descriptive exceptions."""
    with pytest.raises(ValueError, match="Image input cannot be None"):
        load_image_to_pil(None)

    with pytest.raises(ValueError, match="Image byte buffer is empty"):
        load_image_to_pil(b"")

    with pytest.raises(ValueError, match="Failed to decode image"):
        load_image_to_pil(b"not_a_valid_image_header")

    with pytest.raises(FileNotFoundError):
        load_image_to_pil("non_existent_file_path_12345.jpg")


def test_pad_and_resize_aspect_ratio_preservation():
    """
    Verify non-square images are padded with uniform black borders (0, 0, 0)
    to exactly (299, 299) without distorting the internal aspect ratio.
    """
    # Create wide non-square image: 400 wide x 200 tall
    wide_img = Image.new("RGB", (400, 200), color=(200, 200, 200))
    padded = pad_and_resize(wide_img, target_size=(299, 299))

    assert padded.size == (299, 299)
    assert padded.mode == "RGB"

    # Top-center pixel should be black padding
    top_pixel = padded.getpixel((149, 5))
    assert top_pixel == (0, 0, 0), f"Expected black border, got {top_pixel}"

    # Center pixel should contain image content
    center_pixel = padded.getpixel((149, 149))
    assert center_pixel == (200, 200, 200)


def test_normalize_xception():
    """Verify Xception normalization scales [0, 255] strictly into [-1.0, 1.0]."""
    test_arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    norm = normalize_xception(test_arr)

    np.testing.assert_allclose(norm, np.array([-1.0, 0.0, 1.0], dtype=np.float32), atol=1e-5)


def test_preprocess_mri_image_end_to_end_real_file():
    """
    Test end-to-end preprocessing on an actual test dataset image file if present.
    """
    sample_file = PROJECT_ROOT / "data" / "raw" / "mri" / "Testing" / "glioma" / "Te-gl_1.jpg"
    if sample_file.exists():
        tensor = preprocess_mri_image(sample_file)
        assert tensor.shape == (1, 299, 299, 3)
        assert tensor.dtype == np.float32
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0
    else:
        # Fallback synthetic test
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        tensor = preprocess_mri_image(img)
        assert tensor.shape == (1, 299, 299, 3)
        assert tensor.dtype == np.float32
        assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


def test_preprocess_mri_batch():
    """Verify batch preprocessing produces tensor of shape (N, 299, 299, 3)."""
    img1 = Image.new("RGB", (100, 100), color=(10, 10, 10))
    img2 = Image.new("RGB", (200, 150), color=(20, 20, 20))
    batch_tensor = preprocess_mri_batch([img1, img2])

    assert batch_tensor.shape == (2, 299, 299, 3)
    assert batch_tensor.dtype == np.float32


# ==============================================================================
# 2. Predictor & Mock Model Tests
# ==============================================================================

class MockKerasModel:
    """Mock Keras model simulating forward pass predictions."""
    def __init__(self, mocked_probabilities: Sequence[float]):
        self.mocked_probabilities = np.array(mocked_probabilities, dtype=np.float32)

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        # Return batch of predictions matching batch dimension of x
        batch_size = x.shape[0]
        return np.tile(self.mocked_probabilities, (batch_size, 1))


def test_predictor_high_confidence_glioma():
    """Verify high-confidence glioma prediction produces clean, expected output."""
    mock_probs = [0.9650, 0.0150, 0.0050, 0.0150]  # glioma, meningioma, notumor, pituitary
    mock_model = MockKerasModel(mock_probs)
    predictor = MRIPredictor(model_instance=mock_model, confidence_threshold=0.85)

    dummy_img = Image.new("RGB", (256, 256), color=(50, 50, 50))
    result = predictor.predict(dummy_img)

    assert result["success"] is True
    assert result["prediction"] == "glioma"
    assert result["class_index"] == 0
    assert result["confidence"] == 0.9650
    assert result["low_confidence"] is False
    assert result["is_pathology"] is True
    assert result["warning"] is None
    assert result["error"] is None
    assert "probabilities" in result
    assert result["probabilities"]["glioma"] == pytest.approx(0.9650, abs=1e-4)


def test_predictor_low_confidence_triggers_safety_warning():
    """Verify ambiguous probabilities (< 0.85) trigger the low_confidence flag and review warning."""
    # Ambiguous prediction: highest is 0.55 for meningioma (< 0.85 threshold)
    mock_probs = [0.35, 0.55, 0.05, 0.05]
    mock_model = MockKerasModel(mock_probs)
    predictor = MRIPredictor(model_instance=mock_model, confidence_threshold=0.85)

    dummy_img = Image.new("RGB", (256, 256), color=(50, 50, 50))
    result = predictor.predict(dummy_img)

    assert result["success"] is True
    assert result["prediction"] == "meningioma"
    assert result["confidence"] == 0.55
    assert result["low_confidence"] is True
    assert result["warning"] is not None
    assert "below the clinical verification threshold" in result["warning"]


def test_predictor_healthy_scan_triage_flag():
    """Verify notumor scans are marked as non-pathology (is_pathology=False)."""
    mock_probs = [0.01, 0.01, 0.97, 0.01]  # notumor is index 2
    mock_model = MockKerasModel(mock_probs)
    predictor = MRIPredictor(model_instance=mock_model)

    dummy_img = Image.new("RGB", (256, 256), color=(50, 50, 50))
    result = predictor.predict(dummy_img)

    assert result["success"] is True
    assert result["prediction"] == "notumor"
    assert result["class_index"] == 2
    assert result["is_pathology"] is False


def test_predictor_graceful_error_handling_corrupt_input():
    """
    Verify predictor handles unreadable or corrupt payloads gracefully
    without raising an unhandled exception or crashing.
    """
    mock_model = MockKerasModel([0.25, 0.25, 0.25, 0.25])
    predictor = MRIPredictor(model_instance=mock_model)

    result = predictor.predict(b"corrupted_invalid_image_stream")

    assert result["success"] is False
    assert result["prediction"] is None
    assert result["confidence"] == 0.0
    assert result["low_confidence"] is True
    assert result["error"] is not None
    assert "Image processing or inference failed" in result["error"]


def test_predictor_predict_batch():
    """Verify predict_batch correctly processes multiple inputs."""
    mock_probs = [0.10, 0.10, 0.10, 0.70]  # pituitary is index 3
    mock_model = MockKerasModel(mock_probs)
    predictor = MRIPredictor(model_instance=mock_model)

    img1 = Image.new("RGB", (100, 100), color=(10, 10, 10))
    img2 = Image.new("RGB", (100, 100), color=(20, 20, 20))
    results = predictor.predict_batch([img1, img2])

    assert len(results) == 2
    for res in results:
        assert res["success"] is True
        assert res["prediction"] == "pituitary"
        assert res["confidence"] == 0.70


def test_unweighted_predictor_informative_error():
    """Verify an MRIPredictor instantiated without a weights file returns a clear error."""
    predictor = MRIPredictor(model_path="non_existent_weights.keras")
    dummy_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    result = predictor.predict(dummy_img)

    assert result["success"] is False
    assert "Model checkpoint is not loaded" in result["error"]
