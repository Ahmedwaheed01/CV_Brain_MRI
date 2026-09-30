"""
ClinIQ Computer Vision - MRI Preprocessing Pipeline
===================================================
Production-grade preprocessing utility for brain MRI image classification.

Handles:
- Arbitrary input types: file path, raw bytes, PIL.Image, numpy.ndarray.
- Diverse color modes (Grayscale 'L', RGBA, Palette 'P') converted safely to RGB.
- Aspect-ratio preserving padding with black borders ((0, 0, 0)) to target size (299, 299).
- Exact Xception input normalization mapping [0, 255] -> [-1.0, 1.0].
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union, Sequence
import numpy as np
from PIL import Image, ImageOps

DEFAULT_TARGET_SIZE = (299, 299)
SUPPORTED_IMAGE_TYPES = Union[str, Path, bytes, bytearray, io.BytesIO, Image.Image, np.ndarray]


def load_image_to_pil(image_input: SUPPORTED_IMAGE_TYPES) -> Image.Image:
    """
    Safely decodes any supported image input into an RGB PIL Image.

    Parameters
    ----------
    image_input : str | Path | bytes | bytearray | io.BytesIO | Image.Image | np.ndarray
        The input image representation.

    Returns
    -------
    Image.Image
        The decoded PIL Image converted to 'RGB' mode.

    Raises
    ------
    ValueError
        If the image input is empty, invalid, or cannot be identified.
    FileNotFoundError
        If a file path is provided but does not exist.
    """
    if image_input is None:
        raise ValueError("Image input cannot be None.")

    # 1. PIL Image already
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    # 2. NumPy array
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy array is empty.")
        if image_input.dtype != np.uint8:
            # Scale or cast if needed
            if image_input.max() <= 1.0 and image_input.min() >= 0.0:
                image_input = (image_input * 255.0).astype(np.uint8)
            else:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image_input)
        return pil_img.convert("RGB")

    # 3. Raw Bytes / Bytearray / BytesIO
    if isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise ValueError("Image byte buffer is empty (0 bytes).")
            stream = io.BytesIO(image_input)
        else:
            stream = image_input

        try:
            stream.seek(0)
            with Image.open(stream) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to decode image from byte stream: {exc}") from exc

    # 4. File Path (str or Path)
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist at path: {path}")
        if not path.is_file():
            raise ValueError(f"Specified path is not a file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Image file is empty (0 bytes): {path}")

        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to open image file '{path}': {exc}") from exc

    raise TypeError(
        f"Unsupported input type '{type(image_input).__name__}'. "
        "Expected str, Path, bytes, io.BytesIO, PIL.Image, or numpy.ndarray."
    )


def pad_and_resize(
    image: Image.Image,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Pads an image with uniform black borders while preserving its native aspect ratio,
    scaling it to the requested target resolution.

    Parameters
    ----------
    image : Image.Image
        The PIL image in RGB mode.
    target_size : tuple[int, int]
        (width, height) desired output resolution (default: (299, 299)).
    resample : Image.Resampling
        Resampling filter (default: LANCZOS).
    fill_color : tuple[int, int, int]
        Background fill color for the padded margins (default: (0, 0, 0)).

    Returns
    -------
    Image.Image
        Padded and scaled PIL image of exact dimensions `target_size`.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    return ImageOps.pad(
        image,
        size=target_size,
        method=resample,
        color=fill_color,
        centering=(0.5, 0.5),
    )


def normalize_xception(image_array: np.ndarray) -> np.ndarray:
    """
    Normalizes an image array using Xception scaling formula:
    `normalized = (image_array / 127.5) - 1.0`
    Maps pixel values from [0.0, 255.0] to [-1.0, 1.0].

    Parameters
    ----------
    image_array : np.ndarray
        Float32 or uint8 array of pixel intensities.

    Returns
    -------
    np.ndarray
        Float32 array scaled to [-1.0, 1.0].
    """
    arr = image_array.astype(np.float32)
    return (arr / 127.5) - 1.0


def preprocess_mri_image(
    image_input: SUPPORTED_IMAGE_TYPES,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    End-to-end preprocessing pipeline for a single MRI image.
    Performs decoding -> RGB conversion -> aspect-ratio padding -> Xception normalization.

    Parameters
    ----------
    image_input : SUPPORTED_IMAGE_TYPES
        Image path, bytes, PIL Image, or numpy array.
    target_size : tuple[int, int]
        Target resolution (width, height), default (299, 299).

    Returns
    -------
    np.ndarray
        Processed 4D batch tensor of shape `(1, target_size[1], target_size[0], 3)`
        with dtype `float32` and values in `[-1.0, 1.0]`.
    """
    pil_img = load_image_to_pil(image_input)
    padded_img = pad_and_resize(pil_img, target_size=target_size)
    img_array = np.asarray(padded_img, dtype=np.float32)
    normalized_array = normalize_xception(img_array)
    # Return batch dimension: (1, H, W, C)
    return np.expand_dims(normalized_array, axis=0)


def preprocess_mri_batch(
    images: Sequence[SUPPORTED_IMAGE_TYPES],
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    End-to-end preprocessing pipeline for a batch of MRI images.

    Parameters
    ----------
    images : Sequence[SUPPORTED_IMAGE_TYPES]
        List or sequence of image inputs.
    target_size : tuple[int, int]
        Target resolution (width, height), default (299, 299).

    Returns
    -------
    np.ndarray
        Processed 4D batch tensor of shape `(N, target_size[1], target_size[0], 3)`
        with dtype `float32` and values in `[-1.0, 1.0]`.
    """
    if not images:
        raise ValueError("Cannot preprocess an empty batch of images.")

    batch_tensors = [preprocess_mri_image(img, target_size=target_size) for img in images]
    return np.concatenate(batch_tensors, axis=0)
