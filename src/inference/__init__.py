"""
Inference modules for ClinIQ Computer Vision models.
"""

from .mri_predictor import (
    MRIPredictor,
    predict_mri,
    get_mri_predictor,
    DEFAULT_CLASS_NAMES,
)

__all__ = [
    "MRIPredictor",
    "predict_mri",
    "get_mri_predictor",
    "DEFAULT_CLASS_NAMES",
]
