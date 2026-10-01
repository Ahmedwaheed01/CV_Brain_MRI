"""Compatibility imports for the MRI module inference service.

New API integrations should import from ``mri.service``. This module preserves
the previously published Streamlit and Python import paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mri.config import DEFAULT_CHECKPOINT_NAME, MODELS_DIR, find_model_path
from mri.service import DEFAULT_CLASS_NAMES, MRIPredictor, get_mri_predictor, predict_mri


find_default_model_path = find_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain MRI image-classification inference")
    parser.add_argument("--image", "-i", required=True, help="Path to a JPG, JPEG, or PNG image.")
    parser.add_argument("--model", "-m", default=None, help="Optional .keras or .h5 weights path.")
    args = parser.parse_args()
    print(json.dumps(predict_mri(args.image, model_path=args.model), indent=2))


if __name__ == "__main__":
    main()
