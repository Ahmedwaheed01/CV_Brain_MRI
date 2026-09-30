"""
CLI entry point when invoked as a package: `python -m src.inference ...`
"""

from .mri_predictor import main

if __name__ == "__main__":
    main()
