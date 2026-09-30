# ClinIQ — Brain MRI Inference Pipeline (Phase 6 & 7)

Production-grade standalone computer vision inference pipeline for multi-class Brain MRI tumor classification.

---

## 1. Supported Modality & Classes

- **Modality**: Brain Magnetic Resonance Imaging (T1/T2-weighted axial/coronal/sagittal slices)
- **Architecture**: Xception (Selective Fine-Tuning on Exit Flow Block 14)
- **Input Resolution**: `299x299x3` (aspect-ratio preserved with uniform black borders)
- **Classes**:
  - `0`: **glioma**
  - `1`: **meningioma**
  - `2`: **notumor** (Healthy scan)
  - `3`: **pituitary**

---

## 2. Directory Structure

```text
CV feature/
├── models/
│   ├── class_names.json                     # Class index to label mapping
│   ├── preprocess_config.json               # Preprocessing parameters (299x299, Lanczos, padding)
│   ├── mri_model_metadata.json              # Model version, test metrics, and thresholds
│   └── mri_xception_selective_finetuning_best.keras # Checkpoint downloaded from Kaggle
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── mri_preprocessing.py             # Image decoding, aspect-ratio padding, [-1, 1] scaling
│   └── inference/
│       ├── __init__.py
│       ├── __main__.py                      # CLI runner
│       └── mri_predictor.py                 # MRIPredictor OOP class & predict_mri API
└── tests/
    └── test_mri_inference.py                # 16 unit and integration tests (100% passing)
```

---

## 3. Quick Start

### A. Python API Usage

```python
from pathlib import Path
from src.inference.mri_predictor import predict_mri, MRIPredictor

# Option 1: Convenience Function (uses cached singleton)
image_path = "data/raw/mri/Testing/glioma/Te-gl_1.jpg"
result = predict_mri(image_path)
print(result)

# Option 2: Object-Oriented Predictor
predictor = MRIPredictor(
    model_path="models/mri_xception_selective_finetuning_best.keras",
    confidence_threshold=0.85
)

# Accepts file paths, raw bytes (from web uploads), or PIL Images
prediction = predictor.predict(image_path)
```

### B. Standard Output Schema

```json
{
  "success": true,
  "prediction": "glioma",
  "class_index": 0,
  "confidence": 0.9412,
  "low_confidence": false,
  "probabilities": {
    "glioma": 0.9412,
    "meningioma": 0.0315,
    "notumor": 0.0073,
    "pituitary": 0.0200
  },
  "confidence_threshold": 0.85,
  "is_pathology": true,
  "warning": null,
  "error": null
}
```

If the highest confidence is below `0.85`:
```json
{
  "low_confidence": true,
  "warning": "Confidence score (74.20%) is below the clinical verification threshold (85.00%). Manual specialist review is recommended."
}
```

### C. Command Line (CLI) Usage

```bash
# Direct script
python src/inference/mri_predictor.py --image "data/raw/mri/Testing/glioma/Te-gl_1.jpg"

# Or as a module
python -m src.inference --image "path/to/scan.jpg" --threshold 0.85
```

---

## 4. Running Tests

Run the full pytest suite from the project root:

```bash
python -m pytest "CV feature/tests/test_mri_inference.py" -v
```
