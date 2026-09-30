# ClinIQ — Brain MRI Tumor Classification System 🧠

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Modality-Brain%20MRI%20(Validated)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **ClinIQ Computer Vision Subsystem**: An end-to-end, clinically safe deep learning pipeline for multi-class Brain MRI tumor classification and pathology triage. Developed as the Computer Vision core for the ClinIQ AI-Powered Healthcare Marketplace graduation project.

---

## 📌 Executive Summary & Key Results

| Metric | Result | Clinical Significance |
| :--- | :---: | :--- |
| **Held-Out Test Accuracy** | **92.81%** | Evaluated on 1,600 strictly untouched test scans (400 per class) |
| **Macro Average F1-Score** | **0.9263** | Balanced performance across both common and rare tumor morphologies |
| **Tumor Sensitivity (Recall)** | **98.00%** | **1,176 / 1,200** pathological scans detected (only 2.0% false negatives) |
| **Normal Specificity** | **100.00%** | **400 / 400** healthy brains correctly verified (**0 false alarms**) |
| **Inference Safety Threshold** | **0.85** | Flagging ambiguous predictions ($<85\%$) for mandatory specialist review |

---

## 🎯 Supported Pathology Classes

The model classifies axial, coronal, and sagittal brain MRI scans across four distinct categories:

1. **`glioma`**: Intra-axial neuroepithelial tumors originating from glial tissue.
2. **`meningioma`**: Extra-axial tumors arising from the arachnoid layer of the meninges.
3. **`pituitary`**: Sellar and parasellar tumors of the pituitary gland.
4. **`notumor`**: Healthy brain anatomy exhibiting intact symmetry and absence of mass lesions.

---

## 📂 Project Repository Structure

```text
CV feature/
├── app.py                                   # Interactive Streamlit Web Demonstration
├── requirements.txt                         # Pinned production & research dependencies
├── .gitignore                               # Git rules preventing 100MB+ file push errors
├── README.md                                # Comprehensive project documentation
│
├── models/                                  # Packaged artifacts & metadata
│   ├── class_names.json                     # Index-to-label class mapping
│   ├── preprocess_config.json               # Input resolution and normalization config
│   ├── mri_model_metadata.json              # Versioning, hyperparameters & test metrics
│   └── mri_xception_selective_finetuning_best.keras # Checkpoint file (downloaded from Kaggle)
│
├── notebooks/                               # Reproducible Phase-by-Phase Research Notebooks
│   ├── 01_mri_dataset_verification.ipynb    # Phase 1 & 1.5: Image integrity & duplicate audit
│   ├── 02_mri_preprocessing.ipynb           # Phase 2: Stratified train/val split & padding
│   ├── 03_mri_training.ipynb                # Phase 3: Frozen Xception baseline (92.14% val)
│   ├── 04_mri_fine_tuning.ipynb             # Phase 4: Selective fine-tuning (95.80% val)
│   └── 05_mri_final_evaluation.ipynb        # Phase 5: Held-out test evaluation (92.81% test)
│
├── results/                                 # Evaluation figures & scientific reports
│   ├── figures/                             # Training curves, confusion matrices, triage plots
│   └── metrics/                             # CSV history logs and text evaluation reports
│
├── src/                                     # Production-Grade Modular Codebase
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── mri_preprocessing.py             # Aspect-ratio padding & Xception scaler
│   └── inference/
│       ├── __init__.py
│       ├── __main__.py                      # Package CLI runner
│       └── mri_predictor.py                 # MRIPredictor OOP class & predict_mri API
│
└── tests/
    └── test_mri_inference.py                # 16 unit & integration tests (100% passing)
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone the repository and install required packages:

```bash
git clone https://github.com/<your-username>/cliniq-brain-mri.git
cd cliniq-brain-mri

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Streamlit Web UI

Run the user-friendly web interface:

```bash
streamlit run app.py
```

The web application opens automatically in your browser at `http://localhost:8501`, featuring:
- **Instant Scan Upload**: Upload any JPG/PNG brain MRI scan.
- **Benchmark Sample Selector**: Test immediately using pre-loaded glioma, meningioma, pituitary, and normal scans with a single click.
- **Aspect-Ratio Padded Inspection**: Visualizes original vs. preprocessed $299 \times 299$ padded views.
- **Safety Thresholding**: Live slider to simulate clinical review flags.
- **Multi-Class Probability Distribution**: Interactive charts displaying confidence across all conditions.

---

### 3. Python API & Standalone Inference

Use the modular inference pipeline in your own scripts or backend services:

```python
from src.inference.mri_predictor import predict_mri

# Run inference on any image path, byte stream, or PIL Image
result = predict_mri("data/raw/mri/Testing/glioma/Te-gl_1.jpg")

print(result)
```

**Standardized Output Schema**:
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

---

### 4. Command Line (CLI) Usage

Test scans directly from terminal:

```bash
python src/inference/mri_predictor.py --image "path/to/scan.jpg" --threshold 0.85
```

---

### 5. Running the Test Suite

Run the full automated test suite:

```bash
python -m pytest tests/ -v
```

All 16 unit tests validate input decoding (RGB, Grayscale, RGBA, Bytes, NumPy), aspect-ratio padding, Xception normalization, mock predictions, thresholding logic, and error resilience.

---

## 🔬 Model Architecture & Training Methodology

1. **Backbone**: Xception pretrained on ImageNet ($299 \times 299 \times 3$ input).
2. **Selective Fine-Tuning**:
   - Exit Flow **Block 14** (layers 126–131, 2,048 feature channels) selectively unfrozen.
   - Entry Flow and Middle Flow (Blocks 1–12) strictly frozen.
   - All `BatchNormalization` layers kept frozen in inference mode to preserve moving statistics.
   - Trainable parameters: **`4,749,828`** (~22.8% of total model capacity).
3. **Optimization**:
   - Learning Rate: $\eta = 10^{-5}$ (`1e-5`) with Adam optimizer.
   - Objective: Sparse Categorical Crossentropy.
   - Validation Accuracy jumped from **92.14%** (Phase 3 baseline) to **95.80%** (Phase 4).

---

## 🛡️ Clinical AI Safety & Ethical Guardrails

- **Confidence Thresholding**: Predictions with confidence $< 0.85$ are flagged with `low_confidence: True` and trigger an automated warning for secondary specialist review.
- **Triage Flagging**: Distinguishes between healthy scans (`is_pathology: False`) and suspected lesions (`is_pathology: True`).
- **Academic Disclaimer**:
  > **Important Medical Notice**: This software is developed strictly for academic and educational evaluation as part of a computer science graduation thesis. It does not provide medical diagnosis, clinical treatment plans, or autonomous healthcare decisions. All clinical findings must be verified by a board-certified radiologist.

---

## 👥 Authors & Acknowledgments

- **Developer**: Computer Science Graduation Project Team (ClinIQ)
- **Modality**: Brain Tumor MRI Dataset (7,200 balanced multi-source scans)
- **Supervision**: Department of Computer Science & AI Engineering
