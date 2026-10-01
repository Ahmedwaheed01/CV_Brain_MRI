# ClinIQ Computer Vision

Computer Vision component of an AI-based online-clinic graduation project.

The currently implemented component is **Brain MRI image classification**. Future work may add independent Bone Fracture X-ray and Skin Injury models; neither is implemented in this repository yet.

> This is a research and educational prototype. It classifies images and returns model confidence. It is not a medical diagnostic system and does not provide treatment, triage, or clinical advice.

## Current MRI model

The MRI model predicts one dataset class:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

It is an Xception model pretrained on ImageNet and selectively fine-tuned at the top of the backbone. Its image flow is:

```text
Uploaded image
  -> RGB conversion
  -> aspect-ratio-preserving resize + black padding (Lanczos)
  -> 299 x 299
  -> Xception normalization [-1, 1]
  -> Xception + 4-class softmax
  -> predicted class + model confidence
```

Model confidence is the selected softmax output. It is not a clinical probability.

## Evaluation results

- Validation accuracy: **95.80%** on the Phase 2 validation split.
- Final untouched test accuracy: **92.81%** on the original 1,600-image Testing split.

The original dataset’s train/test folders show substantial perceptual and structural similarity. Patient-level leakage cannot be confirmed because patient identifiers are unavailable. Additionally, known exact duplicate groups cross the standard train/validation split, so validation performance may be optimistic. See `results/metrics/` for the experiment reports.

## Repository layout

```text
app/streamlit_app.py             Streamlit MRI demonstration
mri/config.py                    Portable MRI module settings
mri/service.py                   Reusable MRI model loading and prediction service
mri/router.py                    FastAPI APIRouter for the shared AI service
mri/schemas.py                   API request/response schemas
mri/tests/                       MRI module API contract tests
src/preprocessing/               Existing reusable RGB/padding/Xception preprocessing
src/inference/                   Backward-compatible Python inference imports
models/*.json                    Lightweight class, preprocessing, and model metadata
tests/                           Preprocessing and inference tests
results/                         Reproducible training/evaluation artifacts
```

## Shared AI-service integration

The MRI repository does **not** create its own FastAPI application. In the future
shared `cliniq-ai-service/main.py`, include this module's router alongside Voice
and other modules:

```python
from fastapi import FastAPI
from mri.router import router as mri_router

app = FastAPI()
app.include_router(mri_router)
```

This exposes `POST /mri/predict` for a `file` upload (JPG, JPEG, or PNG) and
`GET /mri/health` for model availability. The prediction response contains only
the class, selected softmax model confidence, and all class scores.

Set `CLINIQ_MRI_MODEL_PATH` to an absolute path to the ignored `.keras` or `.h5`
weights when the shared service stores the checkpoint outside `models/`. See
`.env.example`; do not commit the actual `.env` file.

## Run locally

Create a virtual environment, install dependencies, and place the trained model weights locally. The repository deliberately excludes large model binaries and raw medical images.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

The app looks for the checkpoint named in `models/mri_model_metadata.json` under `models/`, or for a path in the `CLINIQ_MRI_MODEL_PATH` environment variable. No model-hosting URL is assumed or provided.

If weights are unavailable, the app shows a clear error and never fabricates a prediction.

## Python inference

```python
from src.inference.mri_predictor import predict_mri

result = predict_mri("path/to/mri_image.jpg")
print(result)
```

The returned result contains `prediction`, `confidence`, and probabilities for all four classes. It contains no clinical or diagnostic interpretation.

## Testing

```bash
python -m pytest
```

The test suite covers RGB, grayscale, RGBA, and palette image conversion; padding and normalization; corrupt-image errors; predictor errors; probability-sum validation; and the MRI FastAPI upload contract using a mock model. Run it locally to verify your environment.

## Data and model files

- Raw medical images are not included and are ignored by Git.
- `.keras`, `.h5`, `.weights.h5`, `.pt`, `.pth`, and `.onnx` model binaries are ignored by Git.
- JSON metadata files remain versioned for reproducibility.

Do not commit raw datasets, extracted images, archives, secrets, or large model binaries.
