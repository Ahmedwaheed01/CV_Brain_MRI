# MRI inference module

`mri_predictor.py` is the reusable interface for the trained Brain MRI image-classification model.

It uses the shared preprocessing module rather than duplicating image handling:

```text
input image -> RGB conversion -> aspect-preserving resize + black padding
-> Xception normalization -> model -> class probabilities
```

## Use from Python

```python
from src.inference.mri_predictor import predict_mri

result = predict_mri("path/to/image.jpg")
```

Successful results contain only classification information:

```json
{
  "success": true,
  "prediction": "glioma",
  "class_index": 0,
  "confidence": 0.94,
  "probabilities": {
    "glioma": 0.94,
    "meningioma": 0.03,
    "notumor": 0.01,
    "pituitary": 0.02
  },
  "error": null
}
```

`confidence` is the selected model softmax output; it is not a clinical probability or diagnosis.

## Model weights

Set `CLINIQ_MRI_MODEL_PATH` to a local `.keras`/`.h5` weights file, or place the expected checkpoint in `models/`. Large model binaries are intentionally ignored by Git.

If weights or an image are unavailable, the module returns `success: false` and a readable `error` message.
