"""Streamlit interface for the existing Brain MRI classification pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.mri_predictor import MRIPredictor, find_default_model_path
from src.preprocessing.mri_preprocessing import load_image_to_pil


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: Optional[str]) -> MRIPredictor:
    """Cache the heavy Keras model across Streamlit reruns."""
    return MRIPredictor(model_path=model_path)


def main() -> None:
    st.set_page_config(page_title="ClinIQ | Brain MRI Classifier", page_icon="🧠", layout="centered")
    st.title("Brain MRI image classifier")
    st.caption("AI research prototype — image classification only")
    st.warning(
        "This system is an AI research prototype for image classification. "
        "The output is not a medical diagnosis or a substitute for professional medical evaluation."
    )

    uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Choose a JPG, JPEG, or PNG image to preview it before analysis.")
        return

    image_bytes = uploaded_file.getvalue()
    try:
        preview = load_image_to_pil(image_bytes)
    except (OSError, TypeError, ValueError) as error:
        st.error(f"The uploaded file is not a readable image: {error}")
        return

    st.image(preview, caption=f"Uploaded image: {uploaded_file.name}", use_container_width=True)
    st.caption(f"Original image: {preview.width} × {preview.height}; converted to RGB for the model.")

    if not st.button("Analyze image", type="primary"):
        return

    # Keep preview/upload available even when a clone does not include model
    # binaries. Inference itself never falls back to invented demo predictions.
    model_path = find_default_model_path()
    if model_path is None:
        st.error(
            "Cannot analyze this image because MRI model weights are unavailable. "
            "Place the trained `.keras` or `.h5` file in `models/`, or set the "
            "`CLINIQ_MRI_MODEL_PATH` environment variable and restart the app."
        )
        st.info("Model binaries are intentionally not committed to this repository.")
        return

    try:
        predictor = load_predictor(str(model_path))
    except Exception:
        st.error("The MRI model could not be loaded. Check the model file and TensorFlow installation.")
        return

    with st.spinner("Running model inference..."):
        result = predictor.predict(image_bytes)

    if not result["success"]:
        st.error(result["error"])
        return

    st.subheader("Model classification result")
    result_columns = st.columns(2)
    result_columns[0].metric("Predicted class", result["prediction"])
    result_columns[1].metric("Model confidence", f"{result['confidence'] * 100:.1f}%")
    st.caption("Model confidence is the selected softmax output, not a clinical probability.")

    probability_table = pd.DataFrame(
        {
            "Class": list(result["probabilities"]),
            "Model confidence (%)": [round(score * 100, 2) for score in result["probabilities"].values()],
        }
    ).set_index("Class")
    st.subheader("Class probabilities")
    # Keep exact values visible; chart axis labels can be compressed on small screens.
    for class_name, row in probability_table.iterrows():
        st.write(f"**{class_name}:** {row['Model confidence (%)']:.2f}%")
    st.bar_chart(probability_table, horizontal=True)


if __name__ == "__main__":
    main()
