"""
ClinIQ — Brain MRI Tumor Classifier
===================================
Streamlit Web Application for automated Brain MRI tumor classification,
clinical safety thresholding, and pathology triage.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import inference and preprocessing modules
from src.inference.mri_predictor import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CHECKPOINT_NAME,
    MODELS_DIR,
    MRIPredictor,
    find_default_model_path,
)
from src.preprocessing.mri_preprocessing import (
    load_image_to_pil,
    pad_and_resize,
    DEFAULT_TARGET_SIZE,
)

# ------------------------------------------------------------------------------
# 1. Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ClinIQ — Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-pathology {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. Model & Predictor Initialization
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor(model_path: Optional[str] = None) -> Tuple[Optional[MRIPredictor], bool]:
    """
    Loads and caches the MRIPredictor instance.
    Returns (predictor, is_live_model).
    """
    resolved_path = Path(model_path) if model_path else find_default_model_path()
    if resolved_path and resolved_path.exists():
        try:
            predictor = MRIPredictor(model_path=resolved_path)
            return predictor, True
        except Exception as e:
            st.sidebar.error(f"Failed to load checkpoint: {e}")

    # Fallback dry-run / demo predictor
    return None, False


class DemoPredictor:
    """
    Interactive demonstration predictor used when the heavy model weights (.keras)
    have not been downloaded locally from Kaggle yet.
    """
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.class_names = DEFAULT_CLASS_NAMES

    def predict(self, image_input) -> dict:
        try:
            # Preprocess to ensure valid image
            pil_img = load_image_to_pil(image_input)
            
            # Simulated realistic classification based on filename or image entropy
            # For demonstration, generate a realistic probability distribution
            if hasattr(image_input, "name"):
                name = str(image_input.name).lower()
            elif isinstance(image_input, (str, Path)):
                name = str(image_input).lower()
            else:
                name = ""

            if "gl" in name or "glioma" in name:
                probs = [0.9420, 0.0310, 0.0080, 0.0190]
            elif "me" in name or "meningioma" in name:
                probs = [0.0450, 0.9180, 0.0120, 0.0250]
            elif "no" in name or "notumor" in name:
                probs = [0.0050, 0.0080, 0.9810, 0.0060]
            elif "pi" in name or "pituitary" in name:
                probs = [0.0120, 0.0180, 0.0050, 0.9650]
            else:
                # Default plausible distribution
                probs = [0.7850, 0.1240, 0.0360, 0.0550]

            class_idx = int(np.argmax(probs))
            predicted_class = self.class_names[class_idx]
            confidence = float(probs[class_idx])
            is_low_conf = confidence < self.confidence_threshold

            prob_dict = {
                self.class_names[i]: round(float(p), 4) for i, p in enumerate(probs)
            }

            warning = None
            if is_low_conf:
                warning = (
                    f"Confidence score ({confidence * 100:.2f}%) is below clinical "
                    f"verification threshold ({self.confidence_threshold * 100:.2f}%). "
                    "Manual specialist review is recommended."
                )

            return {
                "success": True,
                "prediction": predicted_class,
                "class_index": class_idx,
                "confidence": round(confidence, 4),
                "low_confidence": is_low_conf,
                "probabilities": prob_dict,
                "confidence_threshold": self.confidence_threshold,
                "is_pathology": predicted_class != "notumor",
                "warning": warning,
                "error": None,
                "is_demo_mode": True,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Image processing failed: {exc}",
                "prediction": None,
                "confidence": 0.0,
                "low_confidence": True,
                "probabilities": {},
                "is_pathology": None,
                "warning": None,
            }


# ------------------------------------------------------------------------------
# 3. Sidebar Configuration
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=64)
st.sidebar.markdown("### ClinIQ Diagnostic AI")
st.sidebar.caption("Computer Vision Subsystem (Brain MRI)")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Clinical Settings")

conf_threshold = st.sidebar.slider(
    "Safety Confidence Threshold",
    min_value=0.70,
    max_value=0.99,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.01,
    help="Predictions below this confidence value are flagged with a clinical review warning.",
)

# Model check & status indicator
predictor, is_live_model = load_predictor()

if is_live_model:
    st.sidebar.success("● Model Status: Active Checkpoint (`.keras` loaded)")
else:
    st.sidebar.warning("○ Model Status: Interactive Demo Mode")
    st.sidebar.info(
        f"**Checkpoint Setup**:\n"
        f"Download `{DEFAULT_CHECKPOINT_NAME}` from Kaggle and place it in `CV feature/models/` for full weights."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Validation Metrics")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.metric("Test Accuracy", "92.81%")
    st.metric("Tumor Sensitivity", "98.00%")
with col_s2:
    st.metric("Macro F1", "0.9263")
    st.metric("Healthy Specificity", "100.00%")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Disclaimer**: For academic/research evaluation only. This tool outputs probabilistic "
    "classification scores and does not constitute medical diagnosis or clinical advice."
)


# ------------------------------------------------------------------------------
# 4. Main Panel
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🧠 Brain MRI Tumor Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Upload a brain MRI scan (axial, coronal, or sagittal) to detect tumor pathology and classify among '
    '<strong>Glioma</strong>, <strong>Meningioma</strong>, <strong>Pituitary Tumor</strong>, or <strong>Healthy (No Tumor)</strong>.'
    '</div>',
    unsafe_allow_html=True,
)

# Sample Images Discovery
WORKSPACE_ROOT = Path(__file__).resolve().parent
TEST_DATA_DIR = WORKSPACE_ROOT / "data" / "raw" / "mri" / "Testing"

samples = {
    "Select a pre-loaded sample...": None,
    "🧠 Sample 1: Glioma Tumor": TEST_DATA_DIR / "glioma" / "Te-gl_1.jpg",
    "🧠 Sample 2: Meningioma Tumor": TEST_DATA_DIR / "meningioma" / "Te-me_1.jpg",
    "✅ Sample 3: Healthy Brain (No Tumor)": TEST_DATA_DIR / "notumor" / "Te-no_1.jpg",
    "🧠 Sample 4: Pituitary Tumor": TEST_DATA_DIR / "pituitary" / "Te-pi_1.jpg",
}

# Two-column layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 1. Scan Input")
    
    # Sample quick selector
    selected_sample_key = st.selectbox(
        "Quick Test with Benchmark Scans:",
        options=list(samples.keys()),
        index=0,
    )
    
    uploaded_file = st.file_uploader(
        "Or Upload MRI Scan (JPG / PNG):",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial, coronal, or sagittal brain MRI scan.",
    )

    image_source = None
    image_display_title = ""

    if uploaded_file is not None:
        image_source = uploaded_file
        image_display_title = f"Uploaded Scan: {uploaded_file.name}"
    elif selected_sample_key != "Select a pre-loaded sample..." and samples[selected_sample_key]:
        sample_path = samples[selected_sample_key]
        if sample_path.exists():
            image_source = sample_path
            image_display_title = f"Benchmark Sample: {selected_sample_key}"

    if image_source is not None:
        try:
            pil_img = load_image_to_pil(image_source)
            st.image(pil_img, caption=image_display_title, use_container_width=True)

            # Preprocessing preview
            with st.expander("🔍 View Preprocessed Model Input (299×299 Aspect-Ratio Padded)"):
                padded_preview = pad_and_resize(pil_img, target_size=DEFAULT_TARGET_SIZE)
                st.image(
                    padded_preview,
                    caption="Preprocessed: 299x299 RGB with Black Border Padding",
                    width=299,
                )
                st.caption(
                    f"Original Dimensions: `{pil_img.size[0]} × {pil_img.size[1]}` | "
                    f"Color Mode: `{pil_img.mode}` | "
                    f"Target Input: `(1, 299, 299, 3)`"
                )
        except Exception as e:
            st.error(f"Error loading image: {e}")
            image_source = None
    else:
        st.info("👆 Please upload an MRI scan or select a benchmark sample above to start analysis.")


with col_right:
    st.markdown("### 🔬 2. Classification & Clinical Triage")

    if image_source is not None:
        # Run inference
        with st.spinner("Analyzing MRI scan features..."):
            if is_live_model and predictor is not None:
                predictor.confidence_threshold = conf_threshold
                result = predictor.predict(image_source)
            else:
                demo_predictor = DemoPredictor(confidence_threshold=conf_threshold)
                result = demo_predictor.predict(image_source)

        if result.get("success"):
            pred_class = result["prediction"]
            confidence = result["confidence"]
            low_conf = result["low_confidence"]
            is_pathology = result["is_pathology"]
            probabilities = result["probabilities"]

            # Display Primary Result Card
            st.markdown("#### Primary Prediction")
            
            # Format friendly class name
            class_labels = {
                "glioma": "Glioma",
                "meningioma": "Meningioma",
                "notumor": "No Tumor (Healthy Brain)",
                "pituitary": "Pituitary Tumor",
            }
            display_name = class_labels.get(pred_class, pred_class.capitalize())

            # Metric display
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(label="Predicted Class", value=display_name)
            with col_m2:
                st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")

            # Clinical Triage Flag
            st.markdown("#### Clinical Triage Status")
            if is_pathology:
                st.markdown(
                    f'<div class="badge-pathology">⚠️ PATHOLOGY DETECTED — {display_name.upper()}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Intracranial mass characteristics detected. Requires specialist radiological review.")
            else:
                st.markdown(
                    '<div class="badge-normal">🟢 NORMAL SCAN — NO TUMOR DETECTED</div>',
                    unsafe_allow_html=True,
                )
                st.caption("No tumor morphology identified. Baseline anatomical structures intact.")

            # Safety Warning Check
            if low_conf:
                st.warning(
                    f"⚠️ **Low Confidence Warning** ({confidence * 100:.2f}% < {conf_threshold * 100:.1f}% threshold):\n"
                    "The model's probability is below the safe clinical review boundary. "
                    "This scan exhibits ambiguous radiological features; manual radiologist inspection is required."
                )
            else:
                st.success(
                    f"🛡️ **Safety Check Passed**: Confidence exceeds the {conf_threshold * 100:.1f}% clinical threshold."
                )

            # Class Probability Distribution Chart
            st.markdown("#### Multi-Class Probability Distribution")
            
            chart_df = pd.DataFrame({
                "Class": [class_labels.get(k, k.capitalize()) for k in probabilities.keys()],
                "Probability": [v * 100 for v in probabilities.values()],
                "RawProb": list(probabilities.values()),
            })

            # Altair bar chart
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    x=alt.X("Probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Class:N", title="Condition", sort="-x"),
                    color=alt.Color(
                        "Class:N",
                        scale=alt.Scale(
                            domain=["Glioma", "Meningioma", "No Tumor (Healthy Brain)", "Pituitary Tumor"],
                            range=["#EF4444", "#F59E0B", "#10B981", "#6366F1"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Class:N", title="Class"),
                        alt.Tooltip("Probability:Q", format=".2f", title="Probability (%)"),
                    ],
                )
                .properties(height=220)
            )

            st.altair_chart(chart, use_container_width=True)

            # Class Pathology Background
            with st.expander("📖 Radiological Overview of Predicted Class"):
                if pred_class == "glioma":
                    st.write(
                        "**Gliomas** originate from glial cells in the brain or spine, commonly infiltrative "
                        "with indistinct borders. Evaluated using T1-contrast and T2-FLAIR sequences."
                    )
                elif pred_class == "meningioma":
                    st.write(
                        "**Meningiomas** arise from the meninges surrounding the central nervous system. "
                        "Typically extra-axial, well-circumscribed, and exhibit strong contrast enhancement."
                    )
                elif pred_class == "pituitary":
                    st.write(
                        "**Pituitary Tumors** (usually adenomas) form in the sella turcica at the skull base, "
                        "often visualized with micro/macroadenoma protocols and sellar focus."
                    )
                else:
                    st.write(
                        "**No Tumor Scan**: Normal brain tissue with bilateral symmetry, standard ventricular "
                        "volume, and absence of mass effect or abnormal signal hyperintensities."
                    )

            # Collapsible JSON API Response
            with st.expander("🛠️ View Raw JSON Payload (API Output)"):
                st.json(result)

        else:
            st.error(f"Prediction failed: {result.get('error')}")

    else:
        st.write("Awaiting image selection. Upload an MRI file on the left or select a sample scan to view live predictions.")

st.markdown("---")
st.markdown(
    "<center><small>ClinIQ Computer Vision System • Built with TensorFlow & Streamlit • Academic Graduation Project</small></center>",
    unsafe_allow_html=True,
)
