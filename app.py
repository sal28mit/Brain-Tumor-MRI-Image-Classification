"""
Brain Tumor MRI Classifier - Streamlit application
===================================================
Upload one or more brain MRI images and get the predicted tumour type with a
confidence score for every class.

Run with:   streamlit run app.py
Needs (created by the training notebook):
    models/model_config.json   class names, preprocessing and metrics
    models/*.h5                trained models (custom CNN + transfer-learning models)

This is a decision-support prototype - NOT a certified medical device.
"""
import os
import re
import sys
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quiet TensorFlow start-up logs

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="\U0001F9E0", layout="wide")

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as _tf_error:                       # wrong Python environment / broken TensorFlow install
    st.error("TensorFlow could not be loaded in the Python environment that runs this app.")
    st.markdown(
        "**Python being used:** `" + sys.executable + "`\n\n"
        "Start the app from the environment where TensorFlow works (the one used for the training notebook), e.g. in Anaconda Prompt:\n"
        "```\nconda activate brain_tumor\npython -m streamlit run app.py\n```\n"
        "If `streamlit` is missing there: `pip install streamlit`."
    )
    st.code(str(_tf_error)[-600:])
    st.stop()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "data" / "prediction_history.db"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "bmp"]

st.markdown(
    """
    <style>
    .big-pred {font-size: 1.9rem; font-weight: 700; margin-bottom: 0.2rem;}
    .small-note {color: #6b7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Short, plain-language descriptions shown next to a prediction (educational, not medical advice)
CLASS_INFO = {
    "glioma": "Gliomas start in the glial (supporting) cells of the brain or spinal cord. They range from slow-growing "
              "to aggressive tumours and are normally evaluated further with contrast MRI, biopsy and neuro-oncology review.",
    "meningioma": "Meningiomas arise from the meninges, the membranes that surround the brain. Most are benign and slow-growing; "
                  "size and location decide whether monitoring, surgery or radiotherapy is advised.",
    "pituitary": "Pituitary tumours develop in the pituitary gland at the base of the brain. Many are benign adenomas that can "
                 "affect hormone levels and vision; endocrine tests and specialist review are typical next steps.",
    "none": "No tumour pattern was recognised by the model. This does NOT exclude disease - a radiologist must always review the scan.",
    "other": "A specialist should review this scan and decide on further investigation.",
}


def pretty(name: str) -> str:
    """glioma_tumor -> Glioma Tumor, no_tumor -> No Tumor."""
    return name.replace("_", " ").title()


def info_for(class_name: str) -> str:
    c = class_name.lower()
    for key in ("glioma", "meningioma", "pituitary"):
        if key in c:
            return CLASS_INFO[key]
    if re.match(r"^(no|normal|healthy)", c):
        return CLASS_INFO["none"]
    return CLASS_INFO["other"]


# ----------------------------------------------------------------------------------------------
# Loading configuration and models
# ----------------------------------------------------------------------------------------------
@st.cache_data
def load_config():
    """Read models/model_config.json (written by the training notebook); fall back to scanning the folder."""
    cfg_path = MODEL_DIR / "model_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    names_path = MODEL_DIR / "class_names.json"
    files = sorted(MODEL_DIR.glob("*.h5")) + sorted(MODEL_DIR.glob("*.keras"))
    if not names_path.exists() or not files:
        return None
    models = {f.stem: {"file": f.name, "variant": "", "metrics": {}} for f in files if f.stem != "best_model"}
    return {"class_names": json.loads(names_path.read_text(encoding="utf-8")), "img_size": [224, 224],
            "best_model": next(iter(models), files[0].stem), "best_model_file": files[0].name, "models": models}


@st.cache_resource(show_spinner="Loading model ...")
def load_model(file_name: str):
    """Load a saved model (.h5 first, .keras as a fallback)."""
    path = MODEL_DIR / file_name
    try:
        return keras.models.load_model(str(path), compile=False)
    except Exception as first_error:
        alt = path.with_suffix(".keras" if path.suffix == ".h5" else ".h5")
        if alt.exists():
            return keras.models.load_model(str(alt), compile=False)
        raise first_error


def preprocess_bytes(raw: bytes, img_size) -> np.ndarray:
    """Identical to the training pipeline: decode -> RGB -> bilinear resize -> scale to 0-1."""
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    return (tf.cast(img, tf.float32) / 255.0).numpy()


def predict(model, raw: bytes, img_size) -> np.ndarray:
    return model.predict(preprocess_bytes(raw, img_size)[None], verbose=0)[0]


# ----------------------------------------------------------------------------------------------
# Prediction history (SQLite)
# ----------------------------------------------------------------------------------------------
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, file_name TEXT, model TEXT, "
            "predicted_class TEXT, confidence REAL, probabilities TEXT)"
        )
        con.commit()


def log_prediction(file_name, model_name, predicted, confidence, probs_dict):
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute(
            "INSERT INTO predictions (timestamp, file_name, model, predicted_class, confidence, probabilities) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_name, model_name, predicted, float(confidence), json.dumps(probs_dict)),
        )
        con.commit()


def read_history() -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as con:
        return pd.read_sql_query(
            "SELECT id, timestamp, file_name, model, predicted_class, ROUND(confidence * 100, 2) AS confidence_pct "
            "FROM predictions ORDER BY id DESC LIMIT 200", con)


def clear_history():
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("DELETE FROM predictions")
        con.commit()


init_db()

# ----------------------------------------------------------------------------------------------
# Header + configuration check
# ----------------------------------------------------------------------------------------------
st.title("\U0001F9E0 Brain Tumor MRI Classifier")
st.caption("Deep-learning decision support: upload a brain MRI image and get the predicted tumour type with confidence scores.")

cfg = load_config()
if cfg is None:
    st.error("No trained models were found in the 'models' folder. Run **Brain_Tumor_MRI_Classification.ipynb** first "
             "(it creates models/*.h5 and models/model_config.json), then restart this app.")
    st.stop()

class_names = cfg["class_names"]
img_size = tuple(cfg.get("img_size", [224, 224]))
models_info = cfg["models"]

# ----------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_names = list(models_info.keys())
    default_idx = model_names.index(cfg["best_model"]) if cfg.get("best_model") in model_names else 0
    chosen_model = st.selectbox("Model", model_names, index=default_idx,
                                help="The best model (highest validation F1) is pre-selected.")
    if chosen_model == cfg.get("best_model"):
        st.success("Best model of the comparison")
    m = models_info[chosen_model].get("metrics", {})
    if m:
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{m.get('accuracy', 0) * 100:.1f}%")
        c2.metric("Macro F1", f"{m.get('f1_macro', 0) * 100:.1f}%")
    threshold = st.slider("Confidence warning threshold (%)", 30, 95, 70, 5,
                          help="Predictions below this confidence are flagged for extra manual review.")
    save_history = st.checkbox("Save predictions to history", value=True)
    st.markdown("---")
    st.markdown("**Classes:** " + ", ".join(pretty(c) for c in class_names))
    st.warning("Research prototype - **not a medical device**. A qualified radiologist must confirm every result.")

tab_predict, tab_compare, tab_history, tab_about = st.tabs(["Predict", "Model comparison", "History", "About"])

# ----------------------------------------------------------------------------------------------
# Tab 1 - Predict
# ----------------------------------------------------------------------------------------------
with tab_predict:
    files = st.file_uploader("Upload one or more MRI images (JPG, PNG, BMP)", type=ALLOWED_TYPES, accept_multiple_files=True)
    if not files:
        st.info("Upload an MRI image to get a prediction. You can select several files at once.")
    else:
        try:
            model = load_model(models_info[chosen_model]["file"])
        except Exception as exc:
            st.error(f"The model file could not be loaded: {exc}")
            st.stop()

        summary_rows = []
        for f in files:
            st.markdown("---")
            col_img, col_res = st.columns([1, 2])
            try:
                raw = f.getvalue()
                image = Image.open(f).convert("RGB")
                with st.spinner("Analysing ..."):
                    probs = predict(model, raw, img_size)
            except Exception as exc:
                col_img.error(f"'{f.name}' could not be processed: {exc}")
                continue

            k = int(np.argmax(probs))
            predicted, confidence = class_names[k], float(probs[k])
            probs_dict = {c: float(p) for c, p in zip(class_names, probs)}
            summary_rows.append({"File": f.name, "Prediction": pretty(predicted), "Confidence (%)": round(confidence * 100, 2)})
            if save_history:
                log_prediction(f.name, chosen_model, predicted, confidence, probs_dict)

            with col_img:
                st.image(image, caption=f.name, width=320)
            with col_res:
                st.markdown(f"<div class='big-pred'>{pretty(predicted)}</div>", unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.progress(min(max(confidence, 0.0), 1.0))
                if confidence * 100 < threshold:
                    st.warning(f"Low confidence (below {threshold}%). The model is unsure - please have this scan reviewed by a specialist.")
                st.caption(info_for(predicted))
                prob_df = pd.DataFrame({"Class": [pretty(c) for c in class_names],
                                        "Probability (%)": [round(p * 100, 2) for p in probs]}).set_index("Class")
                st.bar_chart(prob_df)
                with st.expander("Show all class probabilities"):
                    st.dataframe(prob_df.sort_values("Probability (%)", ascending=False))

        if len(summary_rows) > 1:
            st.markdown("---")
            st.subheader("Batch summary")
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df)
            st.download_button("Download summary as CSV", summary_df.to_csv(index=False).encode("utf-8"),
                               file_name="mri_predictions.csv", mime="text/csv")

# ----------------------------------------------------------------------------------------------
# Tab 2 - Model comparison
# ----------------------------------------------------------------------------------------------
with tab_compare:
    st.subheader("Trained models")
    rows = []
    for name, info in models_info.items():
        mt = info.get("metrics", {})
        rows.append({"Model": name, "Kept variant": info.get("variant", ""),
                     "Accuracy": mt.get("accuracy"), "Precision (macro)": mt.get("precision_macro"),
                     "Recall (macro)": mt.get("recall_macro"), "F1 (macro)": mt.get("f1_macro"),
                     "ROC-AUC": mt.get("roc_auc"), "Params (M)": mt.get("params_m"), "Inference (ms/img)": mt.get("inference_ms")})
    comp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp_df)
    plot_cols = [c for c in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"] if comp_df[c].notna().any()]
    if plot_cols:
        st.bar_chart(comp_df[plot_cols])
    st.caption(f"Best model (highest validation macro-F1): **{cfg.get('best_model')}**")
    for img_name, title in [("40_model_comparison.png", "Training-notebook comparison charts"),
                            ("41_best_model_confusion.png", "Confusion matrix of the best model")]:
        img_path = OUTPUT_DIR / img_name
        if img_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(img_path))

# ----------------------------------------------------------------------------------------------
# Tab 3 - History
# ----------------------------------------------------------------------------------------------
with tab_history:
    st.subheader("Recent predictions")
    if st.button("Clear history"):
        clear_history()
        st.success("History cleared.")
    hist_df = read_history()
    if hist_df.empty:
        st.info("No predictions saved yet.")
    else:
        st.dataframe(hist_df)
        st.bar_chart(hist_df["predicted_class"].value_counts())

# ----------------------------------------------------------------------------------------------
# Tab 4 - About
# ----------------------------------------------------------------------------------------------
with tab_about:
    st.subheader("About this application")
    st.markdown(
        f"""
**What it does** - classifies a brain MRI image into one of **{len(class_names)} classes**
({", ".join(pretty(c) for c in class_names)}) and reports the confidence of the model for every class.

**How it works**
1. The image is decoded, converted to 3-channel RGB, resized to {img_size[0]} x {img_size[1]} pixels and scaled to the 0-1 range
   (exactly the pipeline used for training).
2. A convolutional neural network (a custom CNN or a fine-tuned ImageNet model such as MobileNetV2, EfficientNetB0 or ResNet50) outputs one probability per class.
3. The class with the highest probability is shown. Predictions below the confidence threshold are flagged for manual review.

**Important limitations**
- Trained on a single public dataset of 2-D slices; performance on scans from other scanners or hospitals may be lower.
- The model does not locate or measure a tumour and is not calibrated for clinical probabilities.
- **This tool is for research and education only. It is not a medical device and must never replace a qualified radiologist.**
        """
    )