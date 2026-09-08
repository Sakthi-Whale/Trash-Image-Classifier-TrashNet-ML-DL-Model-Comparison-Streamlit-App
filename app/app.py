"""
streamlit_app.py
-----------------
Streamlit front-end for the trash image classification project.

Two tabs:
  1. "Classify"  - upload a photo of an item of trash, pick a trained
                    model, and see the predicted category + confidence.
  2. "Compare models" - table + chart comparing every model that has
                    been trained (results/metrics.json).

Run with:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import CLASSES, MODELS_DIR, RESULTS_DIR, IMG_SIZE  # noqa: E402

st.set_page_config(
    page_title="Trash Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_EMOJI = {
    "cardboard": "📦", "glass": "🍾", "metal": "🥫",
    "paper": "📄", "plastic": "🧴", "trash": "🗑️",
}

CLASS_COLOR = {
    "cardboard": "#B08968", "glass": "#2A9D8F", "metal": "#6C757D",
    "paper": "#E9C46A", "plastic": "#4C6EF5", "trash": "#E76F51",
}

# --------------------------------------------------------------------
# Theme / styling
# --------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        :root {
            --brand: #2E7D32;
            --brand-light: #E8F5E9;
            --ink: #1B1F23;
            --muted: #6B7280;
            --card-border: #E5E7EB;
        }

        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

        /* Hero header */
        .app-hero {
            background: linear-gradient(135deg, var(--brand) 0%, #1B5E20 100%);
            border-radius: 16px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.75rem;
            color: white;
            box-shadow: 0 10px 30px -12px rgba(27, 94, 32, 0.45);
        }
        .app-hero h1 {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: white;
            letter-spacing: -0.02em;
        }
        .app-hero p {
            margin: 0;
            font-size: 0.98rem;
            color: rgba(255,255,255,0.88);
            max-width: 640px;
            line-height: 1.5;
        }
        .app-hero .badges { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            color: white;
        }

        /* Cards */
        .app-card {
            background: white;
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            height: 100%;
        }
        .card-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 0.9rem;
        }

        /* Prediction result */
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--brand-light);
            color: var(--brand);
            border: 1px solid #C8E6C9;
            border-radius: 12px;
            padding: 0.8rem 1.1rem;
            font-size: 1.35rem;
            font-weight: 800;
            text-transform: capitalize;
            margin-bottom: 0.9rem;
        }
        .prob-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.55rem; }
        .prob-label { width: 108px; font-size: 0.86rem; font-weight: 500; color: var(--ink); flex-shrink: 0; }
        .prob-track { flex: 1; background: #F1F3F5; border-radius: 999px; height: 10px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 999px; }
        .prob-pct { width: 46px; text-align: right; font-size: 0.82rem; font-weight: 600; color: var(--muted); flex-shrink: 0; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--card-border); }
        .stTabs [data-baseweb="tab"] {
            height: 44px; padding: 0 1.1rem; border-radius: 10px 10px 0 0;
            font-weight: 600; font-size: 0.92rem; color: var(--muted);
        }
        .stTabs [aria-selected="true"] { color: var(--brand) !important; background: var(--brand-light); }

        /* Buttons / inputs */
        div[data-testid="stFileUploader"] { border-radius: 12px; }
        .stSelectbox label, .stFileUploader label { font-weight: 600 !important; color: var(--ink) !important; }

        [data-testid="stMetricValue"] { color: var(--brand); font-weight: 800; }

        .app-footer {
            text-align: center; color: var(--muted); font-size: 0.82rem;
            margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid var(--card-border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# Model registry: discover whatever has actually been trained on disk
# --------------------------------------------------------------------
@st.cache_resource
def discover_models():
    registry = {}

    # Classical ML baselines (joblib: {"model": ..., "scaler": ...})
    for name, fname in [("HOG + SVM", "hog_svm.joblib"), ("HOG + Random Forest", "hog_rf.joblib")]:
        path = MODELS_DIR / fname
        if path.exists():
            registry[name] = {"type": "sklearn", "path": path}

    # Keras models
    for name, fname in [
        ("Simple CNN (from scratch)", "simple_cnn.keras"),
        ("MobileNetV2 (transfer learning)", "mobilenet.keras"),
        ("ResNet50 (transfer learning)", "resnet50.keras"),
    ]:
        path = MODELS_DIR / fname
        if path.exists():
            backbone = "mobilenet" if "mobilenet" in fname else ("resnet50" if "resnet50" in fname else None)
            registry[name] = {"type": "keras", "path": path, "backbone": backbone}

    return registry


@st.cache_resource
def load_sklearn_model(path):
    import joblib
    return joblib.load(path)


@st.cache_resource
def load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def predict_sklearn(bundle, image: Image.Image):
    from skimage.feature import hog
    from skimage.color import rgb2gray

    img = image.convert("RGB").resize((128, 128))
    gray = rgb2gray(np.array(img))
    feats = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), block_norm="L2-Hys")
    feats_scaled = bundle["scaler"].transform([feats])
    probs = bundle["model"].predict_proba(feats_scaled)[0]
    return probs


def predict_keras(model, image: Image.Image, backbone: str):
    import tensorflow as tf

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    if backbone == "mobilenet":
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    elif backbone == "resnet50":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    else:  # simple CNN trained with Rescaling(1/255) baked in? -> normalize manually
        arr = arr / 255.0

    probs = model.predict(arr, verbose=0)[0]
    return probs


def run_prediction(model_name: str, registry: dict, image: Image.Image):
    entry = registry[model_name]
    if entry["type"] == "sklearn":
        bundle = load_sklearn_model(entry["path"])
        return predict_sklearn(bundle, image)
    else:
        model = load_keras_model(entry["path"])
        return predict_keras(model, image, entry["backbone"])


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
registry = discover_models()

st.markdown(
    f"""
    <div class="app-hero">
        <h1>♻️ Trash Image Classifier</h1>
        <p>Classify an item into cardboard, glass, metal, paper, plastic, or trash,
        and compare how different models perform on the TrashNet dataset.</p>
        <div class="badges">
            <span class="hero-badge">{len(registry)} model{'s' if len(registry) != 1 else ''} loaded</span>
            <span class="hero-badge">{len(CLASSES)} classes</span>
            <span class="hero-badge">TrashNet dataset</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_classify, tab_compare = st.tabs(["🔍  Classify an image", "📊  Compare models"])

# ---------------- Tab 1: Classify ----------------
with tab_classify:
    if not registry:
        st.warning(
            "No trained models found in `models/`. Run at least one training script first, "
            "e.g. `python src/train_baseline.py`."
        )
    else:
        col_left, col_right = st.columns([1, 1], gap="medium")

        with col_left:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Input</div>', unsafe_allow_html=True)
            model_name = st.selectbox("Model", list(registry.keys()))
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            with st.expander("📷 Use camera instead"):
                camera = st.camera_input("Take a photo", label_visibility="collapsed")
            image_source = uploaded or camera

            if image_source is not None:
                image = Image.open(image_source)
                st.image(image, caption="Input image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Result</div>', unsafe_allow_html=True)

            if image_source is not None:
                with st.spinner("Classifying..."):
                    probs = run_prediction(model_name, registry, image)

                pred_idx = int(np.argmax(probs))
                pred_class = CLASSES[pred_idx]
                confidence = float(probs[pred_idx])

                st.markdown(
                    f'<div class="result-badge">{CLASS_EMOJI.get(pred_class, "")} {pred_class} '
                    f'&nbsp;·&nbsp; {confidence * 100:.1f}% confidence</div>',
                    unsafe_allow_html=True,
                )

                order = np.argsort(probs)[::-1]
                rows_html = ""
                for i in order:
                    cls = CLASSES[i]
                    pct = float(probs[i]) * 100
                    color = CLASS_COLOR.get(cls, "#4C6EF5")
                    rows_html += f"""
                    <div class="prob-row">
                        <div class="prob-label">{CLASS_EMOJI.get(cls, '')} {cls}</div>
                        <div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
                        <div class="prob-pct">{pct:.1f}%</div>
                    </div>
                    """
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.info("Upload an image or take a photo to see the prediction here.")
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 2: Compare models ----------------
with tab_compare:
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        st.warning("No results yet — run the training scripts to populate `results/metrics.json`.")
    else:
        all_results = json.loads(metrics_path.read_text())

        rows = []
        for name, m in all_results.items():
            rows.append({
                "Model": name,
                "Val Accuracy": round(m["val_accuracy"], 4),
                "Test Accuracy": round(m["test_accuracy"], 4),
                "Test F1 (macro)": round(m["test_f1_macro"], 4),
                "Train Time (s)": round(m["train_time_sec"], 1),
                "# Params": m.get("n_params") or "—",
            })
        df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)

        best = df.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Top model", best["Model"])
        m2.metric("Best test accuracy", f"{best['Test Accuracy'] * 100:.1f}%")
        m3.metric("Best F1 (macro)", f"{best['Test F1 (macro)']:.3f}")

        st.markdown('<div class="app-card" style="margin-top:0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Leaderboard</div>', unsafe_allow_html=True)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Val Accuracy": st.column_config.ProgressColumn("Val Accuracy", format="%.3f", min_value=0, max_value=1),
                "Test Accuracy": st.column_config.ProgressColumn("Test Accuracy", format="%.3f", min_value=0, max_value=1),
            },
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Accuracy vs. F1 by model</div>', unsafe_allow_html=True)
        chart_df = df.set_index("Model")[["Test Accuracy", "Test F1 (macro)"]]
        st.bar_chart(chart_df, color=["#2E7D32", "#A5D6A7"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Per-model detail</div>', unsafe_allow_html=True)
        selected = st.selectbox("View per-class report for:", df["Model"].tolist())

        detail_left, detail_right = st.columns([1, 1], gap="medium")
        with detail_left:
            st.caption("Classification report")
            report = all_results[selected]["classification_report"]
            report_df = pd.DataFrame(report).T
            report_df = report_df[report_df.index.isin(CLASSES)]
            st.dataframe(report_df.round(3), use_container_width=True)

        with detail_right:
            st.caption("Confusion matrix (rows = true label, columns = predicted)")
            cm = np.array(all_results[selected]["confusion_matrix"])
            cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
            st.dataframe(cm_df.style.background_gradient(cmap="Greens"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="app-footer">Dataset: TrashNet-style <code>dataset-resized</code> '
    '(cardboard, glass, metal, paper, plastic, trash).</div>',
    unsafe_allow_html=True,
)
