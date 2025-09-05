import streamlit as st
import pandas as pd
import json
from app.main import train_from_file
from app.detection import detect_and_interpret
from app.visualization import visualize_baseline
from app.config import MODELS_DIR

st.set_page_config(page_title="MindTrack Dashboard", layout="wide")

st.title("🧠 MindTrack – Mental Health Anomaly Detection")

# ---- Upload baseline CSV ----
st.header("1. Train Baseline Model")
csv_file = st.file_uploader("Upload baseline CSV", type="csv")

user_id = st.text_input("User ID for training", value="demo_user")

if csv_file is not None and st.button("Train Model"):
    train_from_file(
        csv_file,
        usercol=None,
        outdir=MODELS_DIR,
        contamination=0.05,
    )
    model_path = f"{MODELS_DIR}/{user_id}_baseline_bundle.joblib"
    st.success(f"Model trained and saved for {user_id} ✅\n📂 Path: {model_path}")

# ---- New Entry Detection ----
st.header("2. Detect Anomaly from New Entry")
entry_text = st.text_area("Paste new entry JSON here")

if st.button("Run Detection"):
    try:
        if not entry_text.strip():
            st.error("Please paste a valid JSON entry first.")
        else:
            entry = json.loads(entry_text)
            result = detect_and_interpret(entry, user_id=user_id, outdir=MODELS_DIR)
            st.json(result)
    except Exception as e:
        st.error(f"Error: {e}")

# ---- Visualize ----
st.header("3. Visualize Baseline vs Anomalies")
if csv_file is not None and st.button("Generate Visualizations"):
    try:
        df = pd.read_csv(csv_file)
        new_entry = None
        if entry_text.strip():
            try:
                new_entry = json.loads(entry_text)
            except:
                st.warning("⚠️ Could not parse new entry JSON, skipping overlay.")
        figs = visualize_baseline(df, user_id=user_id, new_entry=new_entry, save_plots=False)
        for fig in figs:
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Visualization error: {e}")
