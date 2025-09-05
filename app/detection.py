# app/detection.py
import os
from joblib import load
import pandas as pd
from typing import Dict, List
from .config import FEATURE_COLS, MODELS_DIR
from .utils import ensure_raw_columns, add_derived, build_feature_matrix, prepare_new_entry

def load_bundle(user_id, outdir=MODELS_DIR):
    path = os.path.join(outdir, f"{user_id}_baseline_bundle.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found for user {user_id} at {path}")
    return load(path)


def detect_and_interpret(entry: dict, user_id: str, outdir=MODELS_DIR):
    # Convert single entry dict to DataFrame
    entry_df = pd.DataFrame([entry["feature_values"]])

    # Ensure all raw columns exist
    entry_df = ensure_raw_columns(entry_df)

    # ADD THIS: compute derived features
    entry_df = add_derived(entry_df)

    # Build feature matrix (all columns used by model)
    X = build_feature_matrix(entry_df, FEATURE_COLS)

    # Load trained model bundle
    bundle = load_bundle(user_id, outdir)
    model = bundle["model"]
    scaler = bundle["scaler"]

    # Scale and predict
    Xs = scaler.transform(X)
    preds = model.predict(Xs)

    # Return result
    result = entry.copy()
    result["is_anomaly"] = bool(preds[0] == -1)
    return result

