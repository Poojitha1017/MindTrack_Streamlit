# app/visualization.py
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from app.utils import ensure_raw_columns, add_derived, build_feature_matrix, prepare_new_entry
from app.detection import load_bundle
from app.config import FEATURE_COLS, MODELS_DIR

sns.set(style="whitegrid")


def visualize_baseline(
    df: pd.DataFrame,
    user_id: str,
    outdir: str = MODELS_DIR,
    feature_cols: List[str] = FEATURE_COLS,
    new_entry: Dict = None,
    save_plots: bool = False
):
    """
    Return a list of matplotlib figures for visualization:
      1) Screen time vs mood
      2) PCA(2D) projection
      3) Decision-function distribution
      4) Feature-by-feature comparison
    """
    # Load user's trained model bundle
    bundle = load_bundle(user_id, outdir)
    model = bundle["model"]
    scaler = bundle["scaler"]

    # Preprocess baseline data
    df = ensure_raw_columns(df)
    df = add_derived(df)

    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns in dataframe: {missing_cols}")

    X = build_feature_matrix(df, feature_cols)
    Xs = scaler.transform(X)
    preds = model.predict(Xs)

    df_plot = X.copy()
    df_plot["is_anomaly"] = preds == -1

    figs = []

    # --- 1) Scatter: screen_time vs mood ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_plot,
        x="screen_time_min",
        y="mood_score",
        hue="is_anomaly",
        palette={False: "green", True: "red"},
        ax=ax1
    )
    if new_entry:
        new_row = prepare_new_entry(new_entry, feature_cols)
        ax1.scatter(
            [new_row["screen_time_min"]],
            [new_row["mood_score"]],
            s=120, marker="*", label="New entry"
        )
    ax1.set_title(f"{user_id}: Screen time vs Mood")
    ax1.legend()
    figs.append(fig1)

    # --- 2) PCA 2D ---
    pca = PCA(n_components=2)
    comp = pca.fit_transform(Xs)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=comp[:, 0], y=comp[:, 1],
        hue=df_plot["is_anomaly"],
        palette={False: "green", True: "red"},
        ax=ax2
    )
    if new_entry:
        new_row = prepare_new_entry(new_entry, feature_cols)
        new_scaled = scaler.transform([new_row.values.astype(float)])
        new_comp = pca.transform(new_scaled)
        ax2.scatter(new_comp[0, 0], new_comp[0, 1], s=140, marker="*", label="New entry")
    ax2.set_title(f"{user_id}: PCA (2D) projection")
    ax2.legend()
    figs.append(fig2)

    # --- 3) Decision function histogram ---
    dec = model.decision_function(Xs)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.histplot(dec, bins=40, kde=True, ax=ax3)
    ax3.set_title("Decision function (higher = more normal)")
    if new_entry:
        new_row = prepare_new_entry(new_entry, feature_cols)
        new_scaled = scaler.transform([new_row.values.astype(float)])
        new_dec = model.decision_function(new_scaled)[0]
        ax3.axvline(new_dec, linestyle="--", color="red", label="New entry")
        ax3.legend()
    figs.append(fig3)

    # --- 4) Feature-by-feature comparison ---
    if new_entry:
        new_row = prepare_new_entry(new_entry, feature_cols)
        means = X.mean()
        df_compare = pd.DataFrame({
            "feature": feature_cols,
            "baseline_mean": means.values,
            "new_entry": new_row.values
        })

        fig4, ax4 = plt.subplots(figsize=(12, 6))
        df_compare.set_index("feature").plot(kind="bar", ax=ax4)
        ax4.set_title(f"{user_id}: Baseline vs New Entry")
        ax4.set_ylabel("Value")
        ax4.tick_params(axis="x", rotation=45)
        fig4.tight_layout()
        figs.append(fig4)

    return figs

