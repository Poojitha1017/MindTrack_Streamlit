import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump
from .utils import ensure_raw_columns, add_derived, build_feature_matrix, load_df
from .config import FEATURE_COLS, MODELS_DIR

def train_user_baseline(df, user_id, feature_cols=FEATURE_COLS,
                        outdir=MODELS_DIR, contamination=0.05):
    df = ensure_raw_columns(df)
    df = add_derived(df)
    X = build_feature_matrix(df, feature_cols)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(Xs)

    meta = {
        "feature_cols": feature_cols,
        "mean": X.mean().to_dict(),
        "std": X.std().to_dict(),
        "n_samples": len(X)
    }

    os.makedirs(outdir, exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "meta": meta}
    path = os.path.join(outdir, f"{user_id}_baseline_bundle.joblib")
    dump(bundle, path)
    print(f"Saved baseline for {user_id} -> {path} (n={meta['n_samples']})")
    return path

def train_from_file(path, usercol=None, outdir=MODELS_DIR, contamination=0.05):
    df = load_df(path)
    if usercol and usercol in df.columns:
        for uid, group in df.groupby(usercol):
            print(f"Training for user {uid} ({len(group)} rows)...")
            train_user_baseline(group, uid, outdir=outdir, contamination=contamination)
    else:
        uid = "user123"
        train_user_baseline(df, uid, outdir=outdir, contamination=contamination)
