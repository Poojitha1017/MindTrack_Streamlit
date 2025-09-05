# app/utils.py
import pandas as pd
import numpy as np

# -----------------------
# Column preprocessing
# -----------------------
def ensure_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        "screen_time_min", "social_media_min", "work_related_min",
        "unlock_count", "app_usage_var", "distinct_locations",
        "steps", "sleep_hours", "calls_made", "calls_missed",
        "avg_heart_rate", "mood_score"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df

# -----------------------
# Derived features
# -----------------------
# utils.py
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features only if they don't exist.
    """
    # Only compute app usage ratios if missing
    if "social_ratio" not in df or "productivity_ratio" not in df or "entertainment_ratio" not in df:
        # Use 0 Series if column missing
        social = df["social_usage"] if "social_usage" in df else 0
        prod = df["productivity_usage"] if "productivity_usage" in df else 0
        ent = df["entertainment_usage"] if "entertainment_usage" in df else 0

        # Ensure these are Series of the same length as df
        if isinstance(social, int):
            social = pd.Series([social]*len(df))
        if isinstance(prod, int):
            prod = pd.Series([prod]*len(df))
        if isinstance(ent, int):
            ent = pd.Series([ent]*len(df))

        total_app_usage = social + prod + ent
        total_app_usage = total_app_usage.replace(0, 1)  # avoid division by zero

        df["social_ratio"] = social / total_app_usage
        df["productivity_ratio"] = prod / total_app_usage
        df["entertainment_ratio"] = ent / total_app_usage

    # Only compute check_leave_ratio if missing
    if "check_leave_ratio" not in df:
        check_leave = df["check_leave_count"] if "check_leave_count" in df else 0
        total_checks = df["total_checks"] if "total_checks" in df else 1

        if isinstance(check_leave, int):
            check_leave = pd.Series([check_leave]*len(df))
        if isinstance(total_checks, int):
            total_checks = pd.Series([total_checks]*len(df))

        df["check_leave_ratio"] = check_leave / total_checks

    return df

# -----------------------
# Z-score computation
# -----------------------
def compute_z_scores(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)

# -----------------------
# Prepare new entry
# -----------------------
def prepare_new_entry(entry: dict, feature_cols: list):
    """
    Convert new entry dict (with 'feature_values' key) into a Series
    compatible with FEATURE_COLS.
    """
    features = entry.get("feature_values", entry)  # fallback to entry itself if already flat
    return pd.Series([features[c] for c in feature_cols], index=feature_cols)


# -----------------------
# Build feature matrix
# -----------------------
def build_feature_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Return DataFrame with only the feature columns in correct order.
    """
    return df[feature_cols].copy()

# -----------------------
# Load CSV/XLS(X) dataset
# -----------------------
def load_df(path):
    """
    Load dataset from a path or buffer.
    Supports CSV and Excel.
    """
    import pandas as pd

    # Case 1: path is a Streamlit UploadedFile (or file-like buffer)
    if hasattr(path, "read"):  
        # Reset file pointer to the beginning in case it's been read already
        path.seek(0)  
        filename = getattr(path, "name", "")
        if filename.endswith(".csv"):
            return pd.read_csv(path)
        elif filename.endswith(".xlsx"):
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported uploaded file type. Please upload CSV or Excel.")

    # Case 2: path is a string (filepath)
    if isinstance(path, str):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".xlsx"):
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Must be .csv or .xlsx")

    raise TypeError(f"Unsupported path type: {type(path)}")


