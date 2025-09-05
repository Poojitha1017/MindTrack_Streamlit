import os

MODELS_DIR = "mindtrack_models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "screen_time_min",
    "unlock_count",
    "app_usage_var",
    "distinct_locations",
    "steps",
    "sleep_hours",
    "calls_made",
    "calls_missed",
    "avg_heart_rate",
    "social_ratio",
    "productivity_ratio",
    "entertainment_ratio",
    "check_leave_ratio",
    "mood_score"
]

