import os
import json
from typing import Dict

def collect_feedback(detection_result: Dict, feedback_store: str = "user_feedback.json"):
    """
    Save user feedback about anomaly detection.
    detection_result: output dict from detect_and_interpret
    feedback_store: path to JSON log file
    """
    print("\n🤔 Do you agree this was an anomaly?")
    print("Type 'y' for YES (true anomaly), 'n' for NO (false positive).")
    
    user_input = input("Your feedback: ").strip().lower()
    if user_input == "y":
        detection_result["user_feedback"] = "true_anomaly"
    else:
        detection_result["user_feedback"] = "false_positive"
    
    # Append to feedback log
    if os.path.exists(feedback_store):
        with open(feedback_store, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(detection_result)
    with open(feedback_store, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)  # ensure_ascii=False prevents unicode escapes
    
    print("✅ Feedback recorded!")
    return detection_result
