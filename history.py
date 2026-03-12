"""
Mental Health Fitness — Assessment History Module
Stores and retrieves per-person assessment history using a local JSON file.
"""

import os
import json
from datetime import datetime

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HISTORY_FILE = os.path.join(HISTORY_DIR, "assessment_history.json")


def _ensure_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _load_history():
    """Load the full history dict from disk."""
    _ensure_dir()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_history(data):
    """Save the full history dict to disk."""
    _ensure_dir()
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_assessment(user_name, features, prediction, confidence, honesty_score, probabilities, flags_count):
    """
    Save one assessment record for a user.

    Args:
        user_name: The person's name (case-insensitive key)
        features: dict of model input features
        prediction: predicted condition string
        confidence: confidence percentage
        honesty_score: 0-100
        probabilities: dict of {condition: probability}
        flags_count: number of honesty flags
    """
    history = _load_history()
    key = user_name.strip().lower()

    if key not in history:
        history[key] = {
            "display_name": user_name.strip(),
            "assessments": [],
        }

    record = {
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "honesty_score": round(honesty_score, 1),
        "flags_count": flags_count,
        "probabilities": probabilities,
        "features": features,
    }

    history[key]["assessments"].append(record)
    _save_history(history)
    return record


def get_user_history(user_name):
    """
    Get all assessment records for a user.
    Returns list of records (newest first), or empty list.
    """
    history = _load_history()
    key = user_name.strip().lower()
    user_data = history.get(key, {})
    assessments = user_data.get("assessments", [])
    return list(reversed(assessments))  # newest first


def get_all_users():
    """Return list of all user display names that have history."""
    history = _load_history()
    return [v["display_name"] for v in history.values() if v.get("assessments")]


def get_user_trend_data(user_name):
    """
    Get trend data for charts. Returns a list of dicts with
    timestamp, prediction, confidence, honesty_score, and key feature scores.
    """
    records = get_user_history(user_name)
    if not records:
        return []

    trend = []
    for r in reversed(records):  # chronological order
        entry = {
            "Date": r["timestamp"][:10],
            "Time": r["timestamp"][11:16],
            "Prediction": r["prediction"],
            "Confidence": r["confidence"],
            "Honesty Score": r["honesty_score"],
        }
        # Add key feature scores if available
        feats = r.get("features", {})
        for key in ["Mood_Score (0-100)", "Anxiety_Score (0-100)", "Depression_Score (0-100)",
                     "Stress_Score (0-100)", "Energy_Level (0-100)", "Life_Satisfaction (0-100)"]:
            short_name = key.split("_Score")[0].split("_Level")[0].split("_Satisfaction")[0].replace("_", " ")
            entry[short_name] = feats.get(key, None)
        trend.append(entry)

    return trend
