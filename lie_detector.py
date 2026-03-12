"""
Mental Health Fitness — Lie / Inconsistency Detection Engine
Detects contradictions, statistical anomalies, and social desirability bias.
"""

import numpy as np
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mental_health_dataset.csv")


# ─── Contradiction Pairs ─────────────────────────────────────
# Each pair: (feature_A, feature_B, expected_direction, description)
# "positive" = they should move together (both high or both low)
# "negative" = they should be inverse (one high → other low)

CONTRADICTION_PAIRS = [
    {
        "a": "Mood_Score (0-100)",
        "b": "Depression_Score (0-100)",
        "direction": "negative",
        "desc": "High mood with high depression is contradictory",
        "icon": "😊↔️😢",
    },
    {
        "a": "Energy_Level (0-100)",
        "b": "Depression_Score (0-100)",
        "direction": "negative",
        "desc": "High energy with high depression is unusual",
        "icon": "⚡↔️😢",
    },
    {
        "a": "Mood_Score (0-100)",
        "b": "Anxiety_Score (0-100)",
        "direction": "negative",
        "desc": "Very high mood with very high anxiety is contradictory",
        "icon": "😊↔️😰",
    },
    {
        "a": "Life_Satisfaction (0-100)",
        "b": "Depression_Score (0-100)",
        "direction": "negative",
        "desc": "High life satisfaction with high depression is contradictory",
        "icon": "🌟↔️😢",
    },
    {
        "a": "Life_Satisfaction (0-100)",
        "b": "Stress_Score (0-100)",
        "direction": "negative",
        "desc": "High life satisfaction with extreme stress is unusual",
        "icon": "🌟↔️🔥",
    },
    {
        "a": "Social_Interaction (1-10)",
        "b": "Depression_Score (0-100)",
        "direction": "negative",
        "desc": "Very high socializing with high depression is uncommon",
        "icon": "👥↔️😢",
    },
    {
        "a": "Energy_Level (0-100)",
        "b": "Stress_Score (0-100)",
        "direction": "negative",
        "desc": "Very high energy with extreme stress is unusual",
        "icon": "⚡↔️🔥",
    },
    {
        "a": "Concentration (1-10)",
        "b": "Anxiety_Score (0-100)",
        "direction": "negative",
        "desc": "Excellent concentration with severe anxiety is contradictory",
        "icon": "🎯↔️😰",
    },
    {
        "a": "Mood_Score (0-100)",
        "b": "Life_Satisfaction (0-100)",
        "direction": "positive",
        "desc": "Very different mood and life satisfaction levels is unusual",
        "icon": "😊↔️🌟",
    },
    {
        "a": "Sleep_Hours_Per_Day",
        "b": "Energy_Level (0-100)",
        "direction": "positive",
        "desc": "Very little sleep but very high energy is suspicious",
        "icon": "😴↔️⚡",
    },
]


def _normalize_value(val, feature_name, df_stats):
    """Normalize a value to 0-1 range based on dataset min/max."""
    stats = df_stats.get(feature_name)
    if stats is None:
        return 0.5
    mn, mx = stats["min"], stats["max"]
    if mx == mn:
        return 0.5
    return (val - mn) / (mx - mn)


def _load_stats():
    """Load dataset statistics for normalization."""
    try:
        df = pd.read_csv(DATA_PATH)
        stats = {}
        for col in df.select_dtypes(include=np.number).columns:
            stats[col] = {"min": df[col].min(), "max": df[col].max(), "mean": df[col].mean(), "std": df[col].std()}
        return stats, df
    except Exception:
        return {}, None


# ─── 1. Contradiction Detection ──────────────────────────────
def check_contradictions(features):
    """
    Check for contradictory feature pairs.
    Returns list of flags with severity (0-1).
    """
    stats, _ = _load_stats()
    flags = []

    for pair in CONTRADICTION_PAIRS:
        a_name, b_name = pair["a"], pair["b"]
        if a_name not in features or b_name not in features:
            continue

        a_val = _normalize_value(features[a_name], a_name, stats)
        b_val = _normalize_value(features[b_name], b_name, stats)

        if pair["direction"] == "negative":
            # Both should not be high simultaneously
            # Severity = how much both are on the same extreme
            if a_val > 0.7 and b_val > 0.7:
                severity = (a_val + b_val - 1.4) / 0.6  # 0 to 1
                flags.append({
                    "type": "contradiction",
                    "description": pair["desc"],
                    "icon": pair["icon"],
                    "severity": min(severity, 1.0),
                    "details": f"{a_name}: high ({features[a_name]}) vs {b_name}: high ({features[b_name]})",
                })
            elif a_val < 0.3 and b_val < 0.3:
                severity = (0.6 - a_val - b_val) / 0.6
                flags.append({
                    "type": "contradiction",
                    "description": pair["desc"],
                    "icon": pair["icon"],
                    "severity": min(severity * 0.5, 1.0),  # less severe for both-low
                    "details": f"{a_name}: low ({features[a_name]}) vs {b_name}: low ({features[b_name]})",
                })
        elif pair["direction"] == "positive":
            # They should move together, flag if very different
            diff = abs(a_val - b_val)
            if diff > 0.5:
                severity = (diff - 0.5) / 0.5
                flags.append({
                    "type": "contradiction",
                    "description": pair["desc"],
                    "icon": pair["icon"],
                    "severity": min(severity, 1.0),
                    "details": f"{a_name}: {features[a_name]} vs {b_name}: {features[b_name]} (large gap)",
                })

    return flags


# ─── 2. Trap Question Consistency ────────────────────────────
def check_trap_questions(answers, questions):
    """
    Check if trap question answers are consistent with their paired questions.
    Returns list of flags.
    """
    flags = []

    trap_questions = [q for q in questions if q.get("is_trap")]

    for trap in trap_questions:
        trap_id = trap["id"]
        orig_id = trap["checks_against"]

        if trap_id not in answers or orig_id not in answers:
            continue

        trap_val = answers[trap_id]
        orig_val = answers[orig_id]
        correlation = trap["expected_correlation"]

        # Both are on scale5
        if correlation == "positive":
            # They should agree
            diff = abs(trap_val - orig_val)
            if diff >= 3:
                severity = (diff - 2) / 3
                flags.append({
                    "type": "trap_inconsistency",
                    "description": f'Your answer to "{trap["text"]}" contradicts your earlier response',
                    "icon": "🔍",
                    "severity": min(severity, 1.0),
                    "details": f"Q: \"{trap['text']}\" = {trap_val}/5 vs Related Q = {orig_val}/5 (expected similar)",
                })
        elif correlation == "negative":
            # They should be opposite (sum should be ~6)
            agreement = trap_val + orig_val
            if agreement <= 3 or agreement >= 9:
                # Both extremes going same direction = consistent
                pass
            else:
                # Middle ground is fine, but same direction extremes are suspicious
                if (trap_val >= 4 and orig_val >= 4) or (trap_val <= 2 and orig_val <= 2):
                    severity = 0.6 + (min(trap_val, orig_val) - 1) * 0.1
                    flags.append({
                        "type": "trap_inconsistency",
                        "description": f'Your answer to "{trap["text"]}" seems inconsistent with your earlier response',
                        "icon": "🔍",
                        "severity": min(severity, 1.0),
                        "details": f"Q: \"{trap['text']}\" = {trap_val}/5 vs Related Q = {orig_val}/5 (expected opposite)",
                    })

    return flags


# ─── 3. Social Desirability Bias ─────────────────────────────
def check_social_desirability(features):
    """
    Detect if someone is answering too positively across the board.
    A pattern of 'everything is perfect' is suspicious.
    """
    stats, _ = _load_stats()
    flags = []

    positive_features = [
        "Mood_Score (0-100)",
        "Energy_Level (0-100)",
        "Life_Satisfaction (0-100)",
        "Concentration (1-10)",
        "Appetite (1-10)",
        "Social_Interaction (1-10)",
    ]
    negative_features = [
        "Anxiety_Score (0-100)",
        "Depression_Score (0-100)",
        "Stress_Score (0-100)",
    ]

    # Count how many positive features are in top 20%
    positive_count = 0
    total_positive = 0
    for feat in positive_features:
        if feat in features:
            total_positive += 1
            norm = _normalize_value(features[feat], feat, stats)
            if norm > 0.8:
                positive_count += 1

    # Count how many negative features are in bottom 20%
    negative_count = 0
    total_negative = 0
    for feat in negative_features:
        if feat in features:
            total_negative += 1
            norm = _normalize_value(features[feat], feat, stats)
            if norm < 0.2:
                negative_count += 1

    # If almost everything is "perfect"
    if total_positive > 0 and total_negative > 0:
        positive_ratio = positive_count / total_positive
        negative_ratio = negative_count / total_negative

        if positive_ratio >= 0.8 and negative_ratio >= 0.8:
            flags.append({
                "type": "social_desirability",
                "description": "Your responses suggest an unusually perfect profile — all positives are very high and all negatives are very low",
                "icon": "✨",
                "severity": 0.7,
                "details": f"{positive_count}/{total_positive} positive traits at maximum, {negative_count}/{total_negative} negative traits at minimum",
            })
        elif positive_ratio >= 0.6 and negative_ratio >= 0.6:
            flags.append({
                "type": "social_desirability",
                "description": "Your responses lean towards a very optimistic self-assessment",
                "icon": "✨",
                "severity": 0.4,
                "details": f"{positive_count}/{total_positive} positive traits very high, {negative_count}/{total_negative} negative traits very low",
            })

    return flags


# ─── 4. Statistical Anomaly ──────────────────────────────────
def check_statistical_anomaly(features):
    """
    Check if the combination of features is statistically very unlikely.
    Uses z-score analysis on key features.
    """
    stats, df = _load_stats()
    if df is None:
        return []

    flags = []
    extreme_count = 0
    extreme_features = []

    key_features = [
        "Mood_Score (0-100)", "Anxiety_Score (0-100)", "Depression_Score (0-100)",
        "Stress_Score (0-100)", "Energy_Level (0-100)", "Life_Satisfaction (0-100)",
    ]

    for feat in key_features:
        if feat not in features or feat not in stats:
            continue
        s = stats[feat]
        if s["std"] == 0:
            continue
        z = abs(features[feat] - s["mean"]) / s["std"]
        if z > 2.5:
            extreme_count += 1
            extreme_features.append(feat.split("(")[0].strip())

    if extreme_count >= 3:
        flags.append({
            "type": "statistical_anomaly",
            "description": f"Multiple scores ({extreme_count}) are statistically extreme compared to the dataset",
            "icon": "📊",
            "severity": min(0.3 + extreme_count * 0.15, 1.0),
            "details": f"Extreme values in: {', '.join(extreme_features)}",
        })

    return flags


# ─── Main Analysis Function ──────────────────────────────────
def analyze_honesty(features, answers=None, questions=None):
    """
    Run all detection methods and compute an overall honesty score.

    Returns:
        dict with:
            - honesty_score: 0-100 (100 = fully consistent/trustworthy)
            - flags: list of individual flag dicts
            - summary: text summary
            - risk_level: "low", "medium", "high"
    """
    all_flags = []

    # 1. Contradiction pairs
    all_flags.extend(check_contradictions(features))

    # 2. Trap questions
    if answers is not None and questions is not None:
        all_flags.extend(check_trap_questions(answers, questions))

    # 3. Social desirability
    all_flags.extend(check_social_desirability(features))

    # 4. Statistical anomaly
    all_flags.extend(check_statistical_anomaly(features))

    # Calculate overall honesty score
    if not all_flags:
        honesty_score = 100.0
        risk_level = "low"
        summary = "✅ Responses appear consistent and trustworthy. No contradictions detected."
    else:
        # Weighted penalty: each flag reduces score based on severity
        total_penalty = sum(f["severity"] * 15 for f in all_flags)
        honesty_score = max(0, 100 - total_penalty)

        if honesty_score >= 75:
            risk_level = "low"
            summary = "🟢 Minor inconsistencies detected, but responses are mostly trustworthy."
        elif honesty_score >= 50:
            risk_level = "medium"
            summary = "🟡 Several inconsistencies detected. The person may be minimizing or hiding some symptoms."
        elif honesty_score >= 25:
            risk_level = "high"
            summary = "🟠 Significant inconsistencies detected. Responses may not accurately reflect the person's true condition."
        else:
            risk_level = "critical"
            summary = "🔴 Major contradictions found. The person appears to be significantly misrepresenting their condition."

    return {
        "honesty_score": round(honesty_score, 1),
        "flags": all_flags,
        "summary": summary,
        "risk_level": risk_level,
        "flag_count": len(all_flags),
    }
