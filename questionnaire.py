"""
Mental Health Fitness — Questionnaire Module
Defines natural-language questions and maps responses to model features.
Includes trap questions for lie detection.
"""

# ─── Question Definitions ────────────────────────────────────
# Each question maps to one or more model features.
# response_type: "scale5" (1-5), "yesno", "choice"
# trap_for: if set, this is a duplicate check for consistency

QUESTIONS = [
    # ── Demographics ──
    {
        "id": "q_age",
        "section": "Demographics",
        "text": "What is your age?",
        "response_type": "number",
        "min": 18, "max": 65,
        "maps_to": "Age",
    },
    {
        "id": "q_gender",
        "section": "Demographics",
        "text": "What is your gender?",
        "response_type": "choice",
        "options": ["Male", "Female", "Other"],
        "maps_to": "Gender",
    },

    # ── Lifestyle ──
    {
        "id": "q_sleep",
        "section": "Lifestyle",
        "text": "How many hours do you sleep per day on average?",
        "response_type": "scale_range",
        "min": 3.0, "max": 10.0, "step": 0.5,
        "maps_to": "Sleep_Hours_Per_Day",
    },
    {
        "id": "q_work",
        "section": "Lifestyle",
        "text": "How many hours do you work per week?",
        "response_type": "number",
        "min": 0, "max": 60,
        "maps_to": "Work_Hours_Per_Week",
    },
    {
        "id": "q_exercise",
        "section": "Lifestyle",
        "text": "How often do you exercise per week?",
        "response_type": "choice",
        "options": ["Never (0)", "1-2 times", "3-4 times", "5+ times", "Daily (7)"],
        "maps_to": "Exercise_Frequency_Per_Week",
        "value_map": {"Never (0)": 0, "1-2 times": 1, "3-4 times": 3, "5+ times": 5, "Daily (7)": 7},
    },

    # ── Mood & Emotions ──
    {
        "id": "q_mood",
        "section": "How You Feel",
        "text": "How would you rate your overall mood recently?",
        "help": "1 = Very low / sad, 5 = Very happy / content",
        "response_type": "scale5",
        "maps_to": "Mood_Score (0-100)",
        "scale_to_100": True,
    },
    {
        "id": "q_anxiety",
        "section": "How You Feel",
        "text": "How often do you feel anxious or worried?",
        "help": "1 = Rarely / Never, 5 = Almost constantly",
        "response_type": "scale5",
        "maps_to": "Anxiety_Score (0-100)",
        "scale_to_100": True,
    },
    {
        "id": "q_depression",
        "section": "How You Feel",
        "text": "How often do you feel hopeless or deeply sad?",
        "help": "1 = Rarely / Never, 5 = Almost all the time",
        "response_type": "scale5",
        "maps_to": "Depression_Score (0-100)",
        "scale_to_100": True,
    },
    {
        "id": "q_stress",
        "section": "How You Feel",
        "text": "How stressed do you feel in your daily life?",
        "help": "1 = Very relaxed, 5 = Extremely stressed",
        "response_type": "scale5",
        "maps_to": "Stress_Score (0-100)",
        "scale_to_100": True,
    },
    {
        "id": "q_energy",
        "section": "How You Feel",
        "text": "How would you rate your energy levels?",
        "help": "1 = Always exhausted, 5 = Very energetic",
        "response_type": "scale5",
        "maps_to": "Energy_Level (0-100)",
        "scale_to_100": True,
    },

    # ── Social & Cognitive ──
    {
        "id": "q_social",
        "section": "Social & Daily Life",
        "text": "How often do you socialize or interact with others?",
        "help": "1 = Almost never, 5 = Very frequently",
        "response_type": "scale5",
        "maps_to": "Social_Interaction (1-10)",
        "scale_to_10": True,
    },
    {
        "id": "q_concentration",
        "section": "Social & Daily Life",
        "text": "How well can you concentrate on tasks?",
        "help": "1 = Cannot focus at all, 5 = Excellent focus",
        "response_type": "scale5",
        "maps_to": "Concentration (1-10)",
        "scale_to_10": True,
    },
    {
        "id": "q_appetite",
        "section": "Social & Daily Life",
        "text": "How is your appetite?",
        "help": "1 = Very poor / no appetite, 5 = Healthy and regular",
        "response_type": "scale5",
        "maps_to": "Appetite (1-10)",
        "scale_to_10": True,
    },
    {
        "id": "q_life_satisfaction",
        "section": "Social & Daily Life",
        "text": "Overall, how satisfied are you with your life?",
        "help": "1 = Very dissatisfied, 5 = Very satisfied",
        "response_type": "scale5",
        "maps_to": "Life_Satisfaction (0-100)",
        "scale_to_100": True,
    },

    # ── Clinical ──
    {
        "id": "q_therapy",
        "section": "Clinical History",
        "text": "How many therapy sessions do you attend per month?",
        "response_type": "number",
        "min": 0, "max": 10,
        "maps_to": "Therapy_Sessions_Per_Month",
    },
    {
        "id": "q_medication",
        "section": "Clinical History",
        "text": "Are you currently taking any mental health medication?",
        "response_type": "yesno",
        "maps_to": "Medication",
    },
    {
        "id": "q_family",
        "section": "Clinical History",
        "text": "Does anyone in your family have a history of mental illness?",
        "response_type": "yesno",
        "maps_to": "Family_History",
    },
    {
        "id": "q_hallucinations",
        "section": "Clinical History",
        "text": "Have you ever experienced seeing or hearing things that others couldn't?",
        "response_type": "yesno",
        "maps_to": "Hallucinations",
    },
    {
        "id": "q_manic",
        "section": "Clinical History",
        "text": "How many times in the past year have you experienced episodes of extremely high energy, reduced need for sleep, or impulsive behavior?",
        "response_type": "number",
        "min": 0, "max": 6,
        "maps_to": "Manic_Episodes_Per_Year",
    },

    # ── TRAP QUESTIONS (for lie detection) ──
    {
        "id": "trap_mood",
        "section": "Additional",
        "text": "Do you generally feel positive and optimistic about things?",
        "help": "1 = Not at all, 5 = Absolutely yes",
        "response_type": "scale5",
        "is_trap": True,
        "checks_against": "q_mood",
        "expected_correlation": "positive",  # should agree with mood
    },
    {
        "id": "trap_energy",
        "section": "Additional",
        "text": "Do you often feel tired or drained during the day?",
        "help": "1 = Rarely, 5 = Almost always",
        "response_type": "scale5",
        "is_trap": True,
        "checks_against": "q_energy",
        "expected_correlation": "negative",  # high tiredness should mean low energy
    },
    {
        "id": "trap_social",
        "section": "Additional",
        "text": "Do you prefer to stay alone and avoid meeting people?",
        "help": "1 = No, I love socializing, 5 = Yes, I always stay alone",
        "response_type": "scale5",
        "is_trap": True,
        "checks_against": "q_social",
        "expected_correlation": "negative",  # avoiding people ↔ low social
    },
    {
        "id": "trap_depression",
        "section": "Additional",
        "text": "Do you enjoy activities and hobbies that you used to love?",
        "help": "1 = Not at all anymore, 5 = Yes, very much",
        "response_type": "scale5",
        "is_trap": True,
        "checks_against": "q_depression",
        "expected_correlation": "negative",  # enjoying things ↔ low depression
    },
]


def get_sections():
    """Return ordered list of unique sections."""
    seen = []
    for q in QUESTIONS:
        if q["section"] not in seen:
            seen.append(q["section"])
    return seen


def get_questions_by_section(section):
    """Return questions for a specific section."""
    return [q for q in QUESTIONS if q["section"] == section]


def convert_scale5_to_range(value, to_100=False, to_10=False):
    """Convert a 1-5 scale answer to the target range."""
    # 1→0, 2→25, 3→50, 4→75, 5→100 (for 0-100)
    # 1→1, 2→3, 3→5, 4→7, 5→10 (for 1-10)
    if to_100:
        return int((value - 1) * 25)
    elif to_10:
        return int(1 + (value - 1) * 2.25)
    return value


def answers_to_features(answers):
    """
    Convert questionnaire answers dict to the 19-feature input array.
    Returns dict of {feature_name: value}.
    """
    features = {}

    for q in QUESTIONS:
        if q.get("is_trap"):
            continue  # skip trap questions for feature mapping

        qid = q["id"]
        if qid not in answers:
            continue

        val = answers[qid]
        feature_name = q["maps_to"]

        if q["response_type"] == "scale5":
            val = convert_scale5_to_range(
                val,
                to_100=q.get("scale_to_100", False),
                to_10=q.get("scale_to_10", False),
            )
        elif q["response_type"] == "yesno":
            val = 1 if val == "Yes" else 0
        elif q["response_type"] == "choice" and "value_map" in q:
            val = q["value_map"].get(val, 0)

        features[feature_name] = val

    return features
