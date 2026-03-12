"""
Mental Health Fitness — Streamlit App
A premium multi-page mental health prediction dashboard.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch
import torch.nn as nn
from questionnaire import QUESTIONS, get_sections, get_questions_by_section, answers_to_features, convert_scale5_to_range
from lie_detector import analyze_honesty
from history import save_assessment, get_user_history, get_all_users, get_user_trend_data

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MindFit — Mental Health Fitness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "mental_health_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1333 40%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1333 0%, #0f0c29 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0d4ff;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #c4b5fd !important;
    }

    /* Headers */
    h1 { color: #e0d4ff !important; font-weight: 800 !important; }
    h2 { color: #c4b5fd !important; font-weight: 700 !important; }
    h3 { color: #a78bfa !important; font-weight: 600 !important; }
    p, li, span { color: #d1d5db !important; }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.15);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(59,130,246,0.08) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Condition badges */
    .badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; margin: 3px; }
    .badge-normal { background: rgba(16,185,129,0.15); color: #34d399 !important; border: 1px solid rgba(16,185,129,0.3); }
    .badge-depression { background: rgba(99,102,241,0.15); color: #818cf8 !important; border: 1px solid rgba(99,102,241,0.3); }
    .badge-anxiety { background: rgba(245,158,11,0.15); color: #fbbf24 !important; border: 1px solid rgba(245,158,11,0.3); }
    .badge-stress { background: rgba(239,68,68,0.15); color: #f87171 !important; border: 1px solid rgba(239,68,68,0.3); }
    .badge-bipolar { background: rgba(236,72,153,0.15); color: #f472b6 !important; border: 1px solid rgba(236,72,153,0.3); }
    .badge-schizophrenia { background: rgba(168,85,247,0.15); color: #c084fc !important; border: 1px solid rgba(168,85,247,0.3); }

    /* Hero */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #9ca3af !important;
        font-weight: 400;
        max-width: 600px;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(16,185,129,0.1) 100%);
        border: 2px solid rgba(139, 92, 246, 0.35);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: 800;
        color: #e0d4ff !important;
    }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #c4b5fd !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #9ca3af !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #a78bfa !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(139,92,246,0.08);
        border-radius: 8px;
        color: #c4b5fd;
        border: 1px solid rgba(139,92,246,0.15);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(139,92,246,0.2) !important;
        border-color: rgba(139,92,246,0.4) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6366f1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 28px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    }

    /* Honesty meter */
    .honesty-meter {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(139,92,246,0.18);
        text-align: center;
    }
    .honesty-score {
        font-size: 3rem;
        font-weight: 900;
        margin: 8px 0;
    }
    .honesty-high { color: #34d399 !important; }
    .honesty-medium { color: #fbbf24 !important; }
    .honesty-low { color: #f97316 !important; }
    .honesty-critical { color: #ef4444 !important; }
    .flag-card {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .flag-card-warn {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .section-header {
        background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(59,130,246,0.06) 100%);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 20px 0 12px 0;
    }
    .progress-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 8px;
        width: 100%;
        margin: 10px 0;
    }
    .progress-bar-fill {
        height: 8px;
        border-radius: 8px;
        background: linear-gradient(90deg, #7c3aed, #a78bfa);
        transition: width 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    gender_enc = joblib.load(os.path.join(MODEL_DIR, "gender_encoder.joblib"))
    target_enc = joblib.load(os.path.join(MODEL_DIR, "target_encoder.joblib"))
    feat_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, gender_enc, target_enc, feat_names


# ─── MLP Model Definition (must match training architecture) ─
class MentalHealthMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MentalHealthMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_mlp_model():
    """Load the trained MLP model. Returns None if not found."""
    mlp_path = os.path.join(MODEL_DIR, "mlp_model.pth")
    if not os.path.exists(mlp_path):
        return None
    checkpoint = torch.load(mlp_path, map_location="cpu", weights_only=True)
    model = MentalHealthMLP(checkpoint["input_size"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


CONDITION_COLORS = {
    "Normal": "#34d399",
    "Depression": "#818cf8",
    "Anxiety": "#fbbf24",
    "Stress": "#f87171",
    "Bipolar Disorder": "#f472b6",
    "Schizophrenia": "#c084fc",
}

CONDITION_ICONS = {
    "Normal": "✅",
    "Depression": "💙",
    "Anxiety": "⚡",
    "Stress": "🔥",
    "Bipolar Disorder": "🎭",
    "Schizophrenia": "🌀",
}

# ─── Sidebar Navigation ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindFit")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📝 Assessment", "📅 History", "📊 Data Explorer", "📈 Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem; color:#6b7280 !important;">Built with ❤️ using Streamlit, scikit-learn & PyTorch</p>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: HOME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Home":
    df = load_data()

    # Hero
    st.markdown('<div class="hero-title">Mental Health Fitness</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">AI-powered mental health condition prediction using machine learning. '
        "Explore data patterns, understand risk factors, and get instant assessments.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Records</div>'
            f'<div class="metric-value">{len(df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Features</div>'
            f'<div class="metric-value">{df.shape[1]-2}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Conditions</div>'
            f'<div class="metric-value">{df["Mental_Health_Condition"].nunique()}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        avg_age = df["Age"].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg Age</div>'
            f'<div class="metric-value">{avg_age:.0f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### Condition Distribution")

    col1, col2 = st.columns([3, 2])
    with col1:
        counts = df["Mental_Health_Condition"].value_counts()
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=CONDITION_COLORS,
            labels={"x": "Condition", "y": "Count"},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd",
            showlegend=False,
            xaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Quick Overview")
        for cond, count in counts.items():
            badge_class = f"badge-{cond.lower().replace(' ', '-').split('-')[0] if 'Disorder' not in cond else 'bipolar'}"
            if cond == "Schizophrenia":
                badge_class = "badge-schizophrenia"
            icon = CONDITION_ICONS.get(cond, "❓")
            pct = count / len(df) * 100
            st.markdown(
                f'{icon} <span class="badge {badge_class}">{cond}</span> — **{count}** ({pct:.1f}%)',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature highlights
    st.markdown("### 📐 Key Feature Ranges")
    highlights = {
        "Age": (df["Age"].min(), df["Age"].max()),
        "Sleep (hrs/day)": (df["Sleep_Hours_Per_Day"].min(), df["Sleep_Hours_Per_Day"].max()),
        "Work (hrs/week)": (df["Work_Hours_Per_Week"].min(), df["Work_Hours_Per_Week"].max()),
        "Mood Score": (df["Mood_Score (0-100)"].min(), df["Mood_Score (0-100)"].max()),
        "Therapy Sessions/mo": (df["Therapy_Sessions_Per_Month"].min(), df["Therapy_Sessions_Per_Month"].max()),
    }
    cols = st.columns(len(highlights))
    for col, (name, (lo, hi)) in zip(cols, highlights.items()):
        with col:
            st.markdown(
                f'<div class="glass-card" style="text-align:center">'
                f'<div style="color:#9ca3af;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px">{name}</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#a78bfa !important">{lo} — {hi}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: ASSESSMENT (Questionnaire + Lie Detection)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📝 Assessment":
    st.markdown("## 📝 Mental Health Assessment")
    st.markdown("Answer the following questions honestly. Our AI will analyze your responses for consistency and provide a prediction.")

    # Name input for history tracking
    st.markdown("")
    user_name = st.text_input("👤 Your Name", placeholder="Enter your name to track history...", help="Your name is used to save and track your assessment history over time.")
    if not user_name.strip():
        st.info("📌 Enter your name above to save this assessment to your personal history.")

    try:
        model, scaler, gender_enc, target_enc, feat_names = load_model()
    except Exception:
        st.error("⚠️ Model not found! Please run `python model_training.py` first.")
        st.stop()

    # Initialize session state for answers
    if "assessment_answers" not in st.session_state:
        st.session_state.assessment_answers = {}
    if "assessment_submitted" not in st.session_state:
        st.session_state.assessment_submitted = False

    sections = get_sections()
    total_qs = len(QUESTIONS)
    answered = sum(1 for q in QUESTIONS if q["id"] in st.session_state.assessment_answers)

    # Progress bar
    progress_pct = int(answered / total_qs * 100) if total_qs > 0 else 0
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<span style="color:#9ca3af;font-size:0.85rem">Progress: {answered}/{total_qs}</span>'
        f'<div class="progress-bar-bg" style="flex:1"><div class="progress-bar-fill" style="width:{progress_pct}%"></div></div>'
        f'<span style="color:#a78bfa;font-weight:600">{progress_pct}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.form("assessment_form"):
        for section in sections:
            section_qs = get_questions_by_section(section)
            icon_map = {
                "Demographics": "👤", "Lifestyle": "🏃", "How You Feel": "💭",
                "Social & Daily Life": "👥", "Clinical History": "🏥", "Additional": "🔍",
            }
            section_icon = icon_map.get(section, "📋")

            st.markdown(
                f'<div class="section-header">{section_icon} <b style="color:#c4b5fd !important">{section}</b></div>',
                unsafe_allow_html=True,
            )

            for q in section_qs:
                qid = q["id"]
                help_text = q.get("help", None)

                if q["response_type"] == "scale5":
                    val = st.select_slider(
                        q["text"],
                        options=[1, 2, 3, 4, 5],
                        value=st.session_state.assessment_answers.get(qid, 3),
                        format_func=lambda x: ["①", "②", "③", "④", "⑤"][x - 1],
                        key=f"assess_{qid}",
                        help=help_text,
                    )
                    st.session_state.assessment_answers[qid] = val

                elif q["response_type"] == "number":
                    val = st.number_input(
                        q["text"],
                        min_value=q.get("min", 0),
                        max_value=q.get("max", 100),
                        value=st.session_state.assessment_answers.get(qid, q.get("min", 0)),
                        key=f"assess_{qid}",
                        help=help_text,
                    )
                    st.session_state.assessment_answers[qid] = val

                elif q["response_type"] == "scale_range":
                    val = st.slider(
                        q["text"],
                        min_value=float(q.get("min", 0)),
                        max_value=float(q.get("max", 10)),
                        value=float(st.session_state.assessment_answers.get(qid, (q["min"] + q["max"]) / 2)),
                        step=float(q.get("step", 0.5)),
                        key=f"assess_{qid}",
                        help=help_text,
                    )
                    st.session_state.assessment_answers[qid] = val

                elif q["response_type"] == "choice":
                    opts = q["options"]
                    val = st.selectbox(
                        q["text"],
                        opts,
                        index=opts.index(st.session_state.assessment_answers[qid]) if qid in st.session_state.assessment_answers else 0,
                        key=f"assess_{qid}",
                        help=help_text,
                    )
                    st.session_state.assessment_answers[qid] = val

                elif q["response_type"] == "yesno":
                    val = st.selectbox(
                        q["text"],
                        ["No", "Yes"],
                        index=["No", "Yes"].index(st.session_state.assessment_answers[qid]) if qid in st.session_state.assessment_answers else 0,
                        key=f"assess_{qid}",
                        help=help_text,
                    )
                    st.session_state.assessment_answers[qid] = val

        st.markdown("---")
        submitted = st.form_submit_button("🔍  Analyze My Responses", use_container_width=True)

    if submitted:
        st.session_state.assessment_submitted = True
        answers = st.session_state.assessment_answers

        # Convert answers to model features
        features = answers_to_features(answers)

        # Run lie detection
        honesty_result = analyze_honesty(features, answers, QUESTIONS)

        # ── Honesty Score Display ──
        st.markdown("---")
        st.markdown("## 🕵️ Honesty & Consistency Analysis")

        score = honesty_result["honesty_score"]
        risk = honesty_result["risk_level"]
        score_class = {
            "low": "honesty-high", "medium": "honesty-medium",
            "high": "honesty-low", "critical": "honesty-critical",
        }.get(risk, "honesty-medium")

        col_score, col_summary = st.columns([1, 2])
        with col_score:
            st.markdown(
                f'<div class="honesty-meter">'
                f'<div class="metric-label">Honesty Score</div>'
                f'<div class="honesty-score {score_class}">{score:.0f}%</div>'
                f'<div style="color:#9ca3af;font-size:0.9rem">{honesty_result["flag_count"]} issue(s) detected</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_summary:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"#### Analysis Summary")
            st.markdown(honesty_result["summary"])
            if honesty_result["flags"]:
                st.markdown("")
                st.markdown("**Detected Issues:**")
                for flag in honesty_result["flags"]:
                    card_class = "flag-card" if flag["severity"] > 0.5 else "flag-card-warn"
                    severity_label = "High" if flag["severity"] > 0.7 else "Medium" if flag["severity"] > 0.4 else "Low"
                    st.markdown(
                        f'<div class="{card_class}">'
                        f'{flag["icon"]} <b style="color:#e0d4ff !important">{flag["description"]}</b><br/>'
                        f'<span style="font-size:0.8rem;color:#9ca3af !important">Severity: {severity_label} | {flag["details"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Prediction from questionnaire ──
        st.markdown("---")
        st.markdown("## 🔮 AI Prediction")

        if score < 25:
            st.warning("⚠️ **Low honesty score detected.** The prediction below may not reflect your actual condition due to significant inconsistencies in your responses.")

        # Build feature array in correct order
        gender_encoded = gender_enc.transform([features.get("Gender", "Male")])[0] if isinstance(features.get("Gender"), str) else features.get("Gender", 0)
        # Need to handle gender since it was stored as string from questionnaire
        gender_val_raw = answers.get("q_gender", "Male")
        gender_encoded = gender_enc.transform([gender_val_raw])[0]

        input_data = np.array([[
            features.get("Age", 30),
            gender_encoded,
            features.get("Sleep_Hours_Per_Day", 7.0),
            features.get("Work_Hours_Per_Week", 40),
            features.get("Exercise_Frequency_Per_Week", 3),
            features.get("Mood_Score (0-100)", 50),
            features.get("Anxiety_Score (0-100)", 30),
            features.get("Depression_Score (0-100)", 20),
            features.get("Stress_Score (0-100)", 40),
            features.get("Energy_Level (0-100)", 60),
            features.get("Social_Interaction (1-10)", 5),
            features.get("Concentration (1-10)", 6),
            features.get("Appetite (1-10)", 6),
            features.get("Life_Satisfaction (0-100)", 50),
            features.get("Therapy_Sessions_Per_Month", 1),
            features.get("Medication", 0),
            features.get("Family_History", 0),
            features.get("Hallucinations", 0),
            features.get("Manic_Episodes_Per_Year", 0),
        ]])

        input_scaled = scaler.transform(input_data)

        # ── Random Forest Prediction ──
        rf_prediction = model.predict(input_scaled)[0]
        rf_probabilities = model.predict_proba(input_scaled)[0]
        rf_condition = target_enc.inverse_transform([rf_prediction])[0]
        rf_confidence = rf_probabilities[rf_prediction] * 100

        # ── MLP Prediction ──
        mlp_model = load_mlp_model()
        mlp_available = mlp_model is not None

        if mlp_available:
            with torch.no_grad():
                mlp_input = torch.FloatTensor(input_scaled)
                mlp_output = mlp_model(mlp_input)
                mlp_probs = torch.softmax(mlp_output, dim=1).numpy()[0]
                mlp_pred_idx = mlp_probs.argmax()
                mlp_condition = target_enc.inverse_transform([mlp_pred_idx])[0]
                mlp_confidence = mlp_probs[mlp_pred_idx] * 100

        # ── Display: Side-by-side Model Comparison ──
        if mlp_available:
            st.markdown("### ⚡ Model Comparison")
            col_rf, col_mlp = st.columns(2)

            with col_rf:
                rf_icon = CONDITION_ICONS.get(rf_condition, "❓")
                rf_color = CONDITION_COLORS.get(rf_condition, "#a78bfa")
                st.markdown(
                    f'<div class="prediction-box" style="border-color: rgba(59,130,246,0.35)">'
                    f'<div style="color:#9ca3af;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">🌲 Random Forest (ML)</div>'
                    f'<div style="font-size:2.5rem">{rf_icon}</div>'
                    f'<div class="prediction-label" style="color:{rf_color} !important;font-size:1.5rem">{rf_condition}</div>'
                    f'<div style="color:#9ca3af;font-size:0.9rem;margin-top:4px">Confidence: <b style="color:{rf_color}">{rf_confidence:.1f}%</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_mlp:
                mlp_icon = CONDITION_ICONS.get(mlp_condition, "❓")
                mlp_color = CONDITION_COLORS.get(mlp_condition, "#a78bfa")
                st.markdown(
                    f'<div class="prediction-box" style="border-color: rgba(168,85,247,0.35)">'
                    f'<div style="color:#9ca3af;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">🧠 MLP Neural Network (DL)</div>'
                    f'<div style="font-size:2.5rem">{mlp_icon}</div>'
                    f'<div class="prediction-label" style="color:{mlp_color} !important;font-size:1.5rem">{mlp_condition}</div>'
                    f'<div style="color:#9ca3af;font-size:0.9rem;margin-top:4px">Confidence: <b style="color:{mlp_color}">{mlp_confidence:.1f}%</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # ── Ensemble (Averaged) Prediction ──
            ensemble_probs = (rf_probabilities + mlp_probs) / 2.0
            ensemble_pred_idx = ensemble_probs.argmax()
            ensemble_condition = target_enc.inverse_transform([ensemble_pred_idx])[0]
            ensemble_confidence = ensemble_probs[ensemble_pred_idx] * 100
            ens_icon = CONDITION_ICONS.get(ensemble_condition, "❓")
            ens_color = CONDITION_COLORS.get(ensemble_condition, "#a78bfa")

            agree_text = "✅ Both models agree!" if rf_condition == mlp_condition else "⚠️ Models disagree — ensemble shown below"
            st.markdown(f"<div style='text-align:center;color:#c4b5fd;font-size:1rem;margin-bottom:12px'>{agree_text}</div>", unsafe_allow_html=True)

            st.markdown(
                f'<div class="prediction-box" style="border-color: rgba(16,185,129,0.4);background:linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(139,92,246,0.1) 100%)">'
                f'<div style="color:#34d399;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">🎯 Ensemble Prediction (RF + MLP Averaged)</div>'
                f'<div style="font-size:3rem">{ens_icon}</div>'
                f'<div class="prediction-label" style="color:{ens_color} !important">{ensemble_condition}</div>'
                f'<div style="color:#9ca3af;font-size:1rem;margin-top:4px">Confidence: <b style="color:{ens_color}">{ensemble_confidence:.1f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Use ensemble for final values
            predicted_condition = ensemble_condition
            confidence = ensemble_confidence
            probabilities = ensemble_probs

        else:
            # MLP not available — show RF only
            predicted_condition = rf_condition
            confidence = rf_confidence
            probabilities = rf_probabilities
            icon = CONDITION_ICONS.get(predicted_condition, "❓")
            color = CONDITION_COLORS.get(predicted_condition, "#a78bfa")

            st.markdown(
                f'<div class="prediction-box">'
                f'<div style="font-size:3rem">{icon}</div>'
                f'<div class="prediction-label" style="color:{color} !important">{predicted_condition}</div>'
                f'<div style="color:#9ca3af;font-size:1rem;margin-top:4px">Confidence: <b style="color:{color}">{confidence:.1f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.info("💡 **Tip:** Run `python mlp_training.py` to train the MLP model and unlock dual-model comparison!")

        st.markdown("")

        # Probability chart — show comparison if MLP available
        if mlp_available:
            st.markdown("#### 📊 Probability Comparison (RF vs MLP vs Ensemble)")
            compare_df = pd.DataFrame({
                "Condition": list(target_enc.classes_) * 3,
                "Probability": list(rf_probabilities * 100) + list(mlp_probs * 100) + list(ensemble_probs * 100),
                "Model": ["🌲 Random Forest"] * len(target_enc.classes_) + ["🧠 MLP"] * len(target_enc.classes_) + ["🎯 Ensemble"] * len(target_enc.classes_),
            })

            fig = px.bar(
                compare_df, x="Probability", y="Condition", orientation="h",
                color="Model", barmode="group",
                color_discrete_map={"🌲 Random Forest": "#60a5fa", "🧠 MLP": "#a78bfa", "🎯 Ensemble": "#34d399"},
                text=compare_df["Probability"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c4b5fd",
                xaxis=dict(gridcolor="rgba(139,92,246,0.08)", title="Probability (%)"),
                yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c4b5fd")),
                margin=dict(l=20, r=20, t=10, b=20),
                height=450,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                "Condition": target_enc.classes_,
                "Probability": probabilities * 100,
            }).sort_values("Probability", ascending=True)

            fig = px.bar(
                prob_df, x="Probability", y="Condition", orientation="h",
                color="Condition", color_discrete_map=CONDITION_COLORS,
                text=prob_df["Probability"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c4b5fd", showlegend=False,
                xaxis=dict(gridcolor="rgba(139,92,246,0.08)", title="Probability (%)"),
                yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=350,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("")
        st.warning("⚠️ **Disclaimer:** This is an AI-based screening tool and NOT a clinical diagnosis. Please consult a qualified mental health professional for proper evaluation.")

        # Save to history if name provided
        if user_name.strip():
            prob_dict = {cond: round(float(p) * 100, 1) for cond, p in zip(target_enc.classes_, probabilities)}
            feat_dict = {k: float(v) if not isinstance(v, str) else v for k, v in features.items()}
            save_assessment(
                user_name=user_name,
                features=feat_dict,
                prediction=predicted_condition,
                confidence=confidence,
                honesty_score=honesty_result["honesty_score"],
                probabilities=prob_dict,
                flags_count=honesty_result["flag_count"],
            )
            st.success(f"✅ Assessment saved to **{user_name.strip()}'s** history! View it on the 📅 History page.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📅 History":
    st.markdown("## 📅 Assessment History")
    st.markdown("Track your mental health journey over time. View past assessments, trends, and changes.")

    all_users = get_all_users()

    if not all_users:
        st.info("📝 No assessments saved yet. Complete an assessment on the 📝 Assessment page with your name to start tracking your history.")
    else:
        selected_user = st.selectbox("👤 Select Person", all_users, help="Choose a person to view their assessment history.")

        if selected_user:
            records = get_user_history(selected_user)
            trend_data = get_user_trend_data(selected_user)

            st.markdown("")

            # Summary cards
            total_assessments = len(records)
            latest = records[0] if records else {}
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Total Assessments</div>'
                    f'<div class="metric-value">{total_assessments}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Latest Prediction</div>'
                    f'<div class="metric-value" style="font-size:1.4rem">{latest.get("prediction", "N/A")}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Latest Confidence</div>'
                    f'<div class="metric-value">{latest.get("confidence", 0):.0f}%</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                avg_honesty = sum(r.get("honesty_score", 100) for r in records) / len(records) if records else 0
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Avg Honesty</div>'
                    f'<div class="metric-value">{avg_honesty:.0f}%</div></div>',
                    unsafe_allow_html=True,
                )

            if len(trend_data) >= 2:
                st.markdown("")
                st.markdown("### 📈 Trends Over Time")

                tab1, tab2, tab3 = st.tabs(["🧠 Mental Scores", "🎯 Confidence & Honesty", "📊 Predictions"])

                import pandas as pd
                trend_df = pd.DataFrame(trend_data)
                trend_df["Assessment #"] = range(1, len(trend_df) + 1)

                with tab1:
                    score_cols = [c for c in ["Mood", "Anxiety", "Depression", "Stress", "Energy", "Life"] if c in trend_df.columns and trend_df[c].notna().any()]
                    if score_cols:
                        fig = px.line(
                            trend_df, x="Assessment #", y=score_cols,
                            markers=True,
                            color_discrete_sequence=["#a78bfa", "#fbbf24", "#818cf8", "#f87171", "#34d399", "#60a5fa"],
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#c4b5fd",
                            xaxis=dict(gridcolor="rgba(139,92,246,0.08)", dtick=1),
                            yaxis=dict(gridcolor="rgba(139,92,246,0.08)", title="Score (0-100)"),
                            legend=dict(bgcolor="rgba(0,0,0,0)"),
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Score data not available for trend chart.")

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df["Assessment #"], y=trend_df["Confidence"],
                        mode="lines+markers", name="Confidence",
                        line=dict(color="#a78bfa", width=3), marker=dict(size=8),
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_df["Assessment #"], y=trend_df["Honesty Score"],
                        mode="lines+markers", name="Honesty Score",
                        line=dict(color="#34d399", width=3), marker=dict(size=8),
                    ))
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#c4b5fd",
                        xaxis=dict(gridcolor="rgba(139,92,246,0.08)", title="Assessment #", dtick=1),
                        yaxis=dict(gridcolor="rgba(139,92,246,0.08)", title="Score (%)", range=[0, 105]),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    pred_counts = trend_df["Prediction"].value_counts()
                    fig = px.pie(
                        values=pred_counts.values, names=pred_counts.index,
                        color=pred_counts.index, color_discrete_map=CONDITION_COLORS,
                        hole=0.4,
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#c4b5fd", height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Past records table
            st.markdown("")
            st.markdown("### 📋 Past Assessments")
            for i, record in enumerate(records):
                ts = record["timestamp"][:16].replace("T", " at ")
                pred = record["prediction"]
                conf = record["confidence"]
                honesty = record["honesty_score"]
                flags = record["flags_count"]
                icon = CONDITION_ICONS.get(pred, "❓")
                color = CONDITION_COLORS.get(pred, "#a78bfa")

                with st.expander(f"{icon} **{pred}** — {ts}  |  Confidence: {conf:.0f}%  |  Honesty: {honesty:.0f}%", expanded=(i == 0)):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Prediction:** {icon} {pred}")
                        st.markdown(f"**Confidence:** {conf:.1f}%")
                    with col2:
                        st.markdown(f"**Honesty Score:** {honesty:.1f}%")
                        st.markdown(f"**Flags:** {flags} issue(s)")
                    with col3:
                        st.markdown(f"**Date:** {record['timestamp'][:10]}")
                        st.markdown(f"**Time:** {record['timestamp'][11:19]}")

                    # Show probabilities
                    if "probabilities" in record and record["probabilities"]:
                        probs = record["probabilities"]
                        prob_df_hist = pd.DataFrame({
                            "Condition": list(probs.keys()),
                            "Probability": list(probs.values()),
                        }).sort_values("Probability", ascending=True)
                        fig = px.bar(
                            prob_df_hist, x="Probability", y="Condition", orientation="h",
                            color="Condition", color_discrete_map=CONDITION_COLORS,
                            text=prob_df_hist["Probability"].apply(lambda x: f"{x:.1f}%"),
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#c4b5fd", showlegend=False,
                            xaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
                            yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
                            margin=dict(l=20, r=20, t=10, b=20),
                            height=250,
                        )
                        fig.update_traces(textposition="outside")
                        st.plotly_chart(fig, use_container_width=True, key=f"hist_chart_{i}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: DATA EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Data Explorer":
    df = load_data()
    st.markdown("## 📊 Data Explorer")
    st.markdown("Dive deep into the mental health dataset with interactive visualizations.")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📦 Box Plots", "📋 Raw Data"])

    with tab1:
        feature = st.selectbox(
            "Select feature to explore",
            [c for c in df.columns if c not in ("Patient_ID", "Mental_Health_Condition", "Gender")],
        )
        fig = px.histogram(
            df, x=feature, color="Mental_Health_Condition",
            color_discrete_map=CONDITION_COLORS,
            barmode="overlay", opacity=0.7,
            marginal="box",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd",
            xaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if "Patient_ID" in numeric_cols:
            numeric_cols.remove("Patient_ID")
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=["#1a1333", "#7c3aed", "#f472b6"],
            aspect="auto",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd", margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        box_feature = st.selectbox(
            "Select feature for box plot",
            [c for c in df.columns if c not in ("Patient_ID", "Mental_Health_Condition", "Gender")],
            key="box_feat",
        )
        fig = px.box(
            df, x="Mental_Health_Condition", y=box_feature,
            color="Mental_Health_Condition",
            color_discrete_map=CONDITION_COLORS,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd", showlegend=False,
            xaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(
            df.drop(columns=["Patient_ID"]),
            use_container_width=True,
            height=500,
        )
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: MODEL INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📈 Model Insights":
    st.markdown("## 📈 Model Performance & Insights")

    try:
        model, scaler, gender_enc, target_enc, feat_names = load_model()
    except Exception:
        st.error("⚠️ Model not found! Please run `python model_training.py` first.")
        st.stop()

    df = load_data()

    # Retrain metrics for display
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    df_model = df.drop(columns=["Patient_ID"])
    ge = LabelEncoder()
    df_model["Gender"] = ge.fit_transform(df_model["Gender"])
    X = df_model.drop(columns=["Mental_Health_Condition"])
    y_raw = df_model["Mental_Health_Condition"]
    te = LabelEncoder()
    y = te.fit_transform(y_raw)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Metrics row
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Accuracy</div>'
            f'<div class="metric-value">{acc*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">F1 Score</div>'
            f'<div class="metric-value">{f1*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Precision</div>'
            f'<div class="metric-value">{prec*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Recall</div>'
            f'<div class="metric-value">{rec*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["🔥 Feature Importance", "📊 Confusion Matrix", "📋 Classification Report"])

    with tab1:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#1a1333", "#7c3aed", "#f472b6"],
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd",
            xaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            yaxis=dict(gridcolor="rgba(139,92,246,0.08)"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=500,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        labels = te.classes_
        fig = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale=["#1a1333", "#7c3aed", "#f472b6"],
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd",
            margin=dict(l=20, r=20, t=30, b=20),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        report = classification_report(y_test, y_pred, target_names=te.classes_, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # Model details
    st.markdown("---")
    st.markdown("### 🔧 Model Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Algorithm:** Random Forest Classifier")
        st.markdown(f"**Estimators:** {model.n_estimators}")
        st.markdown(f"**Max Depth:** {model.max_depth}")
        st.markdown(f"**Min Samples Split:** {model.min_samples_split}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**Training Samples:** {len(y_train)}")
        st.markdown(f"**Test Samples:** {len(y_test)}")
        st.markdown(f"**Features:** {len(feat_names)}")
        st.markdown(f"**Classes:** {len(te.classes_)}")
        st.markdown("</div>", unsafe_allow_html=True)
