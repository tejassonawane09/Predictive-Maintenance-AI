import streamlit as st
import pandas as pd
import joblib
import time
import traceback
from datetime import datetime


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    /* ---------- Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---------- App background ---------- */
    .stApp {
        background: #0a0f1c;
        background-image:
            radial-gradient(circle at 15% 10%, rgba(56,189,248,0.10), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(139,92,246,0.10), transparent 40%),
            linear-gradient(180deg, #0a0f1c 0%, #050810 100%);
        color: #e2e8f0;
    }

    /* ---------- Hide only the deploy button and menu ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    [data-testid="stMainMenu"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ---------- Sidebar arrow — always visible ---------- */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 10px !important;
        color: #38bdf8 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.45) !important;
    }

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] svg {
        fill: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #050912 100%);
        border-right: 1px solid rgba(56,189,248,0.12);
    }

    section[data-testid="stSidebar"] * {
        color: #cbd5e1;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }

    section[data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 11px 15px;
        transition: all 0.25s ease;
        cursor: pointer;
        font-weight: 500;
        font-size: 0.92rem;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(56,189,248,0.10);
        border-color: rgba(56,189,248,0.35);
        transform: translateX(3px);
    }

    /* ---------- Headings ---------- */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        font-weight: 700 !important;
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 32px 36px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(56,189,248,0.10) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(56,189,248,0.22);
        box-shadow: 0 12px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        top: -60%; right: -10%;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(139,92,246,0.28), transparent 70%);
        filter: blur(50px);
        pointer-events: none;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.025em;
    }

    .hero p {
        margin: 10px 0 0 0;
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 780px;
    }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,41,59,0.85) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(56,189,248,0.18);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #a855f7);
        opacity: 0.75;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(56,189,248,0.45);
        box-shadow: 0 14px 38px rgba(56,189,248,0.18), inset 0 1px 0 rgba(255,255,255,0.06);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.76rem !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
        letter-spacing: -0.02em;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        box-shadow: 0 8px 24px rgba(99,102,241,0.35);
        transition: all 0.25s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(99,102,241,0.55);
        filter: brightness(1.08);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ---------- Inputs ---------- */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(15,23,42,0.85) !important;
        border: 1px solid rgba(56,189,248,0.22) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        transition: all 0.2s ease;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    }

    label, .stSelectbox label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ---------- Progress bar ---------- */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #22d3ee, #6366f1, #a855f7) !important;
        border-radius: 10px;
        height: 12px;
    }

    .stProgress > div > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px;
        height: 12px;
    }

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        border-left: 4px solid;
        backdrop-filter: blur(8px);
        padding: 16px 20px;
        font-weight: 500;
    }

    /* ---------- Divider ---------- */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);
        margin: 28px 0;
    }

    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(56,189,248,0.15);
    }

    /* ---------- Section card ---------- */
    .section-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.6), rgba(15,23,42,0.75));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 24px 28px;
        margin: 16px 0;
        box-shadow: 0 8px 28px rgba(0,0,0,0.28);
    }

    .section-card h3 {
        margin-top: 0;
        color: #f1f5f9 !important;
    }

    /* ---------- Badge ---------- */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .badge-live {
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.4);
    }

    .badge-info {
        background: rgba(56,189,248,0.15);
        color: #38bdf8;
        border: 1px solid rgba(56,189,248,0.4);
    }

    .badge-warn {
        background: rgba(251,191,36,0.15);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.4);
    }

    /* ---------- Result panel ---------- */
    .result-panel {
        border-radius: 18px;
        padding: 26px 30px;
        margin-top: 12px;
        border: 1px solid;
        animation: fadeIn 0.45s ease;
    }

    .result-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(239,68,68,0.05));
        border-color: rgba(239,68,68,0.45);
        box-shadow: 0 10px 36px rgba(239,68,68,0.15);
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(34,197,94,0.05));
        border-color: rgba(34,197,94,0.45);
        box-shadow: 0 10px 36px rgba(34,197,94,0.15);
    }

    .result-panel h2 {
        margin: 0 0 8px 0;
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .result-panel p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.65;
    }

    /* ---------- Probability ring ---------- */
    .prob-ring {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.03em;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease;
    }

    /* ---------- Risk meter ---------- */
    .feature-bar-wrap {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 6px;
    }

    .feature-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        padding: 26px 0 10px 0;
        border-top: 1px solid rgba(148,163,184,0.1);
        margin-top: 40px;
        letter-spacing: 0.03em;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0b1220; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #38bdf8, #6366f1);
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #a855f7; }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/predictive_maintenance_model.pkl")


model = None
model_loaded = False
model_error = None

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_error = str(e)
    model_loaded = False


# =========================================================
# EXTRACT MODEL METADATA
# =========================================================

model_info = {}

if model_loaded and model is not None:
    try:
        # Model type
        model_info["type"] = type(model).__name__

        # Handle pipelines: find the classifier
        classifier = model
        if hasattr(model, "named_steps"):
            # It's a Pipeline — get final step
            last_step = list(model.named_steps.values())[-1]
            classifier = last_step
            model_info["pipeline_steps"] = len(model.named_steps)

        # Get feature names the model was trained on
        expected_features = None

        # Try pipeline's feature names
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

        # Try nested classifier
        elif hasattr(classifier, "feature_names_in_"):
            expected_features = list(classifier.feature_names_in_)

        # Try n_features_in_
        if expected_features is None and hasattr(classifier, "n_features_in_"):
            model_info["n_features"] = int(classifier.n_features_)
        elif expected_features is not None:
            model_info["n_features"] = len(expected_features)

        # Tree-based params
        if hasattr(classifier, "n_estimators"):
            model_info["n_estimators"] = classifier.n_estimators

        if hasattr(classifier, "max_depth"):
            model_info["max_depth"] = classifier.max_depth

        if hasattr(classifier, "classes_"):
            model_info["classes"] = list(classifier.classes_)

        # Store expected feature names
        model_info["expected_features"] = expected_features

        # Feature importances from model (if available)
        if hasattr(classifier, "feature_importances_"):
            model_info["feature_importances_"] = list(classifier.feature_importances_)

    except Exception:
        pass


# =========================================================
# SESSION STATE — prediction history
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "custom_threshold" not in st.session_state:
    st.session_state.custom_threshold = 0.40


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
        <div style="text-align:center; padding: 12px 0 22px 0;">
            <div style="font-size: 2.4rem; line-height: 1;">⚙️</div>
            <div style="font-weight: 800; font-size: 1.1rem; color:#f1f5f9; margin-top:6px;
                        font-family:'Space Grotesk', sans-serif;">
                Predictive <span style="color:#38bdf8;">AI</span>
            </div>
            <div style="font-size: 0.7rem; color:#64748b; letter-spacing:0.16em;
                        text-transform:uppercase; margin-top:4px;">
                Maintenance System
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.74rem; color:#94a3b8; letter-spacing:0.12em; "
        "text-transform:uppercase; margin-bottom:8px;'>🧭 Navigation</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Go to",
        ["Dashboard", "Failure Prediction", "Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    status_color = "#4ade80" if model_loaded else "#f87171"
    status_text = "Online" if model_loaded else "Offline"
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border:1px solid rgba(148,163,184,0.14);
                    border-radius:12px; padding:14px 16px;">
            <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                        letter-spacing:0.1em;">
                System Status
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-top:8px;">
                <span style="width:9px; height:9px; border-radius:50%; background:{status_color};
                             box-shadow: 0 0 10px {status_color};"></span>
                <span style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">
                    Model {status_text}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- Advanced settings ----------
    with st.expander("⚙️ Advanced Settings"):
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.40,
            step=0.05,
            help="Lower threshold = more sensitive (catches more failures, more false alarms). Higher = stricter."
        )
        st.session_state.custom_threshold = threshold
        st.caption(f"Current: **{threshold:.2f}**")

    st.markdown("---")
    st.caption("v2.1 · Enhanced")


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
    <div class="hero fade-in">
        <span class="badge badge-live">● Live System</span>
        <h1>⚙️ Predictive Maintenance AI</h1>
        <p>
            An intelligent machine-failure prediction system powered by a
            <b style="color:#38bdf8;">Random Forest</b> classifier. Analyze operational and
            sensor parameters to estimate the probability of equipment failure before it happens.
        </p>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("### 📊 Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Records", "10,000", delta="Full dataset")
    with col2:
        st.metric("Failure Rate", "3.39%", delta="Dataset rate")
    with col3:
        st.metric("Model", "Random Forest", delta="Trained")
    with col4:
        st.metric("ROC-AUC", "97.99%", delta="Evaluation score")

    st.markdown("---")

    colA, colB = st.columns([1.3, 1])

    with colA:
        st.markdown("""
            <div class="section-card">
                <span class="badge badge-info">Objective</span>
                <h3>🎯 Project Objective</h3>
                <p style="color:#cbd5e1; line-height:1.75; margin-bottom:0;">
                    The objective of this project is to predict whether an industrial machine
                    is likely to experience a failure based on its operating conditions.
                    The model estimates failure risk to support maintenance investigation
                    and proactive monitoring.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="section-card">
                <span class="badge badge-info">Inputs</span>
                <h3>🔧 Machine Parameters</h3>
                <p style="color:#cbd5e1; line-height:1.75; margin-bottom:0;">
                    The model uses machine type, air temperature, process temperature,
                    rotational speed, torque, tool wear, temperature difference, and a
                    derived power proxy to estimate failure risk.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
            <div class="section-card" style="height:100%;">
                <span class="badge badge-warn">Notice</span>
                <h3>ℹ️ Important</h3>
                <p style="color:#cbd5e1; line-height:1.75;">
                    This application is designed as a <b>predictive analytics demonstration</b>.
                    Predictions should support maintenance investigation and must not be
                    treated as a guaranteed failure diagnosis.
                </p>
                <hr style="margin:18px 0;">
                <p style="color:#94a3b8; font-size:0.85rem; margin:0;">
                    <b style="color:#38bdf8;">Recommendation:</b> Always combine model output
                    with domain expertise and physical inspection.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ---------- Loaded model info ----------
    if model_loaded and model_info:
        st.markdown("---")
        st.markdown("#### 🔎 Loaded Model Info")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)

        with info_col1:
            st.metric("Type", model_info.get("type", "—"))
        with info_col2:
            st.metric("Features", model_info.get("n_features", "—"))
        with info_col3:
            st.metric("Estimators", model_info.get("n_estimators", "—"))
        with info_col4:
            st.metric("Pipeline Steps", model_info.get("pipeline_steps", "—"))


# =========================================================
# FAILURE PREDICTION
# =========================================================

elif page == "Failure Prediction":

    st.markdown("### 🔍 Machine Failure Prediction")
    st.caption("Enter the machine operating parameters below to estimate failure probability.")

    # ---------- Model missing ----------
    if not model_loaded:
        st.error(
            "⚠️ Model could not be loaded. Please verify "
            "`models/predictive_maintenance_model.pkl` exists."
        )
        if model_error:
            with st.expander("Show error details"):
                st.code(model_error)

        st.stop()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🏭 Machine & Temperature")
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temperature = st.number_input(
            "Air Temperature [K]", min_value=250.0, max_value=350.0,
            value=300.0, step=0.1
        )
        process_temperature = st.number_input(
            "Process Temperature [K]", min_value=250.0, max_value=400.0,
            value=310.0, step=0.1
        )
        rotational_speed = st.number_input(
            "Rotational Speed [rpm]", min_value=500, max_value=3000,
            value=1500, step=10
        )

    with col2:
        st.markdown("##### ⚡ Mechanical Load")
        torque = st.number_input(
            "Torque [Nm]", min_value=0.0, max_value=100.0,
            value=40.0, step=0.1
        )
        tool_wear = st.number_input(
            "Tool Wear [min]", min_value=0, max_value=300,
            value=100, step=1
        )
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.info("💡 Adjust values to simulate different operating conditions.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Input range warning ----------
    if tool_wear > 200:
        st.warning("⚠️ High tool wear detected. This is a common failure indicator.")

    if (process_temperature - air_temperature) > 12:
        st.warning("⚠️ High temperature difference. Check cooling system.")

    if torque > 70 and rotational_speed > 2500:
        st.warning("⚠️ Combined high torque and speed. This may increase failure risk.")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    predict_button = st.button(
        "🔮  Predict Machine Failure",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        with st.spinner("Analyzing machine parameters..."):
            time.sleep(0.4)

            temperature_difference = process_temperature - air_temperature
            power_proxy = torque * rotational_speed

            input_data = pd.DataFrame({
                "Type": [machine_type],
                "Air temperature [K]": [air_temperature],
                "Process temperature [K]": [process_temperature],
                "Rotational speed [rpm]": [rotational_speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
                "Temperature Difference [K]": [temperature_difference],
                "Power Proxy": [power_proxy]
            })

            # ---------- Align features with model expectations ----------
            expected = model_info.get("expected_features")

            try:
                # If model expects specific column order/names, align
                if expected:
                    # Add any missing columns with default 0
                    for col in expected:
                        if col not in input_data.columns:
                            input_data[col] = 0
                    # Keep only expected columns in expected order
                    input_data = input_data[expected]

                probability = model.predict_proba(input_data)[0][1]
                prediction_raw = model.predict(input_data)[0]

            except Exception as e:
                st.error("❌ Prediction failed. The model rejected the input.")
                with st.expander("🔧 Technical details (for debugging)"):
                    st.code(traceback.format_exc())
                    st.markdown("**Input sent to model:**")
                    st.dataframe(input_data)
                    st.markdown("**Model expected features:**")
                    st.write(expected if expected else "Unknown")
                st.stop()

        # Use custom threshold from sidebar
        threshold = st.session_state.custom_threshold
        prediction = probability >= threshold

        # ---------- Save to history ----------
        st.session_state.prediction_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": machine_type,
            "probability": probability,
            "result": "Failure" if prediction else "Safe"
        })
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]

        st.markdown("---")
        st.markdown("### 📢 Prediction Result")

        if prediction:
            panel_class = "result-danger"
            icon = "⚠️"
            title = "Potential Machine Failure Detected"
            desc = (f"The predicted failure probability is <b>{probability:.2%}</b>, which is above the "
                    f"configured <b>{threshold:.2f}</b> decision threshold. Further maintenance "
                    f"investigation is strongly recommended.")
            badge = ('<span class="badge" style="background:rgba(239,68,68,0.15); '
                     'color:#f87171; border:1px solid rgba(239,68,68,0.4);">High Risk</span>')
            ring_color = "#f87171"
            bar_gradient = "linear-gradient(90deg, #22d3ee, #6366f1, #ef4444)"
        else:
            panel_class = "result-safe"
            icon = "✅"
            title = "No Failure Predicted"
            desc = (f"The predicted failure probability is <b>{probability:.2%}</b>, which is below the "
                    f"configured <b>{threshold:.2f}</b> decision threshold. The machine appears to be "
                    f"operating within safe parameters.")
            badge = ('<span class="badge" style="background:rgba(34,197,94,0.15); '
                     'color:#4ade80; border:1px solid rgba(34,197,94,0.4);">Low Risk</span>')
            ring_color = "#4ade80"
            bar_gradient = "linear-gradient(90deg, #22d3ee, #6366f1, #22c55e)"

        st.markdown(f"""
            <div class="result-panel {panel_class} fade-in">
                {badge}
                <h2>{icon} {title}</h2>
                <p>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(f"""
                <div class="section-card" style="text-align:center;">
                    <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.1em;
                                text-transform:uppercase;">
                        Failure Probability
                    </div>
                    <div class="prob-ring" style="margin-top:10px; color:{ring_color};">
                        {probability:.2%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
                <div class="section-card">
                    <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:0.1em;
                                text-transform:uppercase;">
                        Risk Meter
                    </div>
                    <div style="margin-top:14px;">
                        <div class="feature-bar-wrap" style="height:14px;">
                            <div class="feature-bar" style="width:{probability*100:.1f}%;
                                 background: {bar_gradient};">
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px;
                                    font-size:0.78rem; color:#94a3b8;">
                            <span>0% · Safe</span>
                            <span>Threshold {threshold:.2f}</span>
                            <span>100% · Critical</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ---------- Confidence interpretation ----------
        if probability < 0.20:
            conf_text = "🟢 Very low risk — machine appears healthy"
        elif probability < 0.40:
            conf_text = "🟢 Low risk — continue normal monitoring"
        elif probability < 0.60:
            conf_text = "🟡 Moderate risk — schedule inspection soon"
        elif probability < 0.80:
            conf_text = "🟠 High risk — maintenance recommended"
        else:
            conf_text = "🔴 Critical risk — immediate inspection advised"

        st.info(f"**Interpretation:** {conf_text}")

        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame({
                "Parameter": [
                    "Machine Type", "Air Temperature", "Process Temperature",
                    "Rotational Speed", "Torque", "Tool Wear",
                    "Temperature Difference", "Power Proxy"
                ],
                "Value": [
                    machine_type,
                    f"{air_temperature:.1f} K",
                    f"{process_temperature:.1f} K",
                    f"{rotational_speed} rpm",
                    f"{torque:.1f} Nm",
                    f"{tool_wear} min",
                    f"{temperature_difference:.1f} K",
                    f"{power_proxy:.0f}"
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ---------- Prediction history ----------
    if st.session_state.prediction_history:
        st.markdown("---")
        with st.expander(f"🕒 Recent Predictions ({len(st.session_state.prediction_history)})"):
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df["probability"] = history_df["probability"].apply(
                lambda x: f"{x:.2%}"
            )
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    st.markdown("### 📈 Model Insights")
    st.caption("Model performance metrics, threshold optimization, and feature-level insights.")

    st.markdown("#### 🧠 Final Model — Random Forest Classifier")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "99.10%")
    with col2:
        st.metric("Precision", "94.64%")
    with col3:
        st.metric("Recall", "77.94%")
    with col4:
        st.metric("ROC-AUC", "97.99%")

    st.info(
        "Since machine failures are relatively rare in the dataset, accuracy alone is not sufficient. "
        "Precision, recall, F1 score, and ROC-AUC provide a more complete picture of model performance."
    )

    st.markdown("---")

    st.markdown("#### 🎚️ Threshold Optimization")
    st.write(
        "The classification threshold was evaluated at multiple values. During threshold analysis "
        "on the current held-out evaluation split, a threshold of **0.40** produced the best F1 score "
        "among the tested thresholds."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Threshold", "0.40")
    with col2:
        st.metric("Precision", "93.33%")
    with col3:
        st.metric("Recall", "82.35%")
    with col4:
        st.metric("F1 Score", "87.50%")

    threshold_data = pd.DataFrame({
        "Metric": ["False Positives", "False Negatives", "True Positives"],
        "Value": [4, 12, 56]
    })
    st.dataframe(threshold_data, use_container_width=True, hide_index=True)
    st.caption("The 0.40 threshold increases failure detection sensitivity compared with the default 0.50 threshold.")

    st.markdown("---")

    st.markdown("#### 🔍 Feature Importance")
    st.write(
        "Feature importance shows which input features the trained Random Forest relied on most "
        "when making predictions. Higher importance indicates greater contribution to the model's "
        "decision process."
    )

    importance_path = "artifacts/feature_importance.csv"
    importance_df = None

    # ---------- Try CSV first ----------
    try:
        importance_df = pd.read_csv(importance_path)
        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )
        st.caption("📁 Source: `artifacts/feature_importance.csv`")

    except FileNotFoundError:
        # ---------- Fall back to model's built-in importances ----------
        fi = model_info.get("feature_importances_")
        expected = model_info.get("expected_features")

        if fi and expected and len(fi) == len(expected):
            importance_df = pd.DataFrame({
                "Feature": expected,
                "Importance": fi
            })
            importance_df["Feature"] = (
                importance_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )
            st.caption("📁 Source: built-in `feature_importances_` from the loaded model")

    if importance_df is not None:
        top_features = (
            importance_df
            .sort_values(by="Importance", ascending=False)
            .head(10)
            .sort_values(by="Importance")
        )

        st.bar_chart(
            top_features.set_index("Feature")["Importance"],
            horizontal=True,
            color="#38bdf8"
        )
        st.caption(
            "Feature importance represents model reliance and should not be "
            "interpreted as proof of causation."
        )
    else:
        st.warning(
            "Feature importance data was not found.\n\n"
            "Run: `python src/model_explainability.py` "
            "or ensure the model exposes `feature_importances_`."
        )

    # ---------- Model metadata table ----------
    if model_info:
        st.markdown("---")
        st.markdown("#### 📦 Loaded Model Metadata")

        meta_rows = [{"Property": "Type", "Value": model_info.get("type", "—")}]

        if "n_features" in model_info:
            meta_rows.append({"Property": "Number of Features", "Value": model_info["n_features"]})
        if "n_estimators" in model_info:
            meta_rows.append({"Property": "Number of Estimators", "Value": model_info["n_estimators"]})
        if "max_depth" in model_info:
            meta_rows.append({"Property": "Max Depth", "Value": model_info["max_depth"]})
        if "classes" in model_info:
            meta_rows.append({"Property": "Classes", "Value": str(model_info["classes"])})
        if "pipeline_steps" in model_info:
            meta_rows.append({"Property": "Pipeline Steps", "Value": model_info["pipeline_steps"]})

        st.dataframe(
            pd.DataFrame(meta_rows),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    st.markdown("#### 📌 Important Note")
    st.write(
        "Model performance is based on the project's held-out evaluation dataset. Real-world "
        "production performance may differ depending on machine type, sensor quality, operating "
        "conditions, and data distribution."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
    <div class="footer">
        ⚙️ <b>Predictive Maintenance AI</b> · Random Forest Powered ·
        Built with Streamlit · © 2026
    </div>
""", unsafe_allow_html=True)
