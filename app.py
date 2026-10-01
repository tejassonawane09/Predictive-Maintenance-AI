import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------------------------
# Load Model
# ---------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/predictive_maintenance_model.pkl"
    )


model = load_model()


# ---------------------------------------
# Header
# ---------------------------------------

st.title("⚙️ Predictive Maintenance AI")

st.markdown(
    """
    ### Machine Failure Prediction System

    Predict the probability of machine failure using
    operational and sensor-related parameters.
    """
)

st.divider()


# ---------------------------------------
# Sidebar
# ---------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Failure Prediction",
        "Model Insights"
    ]
)


# =======================================
# DASHBOARD
# =======================================

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Records",
            "10,000"
        )

    with col2:
        st.metric(
            "Failure Rate",
            "3.39%"
        )

    with col3:
        st.metric(
            "Model",
            "Random Forest"
        )

    with col4:
        st.metric(
            "ROC-AUC",
            "97.99%"
        )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        The objective of this project is to predict whether
        an industrial machine is likely to experience a
        failure based on its operating conditions.
        """
    )

    st.subheader("Machine Parameters")

    st.write(
        """
        The model uses machine type, air temperature,
        process temperature, rotational speed, torque,
        tool wear, temperature difference, and a power proxy.
        """
    )

    st.info(
        """
        This application is designed as a predictive analytics
        demonstration. Predictions should support maintenance
        investigation and should not be treated as a guaranteed
        failure diagnosis.
        """
    )


# =======================================
# FAILURE PREDICTION
# =======================================

elif page == "Failure Prediction":

    st.header("🔍 Machine Failure Prediction")

    st.write(
        "Enter the machine operating parameters below."
    )

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=310.0,
            step=0.1
        )

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=500,
            max_value=3000,
            value=1500,
            step=10
        )

    with col2:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Machine Failure",
        type="primary",
        use_container_width=True
    )


    # -----------------------------------
    # Prediction
    # -----------------------------------

    if predict_button:

        temperature_difference = (
            process_temperature
            - air_temperature
        )

        power_proxy = (
            torque
            * rotational_speed
        )

        input_data = pd.DataFrame({
            "Type": [machine_type],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [
                process_temperature
            ],
            "Rotational speed [rpm]": [
                rotational_speed
            ],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
            "Temperature Difference [K]": [
                temperature_difference
            ],
            "Power Proxy": [power_proxy]
        })

        probability = model.predict_proba(
            input_data
        )[0][1]

        threshold = 0.40

        prediction = (
            probability >= threshold
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Failure Probability",
                f"{probability:.2%}"
            )

        with col2:

            if prediction:

                st.error(
                    "⚠️ Potential Machine Failure"
                )

            else:

                st.success(
                    "✅ No Failure Predicted"
                )

        st.progress(
            float(probability)
        )

        if prediction:

            st.warning(
                """
                The predicted failure probability is above
                the configured 0.40 decision threshold.
                Further maintenance investigation is recommended.
                """
            )

        else:

            st.info(
                """
                The predicted failure probability is below
                the configured 0.40 decision threshold.
                """
            )


# =======================================
# MODEL INSIGHTS
# =======================================

elif page == "Model Insights":

    st.header("📈 Model Insights")

    st.write(
        """
        This section provides model performance metrics,
        threshold analysis, and feature-level insights
        from the trained Random Forest model.
        """
    )

    # -----------------------------------
    # Final Model
    # -----------------------------------

    st.subheader("Final Model")

    st.write(
        "Random Forest Classifier"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "99.10%"
        )

    with col2:
        st.metric(
            "Precision",
            "94.64%"
        )

    with col3:
        st.metric(
            "Recall",
            "77.94%"
        )

    with col4:
        st.metric(
            "ROC-AUC",
            "97.99%"
        )

    st.info(
        """
        Since machine failures are relatively rare in the dataset,
        accuracy alone is not sufficient. Precision, recall, F1 score,
        and ROC-AUC provide additional information about model performance.
        """
    )

    # -----------------------------------
    # Threshold Optimization
    # -----------------------------------

    st.divider()

    st.subheader("Threshold Optimization")

    st.write(
        """
        The classification threshold was evaluated at multiple
        values. During threshold analysis on the current held-out
        evaluation split, a threshold of 0.40 produced the best
        F1 score among the tested thresholds.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Threshold",
            "0.40"
        )

    with col2:
        st.metric(
            "Precision",
            "93.33%"
        )

    with col3:
        st.metric(
            "Recall",
            "82.35%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "87.50%"
        )

    threshold_data = pd.DataFrame({
        "Metric": [
            "False Positives",
            "False Negatives",
            "True Positives"
        ],
        "Value": [
            4,
            12,
            56
        ]
    })

    st.dataframe(
        threshold_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The 0.40 threshold increases failure detection sensitivity "
        "compared with the default 0.50 threshold."
    )

    # -----------------------------------
    # Feature Importance
    # -----------------------------------

    st.divider()

    st.subheader("🔍 Feature Importance")

    st.write(
        """
        Feature importance shows which input features the trained
        Random Forest relied on most when making predictions.
        Higher importance indicates greater contribution to the
        model's decision process.
        """
    )

    importance_path = "artifacts/feature_importance.csv"

    try:

        importance_df = pd.read_csv(
            importance_path
        )

        # Clean feature names
        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace(
                "num__",
                "",
                regex=False
            )
            .str.replace(
                "cat__",
                "",
                regex=False
            )
        )

        # Top 10 features
        top_features = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(10)
            .sort_values(
                by="Importance"
            )
        )

        st.bar_chart(
            top_features.set_index(
                "Feature"
            )["Importance"],
            horizontal=True
        )

        st.caption(
            "Feature importance represents model reliance and "
            "should not be interpreted as proof of causation."
        )

    except FileNotFoundError:

        st.warning(
            """
            Feature importance data was not found.

            Run:

            python src/model_explainability.py
            """
        )

    # -----------------------------------
    # Important Note
    # -----------------------------------

    st.divider()

    st.subheader("Important Note")

    st.write(
        """
        Model performance is based on the project's held-out
        evaluation dataset. Real-world production performance
        may differ depending on machine type, sensor quality,
        operating conditions, and data distribution.
        """
    )