# app.py

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt
from typing import List, Optional


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Make the app look a bit more like a dashboard */
    .main > div {
        padding-top: 0rem;
    }
    .big-title {
        font-size: 2rem;
        font-weight: 700;
    }
    .sub {
        color: #555555;
        font-size: 0.95rem;
    }
    .risk-low {
        color: #1a7f37;
        font-weight: 700;
    }
    .risk-medium {
        color: #c98a00;
        font-weight: 700;
    }
    .risk-high {
        color: #b3261e;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Load model + feature names
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_features():
    """
    Load the trained pipeline and feature names.

    Includes a small compatibility patch: if the local scikit-learn
    version does not define the internal helper class
    `_RemainderColsList`, we create a simple stub so that the
    unpickling process works.
    """
    # --- Compatibility patch for older scikit-learn versions ---
    try:
        from sklearn.compose import _column_transformer

        if not hasattr(_column_transformer, "_RemainderColsList"):
            # Create a simple stub class so unpickling succeeds
            class _RemainderColsList(list):
                """Compatibility stub for older scikit-learn versions."""
                pass

            _column_transformer._RemainderColsList = _RemainderColsList
    except Exception as e:
        # If anything goes wrong here, we just proceed and let joblib raise
        print("Warning: could not apply _RemainderColsList compatibility patch:", e)

    # --- Now safely load the model and feature names ---
    model = joblib.load("best_credit_model.pkl")
    feature_names: List[str] = joblib.load("feature_names.pkl")
    return model, feature_names


model, feature_names = load_model_and_features()


# -------------------------------------------------
# Helper: get feature importance from underlying model
# -------------------------------------------------
def get_feature_importance(pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Extract feature importance or coefficient magnitudes from the model
    inside the pipeline.
    """
    try:
        base_model = pipeline.named_steps.get("model")
    except Exception:
        return None

    if base_model is None:
        return None

    if hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
    elif hasattr(base_model, "coef_"):
        # For linear models, take absolute value of coefficients
        coef = getattr(base_model, "coef_", None)
        if coef is None:
            return None
        values = np.abs(coef[0])
    else:
        return None

    if len(values) != len(feature_names):
        return None

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": values
    }).sort_values("importance", ascending=False)

    return fi_df


feature_importance_df = get_feature_importance(model, feature_names)


# -------------------------------------------------
# Sidebar: input form
# -------------------------------------------------
st.sidebar.title("Client Profile")

st.sidebar.markdown(
    "Enter the customer's financial and credit information below. "
    "Values are approximate and for demonstration purposes."
)

# Friendly labels + some reasonable ranges
label_map = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Utilization of Unsecured Lines",
    "age": "Age (years)",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30–59 Days Past Due (last 2 years)",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income (USD)",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines / Loans",
    "NumberOfTimes90DaysLate": "Times 90+ Days Late (last 2 years)",
    "NumberRealEstateLoansOrLines": "Real Estate Loans / Lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60–89 Days Past Due (last 2 years)",
    "NumberOfDependents": "Number of Dependents"
}

# Min/max defaults for nicer widgets
range_map = {
    "RevolvingUtilizationOfUnsecuredLines": (0.0, 5.0, 0.2),
    "age": (18.0, 100.0, 1.0),
    "NumberOfTime30-59DaysPastDueNotWorse": (0.0, 20.0, 1.0),
    "DebtRatio": (0.0, 5.0, 0.1),
    "MonthlyIncome": (0.0, 30000.0, 500.0),
    "NumberOfOpenCreditLinesAndLoans": (0.0, 40.0, 1.0),
    "NumberOfTimes90DaysLate": (0.0, 20.0, 1.0),
    "NumberRealEstateLoansOrLines": (0.0, 20.0, 1.0),
    "NumberOfTime60-89DaysPastDueNotWorse": (0.0, 20.0, 1.0),
    "NumberOfDependents": (0.0, 10.0, 1.0)
}

# Group fields into logical sections
profile_features = ["age", "NumberOfDependents"]
financial_features = ["MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]
behavior_features = [
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

user_inputs = {}

st.sidebar.markdown("### Demographics")
for f in profile_features:
    if f in feature_names:
        minimum, maximum, step = range_map.get(f, (0.0, 100.0, 1.0))
        user_inputs[f] = st.sidebar.number_input(
            label_map.get(f, f),
            min_value=float(minimum),
            max_value=float(maximum),
            value=float((minimum + maximum) / 2),
            step=float(step)
        )

st.sidebar.markdown("### Financial Position")
for f in financial_features:
    if f in feature_names:
        minimum, maximum, step = range_map.get(f, (0.0, 100.0, 1.0))
        user_inputs[f] = st.sidebar.number_input(
            label_map.get(f, f),
            min_value=float(minimum),
            max_value=float(maximum),
            value=float((minimum + maximum) / 2),
            step=float(step)
        )

st.sidebar.markdown("### Credit Behaviour")
for f in behavior_features:
    if f in feature_names:
        minimum, maximum, step = range_map.get(f, (0.0, 100.0, 1.0))
        user_inputs[f] = st.sidebar.number_input(
            label_map.get(f, f),
            min_value=float(minimum),
            max_value=float(maximum),
            value=float((minimum + maximum) / 4),
            step=float(step)
        )

# For any remaining features not covered above
for f in feature_names:
    if f not in user_inputs:
        minimum, maximum, step = range_map.get(f, (0.0, 100.0, 1.0))
        user_inputs[f] = st.sidebar.number_input(
            label_map.get(f, f),
            min_value=float(minimum),
            max_value=float(maximum),
            value=float((minimum + maximum) / 2),
            step=float(step)
        )

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Risk Assessment", type="primary")


# -------------------------------------------------
# Main layout: tabs
# -------------------------------------------------
st.markdown('<p class="big-title">📊 Credit Risk Scoring Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Estimate the probability of serious delinquency within the next two years '
    'based on customer financial and credit history.</p>',
    unsafe_allow_html=True
)

tab_score, tab_importance, tab_about = st.tabs(
    ["Risk Score", "What Drives the Score?", "About the Model"]
)

# Prepare input dataframe
input_df = pd.DataFrame([[user_inputs[f] for f in feature_names]], columns=feature_names)


# -------------------------------------------------
# Tab 1: Risk Score
# -------------------------------------------------
with tab_score:
    st.subheader("Client Input Summary")
    st.dataframe(input_df.style.format(precision=2), use_container_width=True)

    if run_button:
        proba_default = float(model.predict_proba(input_df)[0, 1])
        proba_no_default = 1.0 - proba_default

        # Risk banding
        if proba_default < 0.10:
            risk_label = "Low Risk"
            risk_class = "risk-low"
        elif proba_default < 0.30:
            risk_label = "Medium Risk"
            risk_class = "risk-medium"
        else:
            risk_label = "High Risk"
            risk_class = "risk-high"

        st.markdown("### 🔎 Risk Assessment Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Default Probability", f"{proba_default:.1%}")
        col2.metric("Risk Category", risk_label)
        col3.metric("Probability of No Default", f"{proba_no_default:.1%}")

        # Progress bar for default probability
        st.markdown("##### Risk Level")
        st.progress(min(max(proba_default, 0.0), 1.0))

        # Probability bar chart with Altair
        chart_df = pd.DataFrame({
            "Outcome": ["No Default", "Default"],
            "Probability": [proba_no_default, proba_default]
        })

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Outcome", sort=None),
                y=alt.Y("Probability", axis=alt.Axis(format="%", title="Predicted Probability")),
                tooltip=["Outcome", alt.Tooltip("Probability", format=".2%")]
            )
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            f'<p class="{risk_class}">'
            f"This score is an estimate based on the historical data used to train the model. "
            f"It should be used as a decision-support tool, not as the only basis for approval or rejection."
            f"</p>",
            unsafe_allow_html=True
        )
    else:
        st.info("Adjust the inputs in the sidebar and click **Run Risk Assessment** to see the results.")


# -------------------------------------------------
# Tab 2: Feature Importance / Drivers
# -------------------------------------------------
with tab_importance:
    st.subheader("Overall Drivers of Risk (Model Perspective)")

    if feature_importance_df is None:
        st.warning(
            "The current model type does not expose feature importances or coefficients "
            "in a straightforward way, so we cannot show a ranked list of drivers."
        )
    else:
        top_n = st.slider("Show top N most important features", 3, min(10, len(feature_importance_df)), 8)

        fi_top = feature_importance_df.head(top_n).copy()
        fi_top["Friendly Name"] = fi_top["feature"].map(lambda f: label_map.get(f, f))

        st.markdown(
            "These bars show which input variables matter most **overall** for the model, "
            "not for a single customer. Higher bars indicate stronger influence on the "
            "predicted risk."
        )

        chart = (
            alt.Chart(fi_top)
            .mark_bar()
            .encode(
                x=alt.X("importance", title="Relative Importance"),
                y=alt.Y("Friendly Name", sort="-x", title="Feature"),
                tooltip=[
                    alt.Tooltip("feature", title="Technical Name"),
                    alt.Tooltip("importance", format=".4f", title="Importance")
                ]
            )
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Raw importance table")
        st.dataframe(fi_top[["feature", "importance"]].reset_index(drop=True), use_container_width=True)


# -------------------------------------------------
# Tab 3: About the Model
# -------------------------------------------------
with tab_about:
    st.subheader("How This Tool Works")

    st.markdown(
        """
        **Purpose**

        This dashboard uses a machine learning model trained on historical credit data 
        to estimate the probability that a customer will experience **serious delinquency 
        within the next two years**. The target variable in the training data is 
        `SeriousDlqin2yrs` (0 = no serious delinquency, 1 = serious delinquency).

        **High-level approach**

        - The model was trained on the public **“Give Me Some Credit”** dataset.
        - We evaluated multiple algorithms (Logistic Regression, Random Forest, 
          Gradient Boosting, XGBoost) and selected the one with the best AUC-ROC.
        - Input features were cleaned, missing values were imputed, and numeric features 
          were standardized before training.
        - The saved pipeline (preprocessing + model) is loaded here and used to score 
          the profile you enter in the sidebar.
        """
    )

    st.markdown(
        """
        **How to interpret the score**

        - The **probability of default** is an estimated probability based on the training data.
        - The **risk category (Low / Medium / High)** is a simple banding to make 
          the result easier to read.
        - This tool is intended as a **decision-support system**. Final decisions 
          should still consider policy, regulations, and human judgement.
        """
    )

    st.markdown(
        """
        **Fairness and limitations**

        - As part of the offline analysis, we examined potential bias across age groups 
          using libraries such as `fairlearn`.
        - This dashboard does **not** show individual-level fairness metrics, but any 
          real deployment should continuously monitor performance and fairness over time.
        - The model cannot see unrecorded factors (e.g., recent job changes, medical 
          events) and reflects patterns in the historical data it was trained on.
        """
    )

    st.markdown(
        """
        **Disclaimer**

        This application is for educational and demonstration purposes. It should not be 
        used as the sole basis for real-world credit decisions without proper validation, 
        governance, and regulatory review.
        """
    )
