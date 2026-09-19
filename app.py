from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OTT Customer Churn Predictor",
    page_icon="📺",
    layout="centered",
)

st.title("📺 OTT Customer Churn Predictor")
st.write(
    "Enter a customer's subscription and viewing behavior to estimate "
    "their probability of churn."
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "churn_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "churn_model.joblib was not found. Place it in the same folder as app.py."
        )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Model Overview")
    st.write("**Model:** Class-balanced Random Forest")
    st.write("**Preprocessing:** Median imputation, scaling, one-hot encoding")
    st.write("**Optimization:** RandomizedSearchCV")
    st.metric("Test Accuracy", "91.2%")
    st.metric("Churn F1", "67.1%")
    st.caption("Metrics shown are from the held-out test set used in the project.")

with st.form("churn_prediction_form"):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=0, max_value=100, value=35, step=1)
        no_of_days_subscribed = st.number_input(
            "Days Subscribed", min_value=0, value=120, step=1
        )
        multi_screen_text = st.selectbox("Uses Multiple Screens?", ["No", "Yes"])
        mail_subscribed_text = st.selectbox(
            "Subscribed to Marketing Emails?", ["No", "Yes"]
        )
        weekly_mins_watched = st.number_input(
            "Weekly Minutes Watched", min_value=0.0, value=270.0, step=1.0
        )

    with col2:
        minimum_daily_mins = st.number_input(
            "Minimum Daily Minutes Watched", min_value=0.0, value=10.0, step=0.5
        )
        maximum_daily_mins = st.number_input(
            "Maximum Daily Minutes Watched", min_value=0.0, value=30.0, step=0.5
        )
        weekly_max_night_mins = st.number_input(
            "Weekly Maximum Night Minutes", min_value=0.0, value=100.0, step=1.0
        )
        videos_watched = st.number_input(
            "Videos Watched", min_value=0, value=5, step=1
        )
        maximum_days_inactive = st.number_input(
            "Maximum Days Inactive", min_value=0.0, value=3.0, step=1.0
        )
        customer_support_calls = st.number_input(
            "Customer Support Calls", min_value=0, value=1, step=1
        )

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    if maximum_daily_mins < minimum_daily_mins:
        st.warning(
            "Maximum Daily Minutes should be greater than or equal to Minimum Daily Minutes."
        )
        st.stop()

    multi_screen = 1 if multi_screen_text == "Yes" else 0
    mail_subscribed = 1 if mail_subscribed_text == "Yes" else 0

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "no_of_days_subscribed": no_of_days_subscribed,
        "multi_screen": multi_screen,
        "mail_subscribed": mail_subscribed,
        "weekly_mins_watched": weekly_mins_watched,
        "minimum_daily_mins": minimum_daily_mins,
        "maximum_daily_mins": maximum_daily_mins,
        "weekly_max_night_mins": weekly_max_night_mins,
        "videos_watched": videos_watched,
        "maximum_days_inactive": maximum_days_inactive,
        "customer_support_calls": customer_support_calls,
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        churn_probability = float(model.predict_proba(input_data)[0][1])

        st.divider()
        st.subheader("Prediction")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Churn Probability", f"{churn_probability:.1%}")
        with metric_col2:
            st.metric("Model Prediction", "Churn" if prediction == 1 else "Stay")

        st.progress(min(max(churn_probability, 0.0), 1.0))

        if prediction == 1:
            st.error("⚠️ This customer is predicted to be at risk of churn.")
        else:
            st.success("✅ This customer is predicted to remain subscribed.")

        with st.expander("View model input"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the app inputs match the features used during training."
        )
        st.exception(exc)

st.divider()
st.caption(
    "Portfolio demonstration of a machine-learning churn classifier. "
    "Predictions are for demonstration purposes."
)
