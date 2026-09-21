import streamlit as st
import joblib
import pandas as pd

MODEL_PATH = "churn_model.pkl"   # produced by joblib.dump(final_log_reg, ...) in the notebook
THRESHOLD = 0.5                   # same decision threshold used in the notebook

st.set_page_config(page_title="UrbanCart Churn Predictor", page_icon="📉")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("UrbanCart Customer Churn Predictor")
st.caption("Enter a customer's recent activity to estimate their churn risk.")

col1, col2 = st.columns(2)
with col1:
    months_active = st.number_input("Months Active", min_value=0.0, value=12.0, step=1.0)
    avg_order_value = st.number_input("Avg Order Value ($)", min_value=0.0, value=55.0, step=1.0)
    num_orders_last_quarter = st.number_input("Orders Last Quarter", min_value=0.0, value=4.0, step=1.0)
with col2:
    days_since_last_order = st.number_input("Days Since Last Order", min_value=0.0, value=20.0, step=1.0)
    support_tickets_filed = st.number_input("Support Tickets Filed", min_value=0.0, value=0.0, step=1.0)

if st.button("Predict Churn Risk", type="primary"):
    # Same engineered feature used in training: recency relative to tenure
    recency_tenure_ratio = days_since_last_order / (months_active * 30) if months_active > 0 else 0.0

    input_df = pd.DataFrame([{
        "MonthsActive": months_active,
        "AvgOrderValue": avg_order_value,
        "NumOrdersLastQuarter": num_orders_last_quarter,
        "DaysSinceLastOrder": days_since_last_order,
        "SupportTicketsFiled": support_tickets_filed,
        "recency_tenure_ratio": recency_tenure_ratio,
    }])

    proba = model.predict_proba(input_df)[0, 1]
    is_churn = proba >= THRESHOLD

    st.metric("Churn Probability", f"{proba:.1%}")
    st.progress(min(proba, 1.0))

    if is_churn:
        st.error("⚠️ Likely to churn — flag for retention outreach")
    else:
        st.success("✅ Likely to stay")
