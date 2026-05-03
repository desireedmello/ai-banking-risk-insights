import joblib
import plotly.express as px
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import NUMERIC_FEATURES

st.set_page_config(
    page_title="AI Banking Risk & Customer Insights",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/sample_banking_data.csv")


@st.cache_resource
def load_models():
    churn_model = joblib.load("models/churn_model.pkl")
    anomaly_model = joblib.load("models/anomaly_model.pkl")
    return churn_model, anomaly_model


df = load_data()
churn_model, anomaly_model = load_models()

st.title("AI Banking Risk & Customer Insights Platform")

branches = ["All"] + sorted(df["branch_id"].unique())
selected_branch = st.sidebar.selectbox("Branch", branches)

if selected_branch != "All":
    filtered_df = df[df["branch_id"] == selected_branch]
else:
    filtered_df = df.copy()

st.sidebar.markdown("### App Selections")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Executive Dashboard",
        "Customer Segmentation",
        "Churn Prediction",
        "Suspicious Transactions",
        "Business Recommendations"
    ]
)

if section == "Executive Dashboard":
    st.subheader("Executive Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{len(filtered_df):,}")
    col2.metric("Average Balance",
                f"${filtered_df['account_balance'].mean():,.0f}")
    col3.metric("Churn Rate", f"{filtered_df['churned'].mean() * 100:.1f}%")
    col4.metric(
        "suspicious Tx Rate",
        f"{filtered_df['is_suspicious_transaction'].mean() * 100:.1f}%"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(
            filtered_df,
            x="account_balance",
            nbins=40,
            title="Account Balance Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.box(
            filtered_df,
            x="branch_id",
            y="transaction_amount",
            title="Transaction Amount by Branch"
        )
        st.plotly_chart(fig, use_container_width=True)

elif section == "Customer Segmentation":
    st.subheader("Customer Segmentation")

    fig = px.scatter(
        filtered_df,
        x="income",
        y="account_balance",
        color="churned",
        size="num_products",
        hover_data=["customer_id", "credit_score", "digital_banking_usage"],
        title="Customer Value and Churn Risk"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
                **Interpretation:**
                Customers with low product count, lower digital usage, and higher complaint counts are more likely to churn.
                """)

elif section == "Churn Prediction":
    st.subheader("Customer Churn Prediction")

    col1, col2, col3 = st.columns(3)

    age = col1.slider("Age", 18, 75, 35)
    income = col2.number_input("Income", 18000, 180000, 65000)
    balance = col3.number_input("Account Balance", 0, 90000, 12000)

    col4, col5, col6 = st.columns(3)

    transaction_amount = col4.number_input("Transaction Amount", 5, 15000, 400)
    tenure = col5.slider("Tenure Months", 1, 180, 36)
    num_products = col6.slider("Number of Products", 1, 5, 2)

    col7, col8, col9 = st.columns(3)

    credit_score = col7.slider("Credit Score", 300, 900, 690)
    digital_usage = col8.slider("Digital Banking Usage", 0, 40, 12)
    complaints = col9.slider("Complaint Count", 0, 10, 1)

    branch_id = st.selectbox("Branch ID", sorted(df["branch_id"].unique()))
    transaction_type = st.selectbox(
        "Transaction Type", sorted(df["transaction_type"].unique()))

    customer = pd.DataFrame([{
        "age": age,
        "income": income,
        "account_balance": balance,
        "transaction_amount": transaction_amount,
        "tenure_months": tenure,
        "num_products": num_products,
        "credit_score": credit_score,
        "digital_banking_usage": digital_usage,
        "complaint_count": complaints,
        "branch_id": branch_id,
        "transaction_type": transaction_type,
    }])

    if st.button("Predict Churn Risk"):
        probability = churn_model.predict_proba(customer)[0][1]
        st.metric("Predicted Churn Risk", f"{probability * 100:.1f}%")

        if probability >= 0.6:
            st.warning(
                "High churn risk. Recommend proactive retention outreach.")
        elif probability >= 0.3:
            st.info("Moderate churn risk. Monitor engagement and product usage.")
        else:
            st.success("Low churn risk")

elif section == "Suspicious Transactions":
    st.subheader("Suspicious Transaction Detection")

    anomaly_scores = anomaly_model.predict(filtered_df[NUMERIC_FEATURES])
    results = filtered_df.copy()
    results["model_flag"] = anomaly_scores
    results["risk_label"] = results["model_flag"].map({
        -1: "Unusual",
        1: "Normal"
    })

    st.dataframe(
        results.sort_values("transaction_amount", ascending=False).head(100),
        use_container_width=True
    )

    fig = px.scatter(
        results,
        x="account_balance",
        y="transaction_amount",
        color="risk_label",
        hover_data=["customer_id", "branch_id", "transaction_type"],
        title="Transaction Risk View"
    )

    st.plotly_chart(fig, use_container_width=True)

elif section == "Business Recommendations":
    st.subheader("Business Recommendations")

    churn_rate = filtered_df["churned"].mean() * 100
    suspicious_rate = filtered_df["is_suspicious_transaction"].mean() * 100
    avg_balance = filtered_df["account_balance"].mean()

    st.markdown(f"""
    ### Summary
    
    This branch/customer group has a churn rate of **{churn_rate:.1f}%**,
    a suspicious transaction rate of **{suspicious_rate:.1f}%**,
    and an average account balance of **${avg_balance:,.0f}**.
    
    ### Recommended Actions
     
    - Prioritize retention outreach for customers with high complaints and low digital banking usage.
    - Review large transactions from low-balance accounts for potential risk.
    - Promote bundled banking products to customers with only one active product.
    - Encourage digital banking adoption among low-engagement customers.
    - Monitor branches with above-average churn or suspicious transaction rates.
    """)
