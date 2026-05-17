import joblib
import plotly.express as px
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PINK = "#ffcfcf"
PINK_DARK = "#ff9f9f"
PINK_STRONG = "#ff7f7f"
PINK_LIGHT = "rgba(255, 207, 207, 0.18)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import NUMERIC_FEATURES

st.set_page_config(
    page_title="AI Banking Risk & Customer Insights",
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        --soft-pink: #ffcfcf;
        --soft-pink-dark: #ff9f9f;
        --soft-pink-strong: #ff7f7f;
        --soft-pink-light: rgba(255, 207, 207, 0.18);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--soft-pink);
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--soft-pink);
        border-radius: 8px;
        padding: 16px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background-color: transparent;
        color: inherit;
        border: 1px solid var(--soft-pink);
        border-radius: 6px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: var(--soft-pink-light);
        color: inherit;
        border-color: var(--soft-pink-dark);
    }

    div.stButton > button:focus,
    div.stDownloadButton > button:focus {
        box-shadow: 0 0 0 0.2rem rgba(255, 207, 207, 0.35);
        border-color: var(--soft-pink-dark);
    }

    div[data-baseweb="select"] > div {
        border-color: var(--soft-pink);
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--soft-pink-dark);
        box-shadow: 0 0 0 1px var(--soft-pink-dark);
    }

    div[data-baseweb="radio"] div[aria-checked="true"] {
        background-color: var(--soft-pink-strong) !important;
        border-color: var(--soft-pink-strong) !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background-color: var(--soft-pink-strong);
        border-color: var(--soft-pink-strong);
    }

    div[data-testid="stSlider"] [data-testid="stTickBar"] {
        background-color: var(--soft-pink);
    }
    
    .pink-alert {
        background-color: rgba(255, 207, 207, 0.25);
        border: 1px solid #ffcfcf;
        border-radius: 8px;
        padding: 1rem;
        color: inherit;
        font-weight: 500;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


DATA_PATH = PROJECT_ROOT / "data" / "sample_banking_data.csv"
CHURN_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    if not CHURN_MODEL_PATH.exists() or not ANOMALY_MODEL_PATH.exists():
        st.error(
            "Model files are missing. Please run 'python src/train_models.py' and make sure the models folder is included before deployment"
        )
        st.stop()
        
    churn_model = joblib.load(CHURN_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    
    return churn_model, anomaly_model    


df = load_data()
churn_model, anomaly_model = load_models()

st.title("AI Banking Risk & Customer Insights Platform")

st.caption(
    "Portfolio project using synthetic banking data for churn prediction, suspicious transaction detection, and business insights."
)

branches = ["All"] + sorted(df["branch_id"].unique())
selected_branch = st.sidebar.selectbox("Branch", branches)

if selected_branch != "All":
    filtered_df = df[df["branch_id"] == selected_branch]
else:
    filtered_df = df.copy()
    
st.sidebar.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_banking_risk_data.csv",
    mime="text/csv"
)

st.sidebar.markdown("### App Selections")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Executive Dashboard",
        "Customer Segmentation",
        "Churn Prediction",
        "Suspicious Transactions",
        "Model Explainability",
        "Business Recommendations",
        "About This Project"
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
            title="Account Balance Distribution",
            color_discrete_sequence=[PINK]
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title=""),
        )   
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.box(
            filtered_df,
            x="branch_id",
            y="transaction_amount",
            title="Transaction Amount by Branch",
            color_discrete_sequence=[PINK]
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title=""),
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
        title="Customer Value and Churn Risk",
        color_continuous_scale=[PINK, PINK_STRONG]
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title=""),
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
        "Transaction Type", sorted(df["transaction_type"].unique())
    )

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

        if probability >= 0.45:
            st.markdown(
                """
                <div class="pink-alert">
                    High churn risk. Recommend proactive retention outreach.
                </div>
                """,
                unsafe_allow_html=True
            )
        elif probability >= 0.25:
            st.markdown(
                """
                <div class="pink-alert">
                    Moderate churn risk. Monitor engagement and product usage.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="pink-alert">
                    Low churn risk.
                </div>
                """,
                unsafe_allow_html=True
            )

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
        title="Transaction Risk View",
        color_discrete_map={
            "Normal": PINK,
            "Unusual": PINK_STRONG
        }
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)
    
elif section == "Model Explainability":
    st.subheader("Model Explainability")

    st.markdown("""
    This section explains the main factors that influence churn and suspicious transaction risk.

    For this portfolio version, the dashboard focuses on beginner-friendly explainability:
    connecting model inputs to business meaning.
    """)

    importance_df = pd.DataFrame({
        "Feature": [
            "complaint_count",
            "digital_banking_usage",
            "num_products",
            "credit_score",
            "tenure_months",
            "account_balance",
            "transaction_amount"
        ],
        "Relative Importance": [
            0.24,
            0.20,
            0.17,
            0.14,
            0.11,
            0.08,
            0.06
        ],
        "Business Meaning": [
            "More complaints may signal dissatisfaction.",
            "Lower digital usage may indicate weaker engagement.",
            "Fewer products may mean a weaker customer relationship.",
            "Lower credit score may indicate higher financial risk.",
            "Shorter tenure may mean the customer is less loyal.",
            "Lower balance may indicate lower relationship value.",
            "Unusual transaction amounts may require review."
        ]
    })

    st.dataframe(importance_df, use_container_width=True)

    fig = px.bar(
        importance_df,
        x="Relative Importance",
        y="Feature",
        orientation="h",
        title="Example Feature Importance Ranking",
        color="Relative Importance",
        color_continuous_scale=[PINK, PINK_STRONG]
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title=""),
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
    
elif section == "About This Project":
    st.subheader("About This Project")
    
    st.markdown("""
                This dashboard is a portfolio project that demonstrates how machine learning can support banking risk and customer insight workflows.
                
                The project includes:
                
                - Synthetic banking data generation
                - Data preprocessing
                - Churn prediction
                - Suspicious transaction detection
                - Streamlit dashboard development
                - Business recommendations
                - Testing with pytest
                
                **Important note:** This project uses synthetic data. It is intended for learning, portfolio presentation, and employer review. It should not be used for real banking decisions
                """)
    
    st.markdown("""
                 ### Project Links

                    - GitHub: https://github.com/desireedmello/ai-banking-risk-insights
                    - Live Streamlit app: https://ai-banking-risk-insights.streamlit.app/
                    - Medium article 1: https://medium.com/@desiree2dmello/how-i-built-an-ai-banking-risk-customer-insights-platform-using-synthetic-data-end-to-end-ml-dba47130a83b
                    - Medium article 2: https://medium.com/@desiree2dmello/turning-my-ai-banking-risk-project-into-a-live-interactive-dashboard-b133d9fce3df
                """)
