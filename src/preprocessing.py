import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

FEATURES = [
    "age",
    "income",
    "account_balance",
    "transaction_amount",
    "tenure_months",
    "num_products",
    "credit_score",
    "digital_banking_usage",
    "complaint_count",
    "branch_id",
    "transaction_type",
]

NUMERIC_FEATURES = [
    "age",
    "income",
    "account_balance",
    "transaction_amount",
    "tenure_months",
    "num_products",
    "credit_score",
    "digital_banking_usage",
    "complaint_count",
]

CATEGORICAL_FEATURES = ["branch_id", "transaction_type"]


def split_churn_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["churned"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
