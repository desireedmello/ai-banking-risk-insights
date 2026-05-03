import pandas as pd
import numpy as np


def generate_banking_data(rows=3000, seed=42):
    np.random.seed(seed)

    data = pd.DataFrame({
        "customer_id": range(1, rows + 1),
        "age": np.random.randint(18, 75, rows),
        "income": np.random.normal(65000, 22000, rows).clip(18000, 180000),
        "account_balance": np.random.normal(12000, 9000, rows).clip(0, 90000),
        "transaction_amount": np.random.exponential(400, rows).clip(5, 15000),
        "tenure_months": np.random.randint(1, 180, rows),
        "num_products": np.random.randint(1, 6, rows),
        "credit_score": np.random.normal(699, 70, rows).clip(300, 900),
        "digital_banking_usage": np.random.randint(0, 40, rows),
        "complaint_count": np.random.poisson(0.6, rows),
        "branch_id": np.random.choice(["BR001", "BR002", "BR003", "BR004"], rows),
        "transaction_type": np.random.choice(["Deposit", "Withdrawal", "Transfer", "Bill Payment"], rows)
    })

    churn_score = (
        0.04 * data["complaint_count"]
        - 0.015 * data["num_products"]
        - 0.002 * data["tenure_months"]
        - 0.01 * data["digital_banking_usage"]
        + np.random.normal(0, 0.4, rows)
    )

    data["churned"] = (churn_score > np.percentile(
        churn_score, 72)).astype(int)

    data["is_suspicious_transaction"] = (
        (data["transaction_amount"] > 5000)
        | ((data["transaction_amount"] > 2500) & (data["account_balance"] < 1000))
    ).astype(int)

    return data


if __name__ == "__main__":
    df = generate_banking_data()
    df.to_csv("data/sample_banking_data.csv", index=False)
    print("Sample banking dataset created.")
