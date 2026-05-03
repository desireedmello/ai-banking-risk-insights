import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from preprocessing import split_churn_data, build_preprocessor, NUMERIC_FEATURES


def train_churn_model():
    df = pd.read_csv("data/sample_banking_data.csv")

    X_train, X_test, y_train, y_test = split_churn_data(df)

    model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    joblib.dump(model, "models/churn_model.pkl")
    print("Churn model saved.")
    print(metrics)


def train_anomaly_model():
    df = pd.read_csv("data/sample_banking_data.csv")

    model = IsolationForest(
        contamination=0.08,
        random_state=42
    )

    model.fit(df[NUMERIC_FEATURES])
    joblib.dump(model, "models/anomaly_model.pkl")

    print("Anomaly model saved.")


if __name__ == "__main__":
    train_churn_model()
    train_anomaly_model()
