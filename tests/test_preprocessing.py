import pandas as pd
from src.preprocessing import FEATURES, split_churn_data

def test_required_features_exist():
    df = pd.read_csv("data/sample_banking_data.csv")
    
    for feature in FEATURES:
        assert feature in df.columns
        
def test_train_test_split_runs():
    df = pd.read_csv("data/sample_banking_data.csv")
    X_train, X_test, y_train, y_test = split_churn_data(df)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)