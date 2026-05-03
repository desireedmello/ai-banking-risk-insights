# AI Banking Risk & Customer Insights Platform

An interactive machine learning dashboard for banking risk monitoring, customer churn prediction, suspicious transaction detection, and business decision support.

## Project Overview

This project uses synthetic banking data to simulate how a financial institution could use data analytics and machine learning to support customer retention, fraud/risk monitoring, and branch-level decision-making.

The platform includes:
- Synthetic banking data generation
- Customer churn prediction model
- Suspicious transaction detection using anomaly detection
- Interactive Streamlit dashboard
- Customer segmentation visualizations
- Business recommendations
- Basic unit tests for reliability

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Pytest

## Project Structure

```text
ai-banking-risk-insights/
├── app/
│   └── main.py
├── data/
│   └── sample_banking_data.csv
├── docs/
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── train_models.py
│   └── predict.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── README.md
└── LICENSE
