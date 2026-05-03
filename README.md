# AI Banking Risk & Customer Insights Platform

An interactive analytics and machine learning dashboard for branch banking risk monitoring, customer churn prediction, and business decision support.

## Project Motivation

This project connects my front-line finance experience as a bank teller with my technical background in computer science, AI programming, and machine learning. The goal is to simulate how a financial institution could use data analytics and ML to improve customer retention, identify unusual transactions, and support branch-level decisions.

## Features

- Executive KPI dashboard
- Customer churn prediction model
- Suspicious transaction detection
- Customer segmentation visualization
- Branch-level filtering
- Business recommendation report
- Synthetic banking dataset
- Reproducible model training pipeline

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Pytest

## How To Run

```bash
git clone https://github.com/YOUR_USERNAME/banking-risk-insights.git
cd banking-risk-insights
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/data_generation.py
python src/train_models.py
streamlit run app/main.py
