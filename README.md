# AI Banking Risk & Customer Insights Platform

An interactive machine learning dashboard built with Python and Streamlit to explore customer churn, suspicious transactions, customer segmentation, branch insights, and business recommendations using synthetic banking data.

## Live Demo

[Launch the Streamlit App](https://ai-banking-risk-insights.streamlit.app)

## Medium Articles

- [Part 1: How I Built an AI Banking Risk & Customer Insights Platform Using Synthetic Data (End-to-End ML Project)![Uploading Executive_Dashboard.png…]()
](https://medium.com/@desiree2dmello/how-i-built-an-ai-banking-risk-customer-insights-platform-using-synthetic-data-end-to-end-ml-dba47130a83b)
- [Part 2: How I Turned My AI Banking Risk Project Into a Live Interactive ML Dashboard](ADD-YOUR-SECOND-MEDIUM-LINK-HERE)

After adding screenshots to that folder, use this section:

### Executive Dashboard

![Executive Dashboard]<img width="1720" height="799" alt="Executive_Dashboard" src="https://github.com/user-attachments/assets/72f05c32-5725-44f5-992f-0676759df974" />

### Customer Segmentation

![Customer Segmentation]<img width="1689" height="733" alt="Customer_Segmentation" src="https://github.com/user-attachments/assets/76d87fe7-a978-4ace-b73b-54d4a1badfb8" />

### Churn Prediction

![Churn Prediction]<img width="1699" height="885" alt="image" src="https://github.com/user-attachments/assets/0e53d8fc-c119-4bd3-90ab-d6c8e4c2358d" />

### Suspicious Transactions

![Suspicious Transactions](docs/screenshots/suspicious-transactions.png)

### Model Explainability

![Model Explainability](docs/screenshots/model-explainability.png)

### Business Recommendations

![Business Recommendations](docs/screenshots/business-recommendations.png)

## Project Overview

This project simulates how a banking analytics team could use machine learning and data visualization to understand customer behavior and risk.

The dashboard helps answer questions such as:

- Which customers are more likely to churn?
- Which transactions may require review?
- Which branches show higher risk patterns?
- How can model outputs be explained in business terms?
- What recommendations can be made from the data?

This project was built as a portfolio project to demonstrate data analytics, machine learning, Streamlit dashboard development, testing, deployment, and business communication.

## Features

- Executive dashboard with key banking KPIs
- Customer segmentation view
- Churn prediction simulator
- Suspicious transaction detection
- Branch-level filtering
- Model explainability section
- Business recommendations
- Interactive Plotly charts
- CSV download option
- Soft pink dashboard theme using `#ffcfcf`
- Streamlit Community Cloud deployment

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib
- Pytest
- GitHub

## Project Structure

```text
ai-banking-risk-insights/
├── .streamlit/
│   └── config.toml
├── app/
│   └── main.py
├── data/
│   └── sample_banking_data.csv
├── docs/
│   ├── data_dictionary.md
│   ├── model_card.md
│   └── screenshots/
├── models/
│   ├── anomaly_model.pkl
│   └── churn_model.pkl
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train_models.py
├── tests/
│   ├── conftest.py
│   └── test_preprocessing.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dashboard Sections

### Executive Dashboard

Shows high-level banking metrics, including:

- Total customers
- Average account balance
- Churn rate
- Suspicious transaction rate
- Account balance distribution
- Transaction amount by branch

### Customer Segmentation

Visualizes customer groups using income, account balance, churn status, number of products, credit score, and digital banking usage.

### Churn Prediction

Allows users to enter customer information and estimate churn risk using a trained machine learning model.

### Suspicious Transactions

Uses an anomaly detection model to identify unusual transaction patterns and display potentially risky activity.

### Model Explainability

Explains important features that may influence churn and suspicious transaction risk, such as complaint count, digital banking usage, product count, credit score, tenure, account balance, and transaction amount.

### Business Recommendations

Translates the dashboard results into practical recommendations for customer retention, transaction review, product engagement, and branch monitoring.

## Data Disclaimer

This project uses synthetic banking data.

The data does not represent real customers, real transactions, or any real financial institution. It is intended only for learning, portfolio demonstration, and machine learning practice.

This dashboard should not be used for real banking or financial decisions.

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/desireedmello/ai-banking-risk-insights.git
cd ai-banking-risk-insights
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the sample data if needed:

```bash
python src/data_generation.py
```

Train the models if needed:

```bash
python src/train_models.py
```

Run the Streamlit app:

```bash
streamlit run app/main.py
```

## Run Tests

```bash
pytest
```

## Deployment

The app is deployed with Streamlit Community Cloud.

Deployment settings:

```text
Repository: desireedmello/ai-banking-risk-insights
Branch: main
Main file path: app/main.py
```

## Business Impact

This project demonstrates how machine learning can support banking teams by:

- Identifying customers who may be at risk of churn
- Highlighting unusual transaction behavior
- Supporting branch-level performance review
- Explaining model outputs in business-friendly language
- Turning analytics into practical recommendations

## Limitations

- The dataset is synthetic.
- The models are simplified for portfolio and learning purposes.
- The dashboard does not connect to a real banking database.
- The app does not include authentication.
- The suspicious transaction model is not a production fraud detection system.
- The project is not intended for real financial decision-making.

## Future Improvements

- Add SHAP-based model explainability
- Add model evaluation metrics to the dashboard
- Add confusion matrix and ROC curve visualizations
- Improve suspicious transaction feature engineering
- Add more advanced customer segmentation
- Add a short demo GIF
- Add GitHub Actions for automated testing
- Improve mobile layout and responsiveness

## Author

Desiree D'Mello

GitHub: [desireedmello](https://github.com/desireedmello)

Medium: [Add your Medium profile link here](ADD-YOUR-MEDIUM-PROFILE-LINK-HERE)

LinkedIn: [Add your LinkedIn profile link here](ADD-YOUR-LINKEDIN-PROFILE-LINK-HERE)

## License

This project is licensed under the MIT License.
