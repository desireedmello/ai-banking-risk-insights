## 🏦 AI Banking Risk & Customer Insights Platform

An end-to-end Machine Learning and Banking Analytics project that simulates how financial institutions can identify customer churn risk, suspicious transactions, and behavioral insights using synthetic banking data.

Built using Python, Scikit-learn, Streamlit, and interactive data visualizations to demonstrate practical AI applications in banking and financial technology.

### Key Features
- Customer churn prediction
- Banking risk analysis
- Synthetic financial data generation
- Interactive Streamlit dashboard
- Executive KPI visualizations
- Machine learning model evaluation
- Suspicious transaction detection

---

## 📌 Project Overview

This project was designed to simulate a real-world banking analytics platform capable of analyzing customer behavior, identifying potential risks, and generating actionable insights using Machine Learning.

Since real banking data is highly sensitive and protected, this platform uses synthetic data generation techniques to mimic realistic banking environments while maintaining privacy and compliance.

The system demonstrates how AI can support:
- Risk assessment
- Customer retention strategies
- Fraud detection workflows
- Business intelligence reporting
- Predictive analytics in banking

---

## 📝 Medium Article

👉 Read the full breakdown of this project:  
[Link to Medium Blog](https://medium.com/@desiree2dmello/how-i-built-an-ai-banking-risk-customer-insights-platform-using-synthetic-data-end-to-end-ml-dba47130a83b?postPublishedType=repub)

---

## 🧠 Key Features

### 📊 Executive Dashboard
- Displays key banking KPIs  
- Customer count, average balance, churn rate  
- Suspicious transaction rate  

### 👥 Customer Segmentation
- Visualizes customer value vs churn risk  
- Uses income, balance, product usage, and engagement  

### 🔮 Churn Prediction
- Random Forest model predicts churn probability  
- Real-time predictions via Streamlit interface  

### 🚨 Suspicious Transaction Detection
- Isolation Forest identifies unusual transactions  
- Highlights high-risk financial activity  

### 💼 Business Recommendations
- Converts insights into actionable strategies  
- Retention, fraud monitoring, and product optimization  

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git & GitHu

---

## 📂 Project Structure


ai-banking-risk-insights/
├── app/
│ └── main.py
├── data/
│ └── sample_banking_data.csv
├── docs/
│ └── screenshots/
├── src/
│ ├── data_generation.py
│ ├── preprocessing.py
│ ├── train_models.py
│ └── predict.py
├── tests/
│ └── test_preprocessing.py
├── models/
├── requirements.txt
├── README.md
└── LICENSE


---

## ⚙️ How to Run Locally

### 1. Clone the repository

git clone https://github.com/desireedmello/ai-banking-risk-insights.git

cd ai-banking-risk-insights


### 2. Create and activate virtual environment

python -m venv .venv
.venv\Scripts\activate


### 3. Install dependencies

pip install -r requirements.txt


### 4. Generate synthetic data

python src/data_generation.py


### 5. Train models

python src/train_models.py


### 6. Run the dashboard

streamlit run app/main.py


### 7. Run tests

pytest tests/


---

## 📸 Dashboard Preview

### Executive Dashboard
<img width="1531" height="744" alt="executive_dashboard" src="https://github.com/user-attachments/assets/0cc28560-031c-4aee-a1f5-28ecf66aad7d" />

### Customer Segmentation
<img width="1535" height="729" alt="customer_segmentation" src="https://github.com/user-attachments/assets/dc47ee58-208e-4cf9-8069-318738891cd3" />

### Churn Prediction
<img width="1550" height="860" alt="churn_predictions" src="https://github.com/user-attachments/assets/64c04feb-8e35-4f83-b963-0d4b59ebe001" />

### Suspicious Transactions
<img width="1524" height="900" alt="suspicious_tranctions" src="https://github.com/user-attachments/assets/ad4b9679-000c-4c85-bb36-9b4432fc1591" />

---

## 🧠 Key Learnings

- Building ML systems using synthetic data  
- Designing preprocessing pipelines  
- Combining supervised and unsupervised models  
- Creating interactive dashboards  
- Ensuring reliability with testing  

---

## 🚀 Future Improvements

- Deploy dashboard (Streamlit Cloud / AWS)  
- Add model explainability (SHAP)  
- Improve anomaly detection models  
- Add real-time data pipelines  

---

## 👩‍💻 Author

**Desiree D'Mello**  
Data Analyst | Machine Learning Enthusiast  

---

## 📄 License

MIT License
