# Telco Customer Churn Prediction App

This Streamlit application predicts customer churn using a dataset from Kaggle. It includes:

- Upload interface for Telco churn data in CSV format
- EDA and preprocessing
- Model selection: Random Forest, Logistic Regression, or SVM
- Performance metrics and confusion matrix
- SHAP explainability visualisation
- Prediction tool for new customer input

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telco-churn-app.git
   cd telco-churn-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run telco_churn_streamlit_app_csv_upload.py
   ```

## Deployment

To deploy on Streamlit Cloud:
- Push this repo to GitHub
- Go to [Streamlit Cloud](https://streamlit.io/cloud) and link your GitHub
- Select `telco_churn_streamlit_app_csv_upload.py` as the app entry point

## Dataset

Download the dataset from Kaggle:
[Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

© 2025 Telco Churn App · For academic use only