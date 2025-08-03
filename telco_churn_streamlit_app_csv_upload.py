
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title='Telco Customer Churn Predictor', layout='wide')
st.title("Telco Customer Churn Prediction")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

uploaded_file = st.file_uploader(
    "Upload Telco Churn Dataset (.csv format, from Kaggle)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.success("✅ File uploaded and loaded successfully.")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"⚠️ Failed to read the uploaded file. Error: {e}")
        st.stop()
else:
    st.info("📄 Please upload the Telco churn dataset in CSV format to continue.")
    st.stop()

# Preprocessing
df.replace(" ", np.nan, inplace=True)
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)

binary_cols = []
df_copy = df.copy()
for column in df_copy.select_dtypes(include="object").columns:
    if df_copy[column].nunique() == 2:
        le = LabelEncoder()
        df_copy[column] = le.fit_transform(df_copy[column])
        binary_cols.append(column)
    else:
        df_copy = pd.get_dummies(df_copy, columns=[column])

X = df_copy.drop("Churn", axis=1)
y = df_copy["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_choice = st.selectbox("Choose ML model", ["Random Forest", "Logistic Regression", "SVM"])
if model_choice == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC(probability=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.subheader("Model Performance")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.text("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

roc_score = roc_auc_score(y_test, y_proba)
st.text(f"ROC AUC Score: {roc_score:.2f}")

# SHAP Explainability
st.subheader("Feature Importance with SHAP")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
fig2 = plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
st.pyplot(fig2)

# Prediction Interface
st.subheader("Predict Churn for a New Customer")
input_dict = {}
for col in X.columns:
    if 'Yes' in col or 'No' in col:
        input_dict[col] = st.selectbox(f"{col}", [0, 1])
    else:
        input_dict[col] = st.number_input(f"{col}", value=0.0)

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

st.markdown(f"### Prediction: {'Churn' if prediction == 1 else 'No Churn'} (Probability: {prediction_proba:.2f})")
