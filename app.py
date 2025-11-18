import streamlit as st
import matplotlib.pyplot as plt
from backend import *
import os

st.set_page_config(page_title="Pancreatic Cancer Detection Dashboard", layout="wide")

st.title("🩺 Pancreatic Cancer Detection Dashboard")
st.markdown("""
This app demonstrates machine learning models for detecting pancreatic cancer using urinary biomarkers.

**Dataset credit:** [Urinary Biomarkers for Pancreatic Cancer (Kaggle)](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer/data)
""")

# Load data using backend.py
df = load_data(os.path.join("assets", "data.csv"))

# Preprocess data using backend.py
X, y = preprocess_data(df, target_column='diagnosis')

# Split the data before scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features using backend.py
scale_features(X_train, X_test)

# Train models using backend.py (on scaled features)
models = train_models(X_train, y_train)

# Output confusion matrices for each model using evaluate_classification_model
st.subheader("Model Confusion Matrices on Test Data")
for model_name, model in models.items():
    st.markdown(f"### {model_name}")
    y_pred = model.predict(X_test)
    # Make the confusion matrix plot smaller
    fig, accuracy, sensitivity, specificity = evaluate_classification_model(y_test, y_pred)
    fig.set_size_inches(4, 3)
    st.pyplot(fig)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Sensitivity: {sensitivity:.2f}")
    st.write(f"Specificity: {specificity:.2f}")
