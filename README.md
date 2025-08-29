# Application Structure

## backend.py
This module contains all core data processing and machine learning logic, including:
- Data loading and preprocessing
- Feature scaling
- Model training (Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, XGBoost)
- Model evaluation (confusion matrix, accuracy, sensitivity, specificity)
- Hyperparameter tuning for XGBoost

## app.py
This is a Streamlit dashboard that:
- Loads and preprocesses the data using `backend.py`
- Splits and scales the data
- Trains all models using `backend.py`
- Displays confusion matrices and metrics for each model
- Credits the Kaggle dataset

## How to Run the Streamlit App
1. Install dependencies (in your virtual environment):
	```bash
	pip install -r requirements.txt
	```
2. Launch the Streamlit app:
	```bash
	streamlit run app.py
	```
3. Open the provided local URL in your browser to interact with the dashboard.

The app will automatically use the data in `assets/data.csv` and display results for all models.
# Pancreatic Cancer Detection

This project aims to detect pancreatic cancer using urinary biomarkers and various machine learning models. The dataset used is publicly available on Kaggle.

## Dataset
- [Urinary Biomarkers for Pancreatic Cancer (Kaggle)](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer/data)

## Machine Learning Models Used
The following models were implemented and evaluated:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Random Forest**
5. **Support Vector Machine (SVM)**
6. **Naive Bayes**
7. **XGBoost** (including hyperparameter tuning with RandomizedSearchCV)

## Workflow Overview
- Data loading and preprocessing
- Exploratory data analysis
- Feature selection and scaling
- Model training and evaluation
- Hyperparameter tuning (for XGBoost)

## Requirements
See `requirements.txt` for all dependencies.

## Usage
Open and run the notebook `pacreatic_cancer.ipynb` to reproduce the analysis and results.

## License
See `LICENSE` for details.
