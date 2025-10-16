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

## Testing

This project includes a pytest suite located at `tests/test_backend.py` which contains unit tests for core functions in `backend.py`.

What the tests cover:

- `load_data`: verifies CSV files are loaded correctly.
- `preprocess_data`: checks that expected columns are dropped and categorical mappings (e.g., `diagnosis`, `sex`) are applied.
- `scale_features`: confirms numeric columns are scaled using sklearn's `StandardScaler`.
- `evaluate_classification_model`: validates confusion-matrix metrics (accuracy, sensitivity, specificity) and that a matplotlib figure is returned.
- `train_models` (smoke test): trains the suite of models using a monkeypatched tuner to avoid long hyperparameter searches and verifies models can make predictions.

How to run the tests locally:

1. Activate your project's virtual environment. Example (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install test requirements (if not already installed):

```bash
pip install pytest
```

3. Run pytest from the project root:

```bash
python -m pytest -q
```

All tests should pass; if any warnings appear they are safe for now but may indicate areas to tidy up.
