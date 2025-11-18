import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Data Resampling
from sklearn.model_selection import train_test_split

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression

# 2. K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier

# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier

# 5. Support Vector Machine (SVM)
from sklearn.svm import SVC

# 6. Naive Bayes
from sklearn.naive_bayes import GaussianNB

# 7. XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# Performance Measures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set higher DPI for better plot quality
plt.rcParams["figure.dpi"] = 150

def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, target_column):
    """
    Preprocess the dataset by handling missing values and encoding categorical variables.
    """
    # Drop irrelevant columns
    data.drop(columns=["sample_id", "patient_cohort", "sample_origin", "stage", "benign_sample_diagnosis", "REG1A"], inplace=True)
    data.dropna(inplace=True)

    # Numerize categorical variables
    data['diagnosis'] = data['diagnosis'].replace({1: 0, 2: 1, 3: 1})
    data['sex'] = data['sex'].replace({"M": 0, "F": 1})

    # Separate features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y

def scale_features(X_train, X_test):

    numeric_cols_to_scale = ["creatinine", "age", "LYVE1", "REG1B", "TFF1"]
    scaler = StandardScaler()
    X_train[numeric_cols_to_scale] = scaler.fit_transform(X_train[numeric_cols_to_scale])
    X_test[numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])

def evaluate_classification_model(y_target, y_predicted):

    # Generate the confusion matrix
    cm = confusion_matrix(y_target, y_predicted)

    # Visualize the confusion matrix and store the plot in a variable
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plot_figure = fig

    tn, fp, fn, tp = cm.ravel()

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return plot_figure, accuracy, sensitivity, specificity

def hyperparameter_tuning_xgboost(X_train, y_train):
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n--- Starting RandomizedSearchCV for XGBoost on Training Data ---")

    param_dist_xgb = {
        'n_estimators': randint(100, 1000),
        'learning_rate': uniform(0.001, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5),
        'reg_alpha': uniform(0, 10),
        'reg_lambda': uniform(0.1, 100)
    }

    xgb_base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    scorers = {
        'accuracy_score': make_scorer(accuracy_score),
        'recall_score': make_scorer(recall_score),
        'precision_score': make_scorer(precision_score),
        'f1_score': make_scorer(f1_score),
        'roc_auc_score': make_scorer(roc_auc_score)
    }

    random_search_xgb = RandomizedSearchCV(
        estimator=xgb_base_model,
        param_distributions=param_dist_xgb,
        n_iter=500,
        scoring=scorers,
        refit='accuracy_score',
        cv=kf,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search_xgb.fit(X_train, y_train)

    return random_search_xgb

def train_models(X_train, y_train):

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', random_state=42),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Hyperparameter Tuned XGBoost": hyperparameter_tuning_xgboost(X_train, y_train)
    }

    for model in list(models.values())[:-1]:
        model.fit(X_train, y_train)

    return models
