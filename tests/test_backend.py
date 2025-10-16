import numpy as np
import pandas as pd
import pytest

from backend import (
	load_data,
	preprocess_data,
	scale_features,
	evaluate_classification_model,
	train_models,
	hyperparameter_tuning_xgboost,
)


def make_minimal_df():
	# Create a small dataframe that matches the expectations of preprocess_data
	df = pd.DataFrame({
		'sample_id': [1, 2, 3, 4],
		'patient_cohort': ['A', 'B', 'A', 'B'],
		'sample_origin': ['X', 'Y', 'X', 'Y'],
		'stage': [1, 2, 1, 2],
		'benign_sample_diagnosis': [0, 0, 0, 0],
		'REG1A': [0.1, 0.2, 0.3, 0.4],
		'diagnosis': [1, 2, 2, 3],  # will be mapped to 0,1,1,1
		'sex': ['M', 'F', 'M', 'F'],
		'creatinine': [0.9, 1.1, 1.0, 1.2],
		'age': [50, 60, 55, 65],
		'LYVE1': [10, 20, 15, 25],
		'REG1B': [0.5, 0.6, 0.55, 0.65],
		'TFF1': [2.0, 2.5, 2.2, 2.6],
		'extra_feature': [9, 8, 7, 6],
	})
	return df


def test_load_data_tmpfile(tmp_path):
	df = make_minimal_df()
	p = tmp_path / "mini.csv"
	df.to_csv(p, index=False)

	loaded = load_data(str(p))
	assert loaded is not None
	assert isinstance(loaded, pd.DataFrame)
	# should have same rows and at least the columns we wrote
	assert loaded.shape[0] == df.shape[0]
	for col in df.columns:
		assert col in loaded.columns


def test_preprocess_data_mappings_and_drop():
	df = make_minimal_df()
	X, y = preprocess_data(df.copy(), target_column='diagnosis')

	# dropped columns should not be in X
	for dropped in ["sample_id", "patient_cohort", "sample_origin", "stage", "benign_sample_diagnosis", "REG1A"]:
		assert dropped not in X.columns

	# mappings: diagnosis values 1 -> 0, 2/3 -> 1
	assert set(y.unique()) <= {0, 1}
	# sex mapping to 0/1
	assert set(X['sex'].unique()) <= {0, 1}


def test_scale_features_transforms():
	df = make_minimal_df()
	X, y = preprocess_data(df.copy(), target_column='diagnosis')

	# create train/test splits
	X_train = X.iloc[:3].copy()
	X_test = X.iloc[3:].copy()

	numeric_cols = ["creatinine", "age", "LYVE1", "REG1B", "TFF1"]

	# preserve originals to compare for test set
	original_test = X_test[numeric_cols].copy()

	scale_features(X_train, X_test)

	# After scaling, train numeric columns should have mean approx 0 and std approx 1
	means = X_train[numeric_cols].mean().abs()
	# Use population std (ddof=0) to match sklearn StandardScaler behavior
	stds = X_train[numeric_cols].std(ddof=0)

	for m in means:
		assert m < 1e-6 or pytest.approx(0, abs=1e-6) == m
	for s in stds:
		assert pytest.approx(1, rel=1e-2) == s

	# Test set should not equal original (should be transformed)
	assert not np.allclose(original_test.values, X_test[numeric_cols].values)


def test_evaluate_classification_model_metrics():
	# Confusion matrix: tn=1, fp=1, fn=1, tp=1 -> accuracy=0.5, sensitivity=0.5, specificity=0.5
	y_true = np.array([0, 0, 1, 1])
	y_pred = np.array([0, 1, 0, 1])

	fig, accuracy, sensitivity, specificity = evaluate_classification_model(y_true, y_pred)

	assert hasattr(fig, 'savefig')
	assert pytest.approx(0.5, rel=1e-6) == accuracy
	assert pytest.approx(0.5, rel=1e-6) == sensitivity
	assert pytest.approx(0.5, rel=1e-6) == specificity


def test_train_models_smoke(monkeypatch):
	# Create a larger synthetic dataset for training
	rng = np.random.RandomState(42)
	n = 50
	X = pd.DataFrame({
		'creatinine': rng.normal(size=n),
		'age': rng.normal(loc=50, scale=10, size=n),
		'LYVE1': rng.normal(size=n),
		'REG1B': rng.normal(size=n),
		'TFF1': rng.normal(size=n),
		'extra_feature': rng.normal(size=n),
	})
	y = rng.randint(0, 2, size=n)

	# Monkeypatch the expensive hyperparameter tuning function so it doesn't run a long search
	def fake_tuner(Xt, yt):
		from xgboost import XGBClassifier

		return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

	monkeypatch.setattr('backend.hyperparameter_tuning_xgboost', fake_tuner)

	models = train_models(X, y)

	# Check expected model keys exist
	expected_keys = [
		"Logistic Regression",
		"K-Nearest Neighbors",
		"Decision Tree",
		"Random Forest",
		"Support Vector Machine",
		"Naive Bayes",
		"XGBoost",
		"Hyperparameter Tuned XGBoost",
	]

	for k in expected_keys:
		assert k in models

	# Ensure the fitted models can predict on training data (except the hyperparameter stub)
	for name, model in models.items():
		if name == "Hyperparameter Tuned XGBoost":
			# tuner returned an unfitted estimator
			continue
		preds = model.predict(X)
		assert len(preds) == len(X)
