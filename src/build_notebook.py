"""Build notebooks/predictive_model.ipynb programmatically (valid nbformat file)."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

BASE = Path(__file__).resolve().parent.parent
NB_PATH = BASE / "notebooks" / "predictive_model.ipynb"

cells = []
md = lambda t: cells.append(new_markdown_cell(t))
code = lambda t: cells.append(new_code_cell(t))

md("# Predictive Modeling Using Machine Learning\n"
   "## Loan Approval Prediction (Classification) + Interest-Rate Regression\n\n"
   "End-to-end supervised ML pipeline:\n"
   "**Raw Data -> Cleaning -> Feature Engineering -> Encode/Scale -> Train/Test Split -> "
   "Model Training -> Hyperparameter Tuning -> Evaluation -> Visualization -> Prediction**\n\n"
   "Run the cells top-to-bottom. The project also ships reusable source modules in `src/`.")

md("## 0. Setup & Imports")
code(
"import sys\n"
"from pathlib import Path\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n\n"
"sns.set_theme(style='whitegrid')\n"
"sys.path.append(str(Path('..') / 'src'))\n\n"
"import config as cfg\n"
"from preprocessing import (load_data, prepare_pipeline, encode_categoricals)\n"
"from feature_engineering import create_features, add_polynomial_features, select_k_best\n"
"from train_model import (train_classification, train_regression,\n"
"                         select_best_classifier, save_model)\n"
"from evaluate import (evaluate_classifier, evaluate_regression,\n"
"                      classification_metrics, regression_metrics)\n"
"from pipeline import run_full_pipeline\n\n"
"print('Environment ready. Project root:', cfg.BASE)")

md("## 1. Load Raw Data")
code(
"train_raw, test_raw = load_data()\n"
"print('train:', train_raw.shape, '| test:', test_raw.shape)\n"
"train_raw.head()")

md("### Quick EDA")
code(
"print('Missing values per column:')\n"
"print(train_raw.isna().sum()[train_raw.isna().sum() > 0])\n"
"print('\\nDuplicate rows:', train_raw.duplicated().sum())\n"
"print('\\nTarget balance:')\n"
"print(train_raw['loan_approved'].value_counts(normalize=True).round(3))\n\n"
"num_cols = cfg.BASE_NUMERIC_COLS\n"
"train_raw[num_cols].describe().round(2)")

md("## 2. Preprocessing\n"
   "Handles duplicates, imputes missing values (using **training** statistics), removes IQR "
   "outliers from the training set only, one-hot encodes categoricals, and scales with "
   "`StandardScaler` (fitted on train only).")
code(
"(X_train, X_val, X_test,\n"
" y_train, y_val, y_test,\n"
" feature_cols, scaler) = prepare_pipeline(\n"
"    train_raw, test_raw, test_size=0.2, random_state=cfg.RANDOM_STATE)\n\n"
"print('Scaled train/val/test shapes:', X_train.shape, X_val.shape, X_test.shape)\n"
"print('Number of features:', len(feature_cols))\n"
"from pathlib import Path\n"
"(cfg.MODELS_DIR / 'features.json').write_text(__import__('json').dumps(feature_cols))")

md("## 3. Feature Engineering\n"
   "New meaningful features are created before encoding (leakage-safe, row-wise):\n"
   "`debt_to_income`, `loan_to_income`, `income_per_dependent`, `loan_per_month`, "
   "`employment_ratio`, `credit_tier`, `age_group`.")
code(
"sample = create_features(train_raw.head(5))\n"
"print(sample[['age','income','dependents','income_per_dependent','credit_score','credit_tier','age_group']])")

md("### (Optional) Polynomial Features\n"
   "Demonstrates interaction terms via `PolynomialFeatures` (fit on train only).")
code(
"Xtr_p, Xva_p, Xte_p, poly_names, poly = add_polynomial_features(\n"
"    X_train, X_val, X_test, feature_cols, degree=2)\n"
"print('Original features:', X_train.shape[1], '-> Polynomial features:', Xtr_p.shape[1])")

md("### Feature Selection (SelectKBest)")
code(
"Xtr_k, Xva_k, Xte_k, selector, mask = select_k_best(\n"
"    X_train, y_train, X_val, X_test, k=20)\n"
"kept = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]\n"
"print('Selected (top 20):', kept)")

md("## 4. Model Training & Hyperparameter Tuning\n"
   "Multiple supervised algorithms are trained with `GridSearchCV` (5-fold cross-validation, scoring=ROC-AUC).")
code(
"clf_results = train_classification(X_train, y_train, cv=5)\n"
"# Per-model best CV scores were printed during training above.")

md("## 5. Model Selection\n"
   "Best classifier chosen by validation ROC-AUC, then saved to `models/trained_model.pkl`.")
code(
"best_name, best_model = select_best_classifier(clf_results, X_val, y_val)\n"
"save_model(best_model, 'trained_model.pkl')\n"
"print('Saved best model:', best_name)")

md("## 6. Evaluation on Held-Out Test Set\n"
   "Generates the confusion matrix, ROC curve, precision-recall curve, feature importance, "
   "learning curve (on the training set), and a text classification report in `reports/`.")
code(
"clf_metrics = evaluate_classifier(best_model, X_test, y_test, feature_cols,\n"
"                                  X_train=X_train, y_train=y_train)\n"
"clf_metrics")

md("### Classification Report (text)")
code(
"print((cfg.REPORTS_DIR / 'classification_report.txt').read_text())")

md("## 7. Secondary Regression Demo (predict `interest_rate`)\n"
   "Compares Linear / Decision Tree / Random Forest regressors (metrics: MAE, MSE, RMSE, R²) "
   "and saves prediction-vs-actual and residual plots.")
code(
"(Xr_tr, Xr_va, Xr_te, yr_tr, yr_va, yr_te, reg_feats, _) = prepare_pipeline(\n"
"    train_raw, test_raw, target='interest_rate',\n"
"    exclude_cols=[], save_scaler=False,\n"
"    test_size=0.2, random_state=cfg.RANDOM_STATE, stratify=None)\n\n"
"reg_results = train_regression(Xr_tr, yr_tr, cv=5)\n"
"best_reg = max(reg_results.values(),\n"
"               key=lambda m: regression_metrics(yr_te, (\n"
"                   m.best_estimator_ if hasattr(m,'best_estimator_') else m).predict(Xr_te))['R2'])\n"
"best_reg_est = best_reg.best_estimator_ if hasattr(best_reg,'best_estimator_') else best_reg\n"
"evaluate_regression(best_reg_est, Xr_te, yr_te)")

md("## 8. Prediction on New Records\n"
   "Loads the saved model + scaler + feature schema and predicts approval for raw inputs.")
code(
"from predict import predict_new\n"
"sample = pd.DataFrame([{\n"
"    'age': 35, 'income': 85000, 'employment_years': 10, 'loan_amount': 15000,\n"
"    'loan_term': 36, 'credit_score': 720, 'existing_loans': 1, 'dependents': 2,\n"
"    'gender': 'Male', 'education': 'Bachelor', 'home_ownership': 'Own',\n"
"    'loan_purpose': 'Home', 'marital_status': 'Married'}])\n"
"predict_new(sample)")

md("## 9. One-Click Full Pipeline\n"
   "Alternatively, run everything end-to-end from the orchestrator module.")
code(
"# summary = run_full_pipeline()  # uncomment to reproduce all artifacts")

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python'}
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebook ->", NB_PATH)
