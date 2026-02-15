import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Project root and paths for saved models (used by run_model)
_MODEL_LOGIC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODEL_LOGIC_DIR.parent
SAVED_MODELS_DIR = _MODEL_LOGIC_DIR  # same directory as this file (models/)
MODEL_NAME_TO_PKL = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbor": "k_nearest_neighbor.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def get_model(name):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbor": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    return models.get(name)
def apply_feature_engineering(df):
    df = df.copy()
    
    # 1. Age Binning: Group ages into meaningful categories
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # 2. Log Transform 'balance': Reduces skewness of financial data
    # We add a constant to handle negative balances
    df['balance_log'] = np.log1p(df['balance'] - df['balance'].min())
    
    # 3. Seasonal Engineering: Map months to seasons
    # Month is originally text (jan, feb...)
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map)
    
    # Cyclical encoding for months (so 12 is near 1)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    # 4. Binary Conversion: Cleanup simple yes/no columns
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    # Drop original columns that were transformed or are redundant
    df = df.drop(columns=['age', 'balance', 'month', 'month_num'])
    return df

def run_pipeline(df, target_col='y', model_name='XGBoost'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1": f1_score(y_test, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    
    if len(np.unique(y)) == 2:
        metrics["AUC"] = roc_auc_score(y_test, y_probs[:, 1])
    else:
        metrics["AUC"] = roc_auc_score(y_test, y_probs, multi_class='ovr')
    print(metrics,"\n",confusion_matrix(y_test, y_pred))
    return metrics, confusion_matrix(y_test, y_pred), y_test, y_pred


def run_model(test_data, model_name, target_col="y", saved_dir=None):
    """
    Load test data (path to test.csv or DataFrame), load the saved .pkl for the
    given model, predict, and compute metrics.

    test_data: path to test.csv (str or Path) or DataFrame with same schema as bank_full.
    model_name: one of get_model keys (e.g. "XGBoost", "Logistic Regression").
    saved_dir: directory containing .pkl and artifacts (default: PROJECT_ROOT/saved_models).

    Returns: metrics (dict), confusion_matrix (ndarray), y_true, y_pred.
    """
    if saved_dir is None:
        saved_dir = SAVED_MODELS_DIR
    saved_dir = Path(saved_dir)
    if not saved_dir.exists():
        raise FileNotFoundError(f"Saved models directory not found: {saved_dir}. Run generate_models.py first.")

    if isinstance(test_data, (str, Path)):
        df = pd.read_csv(test_data)
    else:
        df = pd.DataFrame(test_data)

    # Apply same feature engineering as in generate_models (training data)
    #df = apply_feature_engineering(df)
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Load preprocessing artifacts
    with open(saved_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(saved_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(saved_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    # Same preprocessing as training: get_dummies then align to training columns
    X_dum = pd.get_dummies(X_raw, drop_first=True)
    for col in feature_columns:
        if col not in X_dum.columns:
            X_dum[col] = 0
    X_dum = X_dum[feature_columns]
    y_true = le.transform(y_raw.astype(str))
    X_scaled = scaler.transform(X_dum)

    # Load model and predict
    pkl_name = MODEL_NAME_TO_PKL.get(model_name)
    if not pkl_name:
        raise ValueError(f"Unknown model_name: {model_name}. Use one of {list(MODEL_NAME_TO_PKL.keys())}")
    with open(saved_dir / pkl_name, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_scaled)
    y_probs = model.predict_proba(X_scaled)

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:
        metrics["AUC"] = roc_auc_score(y_true, y_probs[:, 1])
    else:
        metrics["AUC"] = roc_auc_score(y_true, y_probs, multi_class="ovr")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Model: {model_name}")
    print("Metrics:", metrics)
    print("Confusion matrix:\n", cm)
    return metrics, cm, y_true, y_pred