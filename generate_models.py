"""
Train all models on dataset/bank_full.csv; save each model as .pkl,
save preprocessing artifacts, and export test split as test.csv.

Flow: split train/test at the start -> save test.csv -> apply feature
engineering on training data only -> train models and save artifacts.
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.model_logic import get_model, MODEL_NAME_TO_PKL, apply_feature_engineering

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "bank_full.csv"
SAVED_DIR = PROJECT_ROOT / "saved_models"
TEST_CSV_PATH = PROJECT_ROOT / "test.csv"
TARGET_COL = "y"

MODEL_NAMES = list(MODEL_NAME_TO_PKL.keys())


def generate_models():
    """Load bank_full.csv, split test set first, then train on engineered training data."""
    df = pd.read_csv(DATA_PATH)

    # 1. Create test dataset at the beginning (before any feature engineering)
    df_train, df_test = train_test_split(
         df,test_size=0.2, random_state=42
    )
    df_test.to_csv(TEST_CSV_PATH, index=False)
    print(f"Saved test data: {TEST_CSV_PATH} ({len(df_test)} rows)")

    # 3. Preprocessing and training
    X = df_train.drop(columns=[TARGET_COL])
    y = df_train[TARGET_COL]

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_dum = pd.get_dummies(X, drop_first=True)
    feature_columns = list(X_dum.columns)
    '''
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    '''
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dum)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # Save preprocessing artifacts (needed by run_model)
    with open(SAVED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(SAVED_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(SAVED_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    print(f"Saved artifacts in {SAVED_DIR}")

    # Train and save each model
    for name in MODEL_NAMES:
        model = get_model(name)
        model.fit(X_scaled, y_enc)
        path = SAVED_DIR / MODEL_NAME_TO_PKL[name]
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model: {path}")


if __name__ == "__main__":
    generate_models()
