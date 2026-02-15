import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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
    }
    
    if len(np.unique(y)) == 2:
        metrics["AUC"] = roc_auc_score(y_test, y_probs[:, 1])
    else:
        metrics["AUC"] = roc_auc_score(y_test, y_probs, multi_class='ovr')
    print(metrics,"\n",confusion_matrix(y_test, y_pred))
    return metrics, confusion_matrix(y_test, y_pred), y_test, y_pred