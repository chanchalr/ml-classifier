"""Test run_pipeline with dataset/bank_batch_1.csv across all models from get_model."""
import pandas as pd
from models.model_logic import run_pipeline, get_model

# Model names must match keys in get_model()
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]

DATA_PATH = "dataset/bank_batch_1.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {len(df)} rows, target 'y'\n")
    print("=" * 60)

    for name in MODEL_NAMES:
        print(f"\n--- {name} ---")
        try:
            metrics, cm, y_test, y_pred = run_pipeline(df, target_col="y", model_name=name)
            print(f"Metrics: {metrics}")
        except Exception as e:
            print(f"Error: {e}")
        print("=" * 60)


if __name__ == "__main__":
    main()
