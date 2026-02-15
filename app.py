"""Bank Marketing ML: dataset management, run_pipeline, and metrics."""
import pandas as pd
from pathlib import Path
from models.model_logic import run_pipeline, get_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]
# Path relative to this file so it works regardless of cwd
DATASET_DIR = Path(__file__).resolve().parent / "dataset"

def add_dataset_mgmt_sidebar():
    st.sidebar.title("🔍 Dataset Management")

    # List & download CSVs from dataset directory
    if DATASET_DIR.exists():
        st.sidebar.subheader("📁 Datasets in folder")
        csv_files = sorted(DATASET_DIR.glob("*.csv"))
        for path in csv_files:
            data = path.read_bytes()
            st.sidebar.download_button(
                label=f"⬇️ {path.name}",
                data=data,
                file_name=path.name,
                mime="text/csv",
                key=f"download_{path.name}",
            )
        if not csv_files:
            st.sidebar.caption("No CSV files in dataset/")
    else:
        st.sidebar.caption("dataset/ directory not found.")


def main():
    st.set_page_config(page_title="Bank Marketing ML", layout="wide")
    st.title("🏦 Bank Marketing Classification Dashboard")

    for key, default in [
        ("last_metrics", None),
        ("last_cm", None),
        ("last_y_test", None),
        ("last_y_pred", None),
        ("last_model_name", None),
        ("upload_metrics_list", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    add_dataset_mgmt_sidebar()

    # Run pipeline on uploaded file (upload in main area)
    st.subheader("Run pipeline on uploaded file")
    uploaded = st.file_uploader("Upload a CSV to run pipeline", type=["csv"], key="dataset_upload")
    uploaded_df = None
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(uploaded_df)} rows × {len(uploaded_df.columns)} cols")
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    if uploaded_df is not None:
        model_choice = st.selectbox("Model", MODEL_NAMES, key="model_choice")
        run_all = st.checkbox("Run all models and compare metrics", key="run_all_models")

        if st.button("Run pipeline", key="run_pipeline_btn"):
            if run_all:
                all_metrics = []
                with st.spinner("Running all models…"):
                    for name in MODEL_NAMES:
                        try:
                            metrics, cm, y_test, y_pred = run_pipeline(
                                uploaded_df.copy(), target_col="y", model_name=name
                            )
                            all_metrics.append({"Model": name, **metrics})
                        except Exception as e:
                            all_metrics.append({"Model": name, "Error": str(e)})
                st.session_state.upload_metrics_list = all_metrics
                st.session_state.last_metrics = None
                st.session_state.last_cm = None
            else:
                with st.spinner("Running pipeline…"):
                    try:
                        metrics, cm, y_test, y_pred = run_pipeline(
                            uploaded_df.copy(), target_col="y", model_name=model_choice
                        )
                        st.session_state.last_metrics = metrics
                        st.session_state.last_cm = cm
                        st.session_state.last_y_test = y_test
                        st.session_state.last_y_pred = y_pred
                        st.session_state.last_model_name = model_choice
                        st.session_state.upload_metrics_list = None
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

        # Display collected metrics
        if st.session_state.upload_metrics_list:
            st.subheader("Metrics (all models)")
            st.dataframe(pd.DataFrame(st.session_state.upload_metrics_list), width="stretch")
        elif st.session_state.last_metrics is not None:
            st.subheader(f"Metrics — {st.session_state.last_model_name}")
            st.json(st.session_state.last_metrics)
            if st.session_state.last_cm is not None:
                st.subheader("Confusion matrix")
                fig, ax = plt.subplots()
                sns.heatmap(st.session_state.last_cm, annot=True, fmt="d", ax=ax, cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)
                plt.close()
    else:
        st.info("Upload a CSV above to run the pipeline and see metrics.")


if __name__ == "__main__":
    main()