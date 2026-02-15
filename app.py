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
    """Sidebar: download buttons for CSVs in dataset/."""
    st.sidebar.title("🔍 Dataset Management")
    if not DATASET_DIR.exists():
        st.sidebar.caption("dataset/ directory not found.")
        return
    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    if not csv_files:
        st.sidebar.caption("No CSV files in dataset/")
        return
    st.sidebar.subheader("⬇️ Download CSV")
    for path in csv_files:
        data = path.read_bytes()
        st.sidebar.download_button(
            label=path.name,
            data=data,
            file_name=path.name,
            mime="text/csv",
            key=f"download_{path.name}",
        )


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

    # Run pipeline: select dataset from folder or upload a CSV
    st.subheader("Run pipeline")
    csv_files = sorted(DATASET_DIR.glob("*.csv")) if DATASET_DIR.exists() else []
    options = ["(None)"] + [p.name for p in csv_files]
    selected_df = None
    selected_name = None
    uploaded_df = None

    col_left, col_or, col_right = st.columns([2, 1, 2])
    with col_left:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="dataset_upload")
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
                st.success(f"{len(uploaded_df)} rows × {len(uploaded_df.columns)} cols")
            except Exception as e:
                st.error(f"Could not load CSV: {e}")
    with col_or:
        st.markdown("<div style='text-align: center; padding: 2rem 0;'><strong>or</strong></div>", unsafe_allow_html=True)
    with col_right:
        selected = st.selectbox(
            "Select dataset from folder",
            options=options,
            key="run_dataset_select",
        )
        if selected != "(None)" and csv_files:
            path = DATASET_DIR / selected
            try:
                selected_df = pd.read_csv(path)
                selected_name = path.name
                st.caption(f"**{selected_name}** — {len(selected_df)} rows × {len(selected_df.columns)} cols")
            except Exception as e:
                st.error(f"Could not load {selected}: {e}")
        # Spacer so this column height matches the uploader drop zone (~120px)
        st.markdown("<div style='height: 90px;'></div>", unsafe_allow_html=True)

    # Prefer uploaded file; otherwise use selected dataset from folder
    active_df = uploaded_df if uploaded_df is not None else selected_df
    active_source = "uploaded file" if uploaded_df is not None else (f"dataset/{selected_name}" if selected_name else None)

    if active_df is not None:
        st.caption(f"Using: **{active_source}** ({len(active_df)} rows)")
        model_choice = st.selectbox("Model", MODEL_NAMES, key="model_choice")
        run_all = st.checkbox("Run all models and compare metrics", key="run_all_models")

        if st.button("Run pipeline", key="run_pipeline_btn"):
            if run_all:
                all_metrics = []
                with st.spinner("Running all models…"):
                    for name in MODEL_NAMES:
                        try:
                            metrics, cm, y_test, y_pred = run_pipeline(
                                active_df.copy(), target_col="y", model_name=name
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
                            active_df.copy(), target_col="y", model_name=model_choice
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
        st.info("Select a dataset above or upload a CSV to run the pipeline.")


if __name__ == "__main__":
    main()