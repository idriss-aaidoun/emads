"""
EMADS Streamlit UI Dashboard
Provides a premium, structured dashboard to monitor the multi-agent pipeline.
"""

import os
import sys

# Append the project root to sys.path so that 'app' can be found regardless of how Streamlit is launched
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as tf  # Aliased to prevent collision with project namespace 'st' or standard abbreviations
import pandas as pd
from app.core.state.emads_state import EMADSState
from app.core.supervisor.supervisor_agent import SupervisorAgent

# Set a professional wide-page configuration and title theme
tf.set_page_config(
    page_title="EMADS Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom internal CSS styling injection to build a clean dashboard interface
tf.markdown("""
    <style>
    .main-title { font-size: 38px !important; font-weight: 700; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { font-size: 16px !important; color: #4B5563; margin-bottom: 30px; }
    .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 8px; border-left: 5px solid #3B82F6; }
    .metric-card b { color: #6B7280; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card span { color: #7C3AED !important; font-weight: 700; }
    .agent-header { font-size: 20px !important; font-weight: 600; color: #7C3AED; margin-top: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

def main() -> None:
    # Title Block Header
    tf.markdown("<div class='main-title'>EMADS v1.0 Dashboard</div>", unsafe_allow_html=True)
    tf.markdown("<div class='subtitle'>Explainable Multi-Agent Data Science System • Core Prototype Implementation</div>", unsafe_allow_html=True)
    tf.markdown("---")

    # Sidebar Panel Control Area
    tf.sidebar.header("📁 Data Ingestion Control")
    uploaded_file = tf.sidebar.file_uploader("Upload Target Dataset (CSV Format Only)", type=["csv"])

    target_column = None
    if uploaded_file is not None:
        # Secure the uploaded file locally under the data path
        os.makedirs("data", exist_ok=True)
        local_path = os.path.join("data", uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the headers dynamically to let the user pick their target
        try:
            preview_df = pd.read_csv(local_path, nrows=5)
            target_column = tf.sidebar.selectbox("🎯 Target Prediction Variable", options=list(preview_df.columns))
        except Exception as e:
            tf.sidebar.error(f"Error indexing headers: {str(e)}")

        tf.sidebar.markdown("---")
        trigger_pipeline = tf.sidebar.button("⚡ Execute Multi-Agent Engine", use_container_width=True)  # noqa: deprecated but kept for compat
    else:
        tf.sidebar.info("Awaiting structural dataset upload to mount storage pipelines.")
        trigger_pipeline = False

    # Main Application Window Layout Workspace
    if uploaded_file is not None:
        if trigger_pipeline:
            # 1. Prepare Initial State Frame
            initial_state: EMADSState = {
                "dataset_path": local_path,
                "target_column": target_column,
                "schema_info": None,
                "eda_summary": None,
                "generated_plots": None,
                "preprocessed_data_path": None,
                "model_path": None,
                "metrics": None,
                "report_path": None
            }

            # 2. Trigger Supervisor Graph Engine Execution using a spinner loader
            with tf.spinner("Supervisor orchestrating active agent execution..."):
                try:
                    supervisor = SupervisorAgent()
                    final_state = supervisor.run_pipeline(initial_state)
                    tf.success("🎉 Pipeline executed cleanly to termination.")
                    
                    # Store final state context inside Streamlit's temporary session state layout cache
                    tf.session_state["final_state"] = final_state
                except Exception as ex:
                    tf.error(f"Execution Engine Fault: {str(ex)}")

        # 3. Render Execution Pipeline Results Tab Layout Dashboard if a valid state run cache exists
        if "final_state" in tf.session_state:
            state_data = tf.session_state["final_state"]
            
            # Setup professional isolated dashboard viewport tabs
            tab1, tab2, tab3, tab4 = tf.tabs([
                "📊 Structural Overview & EDA", 
                "⚙️ Preprocessing Summary", 
                "🏆 Model Performance Metrics", 
                "📄 Generated Report Outputs"
            ])

            with tab1:
                tf.markdown("<div class='agent-header'>Data Profile Analytics</div>", unsafe_allow_html=True)
                schema = state_data.get("schema_info", {})
                
                col1, col2, col3 = tf.columns(3)
                with col1:
                    tf.markdown(f"<div class='metric-card'><b>Total Dataset Rows</b><br><span style='font-size:24px;'>{schema.get('num_rows', 'N/A')}</span></div>", unsafe_allow_html=True)
                with col2:
                    tf.markdown(f"<div class='metric-card'><b>Total Attributes</b><br><span style='font-size:24px;'>{schema.get('num_cols', 'N/A')}</span></div>", unsafe_allow_html=True)
                with col3:
                    tf.markdown(f"<div class='metric-card'><b>Target Chosen</b><br><span style='font-size:24px; color:#1E3A8A;'>{state_data.get('target_column')}</span></div>", unsafe_allow_html=True)
                
                tf.markdown("---")
                tf.markdown("<div class='agent-header'>LLM-Generated Qualitative Analytical Insights</div>", unsafe_allow_html=True)
                tf.markdown(state_data.get("eda_summary", "_No EDA summary available._"))

                # Render static graphs safely if present on disk
                plots = state_data.get("generated_plots", [])
                if plots:
                    tf.markdown("<div class='agent-header'>Exploratory Distribution Plots</div>", unsafe_allow_html=True)
                    plot_cols = tf.columns(len(plots))
                    for idx, plot_path in enumerate(plots):
                        if os.path.exists(plot_path):
                            with plot_cols[idx]:
                                tf.image(plot_path, caption=os.path.basename(plot_path))

            with tab2:
                tf.markdown("<div class='agent-header'>Data Transformation Matrix</div>", unsafe_allow_html=True)
                prep_path = state_data.get("preprocessed_data_path")
                if prep_path and os.path.exists(prep_path):
                    tf.success(f"Transformed data saved successfully at: `{prep_path}`")
                    sample_df = pd.read_csv(prep_path, nrows=10)
                    tf.markdown("### Processed Matrix Preview (First 10 Encoded Rows)")
                    tf.dataframe(sample_df)

            with tab3:
                tf.markdown("<div class='agent-header'>Model Training Evaluation</div>", unsafe_allow_html=True)
                metrics = state_data.get("metrics", {})
                
                if "accuracy" in metrics:
                    m_col1, m_col2, m_col3, m_col4 = tf.columns(4)
                    with m_col1:
                        tf.metric(label="Validation Accuracy", value=f"{metrics.get('accuracy', 0):.4f}")
                    with m_col2:
                        tf.metric(label="Weighted Precision", value=f"{metrics.get('precision', 0):.4f}")
                    with m_col3:
                        tf.metric(label="Weighted Recall", value=f"{metrics.get('recall', 0):.4f}")
                    with m_col4:
                        tf.metric(label="Calculated F1-Score", value=f"{metrics.get('f1_score', 0):.4f}")

                    tf.markdown("---")
                    tf.markdown("### Native Confusion Matrix Array")
                    tf.json(metrics.get("confusion_matrix"))
                else:
                    m_col1, m_col2, m_col3, m_col4 = tf.columns(4)
                    with m_col1:
                        tf.metric(label="MAE", value=f"{metrics.get('mae', 0):.4f}")
                    with m_col2:
                        tf.metric(label="MSE", value=f"{metrics.get('mse', 0):.4f}")
                    with m_col3:
                        tf.metric(label="RMSE", value=f"{metrics.get('rmse', 0):.4f}")
                    with m_col4:
                        tf.metric(label="R²", value=f"{metrics.get('r2', 0):.4f}")

            with tab4:
                tf.markdown("<div class='agent-header'>Executive Summary Report Downlinks</div>", unsafe_allow_html=True)
                rep_path = state_data.get("report_path")
                if rep_path:
                    txt_summary_path = rep_path.replace(".pdf", "_summary.txt")
                    if os.path.exists(txt_summary_path):
                        with open(txt_summary_path, "r", encoding="utf-8") as f:
                            report_text = f.read()
                        tf.text_area("Final Executed Report Text Stream Output", report_text, height=350)
                        
                        # Provide a native download link for user delivery
                        tf.download_button(
                            label="⬇️ Download Executive Operational Report Bundle",
                            data=report_text,
                            file_name="EMADS_Executive_Summary.txt",
                            mime="text/plain",
                        )
    else:
        tf.warning("Please upload a CSV data file using the left sidebar menu to initialize the system pipeline components.")


if __name__ == "__main__":
    main()