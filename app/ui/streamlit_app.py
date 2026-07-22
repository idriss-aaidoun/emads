"""
EMADS Streamlit UI Dashboard (V2)
====================================

Full dashboard: dataset upload, live per-agent progress tracking (via
LangGraph's .stream()), and a tabbed results view covering every stage
of the pipeline — EDA, preprocessing, model comparison, hyperparameters,
evaluation, explainability, and the final PDF report.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd

from app.core.state.emads_state import create_initial_state
from app.core.supervisor.supervisor_agent import SupervisorAgent
from app.utils.file_utils import ensure_dir
import app.utils.logger as logger_utils

# Pipeline-level logger for the UI layer
_ui_logger = logger_utils.get_logger("emads.ui")

st.set_page_config(page_title="EMADS Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Design tokens — one consistent palette used everywhere below
# ----------------------------------------------------------------------
PRIMARY = "#4338CA"     # indigo
PRIMARY_LIGHT = "#818CF8"
ACCENT = "#06B6D4"      # cyan
SUCCESS = "#10B981"
BG_CARD = "#F8FAFC"
TEXT_MUTED = "#64748B"

STEP_LABELS = {
    "data_understanding": "🔍 Understanding your data",
    "eda": "📊 Exploring & visualizing",
    "preprocessing": "🧹 Cleaning & transforming",
    "model_selection": "🏆 Comparing candidate models",
    "hyperparameter_optimization": "🎯 Tuning hyperparameters",
    "evaluation": "📈 Evaluating performance",
    "explainability": "💡 Explaining model decisions",
    "reporting": "📄 Generating final report",
}

st.markdown(f"""
<style>
    .stApp {{ background-color: #0F172A; color: #E2E8F0; }}
    .stApp * {{ color: #E2E8F0; }}
    .main-header {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;
        border: 1px solid {PRIMARY};
    }}
    .main-header h1 {{ color: white; font-size: 32px; font-weight: 800; margin: 0; }}
    .main-header p {{ color: {TEXT_MUTED}; font-size: 15px; margin-top: 4px; }}
    .metric-card {{
        background-color: {BG_CARD}; border-radius: 12px; padding: 18px 20px;
        border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .metric-card .label {{ color: {TEXT_MUTED}; font-size: 12px; text-transform: uppercase;
        letter-spacing: 0.06em; font-weight: 600; }}
    .metric-card .value {{ color: {PRIMARY_LIGHT}; font-size: 26px; font-weight: 800; margin-top: 4px; }}
    .section-title {{ font-size: 20px; font-weight: 700; color: {TEXT_MUTED}; margin: 18px 0 10px 0; 
        border-bottom: 2px solid {ACCENT}; padding-bottom: 6px; display: inline-block; }}
    .decision-card {{
        background-color: {BG_CARD}; border-radius: 10px; padding: 14px 16px;
        margin-bottom: 10px; border-left: 3px solid {ACCENT};
    }}
    .decision-agent {{ color: {PRIMARY_LIGHT}; font-weight: 700; font-size: 13px; }}
    .decision-reason {{ color: {TEXT_MUTED}; font-size: 13px; margin-top: 4px; }}
    .confidence-badge {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 700; color: #064E3B; background-color: {SUCCESS};
    }}
    div[data-testid="stSidebar"] {{ background-color: #020617; }}
</style>
""", unsafe_allow_html=True)


def header() -> None:
    st.markdown("""
        <div class="main-header">
            <h1>🧠 EMADS — Explainable Multi-Agent Data Science System</h1>
            <p>Upload a dataset → let 8 specialized agents analyze, clean, model, and explain it for you.</p>
        </div>
    """, unsafe_allow_html=True)


def metric_card(col, label: str, value) -> None:
    col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.header("📁 Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])

    local_path, target_column = None, None
    if uploaded_file is not None:
        upload_dir = ensure_dir("uploads")
        local_path = os.path.join(upload_dir, uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            preview_df = pd.read_csv(local_path, nrows=5)
            st.sidebar.markdown("**Preview**")
            st.sidebar.dataframe(preview_df.head(3), use_container_width=True)
            target_column = st.sidebar.selectbox("🎯 Target column", options=list(preview_df.columns))
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button(
        "⚡ Run EMADS Pipeline", use_container_width=True, type="primary", disabled=uploaded_file is None
    )
    if uploaded_file is None:
        st.sidebar.info("Upload a CSV to get started.")

    # ---- Log file download button ----------------------------------------
    st.sidebar.markdown("---")
    log_file = logger_utils.get_log_file_path()
    if os.path.exists(log_file):
        with open(log_file, "rb") as _lf:
            st.sidebar.download_button(
                label="📋 Download Pipeline Log",
                data=_lf.read(),
                file_name=os.path.basename(log_file),
                mime="text/plain",
                use_container_width=True,
            )
    # -----------------------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.caption("EMADS v2.0 • 8 specialized agents • Explainable by design")
    return local_path, target_column, run_clicked


def run_pipeline_with_progress(dataset_path: str, target_column: str) -> dict:
    """Streams the LangGraph execution so the UI can show live, per-agent progress
    instead of a single blocking spinner."""
    # Reset the logger cache so ALL agents (eda, explainability, etc.) get a
    # fresh handler pointing to THIS run's log file — not the previous one.
    reset_session_fn = getattr(logger_utils, "reset_session", None)
    if callable(reset_session_fn):
        reset_session_fn()
    _ui_logger = logger_utils.get_logger("emads.ui")  # re-acquire after reset

    _ui_logger.info(
        "Pipeline run started — dataset=%s target_column=%s",
        os.path.basename(dataset_path), target_column,
    )
    initial_state = create_initial_state(dataset_path, dataset_name=os.path.basename(dataset_path))
    initial_state["target_column"] = target_column

    supervisor = SupervisorAgent()
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_box = st.empty()

    total_steps = len(supervisor.pipeline_steps)
    final_state = dict(initial_state)

    try:
        for i, event in enumerate(supervisor.workflow.stream(initial_state), start=1):
            step_name = list(event.keys())[0]
            final_state = _merge_stream_update(final_state, event[step_name])
            label = STEP_LABELS.get(step_name, step_name)
            _ui_logger.info("Pipeline step completed — step=%s (%s/%s)", step_name, i, total_steps)

            if step_name == "eda":
                eda_summary = final_state.get("eda_summary") or ""
                _ui_logger.info(
                    "LLM payload — eda_summary chars=%s starts_with=%r",
                    len(eda_summary),
                    eda_summary[:80],
                )
            elif step_name == "model_selection":
                model_summary = final_state.get("model_selection_summary") or ""
                _ui_logger.info(
                    "LLM payload — model_selection_summary chars=%s starts_with=%r",
                    len(model_summary),
                    model_summary[:80],
                )
            elif step_name == "explainability":
                explain_summary = final_state.get("explainability_summary") or ""
                _ui_logger.info(
                    "LLM payload — explainability_summary chars=%s starts_with=%r",
                    len(explain_summary),
                    explain_summary[:80],
                )
            elif step_name == "reporting":
                report_path = final_state.get("report_path")
                _ui_logger.info(
                    "Reporting output — report_path=%s",
                    report_path,
                )

            status_box.info(f"{label}...")
            progress_bar.progress(min(i / total_steps, 1.0), text=label)
    except Exception as exc:
        _ui_logger.error("Pipeline stopped with error: %s", exc, exc_info=True)
        progress_bar.empty()
        status_box.error(f"❌ Pipeline stopped: {exc}")
        return None

    _ui_logger.info("Pipeline run completed successfully.")
    progress_bar.progress(1.0, text="Done!")
    status_box.success("🎉 Pipeline completed successfully.")
    return final_state


def _merge_stream_update(current_state: dict, update: dict) -> dict:
    """Applies streamed LangGraph updates while preserving accumulator fields."""
    merged = dict(current_state)
    for key, value in update.items():
        if key in {"agent_decisions", "logs", "errors"}:
            merged[key] = [*(merged.get(key) or []), *(value or [])]
        else:
            merged[key] = value
    return merged


def render_overview_tab(state: dict) -> None:
    schema = state.get("schema_info") or {}
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Rows", schema.get("num_rows", "N/A"))
    metric_card(c2, "Columns", schema.get("num_cols", "N/A"))
    metric_card(c3, "Target", state.get("target_column", "N/A"))
    metric_card(c4, "Problem Type", (state.get("problem_type") or "N/A").title())

    issues = schema.get("quality_issues", [])
    if issues:
        section_title("⚠️ Data Quality Issues")
        for issue in issues:
            st.warning(issue["message"])
    else:
        st.success("✅ No data quality issues detected.")


def render_eda_tab(state: dict) -> None:
    section_title("LLM Insights")
    st.markdown(state.get("eda_summary") or "_No EDA summary available._")

    plots = state.get("generated_plots") or []
    existing = [p for p in plots if os.path.exists(p)]
    if existing:
        section_title("Visualizations")
        cols = st.columns(3)
        for i, plot_path in enumerate(existing):
            with cols[i % 3]:
                st.image(plot_path, use_container_width=True)


def render_preprocessing_tab(state: dict) -> None:
    report = state.get("preprocessing_report") or {}
    if not report:
        st.info("No preprocessing report available.")
        return

    section_title("Decisions Made")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Dropped columns:** {report.get('dropped_columns') or 'None'}")
        st.markdown(f"**Scaling applied:** {report.get('scaling') or 'None'}")
    with c2:
        st.markdown(f"**Imputation:** {report.get('imputation') or 'None'}")
        st.markdown(f"**Encoding:** {report.get('encoding') or 'None'}")

    prep_path = state.get("preprocessed_data_path")
    if prep_path and os.path.exists(prep_path):
        section_title("Cleaned Data Preview")
        st.dataframe(pd.read_csv(prep_path, nrows=10), use_container_width=True)


def render_model_tab(state: dict) -> None:
    section_title("Model Comparison")
    results = state.get("candidate_models_results") or []
    if results:
        df = pd.DataFrame(results)[["model_name", "mean_score", "std_score"]]
        df.columns = ["Model", "Mean CV Score", "Std Dev"]
        st.dataframe(
            df.style.highlight_max(subset=["Mean CV Score"], color=f"{ACCENT}33"),
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(df.set_index("Model")["Mean CV Score"])

    st.success(f"🏆 Selected model: **{state.get('selected_model_name', 'N/A')}**")

    summary = state.get("model_selection_summary")
    if summary:
        section_title("Why this model was selected")
        st.markdown(summary)

    hyperparams = state.get("best_hyperparameters")
    if hyperparams:
        section_title("Optimized Hyperparameters")
        st.json(hyperparams)
        opt_summary = state.get("optimization_summary") or {}
        if opt_summary.get("improvement") is not None:
            st.info(
                f"Tuning changed the score from {opt_summary['baseline_score']:.4f} "
                f"to {opt_summary['best_score']:.4f} "
                f"({opt_summary['improvement']:+.4f}) over {opt_summary['n_trials_run']} trials."
            )


def render_evaluation_tab(state: dict) -> None:
    metrics = state.get("metrics") or {}
    section_title("Metrics")

    if "accuracy" in metrics:
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        metric_card(c2, "Precision", f"{metrics.get('precision', 0):.3f}")
        metric_card(c3, "Recall", f"{metrics.get('recall', 0):.3f}")
        metric_card(c4, "F1-score", f"{metrics.get('f1_score', 0):.3f}")
        if metrics.get("roc_auc") is not None:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        st.caption(
            f"Cross-validation: {metrics.get('cv_mean_accuracy', 0):.3f} "
            f"± {metrics.get('cv_std_accuracy', 0):.3f}"
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "MAE", f"{metrics.get('mae', 0):.3f}")
        metric_card(c2, "MSE", f"{metrics.get('mse', 0):.3f}")
        metric_card(c3, "RMSE", f"{metrics.get('rmse', 0):.3f}")
        metric_card(c4, "R²", f"{metrics.get('r2', 0):.3f}")

    plots = [p for p in (state.get("evaluation_plots") or []) if os.path.exists(p)]
    if plots:
        section_title("Diagnostic Plots")
        cols = st.columns(len(plots))
        for i, p in enumerate(plots):
            cols[i].image(p, use_container_width=True)


def render_explainability_tab(state: dict) -> None:
    section_title("Why the model decides what it decides")
    summary = state.get("explainability_summary")

    if not summary:
        model_name = state.get("selected_model_name", "Trained Model")
        importance = state.get("feature_importance") or {}
        top_features = list(importance.items())[:5]
        if top_features:
            top_name, top_score = top_features[0]
            bullets = [
                f"* **Primary Feature Driver**: The model `{model_name}` relies most heavily on `{top_name}` (relative importance: {top_score:.2%})."
            ]
            if len(top_features) > 1:
                sec = ", ".join([f"`{k}` ({v:.2%})" for k, v in top_features[1:4]])
                bullets.append(f"* **Secondary Key Contributors**: Influential features include {sec}.")
            bullets.append("* **Decision Transparency**: Feature importance reflects predictive correlation in the dataset. Validate against domain expectations.")
            summary = "\n\n".join(bullets)

    if summary and summary.startswith("[Groq API Error]"):
        st.warning(f"⚠️ **LLM Inference Warning**: {summary}")
        st.info("Showing visual feature importance and SHAP analysis below:")
    elif summary:
        st.markdown(summary)
    else:
        st.info("_Explainability summary not available yet. Rerun the pipeline after the model and LLM step complete._")

    importance = state.get("feature_importance") or {}
    if importance:
        section_title("Feature Importance")
        df = pd.DataFrame(list(importance.items())[:10], columns=["Feature", "Importance"])
        st.bar_chart(df.set_index("Feature"))

    for p in (state.get("shap_plots") or []):
        if os.path.exists(p):
            st.image(p, use_container_width=True)


def render_decisions_tab(state: dict) -> None:
    section_title("Full Agent Decision Log")
    decisions = state.get("agent_decisions") or []
    if not decisions:
        st.info("No decisions recorded.")
        return
    for d in decisions:
        confidence_html = (
            f'<span class="confidence-badge">{d.confidence:.0%} confidence</span>'
            if d.confidence is not None else ""
        )
        st.markdown(f"""
            <div class="decision-card">
                <span class="decision-agent">{d.agent_name}</span> {confidence_html}
                <div><b>{d.decision}</b></div>
                <div class="decision-reason">{d.reasoning}</div>
            </div>
        """, unsafe_allow_html=True)


def render_report_tab(state: dict) -> None:
    section_title("Final Report")
    report_path = state.get("report_path")
    if report_path and os.path.exists(report_path):
        st.success(f"Report ready: `{os.path.basename(report_path)}`")
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report", data=f.read(),
                file_name=os.path.basename(report_path), mime="application/pdf",
                use_container_width=True, type="primary",
            )
    else:
        st.warning("Report not generated yet.")


def main() -> None:
    header()
    local_path, target_column, run_clicked = render_sidebar()

    if run_clicked and local_path and target_column:
        final_state = run_pipeline_with_progress(local_path, target_column)
        if final_state:
            st.session_state["final_state"] = final_state

    if "final_state" in st.session_state:
        state = st.session_state["final_state"]
        tabs = st.tabs([
            "🏠 Overview", "📊 EDA", "🧹 Preprocessing", "🏆 Model",
            "📈 Evaluation", "💡 Explainability", "🧾 Decisions Log", "📄 Report",
        ])
        with tabs[0]: render_overview_tab(state)
        with tabs[1]: render_eda_tab(state)
        with tabs[2]: render_preprocessing_tab(state)
        with tabs[3]: render_model_tab(state)
        with tabs[4]: render_evaluation_tab(state)
        with tabs[5]: render_explainability_tab(state)
        with tabs[6]: render_decisions_tab(state)
        with tabs[7]: render_report_tab(state)
    elif not run_clicked:
        st.info("👈 Upload a dataset and click **Run EMADS Pipeline** to get started.")


if __name__ == "__main__":
    main()
