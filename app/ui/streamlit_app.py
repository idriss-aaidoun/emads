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

from app.core.state.emads_state import create_initial_state, UNSUPERVISED_PROBLEM_TYPES
from app.core.supervisor.supervisor_agent import SupervisorAgent
from app.utils.file_utils import ensure_dir, prune_old_files
import app.utils.logger as logger_utils
from app.ui import theme

# Pipeline-level logger for the UI layer
_ui_logger = logger_utils.get_logger("emads.ui")

st.set_page_config(page_title="EMADS Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

STEP_LABELS = {
    "data_understanding": "🔍 Understanding your data",
    "eda": "📊 Exploring & visualizing",
    "preprocessing": "🧹 Cleaning & transforming",
    "model_selection": "🏆 Comparing candidate models",
    "hyperparameter_optimization": "🎯 Tuning hyperparameters",
    "evaluation": "📈 Evaluating performance",
    "explainability": "💡 Explaining model decisions",
    "reporting": "📄 Generating final report",
    "meta_evaluation": "🧭 Auditing overall run confidence",
}

# Short labels + one-line descriptions for the stepper and the empty-state
# pipeline preview — same 9 steps as STEP_LABELS/pipeline_steps, just laid
# out for a compact card instead of a status line.
STEP_PREVIEW = [
    ("🔍", "Data Understanding", "Profiles columns, infers target & problem type."),
    ("📊", "EDA", "Descriptive stats, outliers, correlation plots."),
    ("🧹", "Preprocessing", "Cleans, imputes, encodes, scales."),
    ("🏆", "Model Selection", "Compares candidate models via CV / silhouette."),
    ("🎯", "Hyperparameter Tuning", "Optuna search on the selected model."),
    ("📈", "Evaluation", "Holdout metrics + stability check."),
    ("💡", "Explainability", "SHAP, permutation importance, narration."),
    ("📄", "Reporting", "Assembles the final PDF report."),
    ("🧭", "Meta-Evaluation", "Audits overall run confidence."),
]

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
theme.inject_theme("dark" if st.session_state.dark_mode else "light")


def header() -> None:
    st.markdown("""
        <div class="main-header">
            <h1>🧠 EMADS — Explainable Multi-Agent Data Science System</h1>
            <p>Upload a dataset → let 8 specialized agents analyze, clean, model, and explain it for you.</p>
        </div>
    """, unsafe_allow_html=True)


def metric_card(col, label: str, value, icon: str = "", status: str | None = None) -> None:
    theme.metric_card(col, label, value, icon=icon, status=status)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


PROBLEM_TYPE_OPTIONS = {
    "Auto-detect": None,
    "Classification": "classification",
    "Regression": "regression",
    "Clustering": "clustering",
    "Anomaly Detection": "anomaly_detection",
}


def render_sidebar():
    st.sidebar.toggle("🌙 Dark mode", key="dark_mode")
    st.sidebar.markdown('<div class="sidebar-section-label">📁 Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"], label_visibility="collapsed")

    local_path, target_column, problem_type = None, None, None
    if uploaded_file is not None:
        upload_dir = ensure_dir("uploads")
        local_path = os.path.join(upload_dir, uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            preview_df = pd.read_csv(local_path, nrows=5)
            st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.sidebar.markdown("**Preview**")
            st.sidebar.dataframe(preview_df.head(3), use_container_width=True)

            problem_type_label = st.sidebar.selectbox("🧩 Problem type", options=list(PROBLEM_TYPE_OPTIONS.keys()))
            problem_type = PROBLEM_TYPE_OPTIONS[problem_type_label]

            if problem_type in UNSUPERVISED_PROBLEM_TYPES:
                st.sidebar.caption("Unsupervised problem type — no target column needed; every column is used as a feature.")
            else:
                # Defaults to the last column, matching DataUnderstandingAgent's
                # own fallback heuristic when no target is specified — the user
                # can still override it via the selectbox.
                columns = list(preview_df.columns)
                target_column = st.sidebar.selectbox(
                    "🎯 Target column", options=columns, index=len(columns) - 1
                )
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    is_unsupervised_choice = problem_type in UNSUPERVISED_PROBLEM_TYPES
    run_ready = uploaded_file is not None and (target_column or is_unsupervised_choice)
    run_clicked = st.sidebar.button(
        "⚡ Run EMADS Pipeline", use_container_width=True, type="primary", disabled=not run_ready
    )
    if uploaded_file is None:
        st.sidebar.info("Upload a CSV to get started.")

    # ---- Log file download button ----------------------------------------
    st.sidebar.markdown('<div class="sidebar-section-label">🛠️ Utilities</div>', unsafe_allow_html=True)
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

    # ---- Cleanup button — logs/reports/uploads grow one file per run ------
    if st.sidebar.button("🧹 Clean old files", use_container_width=True):
        n_logs = prune_old_files("logs", keep_last=5, pattern="*.log")
        n_reports = prune_old_files("reports", keep_last=5, pattern="*.pdf")
        n_uploads = prune_old_files(os.path.join("data", "uploads"), keep_last=5, pattern="*")
        st.sidebar.success(
            f"Deleted {n_logs + n_reports + n_uploads} old file(s) "
            f"({n_logs} logs, {n_reports} reports, {n_uploads} uploads)."
        )
    # -----------------------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.caption("EMADS v2.0 • 8 specialized agents • Explainable by design")
    return local_path, target_column, problem_type, run_clicked


def run_pipeline_with_progress(dataset_path: str, target_column: str | None, problem_type: str | None) -> dict:
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
    if target_column:
        initial_state["target_column"] = target_column
    if problem_type:
        initial_state["problem_type"] = problem_type

    supervisor = SupervisorAgent()
    step_keys = [step_name for step_name, _ in supervisor.pipeline_steps]
    stepper_labels = [
        (key, f"{icon} {title}")
        for key, (icon, title, _) in zip(step_keys, STEP_PREVIEW)
    ]
    stepper_box = st.empty()
    stepper_box.markdown(theme.stepper_html(stepper_labels, current_index=0), unsafe_allow_html=True)
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_box = st.empty()

    total_steps = len(supervisor.pipeline_steps)
    final_state = dict(initial_state)

    try:
        for i, event in enumerate(supervisor.workflow.stream(initial_state), start=1):
            step_name = list(event.keys())[0]
            final_state = _merge_stream_update(final_state, event[step_name])
            label = STEP_LABELS.get(step_name, step_name)
            stepper_box.markdown(
                theme.stepper_html(stepper_labels, current_index=i if i < total_steps else None),
                unsafe_allow_html=True,
            )
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
        if key in {"agent_decisions", "logs", "errors", "llm_arbitration_log"}:
            merged[key] = [*(merged.get(key) or []), *(value or [])]
        else:
            merged[key] = value
    return merged


def render_meta_evaluation_badge(state: dict) -> None:
    meta_evaluation = state.get("meta_evaluation")
    if not meta_evaluation:
        return
    confidence_score = meta_evaluation.get("confidence_score")
    recommendation = meta_evaluation.get("recommendation")
    is_reliable = not recommendation
    badge = "✅ Reliable" if is_reliable else "⚠️ Review recommended"
    score_text = f"{confidence_score:.0%}" if confidence_score is not None else "N/A"
    if is_reliable:
        st.success(f"{badge} — overall confidence: {score_text}")
    else:
        st.warning(f"{badge} — overall confidence: {score_text}\n\n{recommendation}")


def render_overview_tab(state: dict) -> None:
    render_meta_evaluation_badge(state)
    schema = state.get("schema_info") or {}
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Rows", schema.get("num_rows", "N/A"), icon="📋")
    metric_card(c2, "Columns", schema.get("num_cols", "N/A"), icon="🗂️")
    metric_card(c3, "Target", state.get("target_column") or "N/A", icon="🎯")
    metric_card(c4, "Problem Type", (state.get("problem_type") or "N/A").title(), icon="🧩")

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


def render_model_selection_verdict(state: dict) -> None:
    """
    Mirrors PDFService._build_model_selection_verdict: separates the
    deterministic statistical verdict (always shown, confidence = 1 -
    p_value even after arbitration) from the conditional LLM arbitration
    narrative (shown ONLY when a Wilcoxon tie actually triggered one) — so
    the UI never shows the LLM's confident tie-break text without the
    confidence score it belongs next to.
    """
    decisions = state.get("agent_decisions") or []
    selection_decision = next(
        (d for d in decisions
         if d.agent_name == "model_selection_agent" and d.decision.startswith("Selected '")),
        None,
    )
    arbitration_entries = [
        e for e in (state.get("llm_arbitration_log") or [])
        if e.get("agent_name") == "model_selection_agent"
    ]

    section_title("Statistical Verdict")
    if selection_decision is None:
        st.info("No model selection decision recorded.")
        return

    confidence_suffix = (
        f" ({selection_decision.confidence:.0%} confidence)" if selection_decision.confidence is not None else ""
    )
    st.markdown(f"**{selection_decision.decision}**{confidence_suffix}")
    st.caption(selection_decision.reasoning)

    if arbitration_entries:
        confidence_pct = (
            f"{selection_decision.confidence:.0%}" if selection_decision.confidence is not None else "N/A"
        )
        section_title("LLM Interpretation")
        st.caption(
            "Note: this narrative explanation should not be read as statistical certainty — it reflects "
            f"the LLM's reasoning for breaking a tie the statistical test alone could not resolve "
            f"(confidence: {confidence_pct})."
        )
        for entry in arbitration_entries:
            st.markdown(entry.get("llm_arbitration") or "")
    else:
        st.caption("The statistical test found a significant difference; no LLM arbitration was needed.")


def render_model_tab(state: dict) -> None:
    section_title("Model Comparison")
    results = state.get("candidate_models_results") or []
    if results:
        df = pd.DataFrame(results)[["model_name", "mean_score", "std_score"]]
        df.columns = ["Model", "Mean CV Score", "Std Dev"]
        st.dataframe(
            theme.style_model_comparison(df, state.get("selected_model_name")),
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(df.set_index("Model")["Mean CV Score"])

    st.success(f"🏆 Selected model: **{state.get('selected_model_name', 'N/A')}**")

    render_model_selection_verdict(state)

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
        acc = metrics.get("accuracy", 0)
        metric_card(c1, "Accuracy", f"{acc:.3f}", icon="🎯", status=theme.status_from_thresholds(acc, 0.8, 0.5))
        metric_card(c2, "Precision", f"{metrics.get('precision', 0):.3f}", icon="🔎")
        metric_card(c3, "Recall", f"{metrics.get('recall', 0):.3f}", icon="🧲")
        metric_card(c4, "F1-score", f"{metrics.get('f1_score', 0):.3f}", icon="⚖️")
        if metrics.get("roc_auc") is not None:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        ci_caption = ""
        if metrics.get("cv_ci_lower") is not None:
            ci_caption = f" (95% CI: [{metrics['cv_ci_lower']:.3f}, {metrics['cv_ci_upper']:.3f}])"
        st.caption(
            f"Cross-validation: {metrics.get('cv_mean_accuracy', 0):.3f} "
            f"± {metrics.get('cv_std_accuracy', 0):.3f}{ci_caption}"
        )
    elif "silhouette_score" in metrics:
        c1, c2, c3 = st.columns(3)
        sil = metrics.get("silhouette_score", 0)
        metric_card(c1, "Silhouette Score", f"{sil:.3f}", icon="🧩", status=theme.status_from_thresholds(sil, 0.5, 0.25))
        db = metrics.get("davies_bouldin_score")
        metric_card(c2, "Davies-Bouldin", f"{db:.3f}" if db is not None else "N/A", icon="📐")
        ch = metrics.get("calinski_harabasz_score")
        metric_card(c3, "Calinski-Harabasz", f"{ch:.1f}" if ch is not None else "N/A", icon="📊")
        if "n_clusters_found" in metrics:
            st.caption(f"Clusters found: {metrics['n_clusters_found']}")
        if "anomaly_rate" in metrics:
            st.caption(
                f"Anomalies detected: {metrics.get('n_anomalies_detected', 0)} "
                f"({metrics['anomaly_rate']:.1%} of rows)"
            )
    else:
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "MAE", f"{metrics.get('mae', 0):.3f}", icon="📉")
        metric_card(c2, "MSE", f"{metrics.get('mse', 0):.3f}", icon="📉")
        metric_card(c3, "RMSE", f"{metrics.get('rmse', 0):.3f}", icon="📉")
        r2 = metrics.get("r2", 0)
        metric_card(c4, "R²", f"{r2:.3f}", icon="📈", status=theme.status_from_thresholds(r2, 0.8, 0.5))
        if metrics.get("cv_ci_lower") is not None:
            st.caption(
                f"Cross-validation (neg RMSE): {metrics.get('cv_mean_neg_rmse', 0):.3f} "
                f"± {metrics.get('cv_std_neg_rmse', 0):.3f} "
                f"(95% CI: [{metrics['cv_ci_lower']:.3f}, {metrics['cv_ci_upper']:.3f}])"
            )

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

    perm_importance = state.get("permutation_importance") or {}
    if perm_importance:
        section_title("Permutation Importance (held-out test set)")
        df_perm = pd.DataFrame(list(perm_importance.items())[:10], columns=["Feature", "Importance"])
        st.bar_chart(df_perm.set_index("Feature"))

    agreement_score = state.get("explanation_agreement_score")
    if agreement_score is not None:
        st.metric("SHAP / Permutation Agreement (Spearman)", f"{agreement_score:.3f}")

    local_explanations = state.get("local_explanations") or []
    if local_explanations:
        section_title("Local Explanations (representative test observations)")
        for entry in local_explanations:
            label = f"Row {entry.get('row_index')} — actual={entry.get('actual')}, predicted={entry.get('predicted')}"
            if entry.get("misclassified"):
                label += " ⚠️ mispredicted"
            st.caption(label)
            plot_path = entry.get("plot_path")
            if plot_path and os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)


def render_arbitration_tab(state: dict) -> None:
    section_title("🧭 LLM Arbitration Log")
    entries = state.get("llm_arbitration_log") or []
    if not entries:
        st.info("No LLM arbitration was triggered for this run — every statistical trigger stayed within its normal range.")
        return
    for entry in entries:
        st.markdown(f"""
            <div class="decision-card">
                <span class="decision-agent">{entry.get('agent_name', 'unknown')}</span>
                <div><b>Trigger:</b> {entry.get('trigger', '')}</div>
                <div class="decision-reason">{entry.get('llm_arbitration', '')}</div>
            </div>
        """, unsafe_allow_html=True)


def render_decisions_tab(state: dict) -> None:
    section_title("Full Agent Decision Log")
    decisions = state.get("agent_decisions") or []
    if not decisions:
        st.info("No decisions recorded.")
        return
    for d in decisions:
        confidence_html = (
            theme.badge(f"{d.confidence:.0%} confidence", theme.status_from_thresholds(d.confidence, 0.8, 0.5) or "neutral")
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


def render_empty_state() -> None:
    st.info("👈 Upload a dataset and click **Run EMADS Pipeline** to get started.")
    section_title("What EMADS does, step by step")
    cards = "".join(
        f'<div class="step-preview-card">'
        f'<span class="step-num">{i}</span>'
        f'<div class="step-title">{icon} {title}</div>'
        f'<div class="step-desc">{desc}</div>'
        f'</div>'
        for i, (icon, title, desc) in enumerate(STEP_PREVIEW, start=1)
    )
    st.markdown(f'<div class="step-preview-grid">{cards}</div>', unsafe_allow_html=True)


def main() -> None:
    header()
    local_path, target_column, problem_type, run_clicked = render_sidebar()
    is_unsupervised_choice = problem_type in UNSUPERVISED_PROBLEM_TYPES

    if run_clicked and local_path and (target_column or is_unsupervised_choice):
        final_state = run_pipeline_with_progress(local_path, target_column, problem_type)
        if final_state:
            st.session_state["final_state"] = final_state
        else:
            # A failed run must not leave a previous successful run's results
            # visible below the error box — the tabs would silently render
            # stale results from an unrelated dataset with no indication
            # they don't belong to the run that just failed.
            st.session_state.pop("final_state", None)

    if "final_state" in st.session_state:
        state = st.session_state["final_state"]
        tabs = st.tabs([
            "🏠 Overview", "📊 EDA", "🧹 Preprocessing", "🏆 Model",
            "📈 Evaluation", "💡 Explainability", "🧭 Arbitrage LLM", "🧾 Decisions Log", "📄 Report",
        ])
        with tabs[0]: render_overview_tab(state)
        with tabs[1]: render_eda_tab(state)
        with tabs[2]: render_preprocessing_tab(state)
        with tabs[3]: render_model_tab(state)
        with tabs[4]: render_evaluation_tab(state)
        with tabs[5]: render_explainability_tab(state)
        with tabs[6]: render_arbitration_tab(state)
        with tabs[7]: render_decisions_tab(state)
        with tabs[8]: render_report_tab(state)
    elif not run_clicked:
        render_empty_state()


if __name__ == "__main__":
    main()
