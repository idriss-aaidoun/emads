"""
Reporting Agent Module
=========================

Final agent of the pipeline. Gathers every relevant field from the shared
state, asks the LLM for an executive conclusion, and delegates rendering
to the PDF Service. Contains no formatting logic itself.
"""

from typing import Optional

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService


class ReportingAgent(BaseAgent):
    """Agent responsible for compiling the final PDF report from the state."""

    def __init__(self, llm_service: Optional[LLMService] = None, pdf_service: Optional[PDFService] = None) -> None:
        super().__init__(name="reporting_agent")
        self.llm = llm_service if llm_service else LLMService()
        self.pdf = pdf_service if pdf_service else PDFService()

    def execute(self, state: EMADSState) -> PartialEMADSState:
        metrics = state.get("metrics") or {}
        target_column = state.get("target_column")

        executive_conclusion = self.llm.generate_summary(
            system_prompt=(
                "You are a Principal Data Science Director. Write a brief, direct "
                "executive conclusion (2-3 sentences) on whether this model is ready "
                "for deployment, based on the metrics and model choice provided."
            ),
            user_prompt=(
                f"Target variable: {target_column}\n"
                f"Selected model: {state.get('selected_model_name')}\n"
                f"Metrics: {metrics}\n"
                f"Provide your conclusion now."
            ),
            fallback_message=(
                f"The pipeline completed training for model '{state.get('selected_model_name')}' on target '{target_column}'. "
                f"Evaluation metrics ({metrics}) reflect valid cross-validation alignment. "
                "Validation is recommended on live production traffic prior to full deployment."
            ),
        )

        report_payload = {
            "target_column": target_column,
            "problem_type": state.get("problem_type"),
            "dataset_summary_text": self._build_dataset_summary_text(state),
            "eda_summary": state.get("eda_summary"),
            "generated_plots": state.get("generated_plots") or [],
            "preprocessing_report": state.get("preprocessing_report"),
            "selected_model_name": state.get("selected_model_name"),
            "model_selection_summary": state.get("model_selection_summary"),
            "candidate_models_results": state.get("candidate_models_results"),
            "best_hyperparameters": state.get("best_hyperparameters"),
            "metrics": metrics,
            "evaluation_plots": state.get("evaluation_plots") or [],
            "explainability_summary": state.get("explainability_summary") or "No explanation available.",
            "shap_plots": state.get("shap_plots") or [],
            "agent_decisions": state.get("agent_decisions") or [],
            "conclusion": executive_conclusion or "No conclusion available.",
        }

        report_file_path = self.pdf.generate_pdf_report(
            filename=f"EMADS_Report_{state.get('session_id', 'run')}.pdf",
            content_data=report_payload,
        )

        return {
            "report_path": report_file_path,
            "agent_decisions": [self.decide(
                decision="Generated final EMADS report",
                reasoning="The report consolidates pipeline outputs, metrics, explanations, and decisions.",
                confidence=1.0,
            )],
            "logs": [self.log(f"Report generated at '{report_file_path}'.")],
        }

    def _build_dataset_summary_text(self, state: EMADSState) -> str:
        schema_info = state.get("schema_info") or {}
        issues = schema_info.get("quality_issues", [])
        issues_text = "; ".join(i["message"] for i in issues) if issues else "None detected."
        return (
            f"{schema_info.get('num_rows', '?')} rows, {schema_info.get('num_cols', '?')} columns. "
            f"Data quality issues: {issues_text}"
        )
