"""
Reporting Agent Module
Consolidates all system insights and generates the final output report artifact.
"""

from typing import Optional
from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService


class ReportingAgent(BaseAgent):
    """
    Agent responsible for compiling state metrics and summaries into a clean output file.
    Utilizes LLM for summary copywriting and PDF service for asset creation.
    """

    def __init__(self, llm_service: Optional[LLMService] = None, pdf_service: Optional[PDFService] = None) -> None:
        super().__init__(name="reporting_agent")
        self.llm = llm_service if llm_service else LLMService()
        self.pdf = pdf_service if pdf_service else PDFService()

    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Gathers metrics, asks the LLM for a conclusion statement, and exports the final report file.
        """
        eda_summary = state.get("eda_summary")
        metrics = state.get("metrics")
        target_col = state.get("target_column")

        if not metrics:
            metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        # 1. Generate an executive conclusion using the LLM
        system_prompt = (
            "You are a Principal AI Director. Write a brief executive summary conclusion "
            "based on model evaluation metrics. Be direct, authoritative, and clear."
        )
        
        user_prompt = f"""
        Model Performance Metrics:
        - Target Variable: {target_col}
        - Accuracy: {metrics.get('accuracy')}
        - F1 Score: {metrics.get('f1_score')}
        
        Provide a 2-sentence conclusion on whether this model is ready for deployment.
        """
        
        executive_conclusion = self.llm.generate_summary(system_prompt, user_prompt)

        # 2. Package everything for the PDF compiler
        report_payload = {
            "target_column": target_col,
            "eda_summary": eda_summary,
            "metrics": metrics,
            "conclusion": executive_conclusion
        }

        # 3. Generate the file asset
        report_file_path = self.pdf.generate_pdf_report(
            filename="EMADS_Run_Report.pdf", 
            content_data=report_payload
        )

        return {
            "report_path": report_file_path
        }