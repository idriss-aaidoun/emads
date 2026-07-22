"""
PDF Service Module
=====================

Renders the final EMADS report as a real, professional PDF using ReportLab.
Takes a structured content dict and lays it out into titled sections, tables,
and embedded figures. Contains no decision-making logic — it only formats
what the Reporting Agent gives it.
"""

import os
from xml.sax.saxutils import escape
from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

REPORTS_DIR = "reports"


class PDFService:
    """Compiles structured report data into a polished PDF document."""

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name="SectionTitle", parent=self.styles["Heading2"],
            spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1F3864"),
        ))
        self.styles.add(ParagraphStyle(
            name="Body", parent=self.styles["BodyText"], spaceAfter=6, leading=14,
        ))

    def generate_pdf_report(self, filename: str, content_data: Dict[str, Any]) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        final_path = os.path.join(REPORTS_DIR, filename)

        doc = SimpleDocTemplate(
            final_path, pagesize=A4,
            topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        )

        story: List[Any] = []
        story += self._build_title_page(content_data)
        story += self._build_section("Dataset Summary", content_data.get("dataset_summary_text", ""))
        story += self._build_eda_section(content_data)
        story += self._build_preprocessing_section(content_data)
        story += self._build_model_section(content_data)
        story += self._build_evaluation_section(content_data)
        story += self._build_explainability_section(content_data)
        story += self._build_decisions_log(content_data)
        story += self._build_section("Conclusion", content_data.get("conclusion", ""))

        try:
            doc.build(story)
        except Exception as exc:
            raise RuntimeError(f"[PDF_SERVICE_ERROR] Compilation failed: {exc}")

        return final_path

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_title_page(self, data: Dict[str, Any]) -> List[Any]:
        return [
            Spacer(1, 4 * cm),
            Paragraph("EMADS — Explainable Data Science Report", self.styles["Title"]),
            Spacer(1, 0.5 * cm),
            Paragraph(f"Target variable: {data.get('target_column', 'N/A')}", self.styles["Body"]),
            Paragraph(f"Problem type: {data.get('problem_type', 'N/A')}", self.styles["Body"]),
            Paragraph(f"Selected model: {data.get('selected_model_name', 'N/A')}", self.styles["Body"]),
            PageBreak(),
        ]

    def _build_section(self, title: str, text: str) -> List[Any]:
        return [Paragraph(title, self.styles["SectionTitle"]), Paragraph(self._safe_text(text), self.styles["Body"])]

    def _build_eda_section(self, data: Dict[str, Any]) -> List[Any]:
        elements = self._build_section("Exploratory Data Analysis", data.get("eda_summary", ""))
        elements += self._embed_images(data.get("generated_plots", []))
        return elements

    def _build_preprocessing_section(self, data: Dict[str, Any]) -> List[Any]:
        report = data.get("preprocessing_report") or {}
        if not report:
            return []
        lines = []
        if report.get("dropped_columns"):
            lines.append(f"Dropped columns: {', '.join(report['dropped_columns'])}")
        if report.get("imputation"):
            lines.append(f"Missing value imputation: {report['imputation']}")
        if report.get("encoding"):
            lines.append(f"Categorical encoding: {report['encoding']}")
        if report.get("scaling"):
            lines.append(f"Numerical scaling: {report['scaling']}")
        text = "<br/>".join(lines)
        return self._build_section("Preprocessing Decisions", text)

    def _build_model_section(self, data: Dict[str, Any]) -> List[Any]:
        elements = [Paragraph("Model Selection", self.styles["SectionTitle"])]

        results = data.get("candidate_models_results") or []
        if results:
            table_data = [["Model", "Mean Score", "Std Dev"]]
            for r in results:
                table_data.append([r["model_name"], f"{r['mean_score']:.4f}", f"{r['std_score']:.4f}"])
            elements.append(self._styled_table(table_data))
            elements.append(Spacer(1, 0.3 * cm))

        hyperparams = data.get("best_hyperparameters")
        if hyperparams:
            elements.append(Paragraph(f"<b>Best hyperparameters:</b> {hyperparams}", self.styles["Body"]))

        model_selection_summary = data.get("model_selection_summary")
        if model_selection_summary:
            elements.append(Paragraph("Model Selection Rationale", self.styles["SectionTitle"]))
            elements.append(Paragraph(self._safe_text(model_selection_summary), self.styles["Body"]))

        return elements

    def _build_evaluation_section(self, data: Dict[str, Any]) -> List[Any]:
        metrics = data.get("metrics") or {}
        elements = [Paragraph("Evaluation", self.styles["SectionTitle"])]

        table_data = [["Metric", "Value"]]
        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table_data.append([key.replace("_", " ").title(), display_value])
        elements.append(self._styled_table(table_data))
        elements.append(Spacer(1, 0.3 * cm))

        elements += self._embed_images(data.get("evaluation_plots", []))
        return elements

    def _build_explainability_section(self, data: Dict[str, Any]) -> List[Any]:
        explainability_text = data.get("explainability_summary") or "No explanation available."
        elements = self._build_section("Explainability", explainability_text)
        elements += self._embed_images(data.get("shap_plots", []))
        return elements

    def _build_decisions_log(self, data: Dict[str, Any]) -> List[Any]:
        decisions = data.get("agent_decisions") or []
        if not decisions:
            return []
        elements = [Paragraph("Agent Decisions Log", self.styles["SectionTitle"])]
        for d in decisions:
            confidence = f" (confidence: {d.confidence:.0%})" if d.confidence is not None else ""
            elements.append(Paragraph(
                f"<b>[{escape(d.agent_name)}]</b> {escape(d.decision)}{confidence}<br/><i>{escape(d.reasoning)}</i>",
                self.styles["Body"],
            ))
        return elements

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_images(self, image_paths: List[str], max_width_cm: float = 14) -> List[Any]:
        elements = []
        for path in image_paths:
            if path and os.path.exists(path):
                elements.append(Image(path, width=max_width_cm * cm, height=max_width_cm * 0.7 * cm, kind="proportional"))
                elements.append(Spacer(1, 0.3 * cm))
        return elements

    def _styled_table(self, table_data: List[List[str]]) -> Table:
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    @staticmethod
    def _safe_text(text: Any) -> str:
        """Escapes external text before ReportLab parses its HTML-like markup."""
        return escape(str(text or "No content available.")).replace("\n", "<br/>")
