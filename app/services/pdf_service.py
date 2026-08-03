"""
PDF Service Module
=====================

Renders the final EMADS report as a real, professional PDF using ReportLab.
Takes a structured content dict and lays it out into titled sections, tables,
and embedded figures. Contains no decision-making logic — it only formats
what the Reporting Agent gives it.
"""

import os
import re
from datetime import datetime
from xml.sax.saxutils import escape
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable,
)

REPORTS_DIR = "reports"

# Single source of truth for the report's palette — every color used below
# is named here so the whole document reads as one consistent system
# instead of ad-hoc hex codes scattered across section builders.
PRIMARY = colors.HexColor("#1F3864")       # navy — banner, section titles, table headers
PRIMARY_DARK = colors.HexColor("#152A4A")  # header/footer band
ACCENT = colors.HexColor("#2E9E94")        # teal — dividers, highlighted rows
ACCENT_LIGHT = colors.HexColor("#E4F5F3")  # pale teal — highlighted table rows, callout boxes
BORDER = colors.HexColor("#D8DEE4")
MUTED_TEXT = colors.HexColor("#5B6B79")
ROW_ALT = colors.HexColor("#F4F6F8")
CONFIDENCE_HIGH = colors.HexColor("#1E8449")
CONFIDENCE_MID = colors.HexColor("#B9770E")
CONFIDENCE_LOW = colors.HexColor("#C0392B")


class PDFService:
    """Compiles structured report data into a polished PDF document."""

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name="SectionTitle", parent=self.styles["Heading2"],
            spaceBefore=16, spaceAfter=2, textColor=PRIMARY, fontName="Helvetica-Bold",
        ))
        self.styles.add(ParagraphStyle(
            name="Body", parent=self.styles["BodyText"], spaceAfter=6, leading=14,
        ))
        self.styles.add(ParagraphStyle(
            name="Caption", parent=self.styles["BodyText"], fontSize=8, leading=10,
            textColor=MUTED_TEXT, alignment=TA_CENTER, spaceAfter=10,
        ))
        self.styles.add(ParagraphStyle(
            name="HeroTitle", parent=self.styles["Title"], textColor=colors.white,
            fontName="Helvetica-Bold", fontSize=25, leading=30, alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name="HeroSubtitle", parent=self.styles["BodyText"], textColor=colors.HexColor("#C9D9E8"),
            fontSize=11.5, alignment=TA_CENTER, spaceBefore=6,
        ))
        self.styles.add(ParagraphStyle(
            name="MetaLabel", parent=self.styles["BodyText"], fontName="Helvetica-Bold",
            fontSize=10, textColor=PRIMARY,
        ))
        self.styles.add(ParagraphStyle(
            name="MetaValue", parent=self.styles["BodyText"], fontSize=10, textColor=colors.HexColor("#222"),
        ))

    def generate_pdf_report(self, filename: str, content_data: Dict[str, Any]) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        final_path = os.path.join(REPORTS_DIR, filename)

        doc = SimpleDocTemplate(
            final_path, pagesize=A4,
            topMargin=2.2 * cm, bottomMargin=2.2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        )

        story: List[Any] = []
        story += self._build_title_page(content_data)
        story += self._section_header("Dataset Summary")
        story.append(Paragraph(self._safe_text(content_data.get("dataset_summary_text", "")), self.styles["Body"]))
        story += self._build_eda_section(content_data)
        story += self._build_preprocessing_section(content_data)
        story += self._build_model_section(content_data)
        story += self._build_evaluation_section(content_data)
        story += self._build_explainability_section(content_data)
        story += self._build_local_explanations_section(content_data)
        story += self._build_decisions_log(content_data)
        story += self._build_arbitration_section(content_data)
        story += self._build_confidence_audit_section(content_data)
        story += self._section_header("Conclusion")
        story.append(Paragraph(self._safe_text(content_data.get("conclusion", "")), self.styles["Body"]))

        try:
            doc.build(
                story,
                onFirstPage=lambda c, d: self._draw_header_footer(c, d, is_first_page=True),
                onLaterPages=lambda c, d: self._draw_header_footer(c, d, is_first_page=False),
            )
        except Exception as exc:
            raise RuntimeError(f"[PDF_SERVICE_ERROR] Compilation failed: {exc}")

        return final_path

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_title_page(self, data: Dict[str, Any]) -> List[Any]:
        banner = Table(
            [[Paragraph("EMADS", self.styles["HeroTitle"])],
             [Paragraph("Explainable Multi-Agent Data Science Report", self.styles["HeroSubtitle"])]],
            colWidths=[17 * cm],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PRIMARY),
            ("TOPPADDING", (0, 0), (-1, 0), 22),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 22),
            ("TOPPADDING", (0, 1), (-1, 1), 2),
        ]))

        meta_rows = [
            ("Target variable", str(data.get("target_column") or "N/A")),
            ("Problem type", str(data.get("problem_type") or "N/A").replace("_", " ").title()),
            ("Selected model", str(data.get("selected_model_name") or "N/A")),
            ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ]
        meta_table_data = [
            [Paragraph(label, self.styles["MetaLabel"]), Paragraph(value, self.styles["MetaValue"])]
            for label, value in meta_rows
        ]
        meta_table = Table(meta_table_data, colWidths=[5.5 * cm, 11.5 * cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, ROW_ALT]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))

        return [
            Spacer(1, 3 * cm),
            banner,
            Spacer(1, 1.2 * cm),
            meta_table,
            Spacer(1, 0.8 * cm),
            Paragraph(
                "Every automatic decision in this report — target detection, preprocessing choices, "
                "model selection, hyperparameter tuning — is explained inline, with a confidence score "
                "grounded in real statistics rather than a fixed constant.",
                self.styles["Caption"],
            ),
            PageBreak(),
        ]

    def _section_header(self, title: str) -> List[Any]:
        return [
            Paragraph(title, self.styles["SectionTitle"]),
            HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=2, spaceAfter=8),
        ]

    def _build_section(self, title: str, text: str) -> List[Any]:
        return self._section_header(title) + [Paragraph(self._safe_text(text), self.styles["Body"])]

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
        # Plain "\n" here, not "<br/>" — _safe_text escapes the whole string
        # (including any literal "<br/>") before converting newlines to real
        # line breaks, so a pre-escaped tag here would show up as literal
        # "&lt;br/&gt;" text in the rendered PDF.
        text = "\n".join(lines)
        return self._build_section("Preprocessing Decisions", text)

    def _build_model_section(self, data: Dict[str, Any]) -> List[Any]:
        elements = self._section_header("Model Selection")

        results = data.get("candidate_models_results") or []
        selected_name = data.get("selected_model_name")
        if results:
            table_data = [["Model", "Mean Score", "Std Dev"]]
            highlight_rows = set()
            for idx, r in enumerate(results, start=1):
                # A candidate that raised during scoring is recorded with
                # mean_score=-inf (see ModelSelectionAgent._score_candidates_
                # supervised) rather than being dropped, so it still shows up
                # here — render that plainly instead of a confusing "-inf".
                failed = r["mean_score"] == float("-inf")
                score_display = "Failed" if failed else f"{r['mean_score']:.4f}"
                std_display = "-" if failed else f"{r['std_score']:.4f}"
                table_data.append([r["model_name"], score_display, std_display])
                if r["model_name"] == selected_name:
                    highlight_rows.add(idx)
            elements.append(self._styled_table(table_data, highlight_rows=highlight_rows))
            elements.append(Spacer(1, 0.3 * cm))

        hyperparams = data.get("best_hyperparameters")
        if hyperparams:
            elements.append(Paragraph(f"<b>Best hyperparameters:</b> {escape(str(hyperparams))}", self.styles["Body"]))

        elements += self._build_model_selection_verdict(data)
        return elements

    def _build_model_selection_verdict(self, data: Dict[str, Any]) -> List[Any]:
        """
        Splits ModelSelectionAgent's explanation into two clearly separated
        blocks so a reader never sees the LLM's confidently-worded tie-break
        narrative without the statistical uncertainty it exists to resolve:
        a "Statistical Verdict" (deterministic — always shown, confidence =
        1 - p_value even after arbitration) and an "LLM Interpretation"
        (the conditional arbitration narrative, shown ONLY when a Wilcoxon
        tie actually triggered it — see ModelSelectionAgent._compare_top_two).
        """
        decisions = data.get("agent_decisions") or []
        selection_decision = next(
            (d for d in decisions
             if d.agent_name == "model_selection_agent" and d.decision.startswith("Selected '")),
            None,
        )
        arbitration_entries = [
            e for e in (data.get("llm_arbitration_log") or [])
            if e.get("agent_name") == "model_selection_agent"
        ]

        elements = self._section_header("Statistical Verdict")
        if selection_decision is None:
            elements.append(Paragraph("No model selection decision recorded.", self.styles["Body"]))
            return elements

        confidence_badge = ""
        if selection_decision.confidence is not None:
            color = self._confidence_color(selection_decision.confidence)
            confidence_badge = (
                f' <font color="{color.hexval()}"><b>({selection_decision.confidence:.0%} confidence)</b></font>'
            )
        elements.append(Paragraph(
            f"{escape(selection_decision.decision)}{confidence_badge}<br/>"
            f"{escape(selection_decision.reasoning)}",
            self.styles["Body"],
        ))

        if arbitration_entries:
            confidence_pct = (
                f"{selection_decision.confidence:.0%}" if selection_decision.confidence is not None else "N/A"
            )
            elements += self._section_header("LLM Interpretation")
            for entry in arbitration_entries:
                elements.append(Paragraph(
                    f"<i>Note: this narrative explanation should not be read as statistical certainty — "
                    f"it reflects the LLM's reasoning for breaking a tie the statistical test alone could "
                    f"not resolve (confidence: {confidence_pct}).</i>",
                    self.styles["Body"],
                ))
                elements.append(Paragraph(self._safe_text(entry.get("llm_arbitration", "")), self.styles["Body"]))
        else:
            elements.append(Paragraph(
                "The statistical test found a significant difference; no LLM arbitration was needed.",
                self.styles["Body"],
            ))
        return elements

    def _build_evaluation_section(self, data: Dict[str, Any]) -> List[Any]:
        metrics = data.get("metrics") or {}
        elements = self._section_header("Evaluation")

        table_data = [["Metric", "Value"]]
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
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

    def _build_local_explanations_section(self, data: Dict[str, Any]) -> List[Any]:
        # ExplainabilityAgent computes these (per-row SHAP waterfall plots on
        # the held-out test set, including a mispredicted example where
        # possible) and ReportingAgent forwards them into the payload, but
        # nothing rendered them into the PDF before — only the Streamlit UI
        # showed them, so anyone reading just the report never saw this part
        # of the pipeline's output.
        local_explanations = data.get("local_explanations") or []
        if not local_explanations:
            return []
        elements = self._section_header("Local Explanations")
        for explanation in local_explanations:
            label = "Mispredicted example" if explanation.get("misclassified") else "Example prediction"
            badge_color = CONFIDENCE_LOW if explanation.get("misclassified") else CONFIDENCE_HIGH
            elements.append(Paragraph(
                f'<font color="{badge_color.hexval()}"><b>{label}</b></font> (row {explanation.get("row_index")}): '
                f"actual={escape(str(explanation.get('actual')))}, "
                f"predicted={escape(str(explanation.get('predicted')))}",
                self.styles["Body"],
            ))
            elements += self._embed_images([explanation.get("plot_path")])
        return elements

    def _build_decisions_log(self, data: Dict[str, Any]) -> List[Any]:
        decisions = data.get("agent_decisions") or []
        if not decisions:
            return []
        elements = self._section_header("Agent Decisions Log")
        for d in decisions:
            confidence = ""
            if d.confidence is not None:
                color = self._confidence_color(d.confidence)
                confidence = f' <font color="{color.hexval()}"><b>({d.confidence:.0%} confidence)</b></font>'
            elements.append(Paragraph(
                f'<font color="{PRIMARY.hexval()}"><b>[{escape(d.agent_name)}]</b></font> '
                f"{escape(d.decision)}{confidence}<br/><i>{escape(d.reasoning)}</i>",
                self.styles["Body"],
            ))
        return elements

    def _build_arbitration_section(self, data: Dict[str, Any]) -> List[Any]:
        entries = data.get("llm_arbitration_log") or []
        if not entries:
            return []
        elements = self._section_header("LLM Arbitration Log")
        for entry in entries:
            callout = Table([[Paragraph(
                f'<font color="{PRIMARY.hexval()}"><b>[{escape(str(entry.get("agent_name", "unknown")))}]</b></font> '
                f"Trigger: {escape(str(entry.get('trigger', '')))}"
                f"<br/><i>{escape(str(entry.get('llm_arbitration', '')))}</i>",
                self.styles["Body"],
            )]], colWidths=[17 * cm])
            callout.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.75, ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            elements.append(callout)
            elements.append(Spacer(1, 0.25 * cm))
        return elements

    def _build_confidence_audit_section(self, data: Dict[str, Any]) -> List[Any]:
        """
        Renders MetaEvaluatorAgent's holistic verdict, if present in the
        payload. Purely informational — mirrors the "never re-execute
        automatically" rule the agent itself follows: this shows the verdict
        and recommendation, nothing more.
        """
        meta_evaluation = data.get("meta_evaluation")
        if not meta_evaluation:
            return []
        elements = self._section_header("System Confidence Audit")
        confidence_score = meta_evaluation.get("confidence_score")
        score_text = f"{confidence_score:.0%}" if confidence_score is not None else "N/A"
        verdict = meta_evaluation.get("verdict") or "No verdict available."
        color = self._confidence_color(confidence_score) if confidence_score is not None else MUTED_TEXT
        elements.append(Paragraph(
            f'<font color="{color.hexval()}"><b>Overall confidence: {score_text}</b></font><br/>{escape(verdict)}',
            self.styles["Body"],
        ))
        recommendation = meta_evaluation.get("recommendation")
        if recommendation:
            elements.append(Paragraph(
                f"<b>Recommendation:</b> {escape(recommendation)}", self.styles["Body"]
            ))
        return elements

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _confidence_color(self, confidence: float):
        if confidence >= 0.75:
            return CONFIDENCE_HIGH
        if confidence >= 0.5:
            return CONFIDENCE_MID
        return CONFIDENCE_LOW

    def _embed_images(self, image_paths: List[str], max_width_cm: float = 14) -> List[Any]:
        elements = []
        for path in image_paths:
            if path and os.path.exists(path):
                elements.append(Image(path, width=max_width_cm * cm, height=max_width_cm * 0.7 * cm, kind="proportional"))
                elements.append(Spacer(1, 0.3 * cm))
        return elements

    def _styled_table(self, table_data: List[List[str]], highlight_rows: Optional[set] = None) -> Table:
        table = Table(table_data, hAlign="LEFT")
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        # Marks the row matching the actually-selected model (which may be
        # the runner-up if LLM arbitration overrode the raw top score) so
        # the reader can see at a glance which candidate won, not just which
        # one scored highest.
        for row_idx in highlight_rows or ():
            style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), ACCENT_LIGHT))
            style.append(("FONTNAME", (0, row_idx), (-1, row_idx), "Helvetica-Bold"))
        table.setStyle(TableStyle(style))
        return table

    def _draw_header_footer(self, canvas_obj, doc, is_first_page: bool) -> None:
        width, height = A4
        canvas_obj.saveState()

        if not is_first_page:
            canvas_obj.setFillColor(PRIMARY_DARK)
            canvas_obj.rect(0, height - 1 * cm, width, 1 * cm, stroke=0, fill=1)
            canvas_obj.setFont("Helvetica-Bold", 9)
            canvas_obj.setFillColor(colors.white)
            canvas_obj.drawString(2 * cm, height - 0.65 * cm, "EMADS")
            canvas_obj.setFont("Helvetica", 8)
            canvas_obj.drawRightString(
                width - 2 * cm, height - 0.65 * cm, "Explainable Multi-Agent Data Science Report"
            )

        canvas_obj.setStrokeColor(BORDER)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(MUTED_TEXT)
        canvas_obj.drawString(2 * cm, 1.1 * cm, "Generated by EMADS")
        canvas_obj.drawRightString(width - 2 * cm, 1.1 * cm, f"Page {doc.page}")

        canvas_obj.restoreState()

    @staticmethod
    def _safe_text(text: Any) -> str:
        """
        Escapes external text before ReportLab parses its HTML-like markup,
        then re-introduces the small subset of markdown the LLM services
        actually produce (**bold**, "* " bullets) as real ReportLab tags —
        otherwise every LLM-written summary in the report shows literal
        asterisks instead of formatting.
        """
        safe = escape(str(text or "No content available."))
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        safe = re.sub(r"(^|\n)\*\s+", r"\1• ", safe)
        return safe.replace("\n", "<br/>")
