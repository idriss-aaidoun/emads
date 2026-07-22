"""
Exploratory Data Analysis (EDA) Agent Module
==============================================

Consumes the schema built by the Data Understanding Agent, computes
descriptive statistics, generates visualizations (correlation, missing
values, distributions, class balance, outliers), and asks the LLM to turn
all of it into a natural language explanation.
"""

import pandas as pd
from typing import List, Optional

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState
from app.services.llm_service import LLMService
from app.utils import plot_utils

# Limit the number of distribution plots generated for wide datasets,
# to avoid producing dozens of figures / a huge report.
MAX_DISTRIBUTION_PLOTS = 6


class EDAAgent(BaseAgent):
    """
    Agent responsible for the exploratory analysis: statistics, plots,
    outlier detection, and an LLM-generated narrative summary.
    """

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        super().__init__(name="eda_agent")
        self.llm = llm_service if llm_service else LLMService()

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        problem_type = state.get("problem_type") or "classification"
        schema_info = state.get("schema_info") or {}
        numerical_columns: List[str] = schema_info.get("numerical_columns", [])

        if not dataset_path:
            self.logger.error("'dataset_path' missing from state — aborting.")
            raise ValueError("'dataset_path' is missing from the state.")

        self.logger.debug("Reading dataset from: %s", dataset_path)
        df = pd.read_csv(dataset_path)

        generated_plots: List[str] = []

        corr_plot = plot_utils.plot_correlation_heatmap(df, numerical_columns)
        if corr_plot:
            generated_plots.append(corr_plot)

        missing_plot = plot_utils.plot_missing_values(df)
        if missing_plot:
            generated_plots.append(missing_plot)

        for col in numerical_columns[:MAX_DISTRIBUTION_PLOTS]:
            generated_plots.append(plot_utils.plot_distribution(df, col))

        if target_column and target_column in df.columns:
            generated_plots.append(
                plot_utils.plot_target_balance(df, target_column, problem_type)
            )

        outliers = plot_utils.detect_outliers_iqr(df, numerical_columns)

        eda_stats = {
            "descriptive_stats": df.describe(include="all").to_dict(),
            "outliers_per_column": outliers,
            "total_missing_values": int(df.isnull().sum().sum()),
        }

        outlier_decision = self.decide(
            decision=f"Flagged {len(outliers)} column(s) with outliers"
            if outliers else "No significant outliers detected",
            reasoning=(
                f"Used the 1.5*IQR rule on numerical columns. "
                f"Columns with outliers: {list(outliers.keys())}" if outliers
                else "All numerical columns fall within 1.5*IQR of their quartiles."
            ),
            confidence=0.7,
        )

        self.logger.info(
            "Calling LLM for EDA summary (model=%s)",
            getattr(self.llm, 'model_name', 'unknown'),
        )
        fallback_summary = self._build_fallback_eda_summary(schema_info, target_column, outliers)
        eda_summary = self.llm.generate_eda_summary(
            {**schema_info, **eda_stats, "target_column": target_column},
            dataset_path,
            fallback_message=fallback_summary,
        )
        if not (eda_summary or "").strip():
            self.logger.warning("LLM returned empty EDA summary; using deterministic fallback text.")
            eda_summary = fallback_summary
        self.logger.info(
            "LLM EDA summary received (chars=%s, starts_with=%r)",
            len(eda_summary) if eda_summary else 0,
            (eda_summary or "")[:80],
        )

        self.logger.info(
            "execute() complete — plots=%s outlier_cols=%s",
            len(generated_plots), len(outliers),
        )
        return {
            "eda_stats": eda_stats,
            "eda_summary": eda_summary,
            "generated_plots": generated_plots,
            "agent_decisions": [outlier_decision],
            "logs": [self.log(
                f"Generated {len(generated_plots)} plot(s), "
                f"detected outliers in {len(outliers)} column(s)."
            )],
        }

    def _build_fallback_eda_summary(self, schema_info: dict, target_column: str | None, outliers: dict) -> str:
        num_rows = schema_info.get("num_rows", "N/A")
        num_cols = schema_info.get("num_cols", "N/A")
        target_str = f"`{target_column}`" if target_column else "N/A"
        num_outliers = len(outliers)
        
        return (
            f"* **Dataset Dimensions**: Analyzed {num_rows} rows and {num_cols} columns.\n"
            f"* **Target Variable**: Designated target variable is {target_str}.\n"
            f"* **Data Quality & Outliers**: Flagged {num_outliers} numerical column(s) containing outliers via the 1.5*IQR rule.\n"
            f"* **Feature Profiling**: Preprocessing recommended to handle categorical encodings and scale numerical distributions."
        )