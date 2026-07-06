"""
EMADS State Module
Defines the schema for the shared short-term memory passed between agents.
"""

from typing import TypedDict, Any, Optional, Dict, List


class EMADSState(TypedDict):
    """
    Represents the shared, centralized state of the EMADS pipeline.
    Each key represents an artifact produced or consumed by the agents.
    """

    # --- Data Input & Specification ---
    dataset_path: str
    """The local file system path to the uploaded CSV/Parquet dataset."""
    
    target_column: Optional[str]
    """The name of the target variable chosen for modeling."""

    # --- Data Understanding Agent Output ---
    schema_info: Optional[Dict[str, Any]]
    """Inferred structural schema containing column names, data types, and null counts."""

    # --- EDA Agent Output ---
    eda_summary: Optional[str]
    """LLM-generated textual summary explaining structural and statistical insights."""
    
    generated_plots: Optional[List[str]]
    """List of local file paths pointing to the generated distribution and correlation plots."""

    # --- Preprocessing Agent Output ---
    preprocessed_data_path: Optional[str]
    """Path to the cleaned, encoded, and normalized dataset stored temporarily."""

    # --- Model Agent Output ---
    model_path: Optional[str]
    """Path to the serialized, trained Random Forest model artifact (.pkl or .joblib)."""

    # --- Evaluation Agent Output ---
    metrics: Optional[Dict[str, Any]]
    """Dictionary containing metrics like Accuracy, Precision, Recall, F1, and Confusion Matrix data."""

    # --- Reporting Agent Output ---
    report_path: Optional[str]
    """The local file path to the final generated PDF report."""