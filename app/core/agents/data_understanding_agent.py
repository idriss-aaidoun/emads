"""
Data Understanding Agent Module
Handles initial data loading, structural profiling, and schema inference.
"""

import pandas as pd
from typing import Any, Dict
from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState


class DataUnderstandingAgent(BaseAgent):
    """
    Agent responsible for analyzing the basic structure and metadata of the dataset.
    Operates deterministically using Pandas without LLM dependencies.
    """

    def __init__(self) -> None:
        super().__init__(name="data_understanding_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Loads the dataset, analyzes its structural schema, and identifies the target.

        Args:
            state (EMADSState): The current global state containing the dataset path.

        Returns:
            PartialEMADSState: A state update containing 'schema_info' and 'target_column'.
        """
        dataset_path = state.get("dataset_path")
        user_target = state.get("target_column")

        if not dataset_path:
            raise ValueError(f"[{self.name.upper()}] Error: 'dataset_path' is missing from the state.")

        # 1. Load data safely (Version 1.0 supports CSV)
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            raise RuntimeError(f"[{self.name.upper()}] Failed to read CSV file at {dataset_path}: {str(e)}")

        # 2. Extract basic structural properties
        num_rows, num_cols = df.shape
        columns_profile: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            columns_profile[col] = {
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isnull().sum()),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col]))
            }

        # 3. Handle target column logic (Fallback to the last column if not specified)
        final_target = user_target if user_target else str(df.columns[-1])

        # 4. Construct structural schema payload
        schema_info = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": columns_profile
        }

        # Return only the keys this agent is responsible for updating
        return {
            "schema_info": schema_info,
            "target_column": final_target
        }