"""
Preprocessing Agent Module
Handles missing values, duplicate removal, categorical encoding, and numeric normalization.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState


class PreprocessingAgent(BaseAgent):
    """
    Agent responsible for preparing raw data for training.
    Operates completely deterministically without an LLM.
    """

    def __init__(self) -> None:
        super().__init__(name="preprocessing_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Cleans and transforms data, saving the resulting clean dataset to disk.
        """
        dataset_path = state.get("dataset_path")
        target_col = state.get("target_column")

        if not dataset_path:
            raise ValueError(f"[{self.name.upper()}] 'dataset_path' is missing from state.")
        if not target_col:
            raise ValueError(f"[{self.name.upper()}] 'target_column' is missing from state.")

        # 1. Load data
        df = pd.read_csv(dataset_path)

        # 2. Remove duplicate rows
        df = df.drop_duplicates().reset_index(drop=True)

        # Separate features and target to avoid manipulating target distributions accidentally
        if target_col not in df.columns:
            raise ValueError(f"[{self.name.upper()}] Target column '{target_col}' not found in dataset.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 3. Simple Imputation Strategy (Fill missing values)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Fill numeric NaNs with the column median
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
            else:
                # Fill categorical NaNs with a 'Missing' string placeholder
                X[col] = X[col].fillna("Missing")

        # 4. Categorical Encoding (Label Encoding for MVP simplicity)
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Handle target label encoding if it is a categorical string column
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))

        # 5. Numerical Normalization (Standard Scaling)
        numeric_cols = X.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Recombine clean features and targets into one dataframe
        processed_df = pd.DataFrame(X)
        processed_df[target_col] = y

        # 6. Save the preprocessed artifact
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        preprocessed_path = os.path.join(processed_dir, "clean_dataset.csv")
        processed_df.to_csv(preprocessed_path, index=False)

        return {
            "preprocessed_data_path": preprocessed_path
        }