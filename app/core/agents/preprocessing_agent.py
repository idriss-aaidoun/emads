"""
Preprocessing Agent Module
============================

Cleans and transforms the raw dataset into a model-ready dataset. Every
decision (which imputation strategy, which encoding, which columns to
drop) is chosen automatically based on the column profile built by the
Data Understanding Agent, and explained via an AgentDecision — this is
what makes the agent generalize across different CSV datasets instead of
being hardcoded for one.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState

OUTPUT_DIR = os.path.join("data", "outputs")

# A categorical column with more unique values than this uses label encoding
# instead of one-hot, to avoid exploding the feature space.
ONE_HOT_MAX_CARDINALITY = 10

# Columns with a higher missing ratio than this are dropped entirely rather
# than imputed, since imputing >60% of a column mostly fabricates data.
DROP_COLUMN_MISSING_THRESHOLD = 0.6


class PreprocessingAgent(BaseAgent):
    """
    Agent responsible for turning the raw dataset into a clean, numeric,
    model-ready dataset — fully deterministic, no LLM involved.
    """

    def __init__(self) -> None:
        super().__init__(name="preprocessing_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        schema_info = state.get("schema_info") or {}

        if not dataset_path:
            self.logger.error("'dataset_path' missing from state — aborting.")
            raise ValueError("'dataset_path' is missing from the state.")
        if not target_column:
            self.logger.error("'target_column' missing from state — aborting.")
            raise ValueError("'target_column' is missing from the state.")

        self.logger.debug("Loading dataset from: %s", dataset_path)
        df = pd.read_csv(dataset_path)
        if target_column not in df.columns:
            self.logger.error("Target column '%s' not found in dataset columns: %s", target_column, list(df.columns))
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        decisions = []
        report: dict = {"dropped_columns": [], "imputation": {}, "encoding": {}, "scaling": None}

        # 1. Remove duplicate rows
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed > 0:
            decisions.append(self.decide(
                decision=f"Removed {n_removed} duplicate row(s)",
                reasoning="Exact duplicate rows add no information and can bias training.",
                confidence=1.0,
            ))

        # 2. Drop rows with missing target (can't train on unlabeled rows)
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows remain after dropping missing target values.")

        # 3. Drop useless columns (constant / near-empty / flagged by Data Understanding)
        columns_to_drop = self._select_columns_to_drop(df, target_column, schema_info)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            report["dropped_columns"] = columns_to_drop
            decisions.append(self.decide(
                decision=f"Dropped column(s): {columns_to_drop}",
                reasoning=(
                    "These columns were constant, had >{:.0%} missing values, or were "
                    "flagged as likely ID columns by the Data Understanding Agent."
                ).format(DROP_COLUMN_MISSING_THRESHOLD),
                confidence=0.85,
            ))

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 4. Missing value imputation (per-column strategy, explained)
        X, imputation_report = self._impute_missing_values(X)
        report["imputation"] = imputation_report
        if imputation_report:
            decisions.append(self.decide(
                decision=f"Imputed missing values in {len(imputation_report)} column(s)",
                reasoning="Numeric columns use median imputation (robust to outliers); "
                          "categorical columns use a 'Missing' placeholder category.",
                confidence=0.8,
            ))

        # 5. Categorical encoding (one-hot for low cardinality, label encoding otherwise)
        X, encoding_report = self._encode_categoricals(X)
        report["encoding"] = encoding_report
        if encoding_report:
            decisions.append(self.decide(
                decision=f"Encoded {len(encoding_report)} categorical column(s)",
                reasoning=(
                    f"One-hot encoding used for columns with <= {ONE_HOT_MAX_CARDINALITY} "
                    f"categories; label encoding used for higher-cardinality columns to "
                    f"avoid excessive dimensionality."
                ),
                confidence=0.75,
            ))

        # 6. Encode target if it's categorical (classification)
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y.astype(str))

        # 7. Scale numerical features
        numeric_cols = X.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            report["scaling"] = "StandardScaler"
            decisions.append(self.decide(
                decision=f"Applied StandardScaler to {len(numeric_cols)} numeric column(s)",
                reasoning="Standardization (zero mean, unit variance) benefits distance-based "
                          "and gradient-based models without harming tree-based models.",
                confidence=0.8,
            ))

        processed_df = X.copy()
        processed_df[target_column] = y

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        preprocessed_path = os.path.join(OUTPUT_DIR, "clean_dataset.csv")
        processed_df.to_csv(preprocessed_path, index=False)

        self.logger.info(
            "execute() complete — final_shape=%s dropped=%s imputed=%s encoded=%s scaling=%s",
            processed_df.shape, len(columns_to_drop),
            len(report.get('imputation', {})), len(report.get('encoding', {})),
            report.get('scaling'),
        )
        self.logger.debug("Cleaned dataset saved to: %s", preprocessed_path)
        return {
            "preprocessed_data_path": preprocessed_path,
            "preprocessing_report": report,
            "selected_features": [c for c in processed_df.columns if c != target_column],
            "agent_decisions": decisions,
            "logs": [self.log(
                f"Preprocessing complete. Final shape: {processed_df.shape}. "
                f"{len(columns_to_drop)} column(s) dropped."
            )],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _select_columns_to_drop(self, df: pd.DataFrame, target_column: str, schema_info: dict) -> list[str]:
        flagged = set()
        for issue in schema_info.get("quality_issues", []):
            if issue.get("type") in ("constant_column", "high_cardinality"):
                msg = issue.get("message", "")
                if "'" in msg:
                    parts = msg.split("'")
                    if len(parts) >= 2:
                        flagged.add(parts[1])
        to_drop = []
        for col in df.columns:
            if col == target_column:
                continue
            missing_ratio = df[col].isnull().mean()
            if col in flagged or missing_ratio > DROP_COLUMN_MISSING_THRESHOLD:
                to_drop.append(col)
        return to_drop

    def _impute_missing_values(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        report = {}
        for col in X.columns:
            if X[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
                report[col] = "median"
            else:
                X[col] = X[col].fillna("Missing")
                report[col] = "constant_placeholder"
        return X, report

    def _encode_categoricals(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        report = {}
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique <= ONE_HOT_MAX_CARDINALITY:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                report[col] = "one_hot"
            else:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                report[col] = "label_encoding"
        return X, report