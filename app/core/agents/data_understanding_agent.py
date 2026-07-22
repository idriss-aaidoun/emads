"""
Data Understanding Agent Module
=================================

First agent of the EMADS pipeline. Loads the raw dataset and builds a
complete structural profile of it: column types, missing values, target
column, problem type (classification/regression), and data quality issues.

This agent is fully deterministic (no LLM calls) — its job is to describe
facts about the data, not to interpret them narratively. The EDA Agent
picks up from here to generate visual + LLM explanations.
"""

import pandas as pd
from typing import Any, Dict, List

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState , AgentDecision

# Column names commonly used for the target variable — helps the heuristic
# below make a better guess than "always pick the last column".
COMMON_TARGET_NAMES = {"target", "label", "class", "y", "outcome", "output"}

# A categorical column with more unique values than this is flagged as a
# data quality issue (likely an ID column, free text, or needs special encoding).
HIGH_CARDINALITY_THRESHOLD = 50

# A column with more missing values than this ratio is flagged.
HIGH_MISSING_RATIO_THRESHOLD = 0.4

# A numeric target with more unique values than this is treated as regression
# rather than classification.
MAX_UNIQUE_FOR_CLASSIFICATION = 20


class DataUnderstandingAgent(BaseAgent):
    """
    Agent responsible for loading the dataset and producing the first
    structural understanding of it: schema, column types, target column,
    problem type, and data quality warnings.
    """

    def __init__(self) -> None:
        super().__init__(name="data_understanding_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        dataset_path = state.get("dataset_path")
        user_target = state.get("target_column")

        if not dataset_path:
            self.logger.error("'dataset_path' missing from state — aborting.")
            raise ValueError("'dataset_path' is missing from the state.")

        self.logger.debug("Loading dataset from: %s", dataset_path)
        df = self._load_dataset(dataset_path)
        self.logger.debug("Dataset loaded — shape: %s", df.shape)

        columns_profile = self._profile_columns(df)
        numerical_columns = [c for c, p in columns_profile.items() if p["is_numeric"]]
        categorical_columns = [c for c, p in columns_profile.items() if not p["is_numeric"]]

        target_column, target_decision = self._resolve_target_column(
            df, user_target, categorical_columns, numerical_columns
        )
        user_problem_type = state.get("problem_type")
        problem_type, problem_decision = self._infer_problem_type(df, target_column, user_problem_type)
        self._validate_target_for_modeling(df, target_column, problem_type)
        quality_issues = self._detect_quality_issues(df, columns_profile)

        schema_info: Dict[str, Any] = {
            "num_rows": int(df.shape[0]),
            "num_cols": int(df.shape[1]),
            "columns": columns_profile,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "quality_issues": quality_issues,
        }

        self.logger.info(
            "execute() complete — rows=%s cols=%s target='%s' problem_type='%s' quality_issues=%s",
            df.shape[0], df.shape[1], target_column, problem_type, len(quality_issues),
        )
        return {
            "schema_info": schema_info,
            "target_column": target_column,
            "problem_type": problem_type,
            "agent_decisions": [target_decision, problem_decision],
            "logs": [self.log(
                f"Analyzed {df.shape[0]} rows / {df.shape[1]} columns. "
                f"Target='{target_column}', problem_type='{problem_type}'."
            )],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Loads the CSV, trying a couple of common separators/encodings
        so the agent works on datasets other than the one it was first
        tested on."""
        try:
            return pd.read_csv(dataset_path)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(dataset_path, encoding="latin1")
            except Exception as exc:
                raise RuntimeError(f"Failed to read CSV at {dataset_path}: {exc}") from exc
        except pd.errors.ParserError:
            try:
                return pd.read_csv(dataset_path, sep=";")
            except Exception as exc:
                raise RuntimeError(f"Failed to read CSV at {dataset_path}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV at {dataset_path}: {exc}") from exc

    def _profile_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        profile: Dict[str, Dict[str, Any]] = {}
        n_rows = len(df)

        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            n_unique = int(df[col].nunique(dropna=True))

            profile[col] = {
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_ratio": round(missing_count / n_rows, 4) if n_rows else 0.0,
                "n_unique": n_unique,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
            }
        return profile

    def _resolve_target_column(
        self,
        df: pd.DataFrame,
        user_target: str | None,
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> tuple[str, AgentDecision]:
        """
        Decides which column is the target. Priority:
          1. User-provided target (if it exists in the dataset)
          2. A column whose name matches a common target name
          3. Fallback: the last column of the dataset
        Every path is explained via an AgentDecision.
        """
        if user_target and user_target in df.columns:
            return user_target, self.decide(
                decision=f"Use user-specified target column '{user_target}'",
                reasoning="The target column was explicitly provided in the state.",
                confidence=1.0,
            )

        for col in df.columns:
            if col.strip().lower() in COMMON_TARGET_NAMES:
                return col, self.decide(
                    decision=f"Selected '{col}' as target column",
                    reasoning=(
                        f"Column name '{col}' matches a common target naming "
                        f"convention ({sorted(COMMON_TARGET_NAMES)})."
                    ),
                    confidence=0.8,
                )

        fallback = str(df.columns[-1])
        return fallback, self.decide(
            decision=f"Selected '{fallback}' as target column (fallback)",
            reasoning=(
                "No target was specified and no column matched a common target "
                "name, so the last column of the dataset was assumed to be the "
                "target. This should be confirmed by the user."
            ),
            confidence=0.4,
        )

    def _infer_problem_type(self, df: pd.DataFrame, target_column: str, user_problem_type: str | None = None) -> tuple[str, AgentDecision]:
        if user_problem_type in ("classification", "regression"):
            return user_problem_type, self.decide(
                decision=f"Problem type: {user_problem_type}",
                reasoning=f"Problem type was explicitly set to '{user_problem_type}'.",
                confidence=1.0,
            )

        target_series = df[target_column]
        n_unique = target_series.nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(target_series)

        if not is_numeric:
            return "classification", self.decide(
                decision="Problem type: classification",
                reasoning=f"Target column '{target_column}' is non-numeric ({target_series.dtype}).",
                confidence=0.95,
            )

        is_float = pd.api.types.is_float_dtype(target_series)
        val_range = float(target_series.max() - target_series.min()) if (n_unique > 1 and is_numeric) else 0

        # If floating point, or range > 50, or all unique numbers, treat as regression
        if is_float or val_range > 50 or (n_unique > 10 and n_unique == len(target_series)):
            return "regression", self.decide(
                decision="Problem type: regression",
                reasoning=(
                    f"Target column '{target_column}' is numeric with range {val_range} and {n_unique} unique values, "
                    f"suggesting a continuous numeric target."
                ),
                confidence=0.85,
            )

        if n_unique <= MAX_UNIQUE_FOR_CLASSIFICATION:
            return "classification", self.decide(
                decision="Problem type: classification",
                reasoning=(
                    f"Target column '{target_column}' is numeric but has only "
                    f"{n_unique} unique values (<= {MAX_UNIQUE_FOR_CLASSIFICATION}), "
                    f"suggesting discrete classes."
                ),
                confidence=0.75,
            )

        return "regression", self.decide(
            decision="Problem type: regression",
            reasoning=f"Target column '{target_column}' is numeric with {n_unique} unique values.",
            confidence=0.85,
        )

    def _validate_target_for_modeling(
        self, df: pd.DataFrame, target_column: str, problem_type: str
    ) -> None:
        """Rejects class distributions that cannot support holdout and CV.

        Three examples per class leave one for the untouched test set and at
        least two for stratified cross-validation on the training data.
        """
        if problem_type != "classification":
            return
        counts = df[target_column].dropna().value_counts()
        if len(counts) < 2:
            raise ValueError("Classification requires at least two target classes.")
        if int(counts.min()) < 3:
            raise ValueError(
                "Classification requires at least 3 rows in every target class "
                "for a holdout set and stratified cross-validation."
            )

    def _detect_quality_issues(
        self, df: pd.DataFrame, columns_profile: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Collects human-readable data quality warnings for the report and UI."""
        issues: List[Dict[str, str]] = []

        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            issues.append({
                "type": "duplicate_rows",
                "message": f"{n_duplicates} duplicate row(s) found.",
            })

        for col, profile in columns_profile.items():
            if profile["missing_ratio"] > HIGH_MISSING_RATIO_THRESHOLD:
                issues.append({
                    "type": "high_missing_ratio",
                    "message": f"Column '{col}' has {profile['missing_ratio']*100:.1f}% missing values.",
                })
            if profile["n_unique"] <= 1:
                issues.append({
                    "type": "constant_column",
                    "message": f"Column '{col}' has a single unique value (no predictive power).",
                })
            if not profile["is_numeric"] and profile["n_unique"] > HIGH_CARDINALITY_THRESHOLD:
                issues.append({
                    "type": "high_cardinality",
                    "message": (
                        f"Column '{col}' has {profile['n_unique']} unique categories "
                        f"(possible ID column or needs special encoding)."
                    ),
                })

        return issues
