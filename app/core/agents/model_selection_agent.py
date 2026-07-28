"""
Model Selection Agent Module
==============================

Replaces the V1 "always Random Forest" approach. Compares several candidate
algorithms using cross-validation on the training split, selects the best
one, and explains WHY it was chosen over the alternatives — the core
"explainable AutoML" contribution of the project.

XGBoost and LightGBM are used only if installed; the agent degrades
gracefully to scikit-learn-only candidates otherwise, so the pipeline never
breaks because of an optional dependency.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState, UNSUPERVISED_PROBLEM_TYPES
from app.services.llm_service import LLMService

CV_FOLDS = 5
RANDOM_STATE = 42

# Default configs used ONLY for the family-vs-family comparison below — the
# winning algorithm's real hyperparameters are tuned afterwards by
# HyperparameterAgent. eps=0.5 is workable because PreprocessingAgent already
# StandardScales numeric columns, so 0.5 is in standardized-unit space.
DEFAULT_N_CLUSTERS = 3
DEFAULT_DBSCAN_EPS = 0.5
DEFAULT_DBSCAN_MIN_SAMPLES = 5

# Human-readable labels for each scoring metric, used in LLM prompts/fallback
# text — centralized here instead of repeating the same ternary in 3 methods.
METRIC_LABELS = {
    "accuracy": "accuracy",
    "neg_root_mean_squared_error": "RMSE (negated)",
    "silhouette": "silhouette score",
}

# Optional boosted-tree libraries — only used if actually installed.
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


class ModelSelectionAgent(BaseAgent):
    """
    Agent responsible for comparing candidate ML algorithms via
    cross-validation and selecting the best-performing one.
    """

    def __init__(self) -> None:
        super().__init__(name="model_selection_agent")
        self.llm = LLMService()

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        data_path = state.get("preprocessed_data_path")
        target_column = state.get("target_column")
        problem_type = state.get("problem_type")
        is_unsupervised = problem_type in UNSUPERVISED_PROBLEM_TYPES

        if not data_path or not problem_type:
            self.logger.error(
                "Missing required state keys — data_path=%s problem_type=%s",
                bool(data_path), bool(problem_type),
            )
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path' or 'problem_type'."
            )
        # DataUnderstandingAgent always resolves problem_type to one of the four
        # concrete values before this agent runs (linear pipeline) — only the
        # supervised ones require a target_column.
        if not is_unsupervised and not target_column:
            self.logger.error("'target_column' missing from state for supervised problem_type — aborting.")
            raise ValueError("Missing required state input: 'target_column'.")

        self.logger.debug("Loading preprocessed data from: %s", data_path)
        df = pd.read_csv(data_path)
        candidates = self._build_candidates(problem_type)

        if is_unsupervised:
            scoring = "silhouette"
            results = self._score_candidates_unsupervised(candidates, df)
        else:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Recreate the exact same split used by the Evaluation Agent, so the
            # test set is never touched here (no leakage) and metrics stay comparable.
            stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
            )

            cv = self._build_cv_splitter(problem_type, y_train)
            scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
            results = self._score_candidates_supervised(candidates, X_train, y_train, cv, scoring)

        results.sort(key=lambda r: r["mean_score"], reverse=True)
        best_result = results[0]
        best_model_name = best_result["model_name"]
        best_model = candidates[best_model_name]

        selection_decision = self.decide(
            decision=f"Selected '{best_model_name}' as the final model",
            reasoning=self._build_reasoning(best_result, results, scoring),
            confidence=self._confidence_from_margin(results),
        )

        self.logger.info(
            "Calling LLM for model selection summary (model=%s)",
            getattr(self.llm, "model_name", "unknown"),
        )
        fallback_summary = self._build_fallback_summary(best_model_name, best_result, results, scoring)
        model_selection_summary = self.llm.generate_summary(
            system_prompt=(
                "You are a senior machine learning engineer. Explain in 4-6 concise bullet points "
                "why a particular model was selected over the alternatives, using the comparison "
                "results and the confidence in the choice. Be clear and non-technical."
            ),
            user_prompt=self._build_llm_prompt(best_model_name, best_result, results, scoring),
            fallback_message=fallback_summary,
        )
        if not (model_selection_summary or "").strip():
            self.logger.warning("LLM returned empty model selection summary; using deterministic fallback text.")
            model_selection_summary = fallback_summary
        self.logger.info(
            "LLM model selection summary received (chars=%s, starts_with=%r)",
            len(model_selection_summary) if model_selection_summary else 0,
            (model_selection_summary or "")[:80],
        )

        self.logger.info(
            "Model selection complete: selected_model=%s score=%.4f",
            best_model_name, best_result["mean_score"],
        )
        return {
            "candidate_models_results": results,
            "selected_model_name": best_model_name,
            "model_selection_summary": model_selection_summary,
            "agent_decisions": [selection_decision],
            "logs": [self.log(
                f"Compared {len(candidates)} model(s) via "
                f"{'a single unsupervised fit' if is_unsupervised else f'{CV_FOLDS}-fold CV'}. "
                f"Selected '{best_model_name}' (score={best_result['mean_score']:.4f})."
            )],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _build_candidates(self, problem_type: str) -> Dict[str, Any]:
        if problem_type == "classification":
            candidates: Dict[str, Any] = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            }
            if _HAS_XGBOOST:
                candidates["XGBoost"] = XGBClassifier(
                    random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
                )
            if _HAS_LIGHTGBM:
                candidates["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
        elif problem_type == "regression":
            candidates = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            }
            if _HAS_XGBOOST:
                candidates["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
            if _HAS_LIGHTGBM:
                candidates["LightGBM"] = LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1)
        elif problem_type == "clustering":
            candidates = {
                "KMeans": KMeans(n_clusters=DEFAULT_N_CLUSTERS, random_state=RANDOM_STATE, n_init=10),
                "DBSCAN": DBSCAN(eps=DEFAULT_DBSCAN_EPS, min_samples=DEFAULT_DBSCAN_MIN_SAMPLES),
            }
        else:  # anomaly_detection
            candidates = {
                "IsolationForest": IsolationForest(random_state=RANDOM_STATE, contamination="auto"),
                "LocalOutlierFactor": LocalOutlierFactor(novelty=False, contamination="auto"),
            }
        return candidates

    def _score_candidates_supervised(
        self, candidates: Dict[str, Any], X_train, y_train, cv, scoring: str
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for model_name, model in candidates.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                mean_s, std_s = float(np.mean(scores)), float(np.std(scores))
                self.logger.info(
                    "CV result — model=%s score=%.4f±%.4f metric=%s",
                    model_name, mean_s, std_s, scoring,
                )
                results.append({
                    "model_name": model_name,
                    "mean_score": mean_s,
                    "std_score": std_s,
                    "scoring_metric": scoring,
                })
            except Exception as exc:
                self.logger.warning("CV failed for model=%s error=%s", model_name, exc)
                results.append({
                    "model_name": model_name,
                    "mean_score": float("-inf"),
                    "std_score": 0.0,
                    "scoring_metric": scoring,
                    "error": str(exc),
                })
        return results

    def _score_candidates_unsupervised(self, candidates: Dict[str, Any], X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compares clustering/anomaly detection algorithms via a single fit +
        silhouette score on the full preprocessed dataset, instead of k-fold CV.

        DBSCAN and LocalOutlierFactor (novelty=False) have no real out-of-sample
        .predict() — only fit_predict — so there is no sklearn-idiomatic way to
        do k-fold cross-validation here the way the supervised branch does.
        Anomaly detection reuses this same silhouette metric on the binary
        inlier(1)/outlier(-1) partition rather than inventing a third metric
        family — imperfect on an imbalanced partition, but there is no ground
        truth available to do better with.
        """
        results: List[Dict[str, Any]] = []
        for model_name, model in candidates.items():
            try:
                labels = model.fit_predict(X)
                # silhouette_score requires >=2 distinct labels; a degenerate fit
                # (everything in one cluster, or DBSCAN labeling everything noise)
                # gets the metric's true worst value instead of crashing.
                score = float(silhouette_score(X, labels)) if len(set(labels)) >= 2 else -1.0
            except Exception as exc:
                self.logger.warning("Unsupervised scoring failed for model=%s error=%s", model_name, exc)
                score = -1.0
            results.append({
                "model_name": model_name,
                "mean_score": score,
                "std_score": 0.0,
                "scoring_metric": "silhouette",
            })
        return results

    def _build_cv_splitter(self, problem_type: str, y_train: pd.Series):
        if problem_type == "classification":
            min_class_count = y_train.value_counts().min()
            if min_class_count >= 2:
                n_folds = max(2, min(CV_FOLDS, int(min_class_count)))
                return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        n_splits = max(2, min(CV_FOLDS, len(y_train)))
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def _build_reasoning(self, best: Dict[str, Any], all_results: List[Dict[str, Any]], scoring: str) -> str:
        others = [r for r in all_results if r["model_name"] != best["model_name"]]
        comparison = ", ".join(
            f"{r['model_name']}={r['mean_score']:.4f}" for r in others
        )
        metric_label = METRIC_LABELS.get(scoring, scoring)
        method = (
            "a single fit on the full preprocessed dataset (no train/test split "
            "or cross-validation — see ModelSelectionAgent._score_candidates_unsupervised)"
            if scoring == "silhouette" else
            f"{CV_FOLDS}-fold cross-validation on the training set"
        )
        return (
            f"'{best['model_name']}' achieved the best mean {metric_label} "
            f"({best['mean_score']:.4f} ± {best['std_score']:.4f}) via {method}. "
            f"Other candidates scored: {comparison}."
        )

    def _build_llm_prompt(self, best_model_name: str, best_result: Dict[str, Any], all_results: List[Dict[str, Any]], scoring: str) -> str:
        metric_label = METRIC_LABELS.get(scoring, scoring)
        top_rows = []
        for result in sorted(all_results, key=lambda r: r["mean_score"], reverse=True)[:5]:
            top_rows.append(
                f"- {result['model_name']}: mean {metric_label}={result['mean_score']:.4f}, std={result['std_score']:.4f}"
            )
        return (
            f"Best model: {best_model_name}\n"
            f"Winning score: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}\n"
            f"Metric: {metric_label}\n\n"
            f"Candidate comparison:\n{chr(10).join(top_rows)}\n\n"
            "Explain why the selected model is preferable and mention any caveats about close competition."
        )

    def _build_fallback_summary(self, best_model_name: str, best_result: Dict[str, Any], all_results: List[Dict[str, Any]], scoring: str) -> str:
        metric_label = METRIC_LABELS.get(scoring, scoring)
        others = [r for r in all_results if r["model_name"] != best_model_name]
        comparison = ", ".join(f"{r['model_name']}={r['mean_score']:.4f}" for r in others[:4]) or "no alternatives recorded"
        method = "a single fit on the full dataset" if scoring == "silhouette" else "cross-validation"
        return (
            f"* **Selected Model**: `{best_model_name}` was chosen because it achieved the best mean {metric_label} "
            f"({best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}) via {method}.\n"
            f"* **Comparison**: Competing models scored: {comparison}.\n"
            "* **Caution**: If the margin is small, the choice should be validated further on fresh data before deployment."
        )

    def _confidence_from_margin(self, results: List[Dict[str, Any]]) -> float:
        """
        Confidence reflects how clearly the winner beat the runner-up, not
        just the raw score. A close race between the top 2 models means the
        choice is less "confident" even if the winning score is high.
        """
        if len(results) < 2:
            return 0.9
        best_score, second_score = results[0]["mean_score"], results[1]["mean_score"]
        margin = abs(best_score - second_score)
        # Normalize: a margin >= 0.05 (5 accuracy points, or 0.05 RMSE units) is
        # treated as a clearly confident win.
        return float(min(0.95, 0.5 + margin * 10))
