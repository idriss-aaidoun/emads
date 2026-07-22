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

import os
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState

MODELS_DIR = os.path.join("data", "outputs", "models")
CV_FOLDS = 5
RANDOM_STATE = 42

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

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        data_path = state.get("preprocessed_data_path")
        target_column = state.get("target_column")
        problem_type = state.get("problem_type")

        if not data_path or not target_column or not problem_type:
            self.logger.error(
                "Missing required state keys — data_path=%s target_column=%s problem_type=%s",
                bool(data_path), bool(target_column), bool(problem_type),
            )
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'target_column' or 'problem_type'."
            )

        self.logger.debug("Loading preprocessed data from: %s", data_path)
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Recreate the exact same split used by the Evaluation Agent, so the
        # test set is never touched here (no leakage) and metrics stay comparable.
        stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
        )

        candidates = self._build_candidates(problem_type)
        cv = self._build_cv_splitter(problem_type, y_train)
        scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"

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

        results.sort(key=lambda r: r["mean_score"], reverse=True)
        best_result = results[0]
        best_model_name = best_result["model_name"]
        best_model = candidates[best_model_name]

        # Fit the winning model on the full training split (final trained artifact).
        best_model.fit(X_train, y_train)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "selected_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        selection_decision = self.decide(
            decision=f"Selected '{best_model_name}' as the final model",
            reasoning=self._build_reasoning(best_result, results, scoring),
            confidence=self._confidence_from_margin(results),
        )

        self.logger.info(
            "execute() complete — selected_model=%s score=%.4f model_saved=%s",
            best_model_name, best_result['mean_score'], model_path,
        )
        return {
            "candidate_models_results": results,
            "selected_model_name": best_model_name,
            "model_path": model_path,
            "agent_decisions": [selection_decision],
            "logs": [self.log(
                f"Compared {len(candidates)} model(s) via {CV_FOLDS}-fold CV. "
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
        else:
            candidates = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            }
            if _HAS_XGBOOST:
                candidates["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
            if _HAS_LIGHTGBM:
                candidates["LightGBM"] = LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1)
        return candidates

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
        metric_label = "accuracy" if scoring == "accuracy" else "RMSE (negated)"
        return (
            f"'{best['model_name']}' achieved the best mean {metric_label} "
            f"({best['mean_score']:.4f} ± {best['std_score']:.4f}) across {CV_FOLDS}-fold "
            f"cross-validation on the training set. Other candidates scored: {comparison}."
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