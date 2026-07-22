"""
Evaluation Agent Module
==========================

Loads the final (tuned) model and scores it on the untouched holdout test
set, plus a fresh cross-validation pass for stability. Adds ROC AUC and
Cross Validation scores on top of the V1 metrics, and generates the
confusion matrix / ROC curve figures for the report.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
)

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState
from app.utils import plot_utils

RANDOM_STATE = 42
CV_FOLDS = 5


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for computing final validation metrics and generating
    the corresponding diagnostic plots.
    """

    def __init__(self) -> None:
        super().__init__(name="evaluation_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        data_path = state.get("preprocessed_data_path")
        model_path = state.get("model_path")
        target_column = state.get("target_column")
        problem_type = state.get("problem_type")

        if not all([data_path, model_path, target_column, problem_type]):
            self.logger.error(
                "Missing state keys — data_path=%s model_path=%s target=%s problem_type=%s",
                bool(data_path), bool(model_path), bool(target_column), bool(problem_type),
            )
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'model_path', 'target_column' or 'problem_type'."
            )

        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
        )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        evaluation_plots = []

        # Cross-validation on the training set, for a stability check beyond
        # a single train/test split.
        cv_mean, cv_std = self._cross_validate(model, X_train, y_train, problem_type)

        if problem_type == "regression":
            metrics_payload = {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
                "r2": float(r2_score(y_test, y_pred)),
                "cv_mean_neg_rmse": cv_mean,
                "cv_std_neg_rmse": cv_std,
            }
        else:
            accuracy = float(accuracy_score(y_test, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred).tolist()
            evaluation_plots.append(plot_utils.plot_confusion_matrix(cm))

            roc_auc = self._compute_roc_auc(model, X_test, y_test)
            if roc_auc is not None and len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                evaluation_plots.append(plot_utils.plot_roc_curve(y_test, y_proba))

            metrics_payload = {
                "accuracy": accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": roc_auc,
                "confusion_matrix": cm,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
            }

        stability_decision = self.decide(
            decision=f"Test score vs {CV_FOLDS}-fold CV score: consistency check",
            reasoning=self._build_stability_reasoning(metrics_payload, cv_mean, cv_std, problem_type),
            confidence=0.8,
        )

        self.logger.info("Evaluation metrics: %s", metrics_payload)
        return {
            "metrics": metrics_payload,
            "evaluation_plots": evaluation_plots,
            "agent_decisions": [stability_decision],
            "logs": [self.log(f"Evaluation complete. Metrics: {metrics_payload}")],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _cross_validate(self, model, X_train, y_train, problem_type: str) -> tuple[float, float]:
        """Re-runs CV with a fresh (unfitted) clone of the final model, purely
        to report a stability estimate — does not affect the saved model."""
        fresh_model = clone(model)
        scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
        cv = self._build_cv_splitter(problem_type, y_train)
        scores = cross_val_score(fresh_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores)), float(np.std(scores))

    def _build_cv_splitter(self, problem_type: str, y_train: pd.Series):
        if problem_type == "classification":
            min_class_count = y_train.value_counts().min()
            if min_class_count >= 2:
                n_folds = max(2, min(CV_FOLDS, int(min_class_count)))
                return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        n_splits = max(2, min(CV_FOLDS, len(y_train)))
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def _compute_roc_auc(self, model, X_test, y_test) -> float | None:
        """ROC AUC requires predicted probabilities; not every model type
        supports .predict_proba(), and it's undefined for >2 classes without
        extra averaging assumptions — handled safely here."""
        if not hasattr(model, "predict_proba"):
            return None
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                return float(roc_auc_score(y_test, y_proba[:, 1]))
            return float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            return None

    def _build_stability_reasoning(self, metrics: dict, cv_mean: float, cv_std: float, problem_type: str) -> str:
        test_score = metrics.get("accuracy") if problem_type == "classification" else -metrics.get("rmse", 0)
        gap = abs(test_score - cv_mean)
        if gap > 0.1:
            verdict = "The gap between the test score and CV score suggests possible overfitting or an unrepresentative test split."
        else:
            verdict = "The test score is consistent with the cross-validation score, suggesting the model generalizes reliably."
        return (
            f"Test score: {test_score:.4f}. Cross-validation: {cv_mean:.4f} ± {cv_std:.4f} "
            f"across {CV_FOLDS} folds. {verdict}"
        )