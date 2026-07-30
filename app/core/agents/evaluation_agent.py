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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState, AgentDecision, UNSUPERVISED_PROBLEM_TYPES
from app.utils import plot_utils

RANDOM_STATE = 42
CV_FOLDS = 5
PERMUTATION_N_REPEATS = 10


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
        is_unsupervised = problem_type in UNSUPERVISED_PROBLEM_TYPES

        if not all([data_path, model_path, problem_type]) or (not is_unsupervised and not target_column):
            self.logger.error(
                "Missing state keys — data_path=%s model_path=%s target=%s problem_type=%s",
                bool(data_path), bool(model_path), bool(target_column), bool(problem_type),
            )
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'model_path', 'problem_type', or 'target_column' (required "
                "for supervised problem types)."
            )

        df = pd.read_csv(data_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        permutation_importance_payload = None
        if is_unsupervised:
            metrics_payload, evaluation_plots, decision = self._evaluate_unsupervised(df, model, problem_type)
        else:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
            )

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

            permutation_importance_payload = self._compute_permutation_importance(
                model, X_test, y_test, problem_type
            )

            stability_reasoning, gap = self._build_stability_reasoning(metrics_payload, cv_mean, cv_std, problem_type)
            # Smaller test/CV gap means more confident the model generalizes,
            # rather than a fixed confidence regardless of the actual gap.
            stability_confidence = float(max(0.4, min(0.9, 0.9 - gap * 3)))
            decision = self.decide(
                decision=f"Test score vs {CV_FOLDS}-fold CV score: consistency check",
                reasoning=stability_reasoning,
                confidence=stability_confidence,
            )
            permutation_decision = self._build_permutation_decision(permutation_importance_payload)

        self.logger.info("Evaluation metrics: %s", metrics_payload)
        agent_decisions = [decision] + ([permutation_decision] if not is_unsupervised else [])
        return {
            "metrics": metrics_payload,
            "evaluation_plots": evaluation_plots,
            "permutation_importance": permutation_importance_payload,
            "agent_decisions": agent_decisions,
            "logs": [self.log(f"Evaluation complete. Metrics: {metrics_payload}")],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _evaluate_unsupervised(self, df: pd.DataFrame, model, problem_type: str) -> tuple[dict, list, AgentDecision]:
        """Scores the tuned clustering/anomaly detection model on the full
        preprocessed dataset. Refits via fit_predict rather than .predict()
        because DBSCAN/LocalOutlierFactor have no reusable out-of-sample
        prediction — the same constraint ModelSelectionAgent documents for
        its own unsupervised scoring, kept here for a uniform treatment of
        all four candidate model types instead of one-off special-casing."""
        labels = model.fit_predict(df)
        n_labels = len(set(labels))

        if n_labels >= 2:
            metrics_payload = {
                "silhouette_score": float(silhouette_score(df, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(df, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(df, labels)),
            }
        else:
            metrics_payload = {
                "silhouette_score": -1.0,
                "davies_bouldin_score": None,
                "calinski_harabasz_score": None,
            }

        if problem_type == "clustering":
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            metrics_payload["n_clusters_found"] = int(n_labels)
            metrics_payload["cluster_sizes"] = {str(k): int(v) for k, v in cluster_sizes.items()}
        else:  # anomaly_detection
            n_anomalies = int((labels == -1).sum())
            metrics_payload["n_anomalies_detected"] = n_anomalies
            metrics_payload["anomaly_rate"] = round(n_anomalies / len(labels), 4)

        evaluation_plots = [plot_utils.plot_cluster_scatter(df, labels, problem_type.replace("_", " ").title())]

        silhouette = metrics_payload["silhouette_score"]
        # Confidence derived from the silhouette score itself (normalized to
        # [0,1] then clamped) rather than hardcoded — there is no held-out
        # test score to compare against for unsupervised problems, but the
        # silhouette score is a real, computed signal of cluster separation.
        confidence = float(max(0.3, min(0.9, (silhouette + 1) / 2)))
        decision = self.decide(
            decision=f"{problem_type.replace('_', ' ').title()} evaluated via silhouette/Davies-Bouldin/Calinski-Harabasz",
            reasoning=(
                f"Silhouette score={silhouette:.4f}, Davies-Bouldin="
                f"{metrics_payload['davies_bouldin_score']}, Calinski-Harabasz="
                f"{metrics_payload['calinski_harabasz_score']}, on {n_labels} distinct label(s) "
                f"found. No held-out test set exists for unsupervised problems, so these "
                f"metrics — computed on the full dataset — are the final evaluation, not a "
                f"stability check."
            ),
            confidence=confidence,
        )
        return metrics_payload, evaluation_plots, decision

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

    def _compute_permutation_importance(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str
    ) -> dict:
        """Model-agnostic cross-check for ExplainabilityAgent's native/SHAP
        importances: shuffles one feature at a time on the held-out test set
        and measures the resulting drop in score, rather than relying on
        attribution internal to the model. Computed here (not in
        ExplainabilityAgent) because the train/test split already lives in
        this agent."""
        scoring = "accuracy" if problem_type == "classification" else "r2"
        result = permutation_importance(
            model, X_test, y_test, scoring=scoring,
            n_repeats=PERMUTATION_N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
        )
        importance = dict(zip(X_test.columns, [float(v) for v in result.importances_mean]))
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

    def _build_permutation_decision(self, permutation_importance_payload: dict) -> AgentDecision:
        top_feature = next(iter(permutation_importance_payload), "unknown")
        top_score = permutation_importance_payload.get(top_feature, 0.0)
        return self.decide(
            decision=f"Permutation importance top feature: '{top_feature}'",
            reasoning=(
                f"Shuffling '{top_feature}' on the held-out test set dropped the score "
                f"by {top_score:.4f} on average over {PERMUTATION_N_REPEATS} repeats — "
                "the largest drop of any feature. This is a model-agnostic cross-check "
                "for the native/SHAP feature importance computed later by "
                "ExplainabilityAgent."
            ),
            confidence=float(max(0.3, min(0.9, top_score * 5))) if top_score > 0 else 0.3,
        )

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

    def _build_stability_reasoning(self, metrics: dict, cv_mean: float, cv_std: float, problem_type: str) -> tuple[str, float]:
        test_score = metrics.get("accuracy") if problem_type == "classification" else -metrics.get("rmse", 0)
        gap = abs(test_score - cv_mean)
        if gap > 0.1:
            verdict = "The gap between the test score and CV score suggests possible overfitting or an unrepresentative test split."
        else:
            verdict = "The test score is consistent with the cross-validation score, suggesting the model generalizes reliably."
        reasoning = (
            f"Test score: {test_score:.4f}. Cross-validation: {cv_mean:.4f} ± {cv_std:.4f} "
            f"across {CV_FOLDS} folds. {verdict}"
        )
        return reasoning, gap