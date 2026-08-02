"""
Explainability Agent Module
==============================

Answers the core research question of EMADS: "why does the model make its
decisions?" Computes global feature importance, SHAP values, and asks the
LLM to turn both into a plain-language explanation.
"""

import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState, UNSUPERVISED_PROBLEM_TYPES
from app.services.llm_service import LLMService
from app.utils.plot_utils import save_current_figure

RANDOM_STATE = 42
# SHAP is expensive on large datasets; a sample is enough for a reliable
# global explanation and keeps runtime well within the UI's time budget.
SHAP_SAMPLE_SIZE = 100
# Matches EvaluationAgent's split exactly, so local explanations are computed
# on the same held-out rows the model was actually scored on.
TEST_SIZE = 0.2
# Below this SHAP/permutation-importance rank correlation, the two methods
# are considered to disagree enough to warrant an LLM-written caveat rather
# than silently reporting a single ranking as "the" explanation.
AGREEMENT_LOW_THRESHOLD = 0.3
N_LOCAL_EXPLANATIONS = 3

TREE_MODEL_NAMES = {"DecisionTree", "RandomForest", "XGBoost", "LightGBM"}


class ExplainabilityAgent(BaseAgent):
    """
    Agent responsible for producing global model explanations: feature
    importance, SHAP values, and an LLM narrative built from both.
    """

    def __init__(self, llm_service: LLMService | None = None) -> None:
        super().__init__(name="explainability_agent")
        self.llm = llm_service if llm_service else LLMService()

    def execute(self, state: EMADSState) -> PartialEMADSState:
        self.logger.info("execute() started")
        data_path = state.get("preprocessed_data_path")
        model_path = state.get("model_path")
        target_column = state.get("target_column")
        problem_type = state.get("problem_type")
        model_name = state.get("selected_model_name")
        is_unsupervised = problem_type in UNSUPERVISED_PROBLEM_TYPES

        if not all([data_path, model_path, model_name]) or (not is_unsupervised and not target_column):
            self.logger.error(
                "Missing state keys — data_path=%s model_path=%s target=%s model_name=%s",
                bool(data_path), bool(model_path), bool(target_column), bool(model_name),
            )
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'model_path', 'selected_model_name', or 'target_column' "
                "(required for supervised problem types)."
            )

        self.logger.debug("Loading data from %s and model from %s", data_path, model_path)
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column]) if target_column else df.copy()

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        feature_importance = self._compute_feature_importance(model, X.columns)
        permutation_importance = state.get("permutation_importance") or {}

        shap_plots = []
        shap_top_features = []
        shap_importance = {}
        try:
            self.logger.info("Computing SHAP values (model=%s sample_size=%s)", model_name, SHAP_SAMPLE_SIZE)
            shap_plots, shap_top_features, shap_importance = self._compute_shap(model, X, model_name)
            self.logger.info("SHAP computation successful — top_features=%s", shap_top_features)
        except Exception as exc:
            # SHAP can fail on unusual model/data combinations; the agent
            # should still return the feature importance it already has
            # instead of crashing the whole pipeline over an optional analysis.
            self.logger.warning("SHAP computation skipped: %s", exc, exc_info=True)
            state_note = self.log(f"SHAP computation skipped due to error: {exc}")
        else:
            state_note = self.log(f"Computed SHAP values on a sample of {SHAP_SAMPLE_SIZE} rows.")

        arbitration_log = []
        agreement_score = None
        if not is_unsupervised and shap_importance and permutation_importance:
            agreement_score = self._agreement_score(shap_importance, permutation_importance)
            if agreement_score is not None and agreement_score < AGREEMENT_LOW_THRESHOLD:
                self.logger.info(
                    "SHAP/permutation agreement=%.3f below threshold — escalating to LLM.", agreement_score,
                )
                arbitration_response = self.llm.generate_summary(
                    system_prompt=(
                        "You are a senior data scientist. Two independent feature-importance methods "
                        "(SHAP and permutation importance) disagree substantially on which features "
                        "matter most for this model. Explain, in plain English, what could cause this "
                        "disagreement and what it implies for how much the resulting explanation should "
                        "be trusted. Do NOT pick a side between the two methods."
                    ),
                    user_prompt=(
                        f"SHAP top features: {list(shap_importance.items())[:5]}\n"
                        f"Permutation-importance top features: {list(permutation_importance.items())[:5]}\n"
                        f"Spearman rank correlation between the two rankings: {agreement_score:.3f}"
                    ),
                    fallback_message=(
                        f"SHAP and permutation importance rankings disagree substantially "
                        f"(Spearman correlation={agreement_score:.3f}). Treat this explanation as a "
                        "hint rather than a definitive account, and avoid relying on a single top "
                        "feature for high-stakes decisions."
                    ),
                )
                arbitration_log.append({
                    "agent_name": self.name,
                    "trigger": f"SHAP/permutation agreement score={agreement_score:.3f} (< {AGREEMENT_LOW_THRESHOLD})",
                    "llm_arbitration": arbitration_response,
                })

        local_explanations = []
        if not is_unsupervised:
            try:
                local_explanations = self._compute_local_explanations(
                    model, df, target_column, model_name, problem_type
                )
            except Exception as exc:
                self.logger.warning("Local explanations skipped: %s", exc, exc_info=True)

        self.logger.info(
            "Calling LLM for explainability summary (model=%s)",
            getattr(self.llm, 'model_name', 'unknown'),
        )
        fallback_summary = self._build_fallback_summary(
            model_name, feature_importance, shap_top_features, permutation_importance
        )
        explainability_summary = self.llm.generate_summary(
            system_prompt=(
                "You are a senior data scientist explaining a trained model's behavior "
                "to a non-technical stakeholder. Write 4-6 bullet points in plain English: "
                "which features matter most, what that implies about the data, and any "
                "caveats about trusting these explanations. No code, no jargon."
            ),
            user_prompt=self._build_llm_prompt(
                model_name, feature_importance, shap_top_features, permutation_importance
            ),
            fallback_message=fallback_summary,
        )
        if not (explainability_summary or "").strip():
            self.logger.warning("LLM returned empty explainability summary; using deterministic fallback text.")
            explainability_summary = fallback_summary
        self.logger.info(
            "LLM explainability response received (chars=%s, starts_with=%r)",
            len(explainability_summary) if explainability_summary else 0,
            (explainability_summary or "")[:80],
        )

        explanation_source = "SHAP values" if shap_top_features else "the model's native feature importance"
        explanation_scope = (
            f"a sample of {SHAP_SAMPLE_SIZE} rows" if shap_top_features else "the full training data"
        )
        top_feature = self._top_feature(feature_importance)
        cross_check_note = self._build_cross_check_note(top_feature, permutation_importance)
        if agreement_score is not None:
            agreement_note = f" SHAP/permutation-importance rank agreement (Spearman): {agreement_score:.3f}."
            # agreement_score is a Spearman correlation in [-1, 1] — used
            # directly as confidence, a negative correlation (the two methods
            # actively disagreeing) produced a nonsensical negative
            # confidence. Rescale to [0, 1] first, same clamp range as every
            # other agent's grounded confidence.
            confidence = float(max(0.3, min(0.95, (agreement_score + 1) / 2)))
        else:
            # No agreement score could be computed (SHAP or permutation
            # importance unavailable) — fall back to the margin-based signal
            # rather than reporting no confidence at all.
            agreement_note = ""
            confidence = self._confidence_from_importance_margin(feature_importance, permutation_importance, top_feature)
        importance_decision = self.decide(
            decision=f"Top predictive feature: '{top_feature}'",
            reasoning=(
                f"Derived from {explanation_source}, "
                f"computed on {explanation_scope}. {cross_check_note}{agreement_note}"
            ),
            confidence=confidence,
        )

        self.logger.info(
            "execute() complete — top_feature=%s shap_plots=%s",
            self._top_feature(feature_importance), len(shap_plots),
        )
        return {
            "feature_importance": feature_importance,
            "shap_plots": shap_plots,
            "explainability_summary": explainability_summary,
            "local_explanations": local_explanations,
            "explanation_agreement_score": agreement_score,
            "agent_decisions": [importance_decision],
            "llm_arbitration_log": arbitration_log,
            "logs": [state_note],
        }

    def _build_cross_check_note(self, top_feature: str, permutation_importance: dict) -> str:
        if not permutation_importance:
            return ""
        perm_top_feature = next(iter(permutation_importance), None)
        if perm_top_feature == top_feature:
            return (
                f"Permutation importance (a model-agnostic cross-check on the held-out "
                f"test set) agrees: '{perm_top_feature}' is also its top feature."
            )
        return (
            f"Note: permutation importance on the held-out test set instead ranks "
            f"'{perm_top_feature}' highest, which is worth investigating rather than "
            f"trusting a single method blindly."
        )

    def _build_fallback_summary(
        self,
        model_name: str,
        feature_importance: dict,
        shap_top_features: list[str],
        permutation_importance: dict,
    ) -> str:
        top_features = list(feature_importance.items())[:5]
        bullets = []
        if top_features:
            top_name, top_score = top_features[0]
            bullets.append(
                f"* **Primary Predictive Driver**: The model `{model_name}` relies most heavily on `{top_name}` "
                f"(relative importance score: {top_score:.4f})."
            )
            if len(top_features) > 1:
                secondary = ", ".join([f"`{name}` ({score:.4f})" for name, score in top_features[1:4]])
                bullets.append(f"* **Secondary Influential Features**: `{secondary}` also provide predictive signal.")

        if shap_top_features:
            bullets.append(
                f"* **SHAP Feature Impact**: Global SHAP analysis highlights `{', '.join(shap_top_features[:3])}` "
                f"as the top drivers of individual sample predictions."
            )

        if permutation_importance:
            perm_top = list(permutation_importance.items())[:3]
            perm_str = ", ".join([f"`{name}` ({score:.4f})" for name, score in perm_top])
            bullets.append(
                f"* **Permutation Importance Cross-Check**: Shuffling each feature on the held-out "
                f"test set and measuring the score drop independently points to `{perm_str}` "
                f"as most influential — a model-agnostic confirmation of the ranking above."
            )

        bullets.append(
            "* **Decision Transparency & Caution**: Feature importance measures correlation within the dataset. "
            "Always validate top features with domain knowledge before deploying into production decisions."
        )

        return "\n\n".join(bullets)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _compute_feature_importance(self, model, feature_names) -> dict:
        """Uses whichever native importance attribute the model exposes."""
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            values = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return {}

        importance = dict(zip(feature_names, [float(v) for v in values]))
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

    def _compute_shap(self, model, X: pd.DataFrame, model_name: str) -> tuple[list[str], list[str], dict]:
        sample = X.sample(n=min(SHAP_SAMPLE_SIZE, len(X)), random_state=RANDOM_STATE)

        if model_name in TREE_MODEL_NAMES:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, sample)

        shap_values = explainer(sample)

        # For multiclass classification, shap_values has an extra class dimension;
        # collapse it by taking the mean absolute value across classes for the summary plot.
        values_for_plot = shap_values
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            mean_abs = np.abs(shap_values.values).mean(axis=2)
            values_for_plot = shap.Explanation(
                values=mean_abs, data=sample.values, feature_names=list(sample.columns),
                # shap.summary_plot's bar-plot path unconditionally reads
                # base_values.shape — omitting it (defaults to None) crashes
                # with "'NoneType' object has no attribute 'shape'" for any
                # non-tree multiclass model. The per-class baseline is
                # meaningless once collapsed to a single bar chart anyway, so
                # a same-length zero vector is a safe placeholder.
                base_values=np.zeros(len(sample)),
            )

        plt.figure()
        shap.summary_plot(values_for_plot, sample, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {model_name}")
        summary_plot_path = save_current_figure("shap_summary.png")

        mean_abs_shap = np.abs(values_for_plot.values).mean(axis=0)
        shap_importance = dict(sorted(
            zip(sample.columns, [float(v) for v in mean_abs_shap]),
            key=lambda item: item[1], reverse=True,
        ))
        top_features = list(shap_importance)[:5]
        return [summary_plot_path], top_features, shap_importance

    def _agreement_score(self, shap_importance: dict, perm_importance: dict) -> float | None:
        """
        Spearman rank correlation between SHAP and permutation importance —
        a model-agnostic cross-check of how much the two independent
        explanation methods actually agree, rather than assuming SHAP alone
        is trustworthy. Returns None (not a fabricated number) when there
        are too few common features to correlate meaningfully.
        """
        common_features = [f for f in shap_importance if f in perm_importance]
        if len(common_features) < 2:
            return None
        shap_values = [shap_importance[f] for f in common_features]
        perm_values = [perm_importance[f] for f in common_features]
        try:
            correlation, _ = spearmanr(shap_values, perm_values)
        except Exception:
            return None
        if correlation is None or np.isnan(correlation):
            return None
        return float(correlation)

    def _compute_local_explanations(
        self, model, df: pd.DataFrame, target_column: str, model_name: str, problem_type: str
    ) -> list[dict]:
        """
        Global feature importance says WHAT matters on average; this shows
        HOW it played out for a handful of specific predictions on the
        held-out test set — including a mispredicted one where possible,
        since that's exactly the case where trusting the global explanation
        blindly would be misleading.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
        try:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_y
            )
        except ValueError:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
            )
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            is_error = (y_pred != y_test.values)
        else:
            # No ground-truth notion of "wrong" in regression — flag
            # predictions further off than one std of the target as the
            # closest analogue to a misclassification.
            residual_threshold = float(np.std(y_test.values)) or 1.0
            is_error = np.abs(y_pred - y_test.values) > residual_threshold

        n_rows = len(X_test)
        positions = list(range(n_rows))
        chosen: list[int] = []

        misprediction_pos = next((i for i in positions if is_error[i]), None)
        if misprediction_pos is not None:
            chosen.append(misprediction_pos)

        remaining = [i for i in positions if i not in chosen]
        n_extra = min(N_LOCAL_EXPLANATIONS - len(chosen), len(remaining))
        if n_extra > 0:
            rng = np.random.RandomState(RANDOM_STATE)
            chosen.extend(rng.choice(remaining, size=n_extra, replace=False).tolist())

        if not chosen:
            return []

        sample = X_test.iloc[chosen]
        if model_name in TREE_MODEL_NAMES:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)

        local_explanations = []
        for plot_idx, test_pos in enumerate(chosen):
            instance = shap_values[plot_idx]
            # Multiclass: SHAP returns one contribution vector per class per
            # row — narrow down to the class the model actually predicted.
            if hasattr(instance, "values") and np.ndim(instance.values) == 2:
                predicted_class = y_pred[test_pos]
                class_list = list(getattr(model, "classes_", []))
                class_idx = class_list.index(predicted_class) if predicted_class in class_list else 0
                instance = instance[:, class_idx]

            plt.figure()
            try:
                shap.plots.waterfall(instance, show=False)
            except Exception as exc:
                self.logger.warning("Waterfall plot failed for test row=%s: %s", test_pos, exc)
                plt.close()
                continue
            plot_path = save_current_figure(f"shap_waterfall_{plot_idx}.png")

            actual_value = y_test.iloc[test_pos]
            predicted_value = y_pred[test_pos]
            local_explanations.append({
                "row_index": int(X_test.index[test_pos]),
                "actual": actual_value.item() if hasattr(actual_value, "item") else actual_value,
                "predicted": predicted_value.item() if hasattr(predicted_value, "item") else predicted_value,
                "misclassified": bool(is_error[test_pos]),
                "plot_path": plot_path,
            })
        return local_explanations

    def _confidence_from_importance_margin(
        self, feature_importance: dict, permutation_importance: dict, top_feature: str
    ) -> float:
        """
        Confidence reflects how clearly the top feature's importance stands
        out from the runner-up, boosted or penalized by whether the
        permutation-importance cross-check (a model-agnostic, held-out-test-set
        signal) agrees — rather than a fixed value regardless of the margin.
        """
        scores = list(feature_importance.values())
        if len(scores) >= 2 and scores[0] > 0:
            margin_ratio = (scores[0] - scores[1]) / scores[0]
        else:
            margin_ratio = 0.0
        confidence = max(0.5, min(0.9, 0.5 + margin_ratio * 0.4))

        if permutation_importance:
            perm_top_feature = next(iter(permutation_importance), None)
            if perm_top_feature == top_feature:
                confidence = min(0.95, confidence + 0.05)
            else:
                confidence = max(0.4, confidence - 0.1)
        return float(confidence)

    def _top_feature(self, feature_importance: dict) -> str:
        return next(iter(feature_importance), "unknown")

    def _build_llm_prompt(
        self,
        model_name: str,
        feature_importance: dict,
        shap_top_features: list[str],
        permutation_importance: dict,
    ) -> str:
        top_native = list(feature_importance.items())[:5]
        native_lines = "\n".join(f"  - {name}: {score:.4f}" for name, score in top_native)

        prompt = (
            f"Model: {model_name}\n\n"
            f"Top features by native importance:\n{native_lines or '  (unavailable)'}\n\n"
        )
        if shap_top_features:
            prompt += f"Top features by SHAP value: {', '.join(shap_top_features)}\n\n"
        if permutation_importance:
            top_perm = list(permutation_importance.items())[:5]
            perm_lines = ", ".join(f"{name} ({score:.4f})" for name, score in top_perm)
            prompt += f"Top features by permutation importance (held-out test set): {perm_lines}\n\n"
        prompt += "Please provide your structured explanation now."
        return prompt