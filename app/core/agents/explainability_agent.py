"""
Explainability Agent Module
==============================

Answers the core research question of EMADS: "why does the model make its
decisions?" Computes global feature importance, SHAP values, and asks the
LLM to turn both into a plain-language explanation.
"""
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

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState
from app.services.llm_service import LLMService
from app.utils.plot_utils import save_current_figure

RANDOM_STATE = 42
# SHAP is expensive on large datasets; a sample is enough for a reliable
# global explanation and keeps runtime well within the UI's time budget.
SHAP_SAMPLE_SIZE = 100

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
        model_name = state.get("selected_model_name")

        if not all([data_path, model_path, target_column, model_name]):
            self.logger.error(
                "Missing state keys — data_path=%s model_path=%s target=%s model_name=%s",
                bool(data_path), bool(model_path), bool(target_column), bool(model_name),
            )
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'model_path', 'target_column' or 'selected_model_name'."
            )

        self.logger.debug("Loading data from %s and model from %s", data_path, model_path)
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column])

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        feature_importance = self._compute_feature_importance(model, X.columns)

        shap_plots = []
        shap_top_features = []
        try:
            self.logger.info("Computing SHAP values (model=%s sample_size=%s)", model_name, SHAP_SAMPLE_SIZE)
            shap_plots, shap_top_features = self._compute_shap(model, X, model_name)
            self.logger.info("SHAP computation successful — top_features=%s", shap_top_features)
        except Exception as exc:
            # SHAP can fail on unusual model/data combinations; the agent
            # should still return the feature importance it already has
            # instead of crashing the whole pipeline over an optional analysis.
            self.logger.warning("SHAP computation skipped: %s", exc, exc_info=True)
            state_note = self.log(f"SHAP computation skipped due to error: {exc}")
        else:
            state_note = self.log(f"Computed SHAP values on a sample of {SHAP_SAMPLE_SIZE} rows.")

        self.logger.info(
            "Calling LLM for explainability summary (model=%s)",
            getattr(self.llm, 'model_name', 'unknown'),
        )
        explainability_summary = self.llm.generate_summary(
            system_prompt=(
                "You are a senior data scientist explaining a trained model's behavior "
                "to a non-technical stakeholder. Write 4-6 bullet points in plain English: "
                "which features matter most, what that implies about the data, and any "
                "caveats about trusting these explanations. No code, no jargon."
            ),
            user_prompt=self._build_llm_prompt(model_name, feature_importance, shap_top_features),
            fallback_message=self._build_fallback_summary(model_name, feature_importance, shap_top_features),
        )
        self.logger.info(
            "LLM explainability response received (chars=%s, starts_with=%r)",
            len(explainability_summary) if explainability_summary else 0,
            (explainability_summary or "")[:80],
        )

        explanation_source = "SHAP values" if shap_top_features else "the model's native feature importance"
        explanation_scope = (
            f"a sample of {SHAP_SAMPLE_SIZE} rows" if shap_top_features else "the full training data"
        )
        importance_decision = self.decide(
            decision=f"Top predictive feature: '{self._top_feature(feature_importance)}'",
            reasoning=(
                f"Derived from {explanation_source}, "
                f"computed on {explanation_scope}."
            ),
            confidence=0.7,
        )

        self.logger.info(
            "execute() complete — top_feature=%s shap_plots=%s",
            self._top_feature(feature_importance), len(shap_plots),
        )
        return {
            "feature_importance": feature_importance,
            "shap_plots": shap_plots,
            "explainability_summary": explainability_summary,
            "agent_decisions": [importance_decision],
            "logs": [state_note],
        }

    def _build_fallback_summary(
        self, model_name: str, feature_importance: dict, shap_top_features: list[str]
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

    def _compute_shap(self, model, X: pd.DataFrame, model_name: str) -> tuple[list[str], list[str]]:
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
                values=mean_abs, data=sample.values, feature_names=list(sample.columns)
            )

        plt.figure()
        shap.summary_plot(values_for_plot, sample, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {model_name}")
        summary_plot_path = save_current_figure("shap_summary.png")

        mean_abs_shap = np.abs(values_for_plot.values).mean(axis=0)
        top_features = [
            name for name, _ in sorted(
                zip(sample.columns, mean_abs_shap), key=lambda item: item[1], reverse=True
            )[:5]
        ]
        return [summary_plot_path], top_features

    def _top_feature(self, feature_importance: dict) -> str:
        return next(iter(feature_importance), "unknown")

    def _build_llm_prompt(self, model_name: str, feature_importance: dict, shap_top_features: list[str]) -> str:
        top_native = list(feature_importance.items())[:5]
        native_lines = "\n".join(f"  - {name}: {score:.4f}" for name, score in top_native)

        prompt = (
            f"Model: {model_name}\n\n"
            f"Top features by native importance:\n{native_lines or '  (unavailable)'}\n\n"
        )
        if shap_top_features:
            prompt += f"Top features by SHAP value: {', '.join(shap_top_features)}\n\n"
        prompt += "Please provide your structured explanation now."
        return prompt