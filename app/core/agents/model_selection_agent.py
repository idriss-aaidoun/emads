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
from typing import Any, Dict, List, Optional
from scipy.stats import wilcoxon

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

# Floor confidence for an arbitrated tie with no 3rd candidate to measure
# field separation against (e.g. only 2 candidates were compared at all) —
# matches DataUnderstandingAgent's own LLM-arbitration confidence, so the
# project falls back to one consistent "reasoned but unmeasured" value
# instead of each agent inventing its own number for this edge case.
ARBITRATION_CONFIDENCE_FLOOR = 0.6

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
            try:
                X_train, _, y_train, _ = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
                )
            except ValueError:
                # Every class has >=2 rows but sklearn ALSO requires
                # test_size >= n_classes for a stratified split — on a very
                # small dataset (e.g. after PreprocessingAgent drops
                # duplicates) that second condition can still fail. Fall
                # back to an unstratified split rather than crashing.
                X_train, _, y_train, _ = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
                )

            cv = self._build_cv_splitter(problem_type, y_train)
            scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
            results = self._score_candidates_supervised(candidates, X_train, y_train, cv, scoring)

        results.sort(key=lambda r: r["mean_score"], reverse=True)
        if results[0]["mean_score"] == float("-inf"):
            # Every single candidate raised during scoring (e.g. a dtype/shape
            # issue that only manifests inside cross_val_score/fit_predict) —
            # there is no real winner here, so fail loudly instead of quietly
            # "selecting" an unfit, broken model for the rest of the pipeline.
            failures = "; ".join(f"{r['model_name']}: {r.get('error', 'unknown error')}" for r in results)
            raise ValueError(f"All candidate models failed during scoring — {failures}")
        p_value, arbitration_entry, best_result = self._compare_top_two(results)
        best_model_name = best_result["model_name"]
        best_model = candidates[best_model_name]

        reasoning = self._build_reasoning(best_result, results, scoring)
        if p_value is not None:
            reasoning += self._build_significance_note(p_value, results, arbitration_entry, best_model_name)
        if arbitration_entry:
            # 1-p_value would read as ~0% here by construction (Wilcoxon found no
            # significant difference — that's the whole reason arbitration fired).
            # That's confidence-that-this-model-beats-the-runner-up, not confidence
            # in the final decision. Instead, measure how clearly the tied top-2
            # separate from the REST of the field (3rd place onward): arbitrating
            # between two models that are both far ahead of every other candidate
            # is a low-risk coin-flip, arbitrating in a field where five models are
            # all bunched together is not — a real, computed distinction the flat
            # p-value can't see.
            confidence = self._confidence_from_field_separation(results, best_result)
        elif p_value is not None:
            confidence = float(min(0.99, 1 - p_value))
        else:
            confidence = self._confidence_from_margin(results)
        selection_decision = self.decide(
            decision=f"Selected '{best_model_name}' as the final model",
            reasoning=reasoning,
            confidence=confidence,
        )

        self.logger.info(
            "Calling LLM for model selection summary (model=%s)",
            getattr(self.llm, "model_name", "unknown"),
        )
        # If arbitration already broke a statistical tie, the narrative call is told so
        # explicitly — otherwise it independently "explains" a tie as if one candidate
        # clearly outscored the other, which contradicts the arbitration log.
        other_result = results[1] if best_result is results[0] else results[0]
        tie_note = (
            f"Note: '{best_model_name}' and '{other_result['model_name']}' were statistically tied "
            f"(Wilcoxon p={p_value:.3f}); '{best_model_name}' was selected because an LLM arbitration "
            "recommended it on interpretability/computational-cost grounds, not because of a real "
            "performance gap."
        ) if arbitration_entry else None
        fallback_summary = self._build_fallback_summary(best_model_name, best_result, results, scoring, tie_note)
        model_selection_summary = self.llm.generate_summary(
            system_prompt=(
                "You are a senior machine learning engineer. Explain in 4-6 concise bullet points "
                "why a particular model was selected over the alternatives, using the comparison "
                "results and the confidence in the choice. Be clear and non-technical. If told the "
                "top candidates were statistically tied, say so plainly instead of inventing a "
                "performance-based distinction that isn't there."
            ),
            user_prompt=self._build_llm_prompt(best_model_name, best_result, results, scoring, tie_note),
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
            "llm_arbitration_log": [arbitration_entry] if arbitration_entry else [],
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
                    "fold_scores": [float(s) for s in scores],
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

    def _build_llm_prompt(
        self, best_model_name: str, best_result: Dict[str, Any], all_results: List[Dict[str, Any]],
        scoring: str, tie_note: str | None = None,
    ) -> str:
        metric_label = METRIC_LABELS.get(scoring, scoring)
        top_rows = []
        for result in sorted(all_results, key=lambda r: r["mean_score"], reverse=True)[:5]:
            top_rows.append(
                f"- {result['model_name']}: mean {metric_label}={result['mean_score']:.4f}, std={result['std_score']:.4f}"
            )
        prompt = (
            f"Best model: {best_model_name}\n"
            f"Winning score: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}\n"
            f"Metric: {metric_label}\n\n"
            f"Candidate comparison:\n{chr(10).join(top_rows)}\n\n"
        )
        if tie_note:
            prompt += f"{tie_note}\n\n"
        prompt += "Explain why the selected model is preferable and mention any caveats about close competition."
        return prompt

    def _build_fallback_summary(
        self, best_model_name: str, best_result: Dict[str, Any], all_results: List[Dict[str, Any]],
        scoring: str, tie_note: str | None = None,
    ) -> str:
        metric_label = METRIC_LABELS.get(scoring, scoring)
        others = [r for r in all_results if r["model_name"] != best_model_name]
        comparison = ", ".join(f"{r['model_name']}={r['mean_score']:.4f}" for r in others[:4]) or "no alternatives recorded"
        method = "a single fit on the full dataset" if scoring == "silhouette" else "cross-validation"
        bullets = [
            f"* **Selected Model**: `{best_model_name}` was chosen because it achieved the best mean {metric_label} "
            f"({best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}) via {method}.",
            f"* **Comparison**: Competing models scored: {comparison}.",
        ]
        if tie_note:
            bullets.append(f"* **Statistical Tie**: {tie_note}")
        bullets.append("* **Caution**: If the margin is small, the choice should be validated further on fresh data before deployment.")
        return "\n".join(bullets)

    def _confidence_from_margin(self, results: List[Dict[str, Any]]) -> float:
        """
        Fallback confidence for comparisons the Wilcoxon test in
        _compare_top_two() cannot cover — a single candidate, or the
        unsupervised branch, which fits once instead of via k-fold CV and so
        has no per-fold scores to run a signed-rank test on. Reflects how
        clearly the winner beat the runner-up, not just the raw score: a
        close race between the top 2 models means the choice is less
        "confident" even if the winning score is high.
        """
        if len(results) < 2:
            return 0.9
        best_score, second_score = results[0]["mean_score"], results[1]["mean_score"]
        margin = abs(best_score - second_score)
        # Normalize: a margin >= 0.05 (5 accuracy points, or 0.05 RMSE units) is
        # treated as a clearly confident win.
        return float(min(0.95, 0.5 + margin * 10))

    def _confidence_from_field_separation(
        self, results: List[Dict[str, Any]], best_result: Dict[str, Any]
    ) -> float:
        """
        Confidence for an arbitrated tie, grounded in how far the tied top-2
        (results[0] and results[1] — the pair Wilcoxon found indistinguishable)
        lead the rest of the field, rather than a flat constant. A tied pair
        that's still clearly ahead of every other candidate means arbitrating
        between them is low-risk; a tie inside a tightly bunched field of many
        similarly-scoring candidates is a much shakier ground to arbitrate on,
        and this margin — unlike the Wilcoxon p-value, which is 0 by
        construction for the tied pair itself — actually captures that.
        """
        tied_pair_names = {results[0]["model_name"], results[1]["model_name"]}
        others = [r for r in results if r["model_name"] not in tied_pair_names]
        if not others:
            return ARBITRATION_CONFIDENCE_FLOOR
        third_place_score = max(r["mean_score"] for r in others)
        margin = best_result["mean_score"] - third_place_score
        # Same scaling as _confidence_from_margin: a margin >= 0.05 (5 accuracy
        # points, or 0.05 RMSE units) counts as a clearly separated tied pair.
        return float(max(0.5, min(0.95, 0.5 + margin * 10)))

    def _compare_top_two(
        self, results: List[Dict[str, Any]]
    ) -> tuple[Optional[float], Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Statistically tests whether the winning model actually beat the
        runner-up, rather than trusting a mean-score ranking that CV noise
        alone can produce. Only applicable when both models have per-fold
        scores (the supervised branch's k-fold CV) — the unsupervised branch
        fits once on the full dataset and has no folds to compare.

        If the test finds no significant difference (p > 0.05), arbitration
        is escalated to the LLM on interpretability/cost grounds, since
        performance alone can no longer break the tie — and the LLM's
        recommendation actually decides the winner returned here. Asking for
        arbitration and then ignoring its answer would leave the pipeline's
        choice as an arbitrary artifact of dict insertion order, while a
        separate narrative LLM call independently invents its own
        justification for that same arbitrary pick — exactly the kind of
        self-contradictory explanation this project exists to avoid.

        Always returns a winner (results[0] when no arbitration happened or
        it couldn't be parsed) alongside the p-value and arbitration log
        entry, both of which are None when Wilcoxon wasn't applicable.
        """
        if len(results) < 2:
            return None, None, results[0] if results else None
        best_result, second_result = results[0], results[1]
        best_folds = best_result.get("fold_scores")
        second_folds = second_result.get("fold_scores")
        if not best_folds or not second_folds or len(best_folds) < 2 or len(second_folds) < 2:
            return None, None, best_result

        try:
            _, p_value = wilcoxon(best_folds, second_folds)
            p_value = float(p_value)
        except ValueError:
            # wilcoxon raises when all paired differences are zero (identical
            # scores on every fold) — that IS "no significant difference".
            p_value = 1.0

        arbitration_entry = None
        winner = best_result
        if p_value > 0.05:
            self.logger.info(
                "Wilcoxon p=%.4f between '%s' and '%s' — no significant difference, escalating to LLM.",
                p_value, best_result["model_name"], second_result["model_name"],
            )
            arbitration_response = self.llm.generate_summary(
                system_prompt=(
                    "You are a senior machine learning engineer. Two candidate models achieved "
                    "statistically indistinguishable cross-validation scores (Wilcoxon signed-rank "
                    "test, p > 0.05) — performance cannot break the tie. Arbitrate between them "
                    "using ONLY interpretability and computational cost. Answer in 2-4 concise "
                    "sentences with a clear recommendation, using the model's real name (not "
                    "'Model A'/'Model B') when you state it, and a justification."
                ),
                user_prompt=(
                    f"Model A — {best_result['model_name']} (mean {best_result['scoring_metric']}="
                    f"{best_result['mean_score']:.4f})\n"
                    f"Model B — {second_result['model_name']} (mean {second_result['scoring_metric']}="
                    f"{second_result['mean_score']:.4f})\n"
                    f"Wilcoxon p-value: {p_value:.4f}\n\n"
                    "Which would you recommend and why, based on interpretability and cost alone?"
                ),
                fallback_message=(
                    f"No statistically significant difference between '{best_result['model_name']}' "
                    f"and '{second_result['model_name']}' (p={p_value:.3f}). Manual review of "
                    "interpretability and computational cost is recommended; defaulting to "
                    f"'{best_result['model_name']}' — recommend {best_result['model_name']}."
                ),
            )
            recommended_name = self._parse_arbitration_recommendation(
                arbitration_response, best_result["model_name"], second_result["model_name"]
            )
            if recommended_name == second_result["model_name"]:
                winner = second_result
                self.logger.info(
                    "LLM arbitration recommended '%s' over the tie-break default — switching selection.",
                    recommended_name,
                )
            elif recommended_name is None:
                self.logger.warning(
                    "LLM arbitration response did not clearly recommend either model; keeping '%s'.",
                    best_result["model_name"],
                )
            arbitration_entry = {
                "agent_name": self.name,
                "trigger": f"Wilcoxon p={p_value:.3f} (no significant difference)",
                "llm_arbitration": arbitration_response,
            }
        return p_value, arbitration_entry, winner

    def _parse_arbitration_recommendation(self, response: str, name_a: str, name_b: str) -> Optional[str]:
        """
        Finds which of the two candidate names the arbitration response
        actually recommends, by locating whichever name appears first after
        the word "recommend" — robust to free-form LLM phrasing (e.g. "I'd
        recommend Model B: RandomForest") instead of requiring a strict
        output format the LLM might not follow. Returns None if neither name
        can be found, so the caller can fall back to the original tie-break
        instead of guessing.
        """
        if not response:
            return None
        lowered = response.lower()
        idx = lowered.find("recommend")
        search_region = lowered[idx:] if idx != -1 else lowered
        pos_a = search_region.find(name_a.lower())
        pos_b = search_region.find(name_b.lower())
        if pos_a == -1 and pos_b == -1:
            return None
        if pos_b != -1 and (pos_a == -1 or pos_b < pos_a):
            return name_b
        return name_a

    def _build_significance_note(
        self, p_value: float, results: List[Dict[str, Any]],
        arbitration_entry: Optional[Dict[str, Any]], final_model_name: str,
    ) -> str:
        other_name = results[1]["model_name"] if results[0]["model_name"] == final_model_name else results[0]["model_name"]
        if arbitration_entry:
            return (
                f" A Wilcoxon signed-rank test on the per-fold scores found no statistically significant "
                f"difference between '{final_model_name}' and '{other_name}' (p={p_value:.3f}), so the tie "
                f"was broken by LLM arbitration on interpretability/cost, which recommended "
                f"'{final_model_name}': {arbitration_entry['llm_arbitration']}"
            )
        return (
            f" A Wilcoxon signed-rank test on the per-fold scores confirms this winner over the "
            f"runner-up ('{other_name}') is statistically significant (p={p_value:.3f})."
        )
