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
        if p_value is not None:
            # confidence = 1 - p_value in all cases, including after LLM
            # arbitration. A previous version substituted a margin-based
            # heuristic here (see git history) on the theory that 1-p_value
            # is uninformative for an arbitrated tie — but that heuristic was
            # not a recognized statistical measure (not a rank-biserial
            # effect size or any other documented quantity), just an
            # unfounded score-margin invented to look more confident than
            # the honest number. A high p-value IS genuine statistical
            # uncertainty, arbitration or not, and confidence should say so
            # plainly rather than dress it up.
            confidence = float(min(0.99, 1 - p_value))
        else:
            confidence = self._confidence_from_margin(results)
        selection_decision = self.decide(
            decision=f"Selected '{best_model_name}' as the final model",
            reasoning=reasoning,
            confidence=confidence,
        )

        # No separate, unconditional narrative call here anymore (see git
        # history — there used to be a 2nd LLM call on every run, producing a
        # confidently-worded "rationale" that never mentioned the confidence
        # sitting right next to it in the Decisions Log whenever the pick was
        # actually an arbitrated statistical tie). The ONLY LLM narrative for
        # model selection is now the conditional arbitration call already made
        # inside _compare_top_two() when a Wilcoxon tie required one — reused
        # here, not recomputed, so there is exactly one LLM text describing
        # this decision, consistent with its own confidence score.
        model_selection_summary = arbitration_entry["llm_arbitration"] if arbitration_entry else None

        self.logger.info(
            "Model selection complete: selected_model=%s score=%.4f",
            best_model_name, best_result["mean_score"],
        )
        logs = [self.log(
            f"Compared {len(candidates)} model(s) via "
            f"{'a single unsupervised fit' if is_unsupervised else f'{CV_FOLDS}-fold CV'}. "
            f"Selected '{best_model_name}' (score={best_result['mean_score']:.4f})."
        )]
        if arbitration_entry and arbitration_entry.get("parsed_recommendation") is None:
            # Previously only self.logger.warning() saw this (disk log only) — it
            # never reached state["logs"], so the UI/PDF report showed the tie-break
            # as if the LLM had cleanly recommended the winner. See _build_significance_note.
            logs.append(self.log(
                "LLM arbitration response could not be parsed into a clear model "
                f"recommendation — fell back to the higher-CV-score model, '{best_model_name}'."
            ))
        return {
            "candidate_models_results": results,
            "selected_model_name": best_model_name,
            "model_selection_summary": model_selection_summary,
            "agent_decisions": [selection_decision],
            "llm_arbitration_log": [arbitration_entry] if arbitration_entry else [],
            "logs": logs,
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
                # This response now directly decides selected_model_name (see
                # winner-switch below), not just narrative text — temperature=0
                # minimizes run-to-run variance in an actual model-selection
                # decision, unlike the default 0.2 used for prose-only summaries.
                temperature=0,
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
                # Recorded so callers (execute()'s significance note, tests) can
                # tell a genuine parsed recommendation apart from the
                # parse-failure fallback, instead of both looking identical
                # once `winner` has already collapsed both cases to a model name.
                "parsed_recommendation": recommended_name,
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
            if arbitration_entry.get("parsed_recommendation") is None:
                # _compare_top_two() couldn't find either candidate's name in the
                # LLM's response, so it fell back to the higher-CV-score model —
                # saying the LLM "recommended" final_model_name here would be false;
                # the LLM's answer was never actually followed.
                return (
                    f" A Wilcoxon signed-rank test on the per-fold scores found no statistically significant "
                    f"difference between '{final_model_name}' and '{other_name}' (p={p_value:.3f}). LLM "
                    f"arbitration was attempted, but its response could not be parsed into a clear "
                    f"recommendation for either candidate, so the tie was broken deterministically by "
                    f"keeping the higher-CV-score model, '{final_model_name}'. The unparsed LLM response is "
                    f"included below for context only — it did not determine this selection: "
                    f"{arbitration_entry['llm_arbitration']} "
                    f"This low confidence reflects genuine statistical uncertainty — a tie-break was needed "
                    f"precisely because the models could not be statistically distinguished."
                )
            return (
                f" A Wilcoxon signed-rank test on the per-fold scores found no statistically significant "
                f"difference between '{final_model_name}' and '{other_name}' (p={p_value:.3f}), so the tie "
                f"was broken by LLM arbitration on interpretability/cost, which recommended "
                f"'{final_model_name}': {arbitration_entry['llm_arbitration']} "
                f"This low confidence reflects genuine statistical uncertainty — a tie-break was needed "
                f"precisely because the models could not be statistically distinguished. The LLM "
                f"arbitration above explains the tie-break reasoning."
            )
        return (
            f" A Wilcoxon signed-rank test on the per-fold scores confirms this winner over the "
            f"runner-up ('{other_name}') is statistically significant (p={p_value:.3f})."
        )
