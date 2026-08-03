"""
Hyperparameter Optimization Agent Module
==========================================

Runs Optuna to tune ONLY the model selected by the Model Selection Agent.
It does not compare algorithms anymore — that decision was already made.
This agent searches within that single algorithm's hyperparameter space,
then retrains and saves the final tuned model.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState, AgentDecision, UNSUPERVISED_PROBLEM_TYPES

MODELS_DIR = os.path.join("data", "outputs", "models")
RANDOM_STATE = 42

N_TRIALS = 20          # kept low on purpose: enough to improve on defaults
                        # without risking the Streamlit 2-minute UI block.
TIMEOUT_SECONDS = 60    # hard safety cap for the FIRST optimization pass.
CV_FOLDS_DURING_SEARCH = 3  # cheaper than the 5 folds used in Model Selection,
                            # since it's repeated N_TRIALS times.

# Evaluator-optimizer retry: if the first pass barely beats the untuned
# baseline, one — and only one — retry is run with a widened search space,
# on the theory that the initial bounds may simply have been too narrow.
MIN_IMPROVEMENT_THRESHOLD = 0.01  # 1% minimum expected gain over baseline.
SEARCH_SPACE_WIDEN_MULTIPLIER = 1.3  # +30% on numeric range upper bounds.
# Total time budget SHARED across both passes (not 60+60=120) — a retry
# must not double the worst-case pipeline runtime.
TOTAL_TIMEOUT_SECONDS = 90

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep Streamlit console clean

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


class HyperparameterAgent(BaseAgent):
    """
    Agent responsible for tuning the hyperparameters of the model selected
    by the Model Selection Agent, using Optuna.
    """

    def __init__(self) -> None:
        super().__init__(name="hyperparameter_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        data_path = state.get("preprocessed_data_path")
        target_column = state.get("target_column")
        problem_type = state.get("problem_type")
        model_name = state.get("selected_model_name")
        baseline_score = self._get_baseline_score(state, model_name)
        is_unsupervised = problem_type in UNSUPERVISED_PROBLEM_TYPES

        if not all([data_path, problem_type, model_name]) or (not is_unsupervised and not target_column):
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'problem_type', 'selected_model_name', or 'target_column' "
                "(required for supervised problem types)."
            )

        # LinearRegression has essentially no meaningful hyperparameters to tune.
        if model_name == "LinearRegression":
            df = pd.read_csv(data_path)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            # Fit only on the same training split ModelSelectionAgent/
            # EvaluationAgent use — fitting on the full dataframe here would
            # leak the held-out test rows into training and inflate the
            # metrics EvaluationAgent reports afterwards.
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
            final_model = LinearRegression().fit(X_train, y_train)
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, "selected_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(final_model, f)
            return {
                "best_hyperparameters": {},
                "optimization_summary": {"skipped": True, "reason": "LinearRegression has no tunable hyperparameters."},
                "model_path": model_path,
                "agent_decisions": [self.decide(
                    decision="Skipped hyperparameter optimization",
                    reasoning="'LinearRegression' has no hyperparameters worth tuning via Optuna.",
                    confidence=1.0,
                )],
                "logs": [self.log("Skipped optimization: LinearRegression has no tunable hyperparameters.")],
            }

        df = pd.read_csv(data_path)

        if is_unsupervised:
            # No target, no train/test split: mirrors ModelSelectionAgent's
            # _score_candidates_unsupervised, which fits on the full
            # preprocessed dataset for the same reason (DBSCAN/LocalOutlier
            # Factor have no reusable out-of-sample .predict(), so there is
            # no k-fold-CV-idiomatic way to hold out a test split here).
            X_train = df.copy()
            scoring = "silhouette"

            def make_objective(search_space_multiplier: float):
                def objective(trial: optuna.Trial) -> float:
                    model = self._build_model_from_trial(
                        model_name, problem_type, trial, search_space_multiplier
                    )
                    labels = model.fit_predict(X_train)
                    if len(set(labels)) < 2:
                        return -1.0
                    return float(silhouette_score(X_train, labels))
                return objective
        else:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
            try:
                X_train, _, y_train, _ = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
                )
            except ValueError:
                # Every class has >=2 rows but sklearn ALSO requires
                # test_size >= n_classes for a stratified split — on a very
                # small dataset that second condition can still fail. Fall
                # back to an unstratified split rather than crashing.
                X_train, _, y_train, _ = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
                )

            cv = self._build_cv_splitter(problem_type, y_train)
            scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"

            def make_objective(search_space_multiplier: float):
                def objective(trial: optuna.Trial) -> float:
                    model = self._build_model_from_trial(
                        model_name, problem_type, trial, search_space_multiplier
                    )
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    return float(np.mean(scores))
                return objective

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
        search_start_time = time.time()
        study.optimize(make_objective(1.0), n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=False)

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            # Every trial errored (e.g. an unsupported hyperparameter
            # combination) or the 60s timeout hit before any trial finished —
            # study.best_params/.best_value would raise. Degrade to the
            # untuned baseline instead of crashing the pipeline.
            self.logger.warning(
                "No Optuna trial completed for '%s'; keeping untuned baseline.", model_name
            )
            baseline_model = self._build_model_from_params(model_name, problem_type, {})
            if is_unsupervised:
                baseline_model.fit(X_train)
            else:
                baseline_model.fit(X_train, y_train)
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, "selected_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(baseline_model, f)
            return {
                "best_hyperparameters": {},
                "optimization_summary": {
                    "skipped": True,
                    "reason": "No Optuna trial completed successfully; kept the untuned baseline model.",
                },
                "model_path": model_path,
                "agent_decisions": [self.decide(
                    decision=f"Kept untuned baseline for '{model_name}'",
                    reasoning="Every Optuna trial failed or the search timed out before any trial "
                    "completed, so the already-known baseline configuration was kept instead.",
                    confidence=0.5,
                )],
                "logs": [self.log(
                    f"Hyperparameter search for '{model_name}' produced no completed trials; kept baseline."
                )],
            }

        best_params = study.best_params
        best_score = study.best_value
        initial_best_score = best_score

        # Evaluator-optimizer retry: a single bounded retry with a widened
        # search space when the first pass barely improved on the baseline.
        # Never loops more than once, and shares a single 90s budget across
        # both passes (see TOTAL_TIMEOUT_SECONDS) instead of doubling it.
        retry_needed = (
            baseline_score is not None
            and (best_score - baseline_score) < MIN_IMPROVEMENT_THRESHOLD
        )
        retried = False
        retry_skipped_reason = None
        if retry_needed:
            elapsed = time.time() - search_start_time
            remaining_budget = TOTAL_TIMEOUT_SECONDS - elapsed
            if remaining_budget > 1.0:
                retried = True
                self.logger.info(
                    "Initial tuning for '%s' improved score by only %.4f (< %.2f threshold) — "
                    "retrying once with a widened search space (remaining budget=%.1fs).",
                    model_name, best_score - baseline_score, MIN_IMPROVEMENT_THRESHOLD, remaining_budget,
                )
                study.optimize(
                    make_objective(SEARCH_SPACE_WIDEN_MULTIPLIER),
                    n_trials=N_TRIALS,
                    timeout=remaining_budget,
                    show_progress_bar=False,
                )
                retry_completed_trials = [
                    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if retry_completed_trials:
                    # study.best_params/.best_value reflect the best trial across
                    # BOTH passes (Optuna tracks this per-study, not per-optimize()
                    # call) — this is the "results of the last pass performed"
                    # since a widened space is a superset of the first pass's.
                    best_params = study.best_params
                    best_score = study.best_value
            else:
                retry_skipped_reason = (
                    f"time budget exhausted ({elapsed:.1f}s of {TOTAL_TIMEOUT_SECONDS}s already used)"
                )

        # Retrain the final model on the full training split with the best params found.
        final_model = self._build_model_from_params(model_name, problem_type, best_params)
        if is_unsupervised:
            final_model.fit(X_train)
        else:
            final_model.fit(X_train, y_train)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "selected_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)

        improvement = best_score - baseline_score if baseline_score is not None else None

        optimization_summary = {
            "n_trials_run": len(study.trials),
            "best_score": best_score,
            "baseline_score": baseline_score,
            "improvement": improvement,
            "scoring_metric": scoring,
            "retried": retried,
        }

        retry_decision = self._build_retry_decision(
            model_name, retried, retry_needed, retry_skipped_reason,
            initial_best_score, best_score, baseline_score,
        )

        return {
            "best_hyperparameters": best_params,
            "optimization_summary": optimization_summary,
            "model_path": model_path,
            "agent_decisions": [
                self.decide(
                    decision=f"Tuned '{model_name}' hyperparameters: {best_params}",
                    reasoning=self._build_reasoning(model_name, study, baseline_score, scoring),
                    confidence=self._confidence_from_improvement(improvement),
                ),
                retry_decision,
            ],
            "logs": [self.log(
                f"Ran {len(study.trials)} Optuna trial(s) on '{model_name}'"
                f"{' (including one widened-search-space retry)' if retried else ''}. "
                f"Best {scoring}={best_score:.4f}."
            )],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _confidence_from_improvement(self, improvement: float | None) -> float:
        """
        Mirrors ModelSelectionAgent._confidence_from_margin: confidence
        reflects how clearly tuning improved on the untuned baseline, not a
        fixed value. Without a baseline to compare against (e.g. no CV result
        recorded for this model), there is nothing to ground the number in.
        """
        if improvement is None:
            return 0.6
        return float(max(0.4, min(0.95, 0.5 + max(improvement, 0) * 10)))

    def _build_retry_decision(
        self,
        model_name: str,
        retried: bool,
        retry_needed: bool,
        retry_skipped_reason: str | None,
        initial_best_score: float,
        final_best_score: float,
        baseline_score: float | None,
    ) -> AgentDecision:
        """
        Documents whether the evaluator-optimizer retry fired, per the bounded
        retry loop implemented in execute(). Kept separate from the main
        tuning AgentDecision so the retry mechanism itself — a project-level
        reliability behavior, not a per-model tuning outcome — is auditable
        on its own in the report.
        """
        if retried:
            final_gain = final_best_score - initial_best_score
            gain_text = f"a further gain of {final_gain:+.4f}" if final_gain > 0 else "no further improvement"
            return self.decide(
                decision=f"Retried hyperparameter search for '{model_name}' with a widened search space",
                reasoning=(
                    f"Initial tuning improved score by only {(initial_best_score - baseline_score):+.4f}, "
                    f"below the {MIN_IMPROVEMENT_THRESHOLD:.0%} threshold — retried once with numeric "
                    f"search-space bounds widened by {int((SEARCH_SPACE_WIDEN_MULTIPLIER - 1) * 100)}%. "
                    f"Retry produced {gain_text} (best score: {final_best_score:.4f})."
                ),
                confidence=self._confidence_from_improvement(
                    final_best_score - baseline_score if baseline_score is not None else None
                ),
            )
        if retry_needed and retry_skipped_reason:
            return self.decide(
                decision=f"Retry skipped for '{model_name}' despite low improvement",
                reasoning=(
                    f"Initial tuning improved score by only {(initial_best_score - baseline_score):+.4f}, "
                    f"below the {MIN_IMPROVEMENT_THRESHOLD:.0%} threshold, but the retry was skipped: "
                    f"{retry_skipped_reason}."
                ),
                confidence=1.0,
            )
        reasoning = (
            f"Initial tuning improvement ({(initial_best_score - baseline_score):+.4f}) met or exceeded "
            f"the {MIN_IMPROVEMENT_THRESHOLD:.0%} threshold, so no retry was needed."
            if baseline_score is not None
            else "No baseline score was available for comparison, so the retry threshold could not be evaluated."
        )
        return self.decide(
            decision=f"No retry needed for '{model_name}' hyperparameter search",
            reasoning=reasoning,
            confidence=1.0,
        )

    def _get_baseline_score(self, state: EMADSState, model_name: str) -> float | None:
        """Reads the model's un-tuned CV score from Model Selection results, for comparison."""
        for result in state.get("candidate_models_results") or []:
            if result.get("model_name") == model_name:
                return result.get("mean_score")
        return None

    def _build_cv_splitter(self, problem_type: str, y_train: pd.Series):
        n_samples = len(y_train)
        if problem_type == "classification":
            min_class_count = int(y_train.value_counts().min())
            # StratifiedKFold requires at least n_splits samples in every class.
            # When min_class_count < 2, stratification is impossible — fall back
            # to plain KFold so the pipeline doesn't crash on ID-like targets.
            if min_class_count >= 2:
                n_folds = max(2, min(CV_FOLDS_DURING_SEARCH, min_class_count))
                return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            # Fall through to KFold when stratification is impossible.
        n_splits = max(2, min(CV_FOLDS_DURING_SEARCH, n_samples))
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    @staticmethod
    def _widen_int_high(low: int, high: int, multiplier: float, step: int = 1) -> int:
        """
        Scales an Optuna suggest_int's upper bound by `multiplier`, keeping
        the lower bound fixed — widening a search space means "allow more
        capacity" (deeper trees, more estimators), not "also allow less".
        When `step` > 1, Optuna's IntDistribution requires (high - low) to be
        a multiple of step, so the widened bound is rounded to the nearest
        valid step above the naive scaled value instead of raising.
        """
        widened = int(round(high * multiplier))
        if step > 1:
            k = max(1, round((widened - low) / step))
            widened = low + k * step
        return max(widened, high)

    def _build_model_from_trial(
        self, model_name: str, problem_type: str, trial: optuna.Trial,
        search_space_multiplier: float = 1.0,
    ):
        """
        Defines the Optuna search space for each supported algorithm.

        `search_space_multiplier` widens numeric upper bounds (max_depth,
        n_estimators, min_samples_*, n_clusters, n_neighbors, eps) for the
        single bounded retry in execute() when the first pass barely beats
        baseline — left at 1.0 (untouched) for the initial pass. Bounds that
        are inherently capped by their own domain (contamination and
        subsample, both in (0, 1]) are not widened, since growing them past
        that range wouldn't be valid.
        """
        m = search_space_multiplier
        is_clf = problem_type == "classification"

        if model_name == "LogisticRegression":
            params = {"C": trial.suggest_float("C", 1e-3, 10.0 * m, log=True)}
            return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)

        if model_name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, self._widen_int_high(2, 20, m)),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, self._widen_int_high(2, 20, m)),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, self._widen_int_high(1, 10, m)),
            }
            cls = DecisionTreeClassifier if is_clf else DecisionTreeRegressor
            return cls(random_state=RANDOM_STATE, **params)

        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, self._widen_int_high(100, 400, m, step=50), step=50
                ),
                "max_depth": trial.suggest_int("max_depth", 3, self._widen_int_high(3, 20, m)),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, self._widen_int_high(2, 15, m)),
            }
            cls = RandomForestClassifier if is_clf else RandomForestRegressor
            return cls(random_state=RANDOM_STATE, n_jobs=-1, **params)

        if model_name == "XGBoost" and _HAS_XGBOOST:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, self._widen_int_high(100, 400, m, step=50), step=50
                ),
                "max_depth": trial.suggest_int("max_depth", 3, self._widen_int_high(3, 10, m)),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            cls = XGBClassifier if is_clf else XGBRegressor
            extra = {"eval_metric": "logloss", "verbosity": 0} if is_clf else {"verbosity": 0}
            return cls(random_state=RANDOM_STATE, **params, **extra)

        if model_name == "LightGBM" and _HAS_LIGHTGBM:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, self._widen_int_high(100, 400, m, step=50), step=50
                ),
                "max_depth": trial.suggest_int("max_depth", 3, self._widen_int_high(3, 12, m)),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }
            cls = LGBMClassifier if is_clf else LGBMRegressor
            return cls(random_state=RANDOM_STATE, verbosity=-1, **params)

        if model_name == "KMeans":
            n_clusters = trial.suggest_int("n_clusters", 2, self._widen_int_high(2, 10, m))
            return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)

        if model_name == "DBSCAN":
            params = {
                "eps": trial.suggest_float("eps", 0.1, 2.0 * m),
                "min_samples": trial.suggest_int("min_samples", 3, self._widen_int_high(3, 15, m)),
            }
            return DBSCAN(**params)

        if model_name == "IsolationForest":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 50, self._widen_int_high(50, 300, m, step=50), step=50
                ),
                "contamination": trial.suggest_float("contamination", 0.01, 0.3),
            }
            return IsolationForest(random_state=RANDOM_STATE, **params)

        if model_name == "LocalOutlierFactor":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 5, self._widen_int_high(5, 50, m)),
                "contamination": trial.suggest_float("contamination", 0.01, 0.3),
            }
            return LocalOutlierFactor(novelty=False, **params)

        raise ValueError(f"No hyperparameter search space defined for model '{model_name}'.")

    def _build_model_from_params(
        self, model_name: str, problem_type: str, params: dict,
        search_space_multiplier: float = 1.0,
    ):
        """
        Rebuilds the final model from Optuna's best_params dict (post-search).

        `search_space_multiplier` is accepted only for signature symmetry
        with _build_model_from_trial — it's unused here because params are
        already concrete, chosen values, not ranges to widen.
        """
        is_clf = problem_type == "classification"

        if model_name == "LogisticRegression":
            return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)
        if model_name == "DecisionTree":
            cls = DecisionTreeClassifier if is_clf else DecisionTreeRegressor
            return cls(random_state=RANDOM_STATE, **params)
        if model_name == "RandomForest":
            cls = RandomForestClassifier if is_clf else RandomForestRegressor
            return cls(random_state=RANDOM_STATE, n_jobs=-1, **params)
        if model_name == "XGBoost" and _HAS_XGBOOST:
            cls = XGBClassifier if is_clf else XGBRegressor
            extra = {"eval_metric": "logloss", "verbosity": 0} if is_clf else {"verbosity": 0}
            return cls(random_state=RANDOM_STATE, **params, **extra)
        if model_name == "LightGBM" and _HAS_LIGHTGBM:
            cls = LGBMClassifier if is_clf else LGBMRegressor
            return cls(random_state=RANDOM_STATE, verbosity=-1, **params)
        if model_name == "KMeans":
            return KMeans(random_state=RANDOM_STATE, n_init=10, **params)
        if model_name == "DBSCAN":
            return DBSCAN(**params)
        if model_name == "IsolationForest":
            return IsolationForest(random_state=RANDOM_STATE, **params)
        if model_name == "LocalOutlierFactor":
            return LocalOutlierFactor(novelty=False, **params)
        raise ValueError(f"Cannot rebuild model '{model_name}' from params.")

    def _build_reasoning(self, model_name: str, study: optuna.Study, baseline_score, scoring: str) -> str:
        if baseline_score is None:
            return (
                f"Ran {len(study.trials)} Optuna trials on '{model_name}'. "
                f"Best {scoring}={study.best_value:.4f}."
            )
        gain = study.best_value - baseline_score
        direction = "improved" if gain > 0 else "did not improve over"
        return (
            f"Ran {len(study.trials)} Optuna trials on '{model_name}'. "
            f"Tuning {direction} the default configuration: "
            f"{baseline_score:.4f} → {study.best_value:.4f} ({gain:+.4f})."
        )
