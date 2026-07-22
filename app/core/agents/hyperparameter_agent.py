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
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState

MODELS_DIR = os.path.join("data", "outputs", "models")
RANDOM_STATE = 42

N_TRIALS = 20          # kept low on purpose: enough to improve on defaults
                        # without risking the Streamlit 2-minute UI block.
TIMEOUT_SECONDS = 60    # hard safety cap regardless of n_trials.
CV_FOLDS_DURING_SEARCH = 3  # cheaper than the 5 folds used in Model Selection,
                            # since it's repeated N_TRIALS times.

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

        if not all([data_path, target_column, problem_type, model_name]):
            raise ValueError(
                "Missing required state inputs: 'preprocessed_data_path', "
                "'target_column', 'problem_type' or 'selected_model_name'."
            )

        # LinearRegression has essentially no meaningful hyperparameters to tune.
        if model_name == "LinearRegression":
            return {
                "best_hyperparameters": {},
                "optimization_summary": {"skipped": True, "reason": "LinearRegression has no tunable hyperparameters."},
                "agent_decisions": [self.decide(
                    decision="Skipped hyperparameter optimization",
                    reasoning="'LinearRegression' has no hyperparameters worth tuning via Optuna.",
                    confidence=1.0,
                )],
                "logs": [self.log("Skipped optimization: LinearRegression has no tunable hyperparameters.")],
            }

        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_y = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
        )

        cv = self._build_cv_splitter(problem_type, y_train)
        scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"

        def objective(trial: optuna.Trial) -> float:
            model = self._build_model_from_trial(model_name, problem_type, trial)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value

        # Retrain the final model on the full training split with the best params found.
        final_model = self._build_model_from_params(model_name, problem_type, best_params)
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
        }

        return {
            "best_hyperparameters": best_params,
            "optimization_summary": optimization_summary,
            "model_path": model_path,
            "agent_decisions": [self.decide(
                decision=f"Tuned '{model_name}' hyperparameters: {best_params}",
                reasoning=self._build_reasoning(model_name, study, baseline_score, scoring),
                confidence=0.75,
            )],
            "logs": [self.log(
                f"Ran {len(study.trials)} Optuna trial(s) on '{model_name}'. "
                f"Best {scoring}={best_score:.4f}."
            )],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

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

    def _build_model_from_trial(self, model_name: str, problem_type: str, trial: optuna.Trial):
        """Defines the Optuna search space for each supported algorithm."""
        is_clf = problem_type == "classification"

        if model_name == "LogisticRegression":
            params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
            return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)

        if model_name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            cls = DecisionTreeClassifier if is_clf else DecisionTreeRegressor
            return cls(random_state=RANDOM_STATE, **params)

        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            }
            cls = RandomForestClassifier if is_clf else RandomForestRegressor
            return cls(random_state=RANDOM_STATE, n_jobs=-1, **params)

        if model_name == "XGBoost" and _HAS_XGBOOST:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            cls = XGBClassifier if is_clf else XGBRegressor
            extra = {"eval_metric": "logloss", "verbosity": 0} if is_clf else {"verbosity": 0}
            return cls(random_state=RANDOM_STATE, **params, **extra)

        if model_name == "LightGBM" and _HAS_LIGHTGBM:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }
            cls = LGBMClassifier if is_clf else LGBMRegressor
            return cls(random_state=RANDOM_STATE, verbosity=-1, **params)

        raise ValueError(f"No hyperparameter search space defined for model '{model_name}'.")

    def _build_model_from_params(self, model_name: str, problem_type: str, params: dict):
        """Rebuilds the final model from Optuna's best_params dict (post-search)."""
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