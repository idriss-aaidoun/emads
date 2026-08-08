"""
XGBoost Non-Contiguous Class Label Tests
============================================

Regression tests for a confirmed bug: XGBoost's sklearn API requires class
labels to be contiguous integers starting at 0, raising "Invalid classes
inferred from unique values of `y`" otherwise. Real targets routinely violate
this (ratings 1-5, Likert scales, IDs starting at 1) — the dataset that first
revealed the bug (ambiguous_survey_v2.csv) had a 'satisfaction_rating' target
with values 1-5. Before the fix, ModelSelectionAgent's try/except silently
caught the error and reported XGBoost as "Failed" in every such comparison.

ModelSelectionAgent.XGBClassifierWithLabelEncoding and HyperparameterAgent.
XGBClassifierWithLabelEncoding (independent classes, same behavior — see
CLAUDE.md on why this project has no shared tools/ layer) wrap fit/predict to
remap labels internally so cross_val_score, Optuna's objective, and the
pickled final model all keep working in the ORIGINAL label space.
"""

import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.core.agents.model_selection_agent import ModelSelectionAgent, _HAS_XGBOOST
from app.core.agents.hyperparameter_agent import HyperparameterAgent


def _make_classification_csv(tmpdir: str, classes: list, n_per_class: int = 30) -> str:
    """Builds a small, easily-separable classification dataset whose target
    uses exactly the given class labels (contiguous or not)."""
    rng = np.random.RandomState(42)
    rows = []
    for i, cls in enumerate(classes):
        f1 = rng.normal(loc=i * 5, scale=1.0, size=n_per_class)
        f2 = rng.normal(loc=-i * 5, scale=1.0, size=n_per_class)
        for a, b in zip(f1, f2):
            rows.append({"f1": a, "f2": b, "target": cls})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    path = os.path.join(tmpdir, "preprocessed.csv")
    df.to_csv(path, index=False)
    return path


@unittest.skipUnless(_HAS_XGBOOST, "xgboost is not installed")
class ModelSelectionXGBoostNonContiguousClassesTests(unittest.TestCase):
    def test_non_contiguous_classes_no_longer_fail(self):
        """Classes [1,2,3,4,5] (e.g. a 1-5 satisfaction rating) must no
        longer make XGBoost fail during CV — it must appear with a real
        score, not 'Failed'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_classification_csv(tmpdir, classes=[1, 2, 3, 4, 5])
            agent = ModelSelectionAgent()
            state = {
                "preprocessed_data_path": path,
                "target_column": "target",
                "problem_type": "classification",
            }
            result = agent.execute(state)

        by_name = {r["model_name"]: r for r in result["candidate_models_results"]}
        self.assertIn("XGBoost", by_name)
        xgb_result = by_name["XGBoost"]
        self.assertNotIn("error", xgb_result, f"XGBoost failed: {xgb_result.get('error')}")
        self.assertGreater(xgb_result["mean_score"], float("-inf"))
        self.assertGreater(xgb_result["mean_score"], 0.0)

    def test_zero_indexed_contiguous_classes_still_work(self):
        """Regression check: the already-working case (classes [0,1,2])
        must keep working after the fix — no behavior change for the
        common case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_classification_csv(tmpdir, classes=[0, 1, 2])
            agent = ModelSelectionAgent()
            state = {
                "preprocessed_data_path": path,
                "target_column": "target",
                "problem_type": "classification",
            }
            result = agent.execute(state)

        by_name = {r["model_name"]: r for r in result["candidate_models_results"]}
        self.assertIn("XGBoost", by_name)
        xgb_result = by_name["XGBoost"]
        self.assertNotIn("error", xgb_result, f"XGBoost failed: {xgb_result.get('error')}")
        self.assertGreater(xgb_result["mean_score"], 0.0)


@unittest.skipUnless(_HAS_XGBOOST, "xgboost is not installed")
class HyperparameterAgentXGBoostNonContiguousClassesTests(unittest.TestCase):
    def test_tuning_and_final_pickle_survive_non_contiguous_classes(self):
        """HyperparameterAgent must be able to tune XGBoost end-to-end on a
        non-contiguous-class target and produce a model that predicts in the
        ORIGINAL label space once unpickled (this is what EvaluationAgent
        and ExplainabilityAgent consume downstream)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_classification_csv(tmpdir, classes=[1, 2, 3, 4, 5])
            df = pd.read_csv(path)

            agent = HyperparameterAgent()
            state = {
                "preprocessed_data_path": path,
                "target_column": "target",
                "problem_type": "classification",
                "selected_model_name": "XGBoost",
                "candidate_models_results": [
                    {"model_name": "XGBoost", "mean_score": 0.5, "scoring_metric": "accuracy"}
                ],
            }
            result = agent.execute(state)

        self.assertNotIn("error", result.get("optimization_summary", {}))
        with open(result["model_path"], "rb") as f:
            final_model = pickle.load(f)

        X = df.drop(columns=["target"])
        preds = final_model.predict(X)
        # Predictions must land back in the original label space, not the
        # internal 0-indexed encoding XGBoost required to train.
        self.assertTrue(set(np.unique(preds)).issubset({1, 2, 3, 4, 5}))
        self.assertTrue(set(final_model.classes_).issubset({1, 2, 3, 4, 5}))


if __name__ == "__main__":
    unittest.main()
