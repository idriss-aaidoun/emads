"""
Explainability Agent — Unsupervised SHAP Skip Tests
=======================================================

ExplainabilityAgent used to call _compute_shap() unconditionally, protected
only by a generic try/except that happened to catch DBSCAN's failure after
the fact. It now checks `is_unsupervised` explicitly BEFORE attempting SHAP
or native feature importance at all (see execute()) — these tests verify
that guard directly, for every unsupervised candidate model
(KMeans, DBSCAN, IsolationForest, LocalOutlierFactor), by asserting
_compute_shap is never even called, not just that it fails safely.
"""

import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from app.core.agents.explainability_agent import ExplainabilityAgent


class ExplainabilityUnsupervisedSkipTests(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        blob_a = rng.normal(loc=0.0, scale=1.0, size=(20, 2))
        blob_b = rng.normal(loc=10.0, scale=1.0, size=(20, 2))
        data = np.vstack([blob_a, blob_b])
        return pd.DataFrame(data, columns=["feature_1", "feature_2"])

    def _run_execute(self, model, model_name: str, problem_type: str):
        df = self._dataset()
        model.fit(df) if not hasattr(model, "fit_predict") else model.fit_predict(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "preprocessed.csv")
            model_path = os.path.join(tmpdir, "model.pkl")
            df.to_csv(data_path, index=False)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            state = {
                "preprocessed_data_path": data_path,
                "model_path": model_path,
                "selected_model_name": model_name,
                "target_column": None,
                "problem_type": problem_type,
            }
            agent = ExplainabilityAgent()
            with patch.object(agent, "_compute_shap") as mock_compute_shap, \
                 patch.object(agent.llm, "generate_summary", return_value="fallback summary"):
                result = agent.execute(state)
                mock_compute_shap.assert_not_called()
            return result

    def test_kmeans_skips_shap(self):
        result = self._run_execute(
            KMeans(n_clusters=2, random_state=42, n_init=10), "KMeans", "clustering",
        )
        self.assertEqual(result["feature_importance"], {})
        self.assertEqual(result["shap_plots"], [])

    def test_dbscan_skips_shap(self):
        result = self._run_execute(DBSCAN(eps=2.0, min_samples=3), "DBSCAN", "clustering")
        self.assertEqual(result["feature_importance"], {})
        self.assertEqual(result["shap_plots"], [])

    def test_isolation_forest_skips_shap(self):
        result = self._run_execute(
            IsolationForest(random_state=42, contamination="auto"), "IsolationForest", "anomaly_detection",
        )
        self.assertEqual(result["feature_importance"], {})
        self.assertEqual(result["shap_plots"], [])

    def test_local_outlier_factor_skips_shap(self):
        result = self._run_execute(
            LocalOutlierFactor(novelty=False, contamination="auto"), "LocalOutlierFactor", "anomaly_detection",
        )
        self.assertEqual(result["feature_importance"], {})
        self.assertEqual(result["shap_plots"], [])

    def test_skip_reasoning_says_skipped_by_design(self):
        """The log/state note must explain this is a deliberate design
        choice, not a caught failure — distinguishing it from the
        try/except safety net that still guards the supervised path."""
        result = self._run_execute(
            KMeans(n_clusters=2, random_state=42, n_init=10), "KMeans", "clustering",
        )
        combined_logs = " ".join(result["logs"])
        self.assertIn("skipped by design", combined_logs)
        self.assertNotIn("skipped due to error", combined_logs)


if __name__ == "__main__":
    unittest.main()
