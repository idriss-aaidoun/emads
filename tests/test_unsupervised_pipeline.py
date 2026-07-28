"""
Unsupervised Pipeline Tests
=============================

Verifies the full EMADS pipeline runs end-to-end for the two unsupervised
problem types (clustering, anomaly_detection): opt-in via problem_type,
no target_column required, and metrics specific to each type get produced.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.core.state.emads_state import create_initial_state
from app.core.supervisor.supervisor_agent import SupervisorAgent


class UnsupervisedPipelineTests(unittest.TestCase):
    def run_pipeline(self, df: pd.DataFrame, problem_type: str, target_column: str | None = None):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dataset.csv")
            df.to_csv(path, index=False)
            state = create_initial_state(dataset_path=path, dataset_name="dataset.csv")
            state["problem_type"] = problem_type
            if target_column:
                state["target_column"] = target_column
            return SupervisorAgent().run_pipeline(state)

    def _two_blob_dataset(self) -> pd.DataFrame:
        """60 rows, 2 well-separated clusters — enough to satisfy
        MIN_ROWS_FOR_UNSUPERVISED and LocalOutlierFactor's default
        n_neighbors=20 while giving every candidate algorithm a clear
        structure to find."""
        rng = np.random.RandomState(42)
        blob_a = rng.normal(loc=0.0, scale=1.0, size=(30, 2))
        blob_b = rng.normal(loc=15.0, scale=1.0, size=(30, 2))
        data = np.vstack([blob_a, blob_b])
        return pd.DataFrame(data, columns=["feature_1", "feature_2"])

    def test_clustering_produces_silhouette_metrics(self):
        df = self._two_blob_dataset()
        result = self.run_pipeline(df, problem_type="clustering")
        metrics = result["metrics"]
        self.assertIn("silhouette_score", metrics)
        self.assertIn(result["selected_model_name"], {"KMeans", "DBSCAN"})

    def test_anomaly_detection_produces_anomaly_metrics(self):
        df = self._two_blob_dataset()
        result = self.run_pipeline(df, problem_type="anomaly_detection")
        metrics = result["metrics"]
        self.assertIn("n_anomalies_detected", metrics)
        self.assertIn("anomaly_rate", metrics)
        self.assertIn(result["selected_model_name"], {"IsolationForest", "LocalOutlierFactor"})

    def test_unsupervised_requires_minimum_rows(self):
        """Below MIN_ROWS_FOR_UNSUPERVISED, DataUnderstandingAgent should
        reject the run with a clear error rather than let a confusing
        sklearn failure surface later in the pipeline."""
        df = pd.DataFrame({
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [5, 4, 3, 2, 1],
        })
        with self.assertRaises(RuntimeError):
            self.run_pipeline(df, problem_type="clustering")


if __name__ == "__main__":
    unittest.main()
