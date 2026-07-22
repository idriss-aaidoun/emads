"""
Pipeline Edge Case Tests
===========================

Verifies the full EMADS pipeline behaves predictably on tricky datasets:
empty targets, empty datasets, and regression vs classification detection.
"""

import os
import tempfile
import unittest

import pandas as pd

from app.core.state.emads_state import create_initial_state
from app.core.supervisor.supervisor_agent import SupervisorAgent


class PipelineEdgeCaseTests(unittest.TestCase):
    def run_pipeline(self, df: pd.DataFrame, target_column: str = "target"):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dataset.csv")
            df.to_csv(path, index=False)
            state = create_initial_state(dataset_path=path, dataset_name="dataset.csv")
            state["target_column"] = target_column
            return SupervisorAgent().run_pipeline(state)

    def test_empty_target_column_raises_clear_error(self):
        """All-missing target rows are dropped in preprocessing; if nothing
        remains, the pipeline should fail loudly instead of producing fake
        zero-metrics — this is the V2 behavior change vs V1."""
        df = pd.DataFrame({"a": [1, 2], "target": [None, None]})
        with self.assertRaises(RuntimeError):
            self.run_pipeline(df)

    def test_numeric_target_uses_regression_metrics(self):
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "experience": [1, 3, 5, 8, 10, 12, 15, 18],
            "salary": [30000, 40000, 50000, 60000, 65000, 72000, 80000, 90000],
        })
        result = self.run_pipeline(df, target_column="salary")
        metrics = result["metrics"]
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertNotIn("accuracy", metrics)

    def test_categorical_target_uses_classification_metrics(self):
        df = pd.DataFrame({
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature_2": [10, 9, 8, 7, 6, 5, 4, 3],
            "label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        })
        result = self.run_pipeline(df, target_column="label")
        metrics = result["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)


if __name__ == "__main__":
    unittest.main()