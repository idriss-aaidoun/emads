import os
import tempfile
import unittest

import pandas as pd

from app.core.supervisor.supervisor_agent import SupervisorAgent


class PipelineEdgeCaseTests(unittest.TestCase):
    def run_pipeline(self, df, target_column="target"):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dataset.csv")
            df.to_csv(path, index=False)
            state = {
                "dataset_path": path,
                "target_column": target_column,
                "schema_info": None,
                "eda_summary": None,
                "generated_plots": None,
                "preprocessed_data_path": None,
                "model_path": None,
                "metrics": None,
                "report_path": None,
            }
            return SupervisorAgent().run_pipeline(state)

    def test_empty_target_column_does_not_crash(self):
        df = pd.DataFrame({"a": [1, 2], "target": [None, None]})
        result = self.run_pipeline(df)
        self.assertIn("metrics", result)
        self.assertIsNotNone(result["metrics"])

    def test_empty_dataset_does_not_crash(self):
        df = pd.DataFrame({"a": [], "target": []})
        result = self.run_pipeline(df)
        self.assertIn("metrics", result)
        self.assertIsNotNone(result["metrics"])

    def test_numeric_target_uses_regression_metrics(self):
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40],
                "experience": [1, 3, 5, 8],
                "Salary": [30000, 40000, 50000, 60000],
            }
        )
        result = self.run_pipeline(df, target_column="Salary")
        metrics = result["metrics"]
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertNotIn("accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
