"""
Model Selection LLM Arbitration Tests
=========================================

ModelSelectionAgent._compare_top_two() escalates ties (Wilcoxon p > 0.05
between the top two CV-scored candidates) to an LLM arbitration call, and the
LLM's recommendation is meant to actually decide the winner returned to
execute() — not just produce narrative text after a silent fallback to the
higher mean_score. These tests exercise that method directly, since it's
where the real "does the LLM's answer change anything" logic lives.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from app.core.agents.model_selection_agent import ModelSelectionAgent


class ModelSelectionArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.agent = ModelSelectionAgent()
        # Near-identical per-fold scores with mixed-sign differences — too few
        # samples and too little consistent separation for Wilcoxon to find
        # significance, which is exactly the tie condition arbitration exists for.
        self.best_result = {
            "model_name": "RandomForest",
            "mean_score": 0.804,
            "std_score": 0.01,
            "scoring_metric": "accuracy",
            "fold_scores": [0.80, 0.82, 0.79, 0.81, 0.80],
        }
        self.second_result = {
            "model_name": "LogisticRegression",
            "mean_score": 0.802,
            "std_score": 0.01,
            "scoring_metric": "accuracy",
            "fold_scores": [0.81, 0.80, 0.80, 0.79, 0.81],
        }
        self.results = [self.best_result, self.second_result]

    def test_llm_recommendation_overrides_higher_score(self):
        """A clear LLM recommendation for the lower-scored candidate must
        actually become the winner, not just decorate the higher-scored pick
        with explanatory text."""
        with patch.object(
            self.agent.llm, "generate_summary",
            return_value=(
                "Given the tie in performance, I recommend LogisticRegression "
                "for its interpretability, despite RandomForest's marginally "
                "higher score."
            ),
        ):
            p_value, arbitration_entry, winner = self.agent._compare_top_two(self.results)

        self.assertGreater(p_value, 0.05)
        self.assertIsNotNone(arbitration_entry)
        self.assertEqual(arbitration_entry["parsed_recommendation"], "LogisticRegression")
        self.assertEqual(winner["model_name"], "LogisticRegression")
        # Sanity check this really was the lower-scoring candidate of the tie.
        self.assertLess(self.second_result["mean_score"], self.best_result["mean_score"])

    def test_malformed_llm_response_falls_back_to_highest_score(self):
        """A response that names neither candidate must not be guessed at —
        the deterministic fallback (highest mean CV score) must win, and the
        failure must be recorded (not silently swallowed)."""
        with patch.object(
            self.agent.llm, "generate_summary",
            return_value="This is a genuinely difficult call with no clear answer either way.",
        ):
            p_value, arbitration_entry, winner = self.agent._compare_top_two(self.results)

        self.assertGreater(p_value, 0.05)
        self.assertIsNotNone(arbitration_entry)
        self.assertIsNone(arbitration_entry["parsed_recommendation"])
        self.assertEqual(winner["model_name"], "RandomForest")

    def test_significance_note_does_not_claim_recommendation_on_parse_failure(self):
        """Regression test: the reasoning text shown in the PDF/report must
        not claim the LLM 'recommended' the fallback model when parsing
        actually failed — that would misrepresent what happened."""
        with patch.object(
            self.agent.llm, "generate_summary",
            return_value="Hard to say, both seem fine.",
        ):
            p_value, arbitration_entry, winner = self.agent._compare_top_two(self.results)

        note = self.agent._build_significance_note(
            p_value, self.results, arbitration_entry, winner["model_name"]
        )
        self.assertIn("could not be parsed", note)
        self.assertNotIn("which recommended", note)


class ModelSelectionExecuteArbitrationTests(unittest.TestCase):
    """End-to-end proof through execute() itself — not just the internal
    _compare_top_two() helper — that the field the rest of the pipeline
    actually consumes (selected_model_name) is really driven by the parsed
    LLM recommendation, with a working, visible fallback when parsing fails."""

    def _run_execute(self, llm_response: str):
        agent = ModelSelectionAgent()
        df = pd.DataFrame({
            "f1": list(range(1, 21)),
            "target": [0, 1] * 10,
        })
        candidates = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        }
        # Same near-identical, mixed-sign fold profile as the direct
        # _compare_top_two tests above, keyed by real sklearn class name so
        # cross_val_score's mock returns a fixed score regardless of the
        # actual (irrelevant, since mocked) train/test split content.
        fold_scores = {
            "RandomForestClassifier": np.array([0.80, 0.82, 0.79, 0.81, 0.80]),
            "LogisticRegression": np.array([0.81, 0.80, 0.80, 0.79, 0.81]),
        }

        def fake_cross_val_score(model, X, y, cv=None, scoring=None, n_jobs=None):
            return fold_scores[type(model).__name__]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "preprocessed.csv")
            df.to_csv(path, index=False)
            state = {
                "preprocessed_data_path": path,
                "target_column": "target",
                "problem_type": "classification",
            }
            with patch.object(agent, "_build_candidates", return_value=candidates), \
                 patch(
                     "app.core.agents.model_selection_agent.cross_val_score",
                     side_effect=fake_cross_val_score,
                 ), \
                 patch.object(agent.llm, "generate_summary", return_value=llm_response):
                return agent.execute(state)

    def test_selected_model_name_follows_llm_recommendation_over_higher_score(self):
        """RandomForest has the higher mean CV score of the tied pair; the LLM
        explicitly recommends LogisticRegression (the lower-scored one) — the
        state actually returned to the pipeline must select LogisticRegression,
        not silently keep the higher-scoring RandomForest."""
        result = self._run_execute(
            "Given the statistical tie, I recommend LogisticRegression for its "
            "interpretability and lower training cost."
        )
        self.assertEqual(result["selected_model_name"], "LogisticRegression")
        note = result["agent_decisions"][0].reasoning
        self.assertIn("which recommended", note)
        self.assertIn("'LogisticRegression'", note)

    def test_selected_model_name_falls_back_deterministically_on_parse_failure(self):
        """An ambiguous LLM response (no valid candidate name) must not be
        guessed at: selected_model_name must fall back to the higher CV
        score (RandomForest), the Fix-1 case-3 wording must appear in the
        reasoning (never 'recommended'), and the fallback must be visible in
        state['logs'] — not just the disk logger."""
        result = self._run_execute("This is genuinely a toss-up, hard to say.")
        self.assertEqual(result["selected_model_name"], "RandomForest")
        note = result["agent_decisions"][0].reasoning
        self.assertIn("could not be parsed", note)
        self.assertNotIn("which recommended", note)
        self.assertTrue(
            any("could not be parsed" in entry for entry in result["logs"]),
            f"Expected a parse-failure warning in state['logs'], got: {result['logs']}",
        )


if __name__ == "__main__":
    unittest.main()
