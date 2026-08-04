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

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
