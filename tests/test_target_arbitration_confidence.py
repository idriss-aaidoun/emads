"""
Target-Column LLM Arbitration Confidence Tests
==================================================

DataUnderstandingAgent._arbitrate_target_column() used to assign a flat 0.6
confidence whenever the LLM's suggested target column was usable, regardless
of how plausible that column actually looks. It now derives confidence from
the suggested column's own unique/row ratio (the same structural signal
already used to judge the "last column" fallback), so two different
suggestions must produce two different confidences — never a constant
plateau — which is exactly what these tests exercise directly against
_arbitrate_target_column, the method where this logic lives.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from app.core.agents.data_understanding_agent import DataUnderstandingAgent


class TargetArbitrationConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.agent = DataUnderstandingAgent()

    def _arbitrate(self, df: pd.DataFrame, llm_response: str):
        fallback = str(df.columns[-1])
        n_rows = len(df)
        unique_ratio = df[fallback].nunique(dropna=True) / n_rows
        fallback_confidence = max(0.15, min(0.55, 0.55 - unique_ratio * 0.4))
        with patch.object(self.agent.llm, "generate_summary", return_value=llm_response):
            return self.agent._arbitrate_target_column(df, fallback, unique_ratio, fallback_confidence)

    def test_low_and_high_unique_ratio_suggestions_produce_different_confidences(self):
        """A suggested column that looks like a clean class label (few unique
        values relative to rows) must get a HIGHER confidence than one that
        looks like a free-text/ID column (almost every value unique) — not
        the same flat 0.6 regardless of which column was proposed."""
        df = pd.DataFrame({
            "notes": [f"free text entry #{i}" for i in range(40)],  # unique_ratio = 1.0
            "rating": [1, 2, 3, 4, 5] * 8,  # unique_ratio = 5/40 = 0.125
        })

        target_col, decision_low_ratio, _ = self._arbitrate(df, "rating")
        self.assertEqual(target_col, "rating")

        target_col2, decision_high_ratio, _ = self._arbitrate(df, "notes")
        self.assertEqual(target_col2, "notes")

        self.assertNotEqual(
            decision_low_ratio.confidence, decision_high_ratio.confidence,
            "Confidence must depend on the suggested column's own structure, not be a constant plateau.",
        )
        self.assertGreater(decision_low_ratio.confidence, decision_high_ratio.confidence)

    def test_confidence_stays_within_documented_arbitration_band(self):
        """Regardless of the suggested column's unique_ratio, confidence must
        stay within [0.5, 0.75] — above the pure fallback's ceiling (0.55,
        reached with zero corroborating evidence) and below an unambiguous
        common-name match (up to 0.9, a stronger literal signal)."""
        df = pd.DataFrame({
            "id": list(range(50)),  # unique_ratio = 1.0 -> floor
            "class": [0, 1] * 25,   # unique_ratio = 2/50 = 0.04 -> near ceiling
        })

        _, decision_id, _ = self._arbitrate(df, "id")
        _, decision_class, _ = self._arbitrate(df, "class")

        for decision in (decision_id, decision_class):
            self.assertGreaterEqual(decision.confidence, 0.5)
            self.assertLessEqual(decision.confidence, 0.75)

        # The near-every-value-unique column should sit at (or very near) the floor.
        self.assertAlmostEqual(decision_id.confidence, 0.5, places=2)

    def test_reasoning_never_asks_llm_to_self_rate_confidence(self):
        """The confidence must come from unique_ratio_suggested, computed in
        Python — never parsed out of the LLM's own text — so the reasoning
        should report the suggested column's ratio, not a self-reported score."""
        df = pd.DataFrame({
            "notes": [f"entry {i}" for i in range(30)],
            "outcome_value": [1, 2, 3] * 10,
        })
        _, decision, _ = self._arbitrate(df, "outcome_value")
        self.assertIn("unique/row ratio", decision.reasoning)
        self.assertIn("outcome_value", decision.reasoning)


if __name__ == "__main__":
    unittest.main()
