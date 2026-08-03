"""
Meta-Evaluator Agent Module
==============================

Final agent of the pipeline — runs AFTER ReportingAgent, once the PDF has
already been generated. Its single responsibility is to audit the completed
run and produce a holistic confidence verdict, so a reader of the report has
one place that tells them "how much should I trust this end-to-end?" instead
of having to reconcile a dozen scattered per-agent confidence scores
themselves.

This agent NEVER re-runs a previous step. It only reads the final state and
writes a recommendation in plain text — acting on that recommendation (e.g.
re-running with a different target column) is left entirely to the user.
"""

from typing import Any, Dict, List, Optional

from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import AgentDecision, EMADSState

# Weights for the three components of the global confidence score. Kept
# simple and explicit rather than learned/tuned, since the point is that a
# reader of the report can see exactly how the number was built.
WEIGHT_EVALUATION = 0.4
WEIGHT_EXPLANATION = 0.3
WEIGHT_ARBITRATION = 0.3

# Below this score, the run is flagged for manual review instead of being
# reported as reliable outright.
CONFIDENCE_THRESHOLD = 0.6

# Each LLM arbitration escalated somewhere in the pipeline signals accumulated
# uncertainty (a statistical tie, a low-agreement explanation, ...) — this
# penalizes the arbitration component per entry, but never below this floor:
# arbitration is a legitimate, reasoned mechanism, not a run-invalidating flaw.
ARBITRATION_COMPONENT_FLOOR = 0.4
ARBITRATION_PENALTY_PER_ENTRY = 0.1

# If DataUnderstandingAgent's target-column decision had confidence below
# this, a below-threshold run is most plausibly explained by that — mirrors
# DataUnderstandingAgent.TARGET_ARBITRATION_THRESHOLD, the same cutoff it
# uses to decide the fallback target guess was too uncertain to trust.
TARGET_CONFIDENCE_CONCERN_THRESHOLD = 0.5

# If explanation_agreement_score (Spearman, SHAP vs permutation importance)
# is below this, a below-threshold run is more plausibly explained by an
# unreliable explanation than by the model/target itself — mirrors
# ExplainabilityAgent.AGREEMENT_LOW_THRESHOLD.
EXPLANATION_AGREEMENT_CONCERN_THRESHOLD = 0.3


class MetaEvaluatorAgent(BaseAgent):
    """
    Agent responsible for auditing the final state and producing a single
    holistic confidence verdict for the whole run — informational only.
    """

    def __init__(self) -> None:
        super().__init__(name="meta_evaluator_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        problem_type = state.get("problem_type")
        metrics = state.get("metrics") or {}
        explanation_agreement_score = state.get("explanation_agreement_score")
        arbitration_log = state.get("llm_arbitration_log") or []
        agent_decisions = state.get("agent_decisions") or []

        metrics_component = self._metrics_component(problem_type, metrics)
        explanation_component = self._explanation_component(explanation_agreement_score)
        arbitration_component = self._arbitration_component(arbitration_log)

        confidence_score = (
            WEIGHT_EVALUATION * metrics_component
            + WEIGHT_EXPLANATION * explanation_component
            + WEIGHT_ARBITRATION * arbitration_component
        )

        if confidence_score >= CONFIDENCE_THRESHOLD:
            verdict = "Run report as reliable, no specific concerns flagged."
            recommendation = None
        else:
            verdict = (
                f"Review recommended — overall confidence ({confidence_score:.2f}) is below "
                f"the {CONFIDENCE_THRESHOLD:.2f} threshold."
            )
            recommendation = self._build_recommendation(
                agent_decisions, explanation_agreement_score
            )

        meta_evaluation = {
            "confidence_score": round(confidence_score, 4),
            "verdict": verdict,
            "recommendation": recommendation,
        }

        reasoning = (
            f"Confidence = {WEIGHT_EVALUATION:.0%} x evaluation({metrics_component:.2f}) + "
            f"{WEIGHT_EXPLANATION:.0%} x explanation({explanation_component:.2f}) + "
            f"{WEIGHT_ARBITRATION:.0%} x arbitration({arbitration_component:.2f}) = "
            f"{confidence_score:.2f}. Evaluation component from '{problem_type}' metrics; "
            f"explanation component from explanation_agreement_score="
            f"{explanation_agreement_score if explanation_agreement_score is not None else 'N/A'}; "
            f"arbitration component penalized for {len(arbitration_log)} LLM arbitration(s) "
            f"logged during the run."
        )

        return {
            "meta_evaluation": meta_evaluation,
            # This is a deterministic formula applied to already-known values —
            # there is no uncertainty in the computation itself to reflect here,
            # only in the verdict it produces (which is why it's carried as
            # meta_evaluation.confidence_score, not this decision's confidence).
            "agent_decisions": [self.decide(
                decision=f"Meta-evaluation verdict: {verdict}",
                reasoning=reasoning,
                confidence=1.0,
            )],
            "logs": [self.log(
                f"Meta-evaluation complete. confidence_score={confidence_score:.2f}, "
                f"verdict='{verdict}'."
            )],
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _metrics_component(self, problem_type: Optional[str], metrics: Dict[str, Any]) -> float:
        """
        Normalizes the primary evaluation metric to [0, 1] depending on
        problem_type, so it can be combined with the other two components on
        a comparable scale. Defaults to a neutral 0.5 when the expected
        metric is missing (e.g. EvaluationAgent failed to report it) instead
        of crashing or silently zeroing out the whole score.
        """
        if problem_type == "classification":
            value = metrics.get("accuracy")
        elif problem_type == "regression":
            value = metrics.get("r2")
        else:  # clustering / anomaly_detection
            value = metrics.get("silhouette_score")
            if value is not None:
                # silhouette lives in [-1, 1]; rescale to [0, 1] like
                # EvaluationAgent._evaluate_unsupervised already does for its
                # own confidence, for the same reason.
                value = (value + 1) / 2

        if value is None:
            return 0.5
        return float(max(0.0, min(1.0, value)))

    def _explanation_component(self, explanation_agreement_score: Optional[float]) -> float:
        """
        Rescales the Spearman SHAP/permutation-importance agreement score
        ([-1, 1]) to [0, 1]. Defaults to a neutral 0.5 when unavailable
        (unsupervised problems never compute it — there's no target to
        attribute importance to) rather than penalizing runs that never had
        this signal to begin with.
        """
        if explanation_agreement_score is None:
            return 0.5
        return float(max(0.0, min(1.0, (explanation_agreement_score + 1) / 2)))

    def _arbitration_component(self, arbitration_log: List[Dict[str, Any]]) -> float:
        """
        More LLM arbitrations logged during the run means more accumulated
        uncertainty was resolved along the way (statistical ties, low
        explanation agreement, low-confidence target detection, ...) — each
        one nudges this component down, but it never reaches zero: arbitration
        is a documented, reasoned fallback, not evidence the run is broken.
        """
        n = len(arbitration_log)
        return float(max(ARBITRATION_COMPONENT_FLOOR, 1.0 - ARBITRATION_PENALTY_PER_ENTRY * n))

    def _build_recommendation(
        self,
        agent_decisions: List[AgentDecision],
        explanation_agreement_score: Optional[float],
    ) -> str:
        """
        Points at the most plausible root cause for a below-threshold run,
        in priority order: an uncertain target-column detection first (since
        a wrong target invalidates everything downstream), then a low
        explanation agreement, then a generic catch-all. Purely textual —
        no agent is ever re-triggered from here.
        """
        target_decision = self._find_target_decision(agent_decisions)
        if (
            target_decision is not None
            and target_decision.confidence is not None
            and target_decision.confidence < TARGET_CONFIDENCE_CONCERN_THRESHOLD
        ):
            return (
                f"Target column detection had low confidence "
                f"({target_decision.confidence:.0%}): {target_decision.reasoning} "
                "Consider re-running with a different, explicitly-specified target column."
            )

        if (
            explanation_agreement_score is not None
            and explanation_agreement_score < EXPLANATION_AGREEMENT_CONCERN_THRESHOLD
        ):
            return (
                f"SHAP and permutation importance disagreed substantially "
                f"(Spearman={explanation_agreement_score:.3f}). Verify the model's stability "
                "rather than trusting this explanation at face value."
            )

        return (
            "No single dominant cause was identified across target detection, evaluation "
            "metrics, or explanation agreement — a general manual review of the pipeline's "
            "automatic decisions (see Decisions Log) is recommended before trusting this report."
        )

    def _find_target_decision(self, agent_decisions: List[AgentDecision]) -> Optional[AgentDecision]:
        for decision in agent_decisions:
            if decision.agent_name == "data_understanding_agent" and "target column" in decision.decision.lower():
                return decision
        return None
