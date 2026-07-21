"""
Supervisor Agent Module
========================

Orchestrates the full EMADS pipeline as a sequential LangGraph workflow.

The Supervisor's ONLY responsibilities:
  - decide which agent runs, and in what order
  - catch failures so one broken agent doesn't crash the whole app with a
    raw traceback
  - track which step is currently running, for the Streamlit progress UI

It never touches data, trains models, or computes metrics itself.
"""

from typing import Callable

from langgraph.graph import StateGraph, START, END
from app.core.state.emads_state import EMADSState

from app.core.agents.data_understanding_agent import DataUnderstandingAgent
from app.core.agents.eda_agent import EDAAgent
from app.core.agents.preprocessing_agent import PreprocessingAgent
from app.core.agents.model_selection_agent import ModelSelectionAgent
from app.core.agents.hyperparameter_agent import HyperparameterAgent
from app.core.agents.evaluation_agent import EvaluationAgent
from app.core.agents.explainability_agent import ExplainabilityAgent
from app.core.agents.reporting_agent import ReportingAgent


class SupervisorAgent:
    """
    Builds and runs the EMADS pipeline as a linear LangGraph state machine.
    """

    def __init__(self) -> None:
        # Single source of truth for pipeline order. To add/remove/reorder a
        # step later, edit only this list — nothing else needs to change.
        self.pipeline_steps: list[tuple[str, object]] = [
            ("data_understanding", DataUnderstandingAgent()),
            ("eda", EDAAgent()),
            ("preprocessing", PreprocessingAgent()),
            ("model_selection", ModelSelectionAgent()),
            ("hyperparameter_optimization", HyperparameterAgent()),
            ("evaluation", EvaluationAgent()),
            ("explainability", ExplainabilityAgent()),
            ("reporting", ReportingAgent()),
        ]
        self.workflow = self._build_workflow_graph()

    def _wrap(self, step_name: str, agent) -> Callable[[EMADSState], dict]:
        """
        Wraps a single agent's execute() so every node in the graph gets,
        for free, without repeating this code in each agent:
          1. `current_step` updated in the state (drives the UI progress bar)
          2. a standard success log line
          3. exceptions turned into a clear, labeled RuntimeError instead of
             a raw traceback surfacing in Streamlit
        """
        def node(state: EMADSState) -> dict:
            try:
                update = dict(agent.execute(state) or {})
                update["current_step"] = step_name
                update.setdefault("logs", []).append(
                    self._format_log(step_name, "completed successfully")
                )
                return update
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline stopped at step '{step_name}': {exc}"
                ) from exc
        return node

    @staticmethod
    def _format_log(step_name: str, message: str) -> str:
        return f"[SUPERVISOR] Step '{step_name}' {message}."

    def _build_workflow_graph(self):
        builder = StateGraph(EMADSState)

        for step_name, agent in self.pipeline_steps:
            builder.add_node(step_name, self._wrap(step_name, agent))

        builder.add_edge(START, self.pipeline_steps[0][0])
        for (current_step, _), (next_step, _) in zip(self.pipeline_steps, self.pipeline_steps[1:]):
            builder.add_edge(current_step, next_step)
        builder.add_edge(self.pipeline_steps[-1][0], END)

        return builder.compile()

    def run_pipeline(self, initial_state: EMADSState) -> EMADSState:
        """
        Runs the compiled graph from start to end.

        Note on failure handling: the pipeline stops at the first failing
        step rather than trying to "skip and continue". In a Data Science
        pipeline this is the correct behavior — e.g. running Evaluation
        after Model Selection crashed would just produce a second,
        confusing error instead of a useful result.
        """
        try:
            return self.workflow.invoke(initial_state)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"[SUPERVISOR ENGINE ERROR] Unexpected failure: {exc}") from exc