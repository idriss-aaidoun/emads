"""
Base Agent Module

This module defines the abstract *BaseAgent* class, which serves as the contract
for all concrete agents in the EMADS pipeline. Each agent encapsulates a distinct
step of the data‑science workflow (e.g., data understanding, EDA, preprocessing,
model training, evaluation, reporting). By inheriting from *BaseAgent*, agents
ensure a consistent interface for execution and state mutation, facilitating
orchestration by the *SupervisorAgent*.

The module also provides the *PartialEMADSState* type alias, representing the
partial dictionary of state updates that an agent returns to the shared
*EMADSState* container.
"""

from abc import ABC, abstractmethod
from app.core.state.emads_state import EMADSState, AgentDecision
from app.utils.logger import get_logger

# Type alias to represent a partial dictionary update returned by agents to LangGraph
PartialEMADSState = dict


class BaseAgent(ABC):
    """
    Abstract Base Class for all pipeline agents.
    Enforces a unified contract for executing data science tasks within LangGraph.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the agent with a unique identifier name.

        Args:
            name (str): The logical name of the agent (e.g., 'eda_agent').
        """
        self.name: str = name
        # Each agent gets its own named logger (e.g. "emads.eda_agent")
        # so log lines are clearly attributed in the log file.
        self.logger = get_logger(f"emads.{name}")

    def decide(
        self,
        decision: str,
        reasoning: str,
        confidence: float | None = None,
    ) -> AgentDecision:
        """Builds a structured decision record for the shared state."""
        return AgentDecision(
            agent_name=self.name,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
        )

    def log(self, message: str) -> str:
        """Formats a consistent log entry for downstream aggregation."""
        return f"[{self.name}] {message}"

    @abstractmethod
    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Executes the specific core logic of the agent.

        Args:
            state (EMADSState): The current global shared state of the system.

        Returns:
            dict: A partial state dictionary containing only the fields
                  updated or created by this specific agent.
        """
        pass

    def execute_logged(self, state: EMADSState) -> PartialEMADSState:
        """
        Thin wrapper around execute() that adds automatic entry/exit/error
        logging to the persistent log file.  The SupervisorAgent can call
        this instead of execute() to get full traceability without each
        concrete agent needing to repeat boilerplate try/except blocks.
        """
        self.logger.info(">>> START agent=%s", self.name)
        try:
            result = self.execute(state)
            self.logger.info("<<< END   agent=%s  status=OK", self.name)
            return result
        except Exception as exc:
            self.logger.error(
                "<<< END   agent=%s  status=ERROR  error=%s",
                self.name,
                exc,
                exc_info=True,   # writes the full traceback to the log file
            )
            raise