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
from app.core.state.emads_state import EMADSState

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