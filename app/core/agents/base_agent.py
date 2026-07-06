"""
Base Agent Module
Defines the abstract interface that all EMADS worker agents must implement.
"""

from abc import ABC, abstractmethod
from app.core.state.emads_state import EMADSState


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


# Type alias to represent a partial dictionary update returned by agents to LangGraph
PartialEMADSState = dict