"""
Supervisor Agent Module
Orchestrates the sequential multi-agent execution pipeline using LangGraph.
"""

from langgraph.graph import StateGraph, START, END
from app.core.state.emads_state import EMADSState

# Import concrete agent implementations
from app.core.agents.data_understanding_agent import DataUnderstandingAgent
from app.core.agents.eda_agent import EDAAgent
from app.core.agents.preprocessing_agent import PreprocessingAgent
from app.core.agents.model_selection_agent import ModelAgent
from app.core.agents.evaluation_agent import EvaluationAgent
from app.core.agents.reporting_agent import ReportingAgent


class SupervisorAgent:
    """
    Manages the compilation and linear execution of the EMADS multi-agent system.
    Follows a strict sequential pipeline layout for Version 1.0.
    """

    def __init__(self) -> None:
        # Instantiate all worker agents
        self.data_understanding = DataUnderstandingAgent()
        self.eda = EDAAgent()
        self.preprocessing = PreprocessingAgent()
        self.model = ModelAgent()
        self.evaluation = EvaluationAgent()
        self.reporting = ReportingAgent()
        
        # Compile the workflow graph
        self.workflow = self._build_workflow_graph()

    def _build_workflow_graph(self):
        """
        Constructs the internal state graph layout, adding nodes and edges.
        """
        # 1. Initialize the Graph container tied to our State TypedDict schema
        builder = StateGraph(EMADSState)

        # 2. Define and map nodes (LangGraph requires functions or methods matching state signatures)
        builder.add_node("data_understanding", lambda state: self.data_understanding.execute(state))
        builder.add_node("eda", lambda state: self.eda.execute(state))
        builder.add_node("preprocessing", lambda state: self.preprocessing.execute(state))
        builder.add_node("model", lambda state: self.model.execute(state))
        builder.add_node("evaluation", lambda state: self.evaluation.execute(state))
        builder.add_node("reporting", lambda state: self.reporting.execute(state))

        # 3. Create strict sequential edges (Control flow path mapping)
        builder.add_edge(START, "data_understanding")
        builder.add_edge("data_understanding", "eda")
        builder.add_edge("eda", "preprocessing")
        builder.add_edge("preprocessing", "model")
        builder.add_edge("model", "evaluation")
        builder.add_edge("evaluation", "reporting")
        builder.add_edge("reporting", END)

        # 4. Compile into a runnable component application
        return builder.compile()

    def run_pipeline(self, initial_state: EMADSState) -> EMADSState:
        """
        Triggers the workflow synchronously with a given starting state configuration.
        
        Args:
            initial_state (EMADSState): Contains 'dataset_path' and optional 'target_column'.
            
        Returns:
            EMADSState: The complete, updated state with all generated pipeline results.
        """
        try:
            # invoke runs the compiled graph until it hits the END state node
            final_output_state = self.workflow.invoke(initial_state)
            return final_output_state
        except Exception as e:
            raise RuntimeError(f"[SUPERVISOR ENGINE ERROR] Pipeline failed during execution: {str(e)}")