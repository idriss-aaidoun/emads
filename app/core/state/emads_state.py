"""
EMADS State Module
===================

Defines the shared memory (blackboard) passed between all agents of the
EMADS pipeline. Every agent reads from this state and returns a *partial*
update to it — never a full replacement.

Two categories of fields exist:

1. Fields that agents OVERWRITE (e.g. `metrics`, `model_path`) — each agent
   owns its own fields and is the only one allowed to write them.
2. Fields that ACCUMULATE across agents (`logs`, `agent_decisions`, `errors`)
   — every agent appends to them instead of overwriting. These use
   `Annotated[..., operator.add]` so LangGraph merges them automatically
   instead of one agent erasing another's entries.
"""

from typing import TypedDict, Any, Optional, Dict, List, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import operator


@dataclass
class AgentDecision:
    agent_name: str
    decision: str
    reasoning: str
    confidence: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EMADSState(TypedDict, total=False):
    session_id: str
    created_at: str
    current_step: str

    dataset_path: str
    dataset_name: Optional[str]
    target_column: Optional[str]
    problem_type: Optional[str]

    schema_info: Optional[Dict[str, Any]]

    eda_summary: Optional[str]
    eda_stats: Optional[Dict[str, Any]]
    generated_plots: Optional[List[str]]

    preprocessing_report: Optional[Dict[str, Any]]
    preprocessed_data_path: Optional[str]
    selected_features: Optional[List[str]]

    candidate_models_results: Optional[List[Dict[str, Any]]]
    selected_model_name: Optional[str]
    model_path: Optional[str]
    model_selection_summary: Optional[str]

    best_hyperparameters: Optional[Dict[str, Any]]
    optimization_summary: Optional[Dict[str, Any]]

    metrics: Optional[Dict[str, Any]]
    evaluation_plots: Optional[List[str]]

    feature_importance: Optional[Dict[str, float]]
    shap_plots: Optional[List[str]]
    explainability_summary: Optional[str]

    report_path: Optional[str]

    agent_decisions: Annotated[List[AgentDecision], operator.add]
    logs: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]


def create_initial_state(dataset_path: str, dataset_name: Optional[str] = None) -> EMADSState:
    return EMADSState(
        session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        created_at=datetime.now().isoformat(),
        current_step="data_understanding",
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        agent_decisions=[],
        logs=[],
        errors=[],
    )
