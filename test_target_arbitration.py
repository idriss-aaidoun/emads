from app.core.state.emads_state import create_initial_state
from app.core.supervisor.supervisor_agent import SupervisorAgent

state = create_initial_state(dataset_path="data/uploads/ambiguous_survey_v2.csv")
# Volontairement PAS de target_column — force la vraie détection automatique
result = SupervisorAgent().run_pipeline(state)

print("=" * 50)
print("Target détectée:", result.get("target_column"))
print("Problem type:", result.get("problem_type"))
print("=" * 50)
print("Arbitrages LLM:", result.get("llm_arbitration_log"))
print("=" * 50)
print("Meta-évaluation:", result.get("meta_evaluation"))