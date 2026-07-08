"""
EMADS Execution Runner Script
A standalone entry point to test and execute the EMADS v1.0 pipeline.
"""

import os
import pandas as pd
from app.core.state.emads_state import EMADSState
from app.core.supervisor.supervisor_agent import SupervisorAgent


def create_mock_dataset() -> str:
    """
    Generates a tiny baseline dataset to test the pipeline end-to-end.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    mock_csv_path = os.path.join(data_dir, "mock_titanic.csv")
    
    # Create simple dummy classification dataset (Predicting if a user bought a product)
    data = {
        "Age": [22, 38, 26, 35, 54, 2, 27, 14, 4, 58],
        "Fare": [7.25, 71.28, 7.92, 53.10, 51.86, 21.07, 11.13, 30.07, 16.70, 26.55],
        "Gender": ["male", "female", "female", "female", "male", "male", "female", "female", "female", "male"],
        "Purchased": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(mock_csv_path, index=False)
    return mock_csv_path


def main() -> None:
    print("==================================================")
    print("      EMADS VERSION 1.0 - CORE INTEGRATION        ")
    print("==================================================")

    # 1. Prepare raw input data assets
    print("\n[STEP 1] Setting up dummy data assets...")
    dataset_path = create_mock_dataset()
    print(f" -> Mock dataset mounted at: {dataset_path}")

    # 2. Build our initial State dictionary signature
    initial_state: EMADSState = {
        "dataset_path": dataset_path,
        "target_column": "Purchased",
        "schema_info": None,
        "eda_summary": None,
        "generated_plots": None,
        "preprocessed_data_path": None,
        "model_path": None,
        "metrics": None,
        "report_path": None
    }

    # 3. Spin up the Supervisor Engine
    print("\n[STEP 2] Initializing Multi-Agent Supervisor...")
    supervisor = SupervisorAgent()

    # 4. Invoke the LangGraph multi-agent flow
    print("\n[STEP 3] Triggering pipeline execution graph...")
    final_state = supervisor.run_pipeline(initial_state)

    # 5. Output pipeline performance checkpoints
    print("\n==================================================")
    print("       PIPELINE EXECUTION COMPLETED ALIVE!        ")
    print("==================================================")
    print(f" -> Target Column Managed: {final_state.get('target_column')}")
    print(f" -> Total Rows Analyzed : {final_state.get('schema_info', {}).get('num_rows')}")
    print(f" -> Preprocessed Data   : {final_state.get('preprocessed_data_path')}")
    print(f" -> Trained Model File  : {final_state.get('model_path')}")
    print(f" -> Evaluation Metrics  : {final_state.get('metrics')}")
    print(f" -> Final Output Report : {final_state.get('report_path')}")
    print("==================================================\n")


if __name__ == "__main__":
    main()