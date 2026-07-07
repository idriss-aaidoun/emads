"""
Model Agent Module
Handles data splitting and training a baseline Random Forest model.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState


class ModelAgent(BaseAgent):
    """
    Agent responsible for training a standalone baseline Random Forest model.
    Operates strictly deterministically without LLM dependencies.
    """

    def __init__(self) -> None:
        super().__init__(name="model_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Splits the dataset, fits a Random Forest, and serializes the model to disk.
        """
        data_path = state.get("preprocessed_data_path")
        target_col = state.get("target_column")

        if not data_path:
            raise ValueError(f"[{self.name.upper()}] 'preprocessed_data_path' is missing from state.")
        if not target_col:
            raise ValueError(f"[{self.name.upper()}] 'target_column' is missing from state.")

        # 1. Load preprocessed data
        df = pd.read_csv(data_path)

        if target_col not in df.columns:
            raise ValueError(f"[{self.name.upper()}] Target column '{target_col}' not found in data.")

        # 2. Separate features and labels
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 3. Train/Test Split (80% train, 20% validation, fixed random state for reproducibility)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.dtype == 'int' else None
        )

        # 4. Train the baseline model (Random Forest Classifier for MVP)
        # Using fixed hyperparameters to adhere strictly to the Version 1.0 scope
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 5. Serialize and save the model to disk
        models_dir = os.path.join("data", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_file_path = os.path.join(models_dir, "random_forest_mvp.pkl")

        try:
            with open(model_file_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            raise RuntimeError(f"[{self.name.upper()}] Failed to serialize the trained model: {str(e)}")

        # Return the model path so the Evaluation Agent can load it
        return {
            "model_path": model_file_path
        }