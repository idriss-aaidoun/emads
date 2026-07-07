"""
Evaluation Agent Module
Loads the trained model and computes validation metrics on the holdout split.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for scoring the model on validation data.
    Operates completely deterministically without an LLM.
    """

    def __init__(self) -> None:
        super().__init__(name="evaluation_agent")

    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Loads the trained model, recreates the test split, and calculates metrics.
        """
        data_path = state.get("preprocessed_data_path")
        model_path = state.get("model_path")
        target_col = state.get("target_column")

        if not data_path or not model_path or not target_col:
            raise ValueError(
                f"[{self.name.upper()}] Missing baseline state inputs. "
                f"Ensure preprocessing and model training completed successfully."
            )

        # 1. Reload the preprocessed data
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 2. Recreate the exact same test split using the identical random_state seed
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.dtype == 'int' else None
        )

        # 3. Deserialize and load the trained model
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"[{self.name.upper()}] Failed to load trained model binary: {str(e)}")

        # 4. Generate validation predictions
        y_pred = model.predict(X_test)

        # 5. Compute core classification performance metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        
        # Calculate precision, recall, and f1 safely for multiclass or binary targets
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        
        # Build raw confusion matrix array and convert to a standard nested list structure
        cm = confusion_matrix(y_test, y_pred).tolist()

        # 6. Build the state payload structure
        metrics_payload = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm
        }

        return {
            "metrics": metrics_payload
        }