"""
Exploratory Data Analysis (EDA) Agent Module
Computes metrics, generates distribution plots, and uses an LLM to build a text summary.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from app.core.agents.base_agent import BaseAgent, PartialEMADSState
from app.core.state.emads_state import EMADSState
from app.services.llm_service import LLMService


class EDAAgent(BaseAgent):
    """
    Agent handling descriptive statistics, correlation matrices, plotting,
    and delegating qualitative insights generation to the LLM.
    """

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        super().__init__(name="eda_agent")
        # Dependency injection for the LLM infrastructure layer
        self.llm = llm_service if llm_service else LLMService()

    def execute(self, state: EMADSState) -> PartialEMADSState:
        """
        Performs EDA profiling, outputs static graphs, and prompts the LLM for a narrative rundown.
        """
        dataset_path = state.get("dataset_path")
        target_col = state.get("target_column")
        
        if not dataset_path:
            raise ValueError(f"[{self.name.upper()}] 'dataset_path' is missing from the state.")

        # 1. Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Ensure a directory exists for storing intermediate validation plots
        plots_dir = os.path.join("data", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        generated_plots: List[str] = []

        # 2. Compute Mathematical Statistics for the LLM prompt context
        descriptive_stats = df.describe(include='all').to_string()
        
        # 3. Process Visual Graphics (Correlation Matrix for numerical columns)
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            plt.figure(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Numerical Feature Correlations")
            plt.tight_layout()
            
            corr_path = os.path.join(plots_dir, "correlation_matrix.png")
            plt.savefig(corr_path)
            plt.close()
            generated_plots.append(corr_path)

        # Target column distribution tracking
        if target_col and target_col in df.columns:
            plt.figure(figsize=(6, 4))
            if df[target_col].nunique() <= 10 or not pd.api.types.is_numeric_dtype(df[target_col]):
                sns.countplot(x=target_col, data=df)
            else:
                sns.histplot(df[target_col], kde=True)
            plt.title(f"Target Distribution: {target_col}")
            plt.tight_layout()
            
            target_path = os.path.join(plots_dir, "target_distribution.png")
            plt.savefig(target_path)
            plt.close()
            generated_plots.append(target_path)

        # 4. Prompt Engineering for the LLM Tool execution
        system_prompt = (
            "You are an expert Data Scientist. Your task is to analyze raw text summaries of descriptive statistics "
            "and explain the primary patterns, anomalies, and structural insights clearly to an engineering team."
        )
        
        user_prompt = f"""
        Dataset Context:
        - Target Column: {target_col}
        - Total Row Count: {df.shape[0]}
        - Total Column Count: {df.shape[1]}
        
        Descriptive Statistics Table:
        {descriptive_stats}
        
        Provide a brief, high-level summary paragraph highlighting patterns or potential challenges (missing values, skewed features).
        """
        
        eda_summary = self.llm.generate_summary(system_prompt, user_prompt)

        return {
            "eda_summary": eda_summary,
            "generated_plots": generated_plots
        }