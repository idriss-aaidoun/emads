"""
LLM Service Module
Handles live inference text-generation requests natively through the Groq API SDK.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load credentials from .env – override=True ensures .env always wins
# even when a stale OS-level env var is already present in the session.
load_dotenv(override=True)


class LLMService:
    """
    Handles connections and generation tasks for Language Models.
    Optimized for production Groq API client orchestration.

    Supports both GROQ_API_KEY and LLM_API_KEY env variable names
    so existing .env configurations work without modification.
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"
    VALID_GROQ_MODELS = {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3.6-27b",
    }

    def __init__(self, model_name: Optional[str] = None) -> None:
        # Start from the safe default.
        self.model_name = self.DEFAULT_MODEL

        # 1. Allow caller to pass a model name explicitly.
        if model_name and model_name in self.VALID_GROQ_MODELS:
            self.model_name = model_name

        # 2. Allow override via .env (DEFAULT_MODEL_NAME) – useful for quick switching.
        env_model = os.getenv("DEFAULT_MODEL_NAME")
        if env_model and env_model in self.VALID_GROQ_MODELS:
            self.model_name = env_model

        # 3. Final guard – if anything above produced an invalid name, reset to default.
        if self.model_name not in self.VALID_GROQ_MODELS:
            self.model_name = self.DEFAULT_MODEL

        # Accept either key name so legacy .env files work out of the box.
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")

        # Initialize the official Groq SDK client only when a key is available.
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                # groq package not installed; will fall back to mock responses.
                pass

    # ------------------------------------------------------------------
    # Core low-level call – all public methods delegate to this.
    # ------------------------------------------------------------------
    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompts to Groq and return the generated text.

        Falls back to a deterministic mock when the SDK or API key is absent,
        ensuring the pipeline remains fully operational in offline environments.
        """
        if not self.client:
            # Offline / no-key fallback – keeps the MVP runnable without credentials
            return (
                "**EDA Summary (offline mock):** The dataset exhibits typical structural "
                "characteristics. Numerical features show minimal skewness. Categorical "
                "columns may contain class imbalances. Recommend running correlation "
                "analysis and verifying missing-value imputation before modelling."
            )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,   # Low temperature keeps data-science outputs objective
                max_tokens=600,
            )
            return completion.choices[0].message.content.strip()

        except Exception as exc:
            return f"[Groq API Error]: {str(exc)}"

    # ------------------------------------------------------------------
    # EDA-specific public method (called by EDAAgent)
    # ------------------------------------------------------------------
    def generate_eda_summary(self, schema_info: dict, dataset_path: str) -> str:
        """Generate a concise, bullet-point EDA narrative from schema statistics.

        Args:
            schema_info: High-level stats dict from DataUnderstandingAgent
                         (row/col counts, dtypes, missing values, etc.).
            dataset_path: Path to the source file (used as context for the LLM).

        Returns:
            A human-readable markdown string summarising dataset characteristics.
        """
        system_prompt = (
            "You are a senior data scientist reviewing a dataset for a machine learning project. "
            "Analyse the schema statistics provided and write a concise summary (4-6 bullet points) "
            "covering: dataset shape, missing data risks, feature types, potential class imbalances, "
            "and any recommended preprocessing steps. Use plain English – no code."
        )

        num_rows  = schema_info.get("num_rows", "unknown")
        num_cols  = schema_info.get("num_cols", "unknown")
        columns   = schema_info.get("columns", {})

        # Build a readable column profile for the LLM
        col_lines = []
        for col_name, meta in columns.items():
            missing  = meta.get("missing_values", 0)
            dtype    = meta.get("dtype", "unknown")
            numeric  = meta.get("is_numeric", False)
            col_lines.append(
                f"  - {col_name}: dtype={dtype}, missing={missing}, numeric={numeric}"
            )
        column_profile = "\n".join(col_lines) if col_lines else "  (no columns found)"

        user_prompt = (
            f"Dataset file: {dataset_path}\n"
            f"Shape: {num_rows} rows × {num_cols} columns\n"
            f"Target column: {schema_info.get('target_column', 'unknown')}\n"
            f"Total missing values across all columns: {schema_info.get('missing_totals', 'unknown')}\n\n"
            f"Column Profile:\n{column_profile}\n\n"
        )

        # Append descriptive statistics if provided by the EDA agent
        descriptive_stats = schema_info.get("descriptive_stats")
        if descriptive_stats:
            user_prompt += f"Descriptive Statistics:\n{descriptive_stats}\n\n"

        user_prompt += "Please provide your structured EDA summary now."

        return self._call_groq(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Generic wrapper used by ReportingAgent and any future agents
    # ------------------------------------------------------------------
    def generate_summary(self, system_prompt: str, user_prompt: str) -> str:
        """Generic prompt → summary helper. Delegates to _call_groq."""
        return self._call_groq(system_prompt, user_prompt)