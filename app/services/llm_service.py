"""
LLM Service Module
Handles live inference text-generation requests natively through the Groq API SDK.
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from app.utils.logger import get_logger

# Load credentials from .env – override=True ensures .env always wins.
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(find_dotenv(), override=True)

logger = get_logger("emads.llm")


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
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
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

        # Resolve API key with robust fallback and whitespace cleaning.
        self.api_key = self._resolve_api_key()
        self._api_key_source = self._resolve_api_key_source()
        logger.info(
            "LLMService init: model=%s api_key_present=%s api_key_source=%s",
            self.model_name,
            bool(self.api_key),
            self._api_key_source,
        )

        # Initialize the official Groq SDK client only when a key is available.
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully.")
            except ImportError:
                # groq package not installed; will fall back to offline responses.
                self.client = None
                logger.warning("Groq SDK import failed; falling back to offline mode.")
        else:
            logger.warning("No usable API key found in environment (.env or OS env).")

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        candidates = [
            os.getenv("GROQ_API_KEY"),
            os.getenv("LLM_API_KEY"),
        ]
        # First priority: a key starting with gsk_
        for candidate in candidates:
            if candidate:
                key = candidate.strip(" \"'\r\n")
                if key.startswith("gsk_") and len(key) > 10:
                    return key
        # Second priority: any non-placeholder key
        for candidate in candidates:
            if candidate:
                key = candidate.strip(" \"'\r\n")
                if key and "your_" not in key.lower() and len(key) > 5:
                    return key
        return None

    @staticmethod
    def _resolve_api_key_source() -> str:
        groq_key = os.getenv("GROQ_API_KEY")
        llm_key = os.getenv("LLM_API_KEY")
        if groq_key:
            return "GROQ_API_KEY"
        if llm_key:
            return "LLM_API_KEY"
        return "missing"

    # ------------------------------------------------------------------
    # Core low-level call – all public methods delegate to this.
    # ------------------------------------------------------------------
    def _call_groq(self, system_prompt: str, user_prompt: str, fallback_message: str | None = None) -> str:
        if not self.client:
            logger.warning(
                "LLM call skipped: no Groq client available (model=%s, api_key_source=%s).",
                self.model_name,
                self._api_key_source,
            )
            return fallback_message or (
                "[Offline mode] No LLM response available — Groq SDK is missing or the API key is not configured."
            )

        models_to_try = [self.model_name]
        if "llama-3.1-8b-instant" not in models_to_try:
            models_to_try.append("llama-3.1-8b-instant")

        last_exception = None
        for model in models_to_try:
            for attempt in range(2):
                try:
                    logger.info(
                        "LLM request start: model=%s attempt=%s system_chars=%s user_chars=%s",
                        model,
                        attempt + 1,
                        len(system_prompt),
                        len(user_prompt),
                    )
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=600,
                    )
                    content = completion.choices[0].message.content
                    if content and content.strip():
                        logger.info(
                            "LLM response received: model=%s chars=%s",
                            model,
                            len(content.strip()),
                        )
                        return content.strip()
                    logger.warning("LLM response was empty: model=%s attempt=%s", model, attempt + 1)
                except Exception as exc:
                    last_exception = exc
                    logger.exception(
                        "LLM request failed: model=%s attempt=%s error=%s",
                        model,
                        attempt + 1,
                        exc,
                    )
                    time.sleep(0.5)

        if fallback_message:
            logger.warning(
                "Using fallback message after LLM failure/empty response (model=%s, api_key_source=%s).",
                self.model_name,
                self._api_key_source,
            )
            return fallback_message

        logger.error(
            "LLM failed without fallback_message (model=%s, api_key_source=%s).",
            self.model_name,
            self._api_key_source,
        )
        return f"[Groq API Error]: {str(last_exception)}"

    # ------------------------------------------------------------------
    # EDA-specific public method (called by EDAAgent)
    # ------------------------------------------------------------------
    def generate_eda_summary(self, schema_info: dict, dataset_path: str, fallback_message: str | None = None) -> str:
        """Generate a concise, bullet-point EDA narrative from schema statistics."""
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
            missing  = meta.get("missing_count", 0)
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
            f"Total missing values across all columns: {schema_info.get('total_missing_values', 'unknown')}\n\n"
            f"Column Profile:\n{column_profile}\n\n"
        )

        # Append descriptive statistics if provided by the EDA agent
        descriptive_stats = schema_info.get("descriptive_stats")
        if descriptive_stats:
            user_prompt += f"Descriptive Statistics:\n{descriptive_stats}\n\n"

        user_prompt += "Please provide your structured EDA summary now."

        return self._call_groq(system_prompt, user_prompt, fallback_message=fallback_message)

    # ------------------------------------------------------------------
    # Generic wrapper used by ReportingAgent, ExplainabilityAgent, and others
    # ------------------------------------------------------------------
    def generate_summary(self, system_prompt: str, user_prompt: str, fallback_message: str | None = None) -> str:
        """Generic prompt → summary helper. Delegates to _call_groq."""
        return self._call_groq(system_prompt, user_prompt, fallback_message=fallback_message)
