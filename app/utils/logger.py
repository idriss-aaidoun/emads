"""
Logger Utility
================

Provides a single, consistently configured logger for the whole EMADS
pipeline. Writes to both the console (for local dev) and a rolling file
under logs/ (for post-run debugging) — independent of the in-memory
``state["logs"]`` list which is meant for the UI/report, not disk
persistence.

Usage in any module::

    from app.utils.logger import get_logger
    logger = get_logger("emads.my_agent")
    logger.info("Agent started")
    logger.debug("Detail: %s", some_var)
    logger.error("Something failed", exc_info=True)
"""

import logging
import os
from datetime import datetime

# Directory (relative to cwd) where log files are written.
LOGS_DIR = "logs"

# Shared session id so all loggers in one run write to the SAME file.
_SESSION_ID: str | None = None

# Cache so we never attach duplicate handlers.
_configured_loggers: dict[str, logging.Logger] = {}


def reset_session() -> None:
    """
    Clear the logger cache and reset the session ID.

    Call this ONCE at the beginning of each pipeline run (e.g., from the
    Streamlit UI when the user clicks "Run"). Without this, all agents reuse
    loggers that still point to the PREVIOUS run's log file, so their messages
    never appear in the new log file.
    """
    global _SESSION_ID, _configured_loggers
    # Close and detach all existing handlers before we throw away the references.
    for lgr in _configured_loggers.values():
        for handler in lgr.handlers[:]:
            handler.flush()
            handler.close()
            lgr.removeHandler(handler)
    _configured_loggers = {}
    _SESSION_ID = None  # will be re-generated on the next get_logger() call


def _get_session_id() -> str:
    """Returns (and initialises on first call) the shared session id."""
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _SESSION_ID


def get_log_file_path() -> str:
    """Returns the absolute path to the current session's log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    return os.path.abspath(os.path.join(LOGS_DIR, f"{_get_session_id()}.log"))


def get_logger(name: str = "emads", level: int = logging.DEBUG) -> logging.Logger:
    """
    Returns a configured logger.

    Safe to call multiple times with the same name — avoids attaching
    duplicate handlers (a common logging bug when get_logger() is called
    once per agent instantiation).

    Args:
        name:  Logger hierarchy name, e.g. ``"emads.eda_agent"``.
        level: Minimum log level (default DEBUG so all messages are captured
               to the file; the console handler uses INFO for readability).
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    log_file = get_log_file_path()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — DEBUG and above (captures everything)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Console handler — INFO and above (less noise during dev)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    _configured_loggers[name] = logger

    # Print log file location only once (on the root "emads" logger)
    if name == "emads":
        logger.info("=== EMADS pipeline logger initialised | log file: %s ===", log_file)

    return logger


# Convenience alias — all agents can use this directly.
get_pipeline_logger = get_logger