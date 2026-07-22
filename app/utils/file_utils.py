"""
File Utilities
================

Deduplicates the "build output path + ensure directory exists" pattern
repeated across plot_utils, model_selection_agent, hyperparameter_agent,
preprocessing_agent, and pdf_service.
"""

import os

DATA_ROOT = "data"


def ensure_dir(*path_parts: str) -> str:
    """Builds a path from parts under data/, creates it if missing, and returns it."""
    path = os.path.join(DATA_ROOT, *path_parts)
    os.makedirs(path, exist_ok=True)
    return path


def output_path(*path_parts: str) -> str:
    """Shortcut for ensure_dir('outputs', ...) — the most common case."""
    *subdirs, filename = path_parts
    directory = ensure_dir("outputs", *subdirs)
    return os.path.join(directory, filename)