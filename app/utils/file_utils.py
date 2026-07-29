"""
File Utilities
================

Deduplicates the "build output path + ensure directory exists" pattern
repeated across plot_utils, model_selection_agent, hyperparameter_agent,
preprocessing_agent, and pdf_service.
"""

import glob
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


def prune_old_files(directory: str, keep_last: int = 5, pattern: str = "*") -> int:
    """Deletes all but the `keep_last` most recently modified files matching
    `pattern` in `directory`. Used to stop logs/reports/uploads from
    accumulating indefinitely across pipeline runs, since each run writes a
    new timestamped file rather than overwriting one. Returns the number of
    files deleted; a file that fails to delete (e.g. still open) is skipped
    rather than raised, since this is a best-effort cleanup, not a critical
    pipeline step."""
    if not os.path.isdir(directory):
        return 0
    files = sorted(
        glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime, reverse=True
    )
    deleted = 0
    for f in files[keep_last:]:
        try:
            os.remove(f)
            deleted += 1
        except OSError:
            pass
    return deleted