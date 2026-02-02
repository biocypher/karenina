"""File operation utilities for Karenina.

Provides reusable atomic file write functionality.
"""

import contextlib
import os
from pathlib import Path


def atomic_write(filepath: Path, data: str) -> None:
    """Write content to file atomically using write-rename pattern.

    Writes data to a temporary partial file, then atomically renames it
    to the target path. This ensures the file is never in a partially
    written state.

    Args:
        filepath: Target file path
        data: Content to write

    Raises:
        OSError: If the write or rename fails
    """
    partial_path = filepath.with_suffix(filepath.suffix + ".partial")

    try:
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        partial_path.replace(filepath)

    except Exception as e:
        # Clean up partial file on error
        if partial_path.exists():
            with contextlib.suppress(OSError):
                partial_path.unlink()
        raise e
