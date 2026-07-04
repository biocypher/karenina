"""File operation utilities for Karenina.

Provides reusable atomic file write functionality.
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO


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

    except Exception:
        # Clean up partial file on error
        if partial_path.exists():
            with contextlib.suppress(OSError):
                partial_path.unlink()
        raise


@contextlib.contextmanager
def atomic_writer(filepath: Path) -> Iterator[TextIO]:
    """Context-manager twin of atomic_write for streaming writes.

    Yields a text file handle pointing at a ``.partial`` sidecar of
    ``filepath``. On clean exit, fsyncs the handle, closes it, and
    atomically renames the sidecar into place. On any exception inside
    the ``with`` block (including ``KeyboardInterrupt`` / ``SystemExit``
    via ``BaseException``), removes the ``.partial`` file and reraises.

    Args:
        filepath: Target file path.

    Yields:
        An open writable text file handle for the ``.partial`` sidecar.

    Raises:
        OSError: If the partial write, rename, or cleanup fails.
    """
    partial_path = filepath.with_suffix(filepath.suffix + ".partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    f: TextIO | None = None
    try:
        f = open(partial_path, "w", encoding="utf-8")  # noqa: SIM115
        yield f
        f.flush()
        os.fsync(f.fileno())
        f.close()
        partial_path.replace(filepath)
    except BaseException:
        if f is not None and not f.closed:
            with contextlib.suppress(OSError):
                f.close()
        with contextlib.suppress(OSError):
            partial_path.unlink()
        raise
