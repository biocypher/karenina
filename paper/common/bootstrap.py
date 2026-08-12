"""Shared bootstrap for paper experiment entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

DATA_ROOT_ENV = "KARENINA_PAPER_DATA"


def bootstrap(verbose: bool = False) -> None:
    """Load environment variables and configure logging.

    Args:
        verbose: If True, log at DEBUG level. Otherwise INFO.
    """
    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def data_root() -> Path:
    """Resolve the raw data root from the KARENINA_PAPER_DATA variable.

    Returns:
        The resolved data root directory.

    Raises:
        RuntimeError: If the variable is unset or points at a missing directory.
    """
    value = os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"{DATA_ROOT_ENV} is not set. Point it at the paper data root, "
            "see paper/README.md."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"{DATA_ROOT_ENV} points at a missing directory: {root}")
    return root
