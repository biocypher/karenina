"""Shared bootstrap for paper experiment entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from paper.config import DATA_DEPOSIT_DIRNAME, DATA_ROOT_ENV, DEFAULT_DATA_ROOTS

logger = logging.getLogger(__name__)

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
    if not verbose:
        logging.getLogger("karenina.benchmark.verification.stages").setLevel(logging.WARNING)


def data_root() -> Path:
    """Resolve the portable paper data deposit location.

    An explicit ``KARENINA_PAPER_DATA`` value is authoritative. Without one,
    the standard Zenodo extraction folder is discovered either beside the
    cloned Karenina repository or directly inside it.

    Returns:
        The resolved data root directory.

    Raises:
        RuntimeError: If an explicit path is invalid or no standard deposit
            folder can be discovered.
    """
    value = os.environ.get(DATA_ROOT_ENV)
    if value:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"{DATA_ROOT_ENV} points at a missing directory: {root}")
        return root

    for candidate in DEFAULT_DATA_ROOTS:
        if candidate.is_dir():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in DEFAULT_DATA_ROOTS)
    raise RuntimeError(
        f"Paper data deposit not found. Extract {DATA_DEPOSIT_DIRNAME!r} "
        f"beside the cloned repository, or set {DATA_ROOT_ENV}. Searched: {searched}"
    )


def input_path(relative: str) -> Path:
    """Resolve a required archive member below the configured paper data root."""
    path = data_root() / relative
    if not path.exists():
        raise FileNotFoundError(f"Paper input not found: {path}")
    return path
