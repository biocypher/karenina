"""Stream verification results from JSON exports without loading them fully.

Two on-disk layouts exist: the v2.2 export is a top-level object with a
``results`` array, while the legacy export is a bare array whose elements may
carry an injected ``row_index``. Both layouts can be consumed one element at a
time, which keeps multi-gigabyte paper result files practical to process.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, Literal, overload

import ijson

from karenina.schemas.verification.result import VerificationResult


def _item_prefix(handle: BinaryIO) -> str:
    """Return the ijson item prefix for the stream's top-level container."""
    while first_byte := handle.read(1):
        if first_byte.isspace():
            continue
        handle.seek(0)
        if first_byte == b"[":
            return "item"
        if first_byte == b"{":
            return "results.item"
        raise ValueError("Results JSON must contain a top-level object or array")
    raise ValueError("Results JSON file is empty")


@overload
def iter_results_from_json(
    file_path: Path,
    *,
    raw: Literal[False] = False,
) -> Iterator[VerificationResult]: ...


@overload
def iter_results_from_json(
    file_path: Path,
    *,
    raw: Literal[True],
) -> Iterator[dict[str, Any]]: ...


def iter_results_from_json(
    file_path: Path,
    *,
    raw: bool = False,
) -> Iterator[VerificationResult] | Iterator[dict[str, Any]]:
    """Yield result rows from a JSON export one at a time.

    Args:
        file_path: Path to a v2.2 export or a legacy array export.
        raw: Yield dictionaries instead of validating current result models.

    Yields:
        Validated results, or dictionaries when ``raw`` is True.

    Raises:
        ValueError: If the file is empty or has an unsupported top-level value.
    """
    with file_path.open("rb") as handle:
        prefix = _item_prefix(handle)
        for record in ijson.items(handle, prefix):
            record.pop("row_index", None)
            if raw:
                yield record
            else:
                yield VerificationResult.model_validate(record)
