"""JSON persistence for ReplayStore.

The on-disk format is a version-wrapped object with a list of
(key, entry) pairs. In-memory tuples never appear in JSON; the
serializer emits {"key": {...}, "entry": {...}} objects.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from karenina.replay.exceptions import ReplayPersistenceError
from karenina.replay.store import (
    ReplayEntry,
    ReplayKey,
    ReplayMissPolicy,
    ReplayStore,
)

logger = logging.getLogger(__name__)

CURRENT_VERSION = 1


def dump(store: ReplayStore, path: str | Path) -> None:
    """Write store to JSON atomically via tempfile + os.replace().

    Args:
        store: The ReplayStore to serialize.
        path: Destination path for the JSON file.

    Raises:
        OSError: If the write or rename fails. The temp file is always
            cleaned up before the exception propagates.
    """
    path = Path(path)
    payload = {
        "version": CURRENT_VERSION,
        "miss_policy": store.miss_policy,
        "entries": [{"key": key.model_dump(), "entry": entry.model_dump()} for key, entry in store.entries],
    }
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:  # noqa: BLE001
        # Ensure the temp file is removed on any failure (including
        # KeyboardInterrupt) so we never leak half-written files.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def load(
    path: str | Path,
    *,
    miss_policy: ReplayMissPolicy | None = None,
) -> ReplayStore:
    """Load a ReplayStore from JSON.

    Args:
        path: Source path for the JSON file.
        miss_policy: If supplied, overrides the value encoded in the file.

    Returns:
        A ReplayStore populated from the file.

    Raises:
        ReplayPersistenceError: On version mismatch, malformed JSON, or
            schema validation errors.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReplayPersistenceError(f"Cannot read replay file {path}: {e}") from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReplayPersistenceError(f"Malformed JSON in {path}: {e}") from e

    if not isinstance(payload, dict):
        raise ReplayPersistenceError(f"Replay file {path} must contain a JSON object")

    version = payload.get("version")
    if version != CURRENT_VERSION:
        raise ReplayPersistenceError(
            f"Unsupported replay file version {version!r} in {path}; expected {CURRENT_VERSION}"
        )

    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ReplayPersistenceError(f"Replay file {path}: 'entries' must be a list")

    try:
        entries: list[tuple[ReplayKey, ReplayEntry]] = [
            (ReplayKey.model_validate(item["key"]), ReplayEntry.model_validate(item["entry"])) for item in raw_entries
        ]
    except Exception as e:  # noqa: BLE001  # Pydantic ValidationError, KeyError, TypeError
        raise ReplayPersistenceError(f"Schema error in {path}: {e}") from e

    file_policy: ReplayMissPolicy = payload.get("miss_policy", "fall_through")
    effective_policy = miss_policy if miss_policy is not None else file_policy

    store = ReplayStore(miss_policy=effective_policy, entries=entries)
    logger.info("Loaded replay store: %d entries from %s", len(entries), path)
    return store
