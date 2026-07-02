"""Workspace capture helpers for verification runs."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from karenina.schemas.verification import VerificationResult
from karenina.utils.file_ops import atomic_write

logger = logging.getLogger(__name__)

WorkspaceOutputMode = Literal["none", "full", "produced"]

DEFAULT_WORKSPACE_OUTPUT_EXCLUDES = (
    ".venv/",
    "venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".ipynb_checkpoints/",
    "*.pyc",
)


def safe_path_component(value: str | None) -> str:
    """Return a filesystem-safe path component."""

    if not value:
        return "unknown"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("._") or "unknown"


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _pattern_matches(pattern: str, rel_path: str, is_dir: bool) -> bool:
    normalized = pattern.replace(os.sep, "/")
    rel = rel_path.rstrip("/")
    if normalized.endswith("/"):
        prefix = normalized.rstrip("/")
        return is_dir and (rel == prefix or rel.startswith(f"{prefix}/")) or rel.startswith(f"{prefix}/")
    return fnmatch.fnmatch(rel, normalized) or fnmatch.fnmatch(Path(rel).name, normalized)


def is_excluded(rel_path: str, *, is_dir: bool, patterns: list[str] | tuple[str, ...]) -> bool:
    """Return True when a relative path should be excluded."""

    return any(_pattern_matches(pattern, rel_path, is_dir) for pattern in patterns)


def _iter_files(root: Path, patterns: list[str] | tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        kept_dirnames = []
        for dirname in dirnames:
            child = current / dirname
            rel = _rel_posix(child, root)
            if is_excluded(rel, is_dir=True, patterns=patterns):
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in filenames:
            path = current / filename
            rel = _rel_posix(path, root)
            if is_excluded(rel, is_dir=path.is_dir(), patterns=patterns):
                continue
            files.append(path)
    return files


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_workspace(root: Path, patterns: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Snapshot file content identities under a workspace root."""

    snapshot: dict[str, dict[str, Any]] = {}
    for path in _iter_files(root, patterns):
        rel = _rel_posix(path, root)
        try:
            if path.is_symlink():
                snapshot[rel] = {"kind": "symlink", "target": os.readlink(path)}
            elif path.is_file():
                stat = path.stat()
                snapshot[rel] = {"kind": "file", "size": stat.st_size, "sha256": _hash_file(path)}
        except OSError as exc:
            logger.warning("Failed to snapshot workspace file %s: %s", path, exc)
    return snapshot


def changed_workspace_files(
    root: Path,
    baseline: dict[str, dict[str, Any]],
    patterns: list[str] | tuple[str, ...],
) -> tuple[list[Path], list[dict[str, str]]]:
    """Return files that are new or content-modified relative to baseline."""

    changed: list[Path] = []
    errors: list[dict[str, str]] = []
    for path in _iter_files(root, patterns):
        rel = _rel_posix(path, root)
        try:
            previous = baseline.get(rel)
            current: dict[str, Any]
            if path.is_symlink():
                current = {"kind": "symlink", "target": os.readlink(path)}
            elif path.is_file():
                stat = path.stat()
                current = {"kind": "file", "size": stat.st_size, "sha256": _hash_file(path)}
            else:
                continue
            if current != previous:
                changed.append(path)
        except OSError as exc:
            errors.append({"path": rel, "error": str(exc)})
    return changed, errors


def _copy_file_preserving_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.readlink(source))
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def _copy_full_workspace(
    source: Path,
    destination: Path,
    patterns: list[str] | tuple[str, ...],
) -> tuple[int, list[dict[str, str]]]:
    copied = 0
    errors: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        kept_dirnames = []
        for dirname in dirnames:
            child = current / dirname
            rel = _rel_posix(child, source)
            if is_excluded(rel, is_dir=True, patterns=patterns):
                continue
            try:
                if child.is_symlink():
                    _copy_file_preserving_link(child, destination / rel)
                    copied += 1
                    continue
                (destination / rel).mkdir(parents=True, exist_ok=True)
                kept_dirnames.append(dirname)
            except OSError as exc:
                errors.append({"path": rel, "error": str(exc)})
        dirnames[:] = kept_dirnames

        for filename in filenames:
            path = current / filename
            rel = _rel_posix(path, source)
            if is_excluded(rel, is_dir=False, patterns=patterns):
                continue
            try:
                _copy_file_preserving_link(path, destination / rel)
                copied += 1
            except OSError as exc:
                errors.append({"path": rel, "error": str(exc)})
    return copied, errors


def _copy_full_workspace_count(
    source: Path,
    patterns: list[str] | tuple[str, ...],
) -> tuple[int, list[dict[str, str]]]:
    copied = 0
    errors: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        kept_dirnames = []
        for dirname in dirnames:
            child = current / dirname
            rel = _rel_posix(child, source)
            if is_excluded(rel, is_dir=True, patterns=patterns):
                continue
            if child.is_symlink():
                copied += 1
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in filenames:
            path = current / filename
            rel = _rel_posix(path, source)
            if is_excluded(rel, is_dir=False, patterns=patterns):
                continue
            try:
                if path.is_symlink() or path.is_file():
                    copied += 1
            except OSError as exc:
                errors.append({"path": rel, "error": str(exc)})
    return copied, errors


def _replace_dir(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    source.replace(destination)


def _manifest_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "manifest.jsonl", output_dir / "manifest.json"


def append_manifest_row(output_dir: Path, row: dict[str, Any]) -> None:
    """Append one JSONL workspace manifest row."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, _ = _manifest_paths(output_dir)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def compact_manifest(output_dir: Path) -> Path | None:
    """Compact manifest JSONL into manifest.json."""

    jsonl_path, json_path = _manifest_paths(output_dir)
    if not jsonl_path.exists():
        return None
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    payload = {
        "format_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "entries": rows,
    }
    atomic_write(json_path, json.dumps(payload, indent=2, default=str) + "\n")
    return json_path


def capture_workspace(
    *,
    mode: WorkspaceOutputMode,
    output_dir: Path | None,
    workspace_path: Path | None,
    baseline: dict[str, dict[str, Any]] | None,
    result: VerificationResult,
    exclude_patterns: list[str] | tuple[str, ...],
) -> None:
    """Capture a result workspace sidecar and append a manifest row."""

    if mode == "none":
        return

    if output_dir is None:
        logger.warning("Workspace capture requested but workspace_output_dir is not set")
        return

    result_id = result.metadata.result_id
    question_id = result.metadata.question_id
    folder_name = f"{safe_path_component(question_id)}__{safe_path_component(result_id)}"
    destination = output_dir / folder_name
    row: dict[str, Any] = {
        "question_id": question_id,
        "result_id": result_id,
        "run_name": result.metadata.run_name,
        "replicate": result.metadata.replicate,
        "scenario_id": result.metadata.scenario_id,
        "scenario_node": result.metadata.scenario_node,
        "scenario_turn": result.metadata.scenario_turn,
        "answering": result.metadata.answering.model_dump(mode="json"),
        "parsing": result.metadata.parsing.model_dump(mode="json"),
        "capture_mode": mode,
        "source_workspace_path": str(workspace_path) if workspace_path else None,
        "destination": str(destination),
        "status": "captured",
        "copied_file_count": 0,
        "copy_errors": [],
    }

    if workspace_path is None or not workspace_path.exists():
        row["status"] = "skipped"
        row["skip_reason"] = "workspace_not_available"
        append_manifest_row(output_dir, row)
        logger.warning("Workspace capture skipped for %s: no live workspace", question_id)
        return

    try:
        if mode == "full" and workspace_path.resolve() == destination.resolve():
            copied_count, errors = _copy_full_workspace_count(workspace_path, exclude_patterns)
            row["copied_file_count"] = copied_count
            row["copy_errors"] = errors
            if errors:
                row["status"] = "captured_with_errors"
            append_manifest_row(output_dir, row)
            return
    except OSError as exc:
        row["status"] = "skipped"
        row["skip_reason"] = f"{type(exc).__name__}: {exc}"
        append_manifest_row(output_dir, row)
        logger.warning("Workspace capture failed for %s: %s", question_id, exc, exc_info=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_parent = output_dir
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{folder_name}.tmp-", dir=tmp_parent))
    try:
        if mode == "full":
            copied_count, errors = _copy_full_workspace(workspace_path, tmp_path, exclude_patterns)
        elif mode == "produced":
            changed, detected_errors = changed_workspace_files(workspace_path, baseline or {}, exclude_patterns)
            copied_count = 0
            errors = list(detected_errors)
            for path in changed:
                rel = _rel_posix(path, workspace_path)
                try:
                    _copy_file_preserving_link(path, tmp_path / rel)
                    copied_count += 1
                except OSError as exc:
                    errors.append({"path": rel, "error": str(exc)})
        else:
            raise ValueError(f"Unsupported workspace capture mode: {mode}")

        _replace_dir(tmp_path, destination)
        row["copied_file_count"] = copied_count
        row["copy_errors"] = errors
        if errors:
            row["status"] = "captured_with_errors"
            logger.warning("Workspace capture for %s completed with %d copy error(s)", question_id, len(errors))
        append_manifest_row(output_dir, row)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "skipped"
        row["skip_reason"] = f"{type(exc).__name__}: {exc}"
        append_manifest_row(output_dir, row)
        logger.warning("Workspace capture failed for %s: %s", question_id, exc, exc_info=True)
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
