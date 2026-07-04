"""Pre-commit hook for docs notebooks: sync jupytext pairs and ruff-format code cells.

Why this exists: ruff cannot parse jupytext markdown, so a plain ruff-format
hook rewrites code cells only in the .ipynb half of a pair and silently
desyncs it from its markdown source. Piping a formatter through
``jupytext --sync --pipe`` is no alternative for markdown pairing because it
compounds inter-cell blank lines on every pass. The only stable fixpoint is
the one this script produces: pair synced, with both representations
carrying ruff-formatted code.

For every file passed (pre-commit passes the staged paths):

1. Member of a jupytext pair: run ``jupytext --sync`` (mtime decides the
   direction, as in normal jupytext usage), then ``ruff format`` on the
   .ipynb, then propagate the formatted cells back to the markdown with a
   second sync.
2. Unpaired .ipynb: run ``ruff format`` on it.
3. Unpaired .md: skipped.

Exits 1 when any file was modified. Re-stage the reported files and commit
again, as with any other fixing pre-commit hook.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import jupytext
from jupytext.paired_paths import paired_paths


def _digest(path: Path) -> str | None:
    """Return a content hash for the file, or None when it does not exist."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> None:
    """Run a command, forwarding its output, and abort on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    for stream in (result.stdout, result.stderr):
        if stream.strip():
            print(stream.strip())
    if result.returncode != 0:
        print(f"[docs-pairs] command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def _pair_for(path: Path) -> tuple[Path, Path] | None:
    """Return (md, ipynb) for a paired file, or None when the file is unpaired."""
    fmt = path.suffix.lstrip(".")
    try:
        notebook = jupytext.read(path)
    except Exception:
        return None
    formats = (notebook.metadata.get("jupytext") or {}).get("formats") or ""
    if "ipynb" not in formats or "md" not in formats:
        return None
    try:
        members = [Path(p) for p, _ in paired_paths(str(path), fmt, formats)]
    except Exception:
        return None
    md = next((p for p in members if p.suffix == ".md"), None)
    ipynb = next((p for p in members if p.suffix == ".ipynb"), None)
    if md is None or ipynb is None or not md.exists() or not ipynb.exists():
        return None
    return md, ipynb


def main(files: list[str]) -> int:
    """Sync and format the given docs files. Return the process exit code."""
    seen_pairs: set[Path] = set()
    modified: list[str] = []

    for name in files:
        path = Path(name)
        if not path.exists():
            continue

        pair = _pair_for(path)

        if pair is None:
            if path.suffix != ".ipynb":
                continue
            before = _digest(path)
            _run(["ruff", "format", str(path)])
            if _digest(path) != before:
                modified.append(str(path))
            continue

        md, ipynb = pair
        if ipynb in seen_pairs:
            continue
        seen_pairs.add(ipynb)

        before_md, before_ipynb = _digest(md), _digest(ipynb)

        # Sync the pair first so the most recent edits win, then format the
        # notebook, then push the formatted cells back into the markdown.
        _run(["jupytext", "--sync", str(path)])
        pre_format = _digest(ipynb)
        _run(["ruff", "format", str(ipynb)])
        if _digest(ipynb) != pre_format:
            os.utime(ipynb, None)
            _run(["jupytext", "--sync", str(ipynb)])

        if _digest(md) != before_md:
            modified.append(str(md))
        if _digest(ipynb) != before_ipynb:
            modified.append(str(ipynb))

    if modified:
        print("[docs-pairs] files updated, please re-stage and commit again:")
        for name in modified:
            print(f"    {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
