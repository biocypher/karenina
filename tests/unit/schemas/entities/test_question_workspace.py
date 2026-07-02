"""Behavior tests for ``Question.workspace_path``.

``workspace_path`` is a relative path that the verification pipeline joins onto
``workspace_root`` (see ``generate_answer._ensure_workspace``). The interesting
regression signals are:

* the field survives JSON round-trips through Pydantic (so it persists in
  checkpoints and JSON-LD exports),
* unknown fields stay rejected (so ``workspace_path`` cannot smuggle in
  arbitrary keys), and
* the value is stored verbatim — the model must not absolutize or normalize it,
  because the join with ``workspace_root`` happens at runtime.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from karenina.schemas.entities.question import Question


@pytest.mark.unit
class TestQuestionWorkspacePath:
    def _make(self, **overrides) -> Question:
        defaults = {"question": "Fix the bug", "raw_answer": "Fixed"}
        defaults.update(overrides)
        return Question(**defaults)

    def test_default_is_none(self) -> None:
        assert self._make().workspace_path is None

    def test_round_trips_through_json(self) -> None:
        """workspace_path must survive model_dump_json / model_validate_json.

        This is what the checkpoint writer and JSON-LD exporter rely on.
        """
        original = self._make(workspace_path="coding/task_01")
        restored = Question.model_validate_json(original.model_dump_json())
        assert restored.workspace_path == "coding/task_01"

    def test_stored_verbatim_no_absolutization(self) -> None:
        """A leading-slash or dotted path is kept as-is.

        The pipeline joins this onto ``workspace_root`` at runtime; if the model
        normalized it here (e.g. resolved to an absolute path) the join would
        produce a bogus directory and ``_ensure_workspace`` would raise.
        """
        for raw in ("task_01", "coding/task_01", "./tasks/task_01", "../shared/x"):
            q = self._make(workspace_path=raw)
            assert q.workspace_path == raw

    def test_workspace_path_does_not_open_extra_fields(self) -> None:
        """Adding workspace_path must not weaken the extra='forbid' guarantee."""
        with pytest.raises(ValidationError):
            self._make(workspace_path="task_01", bogus="nope")

    def test_join_with_workspace_root_yields_expected_dir(self, tmp_path: Path) -> None:
        """End-to-end-ish check: the stored relative path resolves under a root.

        This is the contract ``_ensure_workspace`` depends on — if the model ever
        stored an absolute path or mangled the string, the join would break.
        """
        root = tmp_path / "ws_root"
        (root / "coding" / "task_01").mkdir(parents=True)
        q = self._make(workspace_path="coding/task_01")

        resolved = (root / q.workspace_path).resolve()
        assert resolved.is_dir()
        assert resolved == (root / "coding" / "task_01").resolve()
