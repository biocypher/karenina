"""Tests for Question.workspace_path field."""

import pytest

from karenina.schemas.entities.question import Question


@pytest.mark.unit
class TestQuestionWorkspacePath:
    def _make_question(self, **overrides):
        defaults = {"question": "Fix the bug", "raw_answer": "Fixed"}
        defaults.update(overrides)
        return Question(**defaults)

    def test_workspace_path_defaults_to_none(self):
        q = self._make_question()
        assert q.workspace_path is None

    def test_workspace_path_accepts_relative_string(self):
        q = self._make_question(workspace_path="task_01")
        assert q.workspace_path == "task_01"

    def test_workspace_path_accepts_nested_relative(self):
        q = self._make_question(workspace_path="coding/task_01")
        assert q.workspace_path == "coding/task_01"

    def test_workspace_path_serializes(self):
        q = self._make_question(workspace_path="task_01")
        data = q.model_dump()
        assert data["workspace_path"] == "task_01"

    def test_workspace_path_roundtrips(self):
        q = self._make_question(workspace_path="task_01")
        q2 = Question.model_validate(q.model_dump())
        assert q2.workspace_path == "task_01"
