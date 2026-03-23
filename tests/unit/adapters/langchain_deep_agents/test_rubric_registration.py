"""Tests for Deep Agents rubric task registration."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestRubricTaskRegistration:
    def test_rubric_dynamic_presence_check_registered(self):
        """Deep Agents should register rubric_dynamic_presence_check like other adapters."""
        from karenina.adapters.langchain_deep_agents.prompts.rubric import _RUBRIC_TASKS

        assert "rubric_dynamic_presence_check" in _RUBRIC_TASKS

    def test_rubric_task_count_matches_other_adapters(self):
        """Deep Agents should register the same number of rubric tasks as Claude Tool."""
        from karenina.adapters.langchain_deep_agents.prompts.rubric import _RUBRIC_TASKS

        assert len(_RUBRIC_TASKS) == 6
