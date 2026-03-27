"""Tests for template generation progress events and concurrency resolution."""

import pytest


@pytest.mark.unit
class TestTemplateProgressEvent:
    def test_construction_with_all_fields(self):
        from karenina.benchmark.benchmark_helpers import TemplateProgressEvent

        event = TemplateProgressEvent(
            event="task_completed",
            question_id="q1",
            processed_count=3,
            total_count=10,
            successful_count=2,
            failed_count=1,
            percentage=30.0,
            error=None,
            template_code="class Answer(BaseAnswer): pass",
            task_duration=1.5,
            in_progress_questions=["q4", "q5"],
        )
        assert event.event == "task_completed"
        assert event.question_id == "q1"
        assert event.percentage == 30.0
        assert event.template_code == "class Answer(BaseAnswer): pass"

    def test_job_level_event_has_no_question_id(self):
        from karenina.benchmark.benchmark_helpers import TemplateProgressEvent

        event = TemplateProgressEvent(
            event="job_started",
            question_id=None,
            processed_count=0,
            total_count=5,
            successful_count=0,
            failed_count=0,
            percentage=0.0,
            error=None,
            template_code=None,
            task_duration=None,
            in_progress_questions=[],
        )
        assert event.question_id is None
        assert event.event == "job_started"


@pytest.mark.unit
class TestResolveMaxWorkers:
    def test_explicit_value_returned(self):
        from karenina.benchmark.benchmark_helpers import _resolve_max_workers

        assert _resolve_max_workers(4) == 4

    def test_none_reads_env_var(self, monkeypatch):
        from karenina.benchmark.benchmark_helpers import _resolve_max_workers

        monkeypatch.setenv("KARENINA_ASYNC_MAX_WORKERS", "3")
        assert _resolve_max_workers(None) == 3

    def test_none_defaults_to_1_without_env(self, monkeypatch):
        from karenina.benchmark.benchmark_helpers import _resolve_max_workers

        monkeypatch.delenv("KARENINA_ASYNC_MAX_WORKERS", raising=False)
        assert _resolve_max_workers(None) == 1

    def test_env_var_non_numeric_defaults_to_1(self, monkeypatch):
        from karenina.benchmark.benchmark_helpers import _resolve_max_workers

        monkeypatch.setenv("KARENINA_ASYNC_MAX_WORKERS", "abc")
        assert _resolve_max_workers(None) == 1
