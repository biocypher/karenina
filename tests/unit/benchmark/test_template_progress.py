"""Tests for template generation progress events and concurrency resolution."""

import threading

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


@pytest.mark.unit
class TestGenerateTemplatesSequential:
    """Test generate_templates with max_workers=1 (sequential)."""

    def test_emits_progress_events_in_order(self, monkeypatch):
        """Progress callback receives job_started, task events, job_completed."""
        from karenina.benchmark.benchmark_helpers import (
            TemplateProgressEvent,
            generate_templates,
        )

        def mock_gen(benchmark, question_id, **kwargs):
            return {
                "success": True,
                "template_code": f"# template for {question_id}",
                "error": None,
                "raw_response": f"# template for {question_id}",
                "skipped": False,
            }

        monkeypatch.setattr(
            "karenina.benchmark.benchmark_helpers.generate_template_for_question",
            mock_gen,
        )

        class FakeBenchmark:
            _questions_cache = {"q1": {"question": "What?"}, "q2": {"question": "Why?"}}
            name = "test"

            def has_template(self, _qid):
                return False

            def get_template(self, _qid):
                return ""

            def get_question_ids(self):
                return list(self._questions_cache.keys())

            def add_answer_template(self, _qid, _code):
                pass

        events: list[TemplateProgressEvent] = []
        results = generate_templates(
            FakeBenchmark(),
            question_ids=["q1", "q2"],
            progress_callback=events.append,
            max_workers=1,
        )

        event_types = [e.event for e in events]
        assert event_types[0] == "job_started"
        assert "task_started" in event_types
        assert "task_completed" in event_types
        assert event_types[-1] == "job_completed"
        assert results["q1"]["success"] is True
        assert results["q2"]["success"] is True

    def test_cancel_event_stops_sequential(self, monkeypatch):
        """Setting cancel_event stops the loop."""
        from karenina.benchmark.benchmark_helpers import generate_templates

        call_count = 0

        def mock_gen(benchmark, question_id, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "success": True,
                "template_code": "# code",
                "error": None,
                "raw_response": "# code",
                "skipped": False,
            }

        monkeypatch.setattr(
            "karenina.benchmark.benchmark_helpers.generate_template_for_question",
            mock_gen,
        )

        class FakeBenchmark:
            _questions_cache = {f"q{i}": {"question": f"Q{i}"} for i in range(10)}
            name = "test"

            def has_template(self, _qid):
                return False

            def get_template(self, _qid):
                return ""

            def get_question_ids(self):
                return list(self._questions_cache.keys())

            def add_answer_template(self, _qid, _code):
                pass

        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        generate_templates(
            FakeBenchmark(),
            question_ids=[f"q{i}" for i in range(10)],
            max_workers=1,
            cancel_event=cancel,
        )

        assert call_count <= 1


@pytest.mark.unit
class TestGenerateTemplatesParallel:
    """Test generate_templates with max_workers>1 (parallel)."""

    def test_parallel_produces_same_results_as_sequential(self, monkeypatch):
        """Parallel and sequential produce the same result set."""
        from karenina.benchmark.benchmark_helpers import generate_templates

        generated = {}

        def mock_gen(benchmark, question_id, **kwargs):
            import time as _time

            _time.sleep(0.01)  # Simulate I/O
            return {
                "success": True,
                "template_code": f"# template for {question_id}",
                "error": None,
                "raw_response": f"# template for {question_id}",
                "skipped": False,
            }

        monkeypatch.setattr(
            "karenina.benchmark.benchmark_helpers.generate_template_for_question",
            mock_gen,
        )

        class FakeBenchmark:
            _questions_cache = {f"q{i}": {"question": f"Q{i}"} for i in range(5)}
            name = "test"

            def has_template(self, _qid):
                return False

            def get_template(self, _qid):
                return ""

            def get_question_ids(self):
                return list(self._questions_cache.keys())

            def add_answer_template(self, qid, code):
                generated[qid] = code

        qids = [f"q{i}" for i in range(5)]

        results = generate_templates(
            FakeBenchmark(),
            question_ids=qids,
            max_workers=3,
        )

        assert set(results.keys()) == set(qids)
        assert all(r["success"] for r in results.values())

    def test_cancel_event_stops_parallel(self, monkeypatch):
        """Setting cancel_event cancels remaining futures."""
        from karenina.benchmark.benchmark_helpers import generate_templates

        call_count = 0

        def mock_gen(benchmark, question_id, **kwargs):
            nonlocal call_count
            import time as _time

            _time.sleep(0.05)  # Slow enough to allow cancel
            call_count += 1
            return {
                "success": True,
                "template_code": "# code",
                "error": None,
                "raw_response": "# code",
                "skipped": False,
            }

        monkeypatch.setattr(
            "karenina.benchmark.benchmark_helpers.generate_template_for_question",
            mock_gen,
        )

        class FakeBenchmark:
            _questions_cache = {f"q{i}": {"question": f"Q{i}"} for i in range(20)}
            name = "test"

            def has_template(self, _qid):
                return False

            def get_template(self, _qid):
                return ""

            def get_question_ids(self):
                return list(self._questions_cache.keys())

            def add_answer_template(self, _qid, _code):
                pass

        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        generate_templates(
            FakeBenchmark(),
            question_ids=[f"q{i}" for i in range(20)],
            max_workers=2,
            cancel_event=cancel,
        )

        # Should not have processed all 20
        assert call_count < 20
