"""End-to-end materializer tests over hand-constructed fixtures."""

from __future__ import annotations

import json

import pytest
import yaml

from karenina.benchmark.error_analysis.exceptions import MaterializationError
from karenina.benchmark.error_analysis.materializer import ErrorAnalysisMaterializer
from karenina.schemas.entities.question import Question
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.results.verification_result_set import VerificationResultSet

from .fixtures import make_failure, make_pass


class _StubBenchmark:
    """Minimal stand-in for Benchmark, carrying just the fields the materializer reads."""

    def __init__(self, name: str, questions: list[Question], rubric_dump: dict | None = None):
        self.name = name
        self._questions = {q.id: q for q in questions}
        self._rubric_dump = rubric_dump

    @property
    def questions(self):
        return list(self._questions.values())

    def get_question(self, question_id: str) -> Question | None:
        return self._questions.get(question_id)

    @property
    def rubric(self):
        if self._rubric_dump is None:
            return None

        class _Dumpable:
            def __init__(self, data):
                self._data = data

            def model_dump(self, **_):
                return self._data

        return _Dumpable(self._rubric_dump)

    def get_template_source(self, question_id: str) -> str | None:
        return f"# template for {question_id}\nclass Answer: pass\n"


@pytest.fixture
def tiny_benchmark():
    q1 = Question(question="2+2?", raw_answer="4", keywords=["arith"])
    q2 = Question(question="Capital of France?", raw_answer="Paris")
    return _StubBenchmark(name="sample", questions=[q1, q2])


@pytest.fixture
def tiny_result_set(tiny_benchmark):
    q1, q2 = tiny_benchmark.questions
    passed = make_pass(question_id=q1.id)
    failed = make_failure(
        question_id=q2.id,
        category=FailureCategory.CONTENT,
        stage="verify_template",
        reason="expected Paris",
    )
    return VerificationResultSet(results=[passed, failed], scenario_results=None)


@pytest.mark.unit
class TestMaterializerBuild:
    def test_builds_expected_tree(self, tmp_path, tiny_benchmark, tiny_result_set):
        out_dir = tmp_path / "analysis"
        materializer = ErrorAnalysisMaterializer()
        materializer.build(tiny_result_set, tiny_benchmark, out_dir)
        assert (out_dir / "INDEX.md").exists()
        assert (out_dir / "benchmark" / "questions.jsonl").exists()
        assert (out_dir / "benchmark" / "rubric.json").exists()
        assert any((out_dir / "passes").glob("q_*.md"))
        assert any((out_dir / "failures" / "content").glob("q_*.md"))

    def test_questions_jsonl_uses_keywords_key(self, tmp_path, tiny_benchmark, tiny_result_set):
        out_dir = tmp_path / "analysis"
        ErrorAnalysisMaterializer().build(tiny_result_set, tiny_benchmark, out_dir)
        lines = (out_dir / "benchmark" / "questions.jsonl").read_text().splitlines()
        parsed = [json.loads(line) for line in lines]
        assert all(set(q.keys()) == {"id", "keywords", "raw_answer", "text"} for q in parsed)
        has_keywords = next(q for q in parsed if q["keywords"])
        assert has_keywords["keywords"] == ["arith"]

    def test_refuses_nonempty_out_dir_without_force(self, tmp_path, tiny_benchmark, tiny_result_set):
        out_dir = tmp_path / "analysis"
        out_dir.mkdir()
        (out_dir / "preexisting.txt").write_text("something")
        with pytest.raises(MaterializationError):
            ErrorAnalysisMaterializer().build(tiny_result_set, tiny_benchmark, out_dir)

    def test_force_preserves_prior_report(self, tmp_path, tiny_benchmark, tiny_result_set):
        out_dir = tmp_path / "analysis"
        out_dir.mkdir()
        (out_dir / "REPORT.md").write_text("previous analysis")
        ErrorAnalysisMaterializer().build(tiny_result_set, tiny_benchmark, out_dir, force=True)
        assert (out_dir / "REPORT.previous.md").read_text() == "previous analysis"
        assert not (out_dir / "REPORT.md").exists()  # launcher has not run

    def test_idempotence_except_index_timestamp(self, tmp_path, tiny_benchmark, tiny_result_set):
        out_dir = tmp_path / "analysis"
        materializer = ErrorAnalysisMaterializer()
        materializer.build(tiny_result_set, tiny_benchmark, out_dir)
        first = {p.relative_to(out_dir): p.read_text() for p in out_dir.rglob("*") if p.is_file()}
        materializer.build(tiny_result_set, tiny_benchmark, out_dir, force=True)
        second = {p.relative_to(out_dir): p.read_text() for p in out_dir.rglob("*") if p.is_file()}
        for rel_path in set(first) | set(second):
            if rel_path.name == "INDEX.md":
                continue
            assert first.get(rel_path) == second.get(rel_path), f"mismatch at {rel_path}"

    def test_bucketing_by_category(self, tmp_path, tiny_benchmark, tiny_result_set):
        out_dir = tmp_path / "analysis"
        ErrorAnalysisMaterializer().build(tiny_result_set, tiny_benchmark, out_dir)
        parsing_bucket = out_dir / "failures" / "parsing"
        assert not parsing_bucket.exists()  # only content failures in this set
        content_bucket = out_dir / "failures" / "content"
        assert content_bucket.exists()
        cases = list(content_bucket.glob("*.md"))
        assert len(cases) == 1
        fm = yaml.safe_load(cases[0].read_text().split("---\n", 2)[1])
        assert fm["category"] == "content"
