"""Materializer integration test against the real ``Benchmark`` class.

The stub-based tests exercise the shape the materializer was originally
coded against, but the real ``Benchmark`` facade exposes a different
API. This module verifies the materializer works with the real class.
"""

from __future__ import annotations

import json

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.benchmark.error_analysis.materializer import ErrorAnalysisMaterializer
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.results.verification_result_set import VerificationResultSet

from .fixtures import make_failure, make_pass

_VALID_TEMPLATE = (
    "from karenina.schemas.entities import BaseAnswer\n"
    "from pydantic import Field\n"
    "\n"
    "class Answer(BaseAnswer):\n"
    "    value: str = Field(description='answer text')\n"
    "    def model_post_init(self, __context):\n"
    "        self.correct = {'value': 'expected'}\n"
    "    def verify(self) -> bool:\n"
    "        return self.value == self.correct['value']\n"
)


@pytest.fixture
def real_benchmark() -> Benchmark:
    """Build a real Benchmark with two questions and a global rubric."""
    bench = Benchmark.create(name="real-sample", description="integration fixture")
    bench.add_question(
        question="2+2?",
        raw_answer="4",
        answer_template=_VALID_TEMPLATE,
    )
    bench.add_question(
        question="Capital of France?",
        raw_answer="Paris",
        answer_template=_VALID_TEMPLATE,
    )
    rubric = Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="is_concise",
                description="Is the response concise?",
                kind="boolean",
            )
        ]
    )
    bench.set_global_rubric(rubric)
    return bench


@pytest.fixture
def real_result_set(real_benchmark: Benchmark) -> VerificationResultSet:
    """Build a result set that references the real benchmark's question IDs."""
    questions = real_benchmark.get_all_questions_as_objects()
    q1, q2 = questions[0], questions[1]
    passed = make_pass(question_id=q1.id)
    failed = make_failure(
        question_id=q2.id,
        category=FailureCategory.CONTENT,
        stage="verify_template",
        reason="expected Paris",
    )
    return VerificationResultSet(results=[passed, failed], scenario_results=None)


@pytest.mark.unit
class TestMaterializerRealBenchmark:
    """End-to-end materializer sanity against the real Benchmark class."""

    def test_builds_tree_against_real_benchmark(
        self,
        tmp_path,
        real_benchmark: Benchmark,
        real_result_set: VerificationResultSet,
    ) -> None:
        """Materializer.build must not crash and must write all artifacts."""
        out_dir = tmp_path / "analysis"
        ErrorAnalysisMaterializer().build(real_result_set, real_benchmark, out_dir)

        # Benchmark artifacts
        questions_jsonl = out_dir / "benchmark" / "questions.jsonl"
        rubric_json = out_dir / "benchmark" / "rubric.json"
        metadata_md = out_dir / "benchmark" / "metadata.md"
        assert questions_jsonl.exists()
        assert rubric_json.exists()
        assert metadata_md.exists()

        # Templates directory should carry one .py per question.
        templates_dir = out_dir / "benchmark" / "templates"
        assert templates_dir.is_dir()
        template_files = list(templates_dir.glob("q_*.py"))
        assert len(template_files) == 2
        for template_file in template_files:
            body = template_file.read_text(encoding="utf-8")
            assert "class Answer(BaseAnswer)" in body

        # questions.jsonl must have one record per question with the expected keys.
        lines = questions_jsonl.read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]
        assert len(parsed) == 2
        for record in parsed:
            assert set(record.keys()) == {"id", "keywords", "raw_answer", "text"}

        # rubric.json must reflect the attached rubric.
        rubric_dump = json.loads(rubric_json.read_text(encoding="utf-8"))
        assert rubric_dump.get("llm_traits"), "global rubric llm_traits must survive dump"

        # Case buckets must have content.
        assert (out_dir / "INDEX.md").exists()
        assert any((out_dir / "passes").glob("q_*.md"))
        assert any((out_dir / "failures" / "content").glob("q_*.md"))

    def test_large_tool_result_is_offloaded_to_case_assets(
        self,
        tmp_path,
        real_benchmark: Benchmark,
    ) -> None:
        """A long tool_result in trace_messages is offloaded and referenced."""
        questions = real_benchmark.get_all_questions_as_objects()
        q1 = questions[0]
        big_blob = "Z" * 5000
        passed = make_pass(question_id=q1.id)
        # QA-style trace: no user role, just assistant text plus a big
        # tool result; exercises both the synthesized turn and offloading.
        passed.template.trace_messages = [
            {"role": "assistant", "content": "let me search"},
            {"role": "assistant", "content": "lookup(q='x')"},
            {"role": "tool", "content": big_blob},
            {"role": "assistant", "content": "done"},
        ]
        result_set = VerificationResultSet(
            results=[passed],
            scenario_results=None,
        )

        out_dir = tmp_path / "analysis"
        ErrorAnalysisMaterializer(max_trace_chars=2000).build(
            result_set,
            real_benchmark,
            out_dir,
        )

        case_files = list((out_dir / "passes").glob("q_*.md"))
        assert len(case_files) == 1
        case_body = case_files[0].read_text(encoding="utf-8")
        assert "# Trace" in case_body
        assert '<turn number="1">' in case_body
        assert "[Content offloaded:" in case_body
        assert 'offloaded="true"' in case_body

        artifacts_dir = out_dir / "case_assets" / case_files[0].stem / "traces" / "artifacts"
        assert artifacts_dir.is_dir()
        artifact_files = list(artifacts_dir.glob("*.txt"))
        assert artifact_files, "expected an offloaded artifact file"
        # The offloaded blob must contain the original content.
        assert any(f.read_text(encoding="utf-8") == big_blob for f in artifact_files)
