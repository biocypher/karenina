"""Tests for the YAML frontmatter emitted by case_renderer."""

from __future__ import annotations

import pytest
import yaml

from karenina.benchmark.error_analysis.case_renderer import render_qa_case
from karenina.schemas.results.failure import FailureCategory

from .fixtures import make_failure, make_pass


def _split_frontmatter(body: str) -> tuple[dict, str]:
    assert body.startswith("---\n"), f"Missing frontmatter prefix: {body[:40]!r}"
    _, fm, rest = body.split("---\n", 2)
    return yaml.safe_load(fm), rest


@pytest.mark.unit
class TestQaCaseFrontmatter:
    def test_pass_frontmatter_keys(self):
        result = make_pass(question_id="q_001", replicate=1)
        body = render_qa_case(result, template_source=None, artifacts_dir=None)
        fm, _ = _split_frontmatter(body)
        assert fm["id"] == "q_q_001__rep_1"
        assert fm["outcome"] == "pass"
        assert fm["question_id"] == "q_001"
        assert fm["replicate"] == 1
        assert fm["model"] == "anthropic:claude-opus-4-6"
        assert fm["parsing_model"] == "anthropic:claude-sonnet-4-6"
        assert fm["category"] is None
        assert fm["group"] is None
        assert fm["stage"] is None

    def test_failure_frontmatter_keys(self):
        result = make_failure(
            question_id="q_003",
            category=FailureCategory.CONTENT,
            stage="verify_template",
            reason="price mismatch",
        )
        body = render_qa_case(result, template_source=None, artifacts_dir=None)
        fm, _ = _split_frontmatter(body)
        assert fm["outcome"] == "failure"
        assert fm["category"] == "content"
        assert fm["group"] == "content"
        assert fm["stage"] == "verify_template"
        assert fm["replicate"] is None

    def test_failure_section_present_only_for_failures(self):
        pass_body = render_qa_case(make_pass(), template_source=None, artifacts_dir=None)
        fail_body = render_qa_case(
            make_failure(reason="x"),
            template_source=None,
            artifacts_dir=None,
        )
        assert "# Failure" not in pass_body
        assert "# Failure" in fail_body

    def test_template_section_inline_under_100_lines(self):
        short_template = "class Answer(BaseAnswer):\n    result: int\n"
        body = render_qa_case(
            make_pass(),
            template_source=short_template,
            template_link="../../benchmark/templates/q_pass.py",
            artifacts_dir=None,
        )
        assert "```python" in body
        assert "class Answer(BaseAnswer):" in body
        assert "../../benchmark/templates/q_pass.py" in body

    def test_template_section_excerpt_over_100_lines(self):
        long_template = "\n".join(f"line_{i} = {i}" for i in range(150))
        body = render_qa_case(
            make_pass(),
            template_source=long_template,
            template_link="../../benchmark/templates/q_pass.py",
            artifacts_dir=None,
        )
        # Only 20 lines of the source appear in the body
        assert body.count("\nline_") == 20
        # The full 150-line source is NOT inlined.
        assert "line_149 = 149" not in body
        # The link is still there.
        assert "../../benchmark/templates/q_pass.py" in body
