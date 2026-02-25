"""Tests for the ground_truth(self) -> model_post_init bridge in BaseAnswer.

Verifies that:
1. ground_truth(self) gets bridged to model_post_init transparently
2. When both are defined, model_post_init takes precedence
3. Neither defined (regex-only template) causes no error
4. exec()-created classes with ground_truth work through validate_answer_template
5. inject_question_id_into_answer_class works with ground_truth-based parent
6. self.regex set in ground_truth works correctly
"""

from pathlib import Path

import pytest

from karenina.benchmark.authoring.answers.generator import (
    inject_question_id_into_answer_class,
)
from karenina.benchmark.verification.utils.template_validation import (
    validate_answer_template,
)
from karenina.schemas.entities import BaseAnswer


@pytest.mark.unit
class TestGroundTruthBridge:
    """Tests for the ground_truth -> model_post_init bridge."""

    def test_ground_truth_gets_bridged(self):
        """ground_truth(self) should be bridged to model_post_init."""

        class Answer(BaseAnswer):
            target: str = "default"

            def ground_truth(self):
                self.correct = {"target": "BCL2"}

        answer = Answer(target="BCL2")
        assert hasattr(answer, "correct")
        assert answer.correct == {"target": "BCL2"}
        # The bridge should have injected model_post_init
        assert "model_post_init" in Answer.__dict__

    def test_model_post_init_takes_precedence(self):
        """When both are defined, model_post_init takes precedence (bridge skipped)."""

        class Answer(BaseAnswer):
            target: str = "default"

            def ground_truth(self):
                self.correct = {"target": "WRONG"}

            def model_post_init(self, __context):
                self.correct = {"target": "CORRECT"}

        answer = Answer(target="test")
        assert answer.correct == {"target": "CORRECT"}

    def test_neither_defined_no_error(self):
        """Regex-only template with neither method should not error."""

        class Answer(BaseAnswer):
            value: str = "default"

            def verify(self) -> bool:
                return True

        answer = Answer(value="test")
        # No correct attribute set, no error
        assert not hasattr(answer, "correct") or answer.correct is None or isinstance(answer.correct, dict)

    def test_exec_created_class_with_ground_truth(self):
        """exec()-created class with ground_truth should work through validate_answer_template."""
        template_code = """class Answer(BaseAnswer):
    target: str = Field(description="The drug target")

    def ground_truth(self):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target == self.correct["target"]
"""
        is_valid, error, answer_cls = validate_answer_template(template_code)
        assert is_valid, f"Validation failed: {error}"
        assert answer_cls is not None

        # Verify the bridged class works
        instance = answer_cls(target="BCL2")
        assert instance.correct == {"target": "BCL2"}
        assert instance.verify() is True

    def test_inject_question_id_works_with_ground_truth(self):
        """inject_question_id_into_answer_class should work with ground_truth-based parent."""

        class Answer(BaseAnswer):
            target: str = "default"

            def ground_truth(self):
                self.correct = {"target": "BCL2"}

            def verify(self) -> bool:
                return self.target == self.correct["target"]

        AnswerWithID = inject_question_id_into_answer_class(Answer, "Q001")
        instance = AnswerWithID(target="BCL2")

        # Check that ground truth was set via the bridge
        assert instance.correct == {"target": "BCL2"}
        # Check that question ID was injected
        assert instance.id == "Q001"
        # Check that verification works
        assert instance.verify() is True

    def test_regex_in_ground_truth(self):
        """self.regex set in ground_truth should work correctly."""

        class Answer(BaseAnswer):
            value: str = "default"

            def ground_truth(self):
                self.correct = {}
                self.regex = {
                    "citation": {
                        "pattern": r"\[\d+\]",
                        "expected": 3,
                        "match_type": "count",
                    }
                }

            def verify(self) -> bool:
                return True

        answer = Answer(value="test")
        assert answer.correct == {}
        assert "citation" in answer.regex
        assert answer.regex["citation"]["expected"] == 3

    def test_ground_truth_fixture_template(self, fixtures_dir: Path):
        """Test that the ground_truth_style fixture template loads and works."""
        import importlib.util
        import sys

        template_path = fixtures_dir / "templates" / "ground_truth_style.py"
        spec = importlib.util.spec_from_file_location("gt_template", template_path)
        assert spec is not None and spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules["gt_template"] = module
        spec.loader.exec_module(module)

        Answer = module.Answer
        assert issubclass(Answer, BaseAnswer)

        # Correct answer
        correct = Answer(target="BCL2")
        assert correct.correct == {"target": "BCL2"}
        assert correct.verify() is True

        # Wrong answer
        wrong = Answer(target="TP53")
        assert wrong.verify() is False

        # Case insensitive
        case_insensitive = Answer(target="bcl2")
        assert case_insensitive.verify() is True
