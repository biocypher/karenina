"""Tests for Question model_config extra='forbid' enforcement.

Verifies that unknown extra fields are rejected while the legacy
``tags`` key continues to work via the mode='before' model_validator.
"""

import pytest
from pydantic import ValidationError

from karenina.schemas.entities.question import Question


@pytest.mark.unit
class TestQuestionExtraForbid:
    """Question rejects unknown extra fields."""

    def _make_question(self, **overrides):
        defaults = {"question": "What is 2+2?", "raw_answer": "4"}
        defaults.update(overrides)
        return Question(**defaults)

    def test_unknown_extra_field_rejected(self):
        """An unrecognized field raises ValidationError."""
        with pytest.raises(ValidationError, match="extra_field"):
            self._make_question(extra_field="not allowed")

    def test_multiple_unknown_fields_rejected(self):
        """Multiple unrecognized fields raise ValidationError."""
        with pytest.raises(ValidationError):
            self._make_question(foo="bar", baz=123)

    def test_legacy_tags_still_converted(self):
        """Legacy ``tags`` key is accepted and converted to ``keywords``."""
        q = self._make_question(tags=["bio", "chem"])
        assert q.keywords == ["bio", "chem"]

    def test_legacy_tags_with_unknown_field_rejected(self):
        """Legacy ``tags`` works, but extra unknown fields are still rejected."""
        with pytest.raises(ValidationError, match="unknown_field"):
            Question(
                question="What is 2+2?",
                raw_answer="4",
                tags=["bio"],
                unknown_field="rejected",
            )

    def test_known_fields_pass(self):
        """All known fields are accepted without error."""
        q = self._make_question(
            keywords=["bio"],
            answer_notes="some notes",
            answer_template="MyTemplate",
        )
        assert q.keywords == ["bio"]
        assert q.answer_notes == "some notes"
