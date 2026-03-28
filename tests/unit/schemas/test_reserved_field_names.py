"""Tests for reserved field name validation in BaseAnswer.

Verifies that BaseAnswer subclasses cannot use field names that collide
with internal attributes (e.g., 'correct'), and that the error is raised
at class definition time with a clear message.
"""

import pytest
from pydantic import Field

from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives import BooleanMatch, ExactMatch


@pytest.mark.unit
class TestReservedFieldNameEnforcement:
    """Defining a BaseAnswer subclass with a reserved field name raises TypeError."""

    def test_correct_verified_field_rejected(self):
        """VerifiedField named 'correct' raises TypeError at class definition."""
        with pytest.raises(TypeError, match="reserved by BaseAnswer"):

            class BadAnswer(BaseAnswer):
                correct: bool = VerifiedField(
                    description="Whether the answer is correct",
                    ground_truth=True,
                    verify_with=BooleanMatch(),
                )

    def test_error_message_mentions_reserved(self):
        """Error message explicitly states the field is reserved by BaseAnswer."""
        with pytest.raises(TypeError, match="reserved by BaseAnswer"):

            class BadAnswer(BaseAnswer):
                correct: str = VerifiedField(
                    description="Correctness indicator",
                    ground_truth="yes",
                    verify_with=ExactMatch(),
                )

    def test_correct_plain_field_also_rejected(self):
        """Plain Pydantic Field named 'correct' also raises TypeError."""
        with pytest.raises(TypeError, match="reserved by BaseAnswer"):

            class BadAnswer(BaseAnswer):
                correct: bool = Field(default=False)

    def test_non_reserved_names_pass(self):
        """Fields named 'is_correct' or 'factual' do not trigger the error."""

        class GoodAnswer1(BaseAnswer):
            is_correct: bool = VerifiedField(
                description="Whether the answer is correct",
                ground_truth=True,
                verify_with=BooleanMatch(),
            )

        class GoodAnswer2(BaseAnswer):
            factual: bool = VerifiedField(
                description="Whether the answer is factual",
                ground_truth=True,
                verify_with=BooleanMatch(),
            )

        # Both classes should be usable without error
        a1 = GoodAnswer1(is_correct=True)
        a2 = GoodAnswer2(factual=True)
        assert a1.is_correct is True
        assert a2.factual is True
