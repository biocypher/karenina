"""Minimal karenina answer template skeleton.

Replace the class name, field definitions, and ground truth values
with your domain-specific evaluation criteria.
"""

from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import BooleanMatch


class Answer(BaseAnswer):
    """Template for evaluating [YOUR DOMAIN]."""

    identifies_key_fact: bool = VerifiedField(
        description=(
            "True if the response correctly identifies [KEY FACT]. "
            "False if [KEY FACT] is not mentioned or a different "
            "[ENTITY] is identified instead."
        ),
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
