"""Shared factory for VerificationContext fixtures across verification tests.

Provides a dependency-free ``make_context`` helper that builds a minimal
``VerificationContext`` suitable for stage-level unit tests. Callers may
override any field via keyword arguments.
"""

from __future__ import annotations

from typing import Any

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.schemas.config import ModelConfig


def make_context(**overrides: Any) -> VerificationContext:
    """Build a minimal ``VerificationContext`` for unit tests.

    The default configuration uses a placeholder ``ModelConfig`` for both the
    answering and parsing models; callers should override those when a stage
    under test touches model identity.

    Args:
        **overrides: Keyword arguments forwarded to ``VerificationContext``;
            any of these take precedence over the defaults below.

    Returns:
        A ``VerificationContext`` instance ready to be passed to stages.
    """
    model = ModelConfig(id="test", model_name="test-model")
    base: dict[str, Any] = {
        "question_id": "q1",
        "template_id": "tpl1",
        "question_text": "What?",
        "template_code": "class Answer: pass",
        "answering_model": model,
        "parsing_model": model,
        "raw_answer": "Y",
    }
    base.update(overrides)
    return VerificationContext(**base)


__all__ = ["make_context"]
