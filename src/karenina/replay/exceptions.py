"""Exception hierarchy for the replay layer."""

from __future__ import annotations

from typing import Any

from karenina.exceptions import KareninaError


class ReplayError(KareninaError):
    """Base exception for the replay layer."""


class ReplayMissError(ReplayError):
    """Raised in strict mode when no matching entry is found.

    Attributes:
        key: The ReplayKey that was looked up (Any to avoid a circular
            import with store.py).
    """

    def __init__(self, message: str, *, key: Any = None) -> None:
        super().__init__(message)
        self.key = key


class ReplayHydrationError(ReplayError):
    """Raised when parsed_answer_fields cannot be validated against the
    current Answer class.

    Attributes:
        captured_fields: The parsed field dict that failed validation.
        inner: The underlying exception (typically ValidationError).
    """

    def __init__(
        self,
        message: str,
        *,
        captured_fields: dict[str, Any] | None = None,
        inner: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.captured_fields = captured_fields
        self.inner = inner


class ReplayPersistenceError(ReplayError):
    """Raised on version mismatch, schema failure, or malformed replay JSON."""


class ProjectionError(ReplayError):
    """Raised by ScenarioReplayBuilder.build(strict=True) when the
    projection report contains unmatched targets or duplicate targets.

    Attributes:
        report: The full ProjectionReport attached to the error for
            post-mortem inspection. Typed ``Any`` to avoid a circular
            import with ``karenina.replay.projection`` (same treatment
            as ``ReplayMissError.key``). May be None when the caller
            did not supply one.
    """

    def __init__(self, message: str, *, report: Any = None) -> None:
        super().__init__(message)
        self.report = report
