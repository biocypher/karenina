"""Behavioral contracts for the port error hierarchy.

The port errors (``PortError`` and its subclasses) are the exception
vocabulary that adapters raise and the verification pipeline catches.
The contracts that matter are:

1. **Catch-all**: a single ``except PortError`` clause must catch every
   failure an adapter can raise, so the pipeline can centralise handling.
2. **Subclass relationships encode failure semantics**: ``AgentTimeoutError``
   is an ``AgentExecutionError`` (a timeout is one kind of execution
   failure) so ``except AgentExecutionError`` catches both; but
   ``AgentResponseError`` and ``ParseError`` are siblings, not children,
   because they describe distinct failure modes that callers branch on.
3. **Diagnostic fields survive raise/catch**: ``reason``,
   ``fallback_interface``, ``stderr``, ``limit_reached`` and
   ``raw_response`` are read by real consumers — the factory fallback
   path, the recursion-limit autofail stage, and the error-analysis
   materializer — so they must round-trip through raise/catch intact.
4. **ErrorRegistry classification**: every port error is PERMANENT for
   retry purposes (adapters raise them for semantic stopping conditions,
   not transient network blips), so RetryExecutor never wastes a budget
   retrying them.
5. **Exception chaining**: ``ParseError`` is documented to wrap an
   underlying cause (e.g. a provider JSON error) and must preserve the
   cause via ``raise ... from`` so error-analysis can walk the chain.

The earlier revision of this file had one assertion per inheritance
relationship (``test_X_inherits_from_Y``) and one per
``test_X_can_be_caught_as_Y``. Those tests described the class hierarchy
to itself and would only fail if Python broke ``issubclass``. The tests
below exercise the contracts the hierarchy exists to support.
"""

from __future__ import annotations

import pytest

from karenina.exceptions import KareninaError
from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
    ParseError,
    PortError,
)
from karenina.utils.errors import ErrorCategory, ErrorRegistry

# Every concrete port error class. Used for parametrised contract checks.
ALL_PORT_ERRORS = [
    AdapterUnavailableError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentResponseError,
    ParseError,
]


# ---------------------------------------------------------------------------
# Contract 1: PortError is the catch-all
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPortErrorIsCatchAll:
    """A single ``except PortError`` must catch every adapter failure."""

    @pytest.mark.parametrize("error_cls", ALL_PORT_ERRORS, ids=lambda c: c.__name__)
    def test_every_port_error_is_caught_as_port_error(self, error_cls: type[PortError]) -> None:
        """The pipeline catches PortError broadly; every subclass must be caught.

        If a new port error is added that does not inherit from PortError,
        pipeline-level handlers will silently miss it.
        """
        # Construct an instance each subclass can build with a single message arg.
        # Parametrise on the class so a future subclass is forced into this test.
        try:
            if error_cls is AdapterUnavailableError:
                raise error_cls("boom", reason="r", fallback_interface="langchain")
            elif error_cls is AgentExecutionError:
                raise error_cls("boom", stderr="trace", limit_reached=False)
            elif error_cls is ParseError:
                raise error_cls("boom", raw_response="raw")
            else:
                raise error_cls("boom")
        except PortError as caught:
            assert type(caught) is error_cls
            assert str(caught) == "boom"

    def test_port_error_inherits_from_karenina_error(self) -> None:
        """PortError is part of the KareninaError umbrella hierarchy.

        This is what lets top-level handlers in the CLI/server catch any
        karenina-originated exception in one clause.
        """
        assert issubclass(PortError, KareninaError)

    def test_port_error_message_round_trips_through_str(self) -> None:
        """str(PortError) returns the message; used by every logger call."""
        message = "Claude SDK returned malformed response"
        assert str(PortError(message)) == message
        assert PortError(message).message == message


# ---------------------------------------------------------------------------
# Contract 2: Subclass relationships encode failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubclassSemantics:
    """Subclass relationships encode the branching callers rely on."""

    def test_agent_timeout_is_agent_execution(self) -> None:
        """A timeout is one kind of execution failure.

        ``except AgentExecutionError`` in the recursion-limit-autofail
        stage catches both process crashes and timeouts; this assertion
        pins that AgentTimeoutError remains a subclass.
        """
        try:
            raise AgentTimeoutError("exceeded max_turns=25")
        except AgentExecutionError as caught:
            assert isinstance(caught, AgentTimeoutError)

    def test_agent_timeout_is_distinguishable_from_generic_execution(self) -> None:
        """Callers branch on AgentTimeoutError specifically.

        Timeout handling (treat as partial-success, surface to user) differs
        from generic AgentExecutionError handling (surface as a hard
        failure). If AgentTimeoutError were folded into the parent, that
        branch would disappear.
        """
        timeout = AgentTimeoutError("slow")
        crash = AgentExecutionError("boom")
        assert isinstance(timeout, AgentTimeoutError)
        assert not isinstance(crash, AgentTimeoutError)

    @pytest.mark.parametrize(
        "sibling_pair",
        [
            (AgentResponseError, AgentExecutionError),
            (ParseError, AgentExecutionError),
            (AdapterUnavailableError, AgentExecutionError),
            (ParseError, AgentResponseError),
        ],
        ids=lambda pair: f"{pair[0].__name__}_vs_{pair[1].__name__}",
    )
    def test_distinct_failure_modes_are_not_subclasses(
        self,
        sibling_pair: tuple[type[PortError], type[PortError]],
    ) -> None:
        """Sibling failure modes must not inherit from each other.

        ``AgentResponseError`` (malformed response) and ``ParseError``
        (structured-output parse failure) are siblings of
        ``AgentExecutionError`` (runtime failure), not children, because
        callers handle them differently.
        """
        left, right = sibling_pair
        assert not issubclass(left, right), f"{left.__name__} should not be a subclass of {right.__name__}"
        assert not issubclass(right, left), f"{right.__name__} should not be a subclass of {left.__name__}"


# ---------------------------------------------------------------------------
# Contract 3: Diagnostic fields survive raise/catch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiagnosticFieldsRoundTrip:
    """Field values set at construction survive being raised and caught.

    These fields are read by real consumers and must round-trip:
    - ``reason`` and ``fallback_interface`` drive factory auto-fallback.
    - ``stderr`` is shown by the error-analysis materializer.
    - ``limit_reached`` triggers the recursion-limit-autofail stage.
    - ``raw_response`` is materialized for debugging parse failures.
    """

    def test_adapter_unavailable_carries_reason_and_fallback(self) -> None:
        try:
            raise AdapterUnavailableError(
                message="Claude Code CLI not found",
                reason="claude executable not in PATH",
                fallback_interface="langchain",
            )
        except PortError as caught:
            assert isinstance(caught, AdapterUnavailableError)
            assert caught.reason == "claude executable not in PATH"
            assert caught.fallback_interface == "langchain"
            # The factory reads fallback_interface to pick the next adapter;
            # truthiness is the contract.
            assert bool(caught.fallback_interface) is True

    def test_adapter_unavailable_reason_defaults_to_message(self) -> None:
        """When no reason is supplied, the message stands in for it.

        Error-analysis indexing reads ``.reason`` unconditionally; the
        default ensures it is never None.
        """
        try:
            raise AdapterUnavailableError("CLI missing")
        except AdapterUnavailableError as caught:
            assert caught.reason == "CLI missing"
            assert caught.fallback_interface is None

    def test_agent_execution_carries_stderr_and_limit_reached(self) -> None:
        try:
            raise AgentExecutionError(
                message="CLI process exited with code 1",
                stderr="Error: connection refused\nRetry failed",
                limit_reached=True,
            )
        except PortError as caught:
            assert isinstance(caught, AgentExecutionError)
            assert caught.stderr == "Error: connection refused\nRetry failed"
            assert caught.limit_reached is True

    def test_agent_execution_defaults_to_no_stderr_no_limit(self) -> None:
        """Default fields must be safe for callers that read them unconditionally."""
        try:
            raise AgentExecutionError("crash")
        except AgentExecutionError as caught:
            assert caught.stderr is None
            assert caught.limit_reached is False

    def test_agent_timeout_inherits_stderr_field(self) -> None:
        """AgentTimeoutError must accept stderr (inherited from parent).

        ``wrap_sdk_error`` constructs ``AgentTimeoutError(stderr=...)``
        when the CLI process is killed; that contract must keep working.
        """
        try:
            raise AgentTimeoutError(
                message="CLI timed out",
                stderr="Killed by SIGKILL",
            )
        except AgentExecutionError as caught:  # caught via the parent
            assert isinstance(caught, AgentTimeoutError)
            assert caught.stderr == "Killed by SIGKILL"

    def test_parse_error_carries_raw_response(self) -> None:
        try:
            raise ParseError(
                message="Could not extract Answer schema",
                raw_response="The model said: I don't know.",
            )
        except PortError as caught:
            assert isinstance(caught, ParseError)
            assert caught.raw_response == "The model said: I don't know."

    def test_parse_error_raw_response_defaults_to_none(self) -> None:
        try:
            raise ParseError("no data")
        except ParseError as caught:
            assert caught.raw_response is None


# ---------------------------------------------------------------------------
# Contract 4: ErrorRegistry classifies every PortError as PERMANENT
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPortErrorsArePermanentForRetry:
    """Port errors represent semantic stopping conditions, not transient blips.

    RetryExecutor asks ErrorRegistry for a category and only retries
    CONNECTION/TIMEOUT/RATE_LIMIT/SERVER_ERROR. If any port error were
    misclassified as a retryable category, the pipeline would burn a retry
    budget re-running an operation that is guaranteed to fail the same
    way (e.g. re-running an agent that hit max_turns, or re-parsing a
    malformed response with the same input).
    """

    @pytest.mark.parametrize(
        "error_cls",
        ALL_PORT_ERRORS,
        ids=lambda c: c.__name__,
    )
    def test_port_error_classifies_as_permanent(self, error_cls: type[PortError]) -> None:
        registry = ErrorRegistry()
        # Construct a representative instance for each subclass.
        if error_cls is AdapterUnavailableError:
            exc = error_cls("x", reason="r", fallback_interface="langchain")
        elif error_cls is AgentExecutionError:
            exc = error_cls("x", stderr=None, limit_reached=False)
        elif error_cls is ParseError:
            exc = error_cls("x", raw_response="raw")
        else:
            exc = error_cls("x")

        category = registry.classify(exc)
        assert category is ErrorCategory.PERMANENT, (
            f"{error_cls.__name__} classified as {category.name}; port errors must be PERMANENT "
            f"so RetryExecutor does not retry semantic failures"
        )

    def test_retry_executor_does_not_retry_port_error(self) -> None:
        """End-to-end: a PortError raised inside RetryExecutor runs exactly once."""
        from karenina.utils.retry_policy import (
            CategoryRetryConfig,
            RetryExecutor,
            RetryPolicy,
        )

        call_count = {"n": 0}

        def fn() -> None:
            call_count["n"] += 1
            raise AgentExecutionError("CLI crashed", stderr="boom")

        # Give every retryable category a generous budget to prove the
        # executor still does not retry a PERMANENT error.
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=5, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=5, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=5, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=5, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())

        with pytest.raises(AgentExecutionError):
            executor.execute(fn)
        assert call_count["n"] == 1, "PortError must not be retried even with a large budget"


# ---------------------------------------------------------------------------
# Contract 5: Exception chaining for ParseError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExceptionChaining:
    """ParseError must preserve its underlying cause.

    The langchain parser adapter raises
    ``ParseError(...) from <provider error>`` so error-analysis can walk
    the chain to find the original JSON/ValidationError. If the cause is
    dropped (e.g. ``raise ParseError(...)`` without ``from``), debugging
    context is lost.
    """

    def test_parse_error_preserves_chained_cause(self) -> None:
        inner = ValueError("invalid JSON at line 3")
        try:
            try:
                raise inner
            except ValueError as exc:
                raise ParseError("Could not parse response", raw_response="{bad json") from exc
        except ParseError as caught:
            assert caught.raw_response == "{bad json"
            assert caught.__cause__ is inner, "ParseError must preserve __cause__ so error-analysis can walk the chain"

    def test_parse_error_prescribes_implicit_context_when_not_chained(self) -> None:
        """Without explicit ``from``, Python sets __context__ but not __cause__.

        Documenting the difference so future code uses ``from`` when the
        underlying exception is the reason for the ParseError.
        """
        try:
            try:
                raise RuntimeError("provider exploded")
            except RuntimeError:
                raise ParseError("parse failed")  # noqa: B904 - intentional demo of the wrong pattern
        except ParseError as caught:
            assert caught.__cause__ is None
            assert isinstance(caught.__context__, RuntimeError)
