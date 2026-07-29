"""Shared wall-clock bounds for sync wrappers around async LLM calls.

Every adapter exposes sync entry points (invoke, parse_to_pydantic) that
dispatch their async counterparts via a portal, a fresh thread, or
asyncio.run. Those dispatch points need a wall-clock bound so a stalled
coroutine cannot wedge the calling thread forever. Historically each site
hardcoded its own constant (300s for most adapters, 600s for deep agents
and the generic portal dispatch, 45s for MCP tool description fetches,
60s for search providers). This module centralizes the constants and the
derivation rule so the bound scales with ModelConfig.request_timeout and
the retry policy instead of silently truncating long-running calls.

The derivation mirrors the worst case of a retried LLM call: per attempt
the (possibly escalated) request timeout applies, attempts are separated
by backoff, and a flow may run several internal call sequences (for
example the langchain parser runs up to 4: native structured, fallback,
null retry, format retry). A grace period absorbs scheduling overhead.
The returned bound never goes below the site's historical floor, so the
bound only loosens when request_timeout demands more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.utils.retry_policy import RetryPolicy

__all__ = [
    "DEEP_AGENTS_SYNC_WRAPPER_FLOOR",
    "DEFAULT_SYNC_WRAPPER_FLOOR",
    "MCP_CONNECT_TIMEOUT",
    "MCP_TOOL_FETCH_FLOOR",
    "PARSE_INTERNAL_CALL_SEQUENCES",
    "PORTAL_DISPATCH_FLOOR",
    "SEARCH_PROVIDER_FLOOR",
    "SYNC_WRAPPER_GRACE_SECONDS",
    "compute_sync_wrapper_timeout",
]

# Historical hardcoded floors, one per call-site family. The helper never
# returns less than the floor passed by the site, so these are the exact
# values the codebase used before centralization.
DEFAULT_SYNC_WRAPPER_FLOOR = 300.0
DEEP_AGENTS_SYNC_WRAPPER_FLOOR = 600.0
PORTAL_DISPATCH_FLOOR = 600.0
SEARCH_PROVIDER_FLOOR = 60.0

# MCP tool description fetches use a pair of related bounds: the inner
# per-server connect timeout fires inside afetch_tool_descriptions, and
# the outer dispatch bound wraps the whole sync wrapper (portal or
# thread). The outer floor is derived from the inner bound plus grace so
# that, for a single server, the inner timeout fires first and surfaces
# the precise per-server error instead of a generic dispatch timeout.
# With multiple servers the inner bounds sum (N x 30s), so the outer
# dispatch bound may fire while later servers are still connecting. The
# values preserve the historical constants (30s inner, 45s outer).
MCP_CONNECT_TIMEOUT = 30.0
MCP_TOOL_FETCH_FLOOR = MCP_CONNECT_TIMEOUT + 15.0

# Grace added on top of the derived retry budget to absorb thread and
# portal scheduling overhead.
SYNC_WRAPPER_GRACE_SECONDS = 30.0

# Worst-case number of sequential LLM call sequences inside a
# parse_to_pydantic flow: native structured, fallback, null-feedback
# retry, format-feedback retry.
PARSE_INTERNAL_CALL_SEQUENCES = 4


def compute_sync_wrapper_timeout(
    request_timeout: float | None,
    *,
    floor: float = DEFAULT_SYNC_WRAPPER_FLOOR,
    retry_policy: RetryPolicy | None = None,
    internal_call_sequences: int = 1,
) -> float:
    """Compute a wall-clock bound for a sync wrapper around an async call.

    The bound covers the worst case of the wrapped flow: each internal
    call sequence runs ``timeout.max_attempts + 1`` attempts, each attempt
    is bounded by the (possibly escalated) request timeout, attempts are
    separated by up to ``backoff_max`` of backoff, and a grace period is
    added on top. The result is clamped to ``max(floor, derived)`` so the
    bound never tightens below the site's historical constant.

    Args:
        request_timeout: The per-attempt request timeout from ModelConfig.
            When None, no budget can be derived and the floor is returned.
        floor: Minimum bound in seconds. Defaults to the 300s used by most
            adapter sync wrappers. Sites with a different historical
            constant pass it explicitly.
        retry_policy: Retry policy of the wrapped call. When None, the
            default RetryPolicy is assumed.
        internal_call_sequences: Number of sequential LLM call sequences
            the wrapped flow may issue (for example 4 for the langchain
            parser flow).

    Returns:
        The wall-clock bound in seconds, never below ``floor``.
    """
    if request_timeout is None:
        return floor

    from karenina.utils.retry_policy import RetryPolicy, compute_escalated_timeout

    policy = retry_policy if retry_policy is not None else RetryPolicy()

    timeout_max_attempts = policy.timeout.max_attempts
    escalation = policy.timeout_escalation
    attempts = timeout_max_attempts + 1  # initial call + timeout retries

    # Sum the worst-case per-attempt timeouts. When escalation is None,
    # this collapses to request_timeout * attempts.
    per_attempt_total = sum(
        compute_escalated_timeout(
            base_timeout=request_timeout,
            timeout_attempt=n,
            config=escalation,
            max_attempts=timeout_max_attempts,
        )
        or request_timeout
        for n in range(attempts)
    )
    per_sequence = per_attempt_total + policy.timeout.backoff_max * timeout_max_attempts
    derived = per_sequence * internal_call_sequences + SYNC_WRAPPER_GRACE_SECONDS
    return max(floor, derived)
