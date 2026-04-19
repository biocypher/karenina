"""Helpers for deciding how to classify adapter-reported usage metadata."""

from typing import Any


def should_mark_usage_unavailable(response: Any) -> bool:
    """Return True when we cannot trust the adapter's usage report.

    An adapter reports zero tokens in two cases that must be distinguished
    downstream: (a) the call genuinely consumed zero tokens (rare, but
    possible with cached or empty completions), and (b) the adapter could
    not capture usage at all (the vLLM streaming-without-stream_options
    case fixed in this patch). The zero counts alone cannot distinguish
    them, so the pipeline conservatively marks all-zero or missing usage
    as unavailable.

    Accepts duck-typed response objects: ``LLMResponse`` exposes both
    ``usage`` (``UsageMetadata | None``) and ``usage_unavailable`` (bool);
    ``AgentResult`` exposes only ``usage``. The check tolerates either.

    Args:
        response: The object returned by ``LLMPort.stream_invoke``,
            ``LLMPort.invoke``, or ``AgentPort.run``.

    Returns:
        True when usage is missing, all-zero, or explicitly flagged
        unavailable. False otherwise.
    """
    if getattr(response, "usage_unavailable", False):
        return True
    usage = getattr(response, "usage", None)
    if usage is None:
        return True
    return (
        getattr(usage, "input_tokens", 0) == 0
        and getattr(usage, "output_tokens", 0) == 0
        and getattr(usage, "total_tokens", 0) == 0
    )
