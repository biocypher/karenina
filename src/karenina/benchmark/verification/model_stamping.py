"""Shared pipeline-default stamping for ModelConfig instances.

Both verification paths stamp the pipeline-level ``request_timeout`` and
``retry_policy`` from :class:`~karenina.schemas.verification.VerificationConfig`
onto every model that does not carry its own value: the QA task-queue
expansion (:mod:`karenina.benchmark.verification.batch_runner`) and the
scenario combo preparation (``Benchmark._run_scenario_verification``,
including per-node ``ModelOverride`` instances). The logic lived as
near-identical private copies at both call sites. This module is the single
home.

Semantics (identical to both prior copies):

- Only-when-unset: a model whose field is already set is never overwritten.
- A ``None`` pipeline value is a no-op for that field.
- When nothing needs stamping the ORIGINAL instance is returned (identity
  preserved, no rebuild). When stamping is needed, a shallow ``model_copy``
  with the updated field is returned and the original is never mutated.
"""

from __future__ import annotations

from typing import Any


def stamp_request_timeout(model: Any, request_timeout: float | None) -> Any:
    """Stamp a pipeline-level ``request_timeout`` onto ``model`` if unset.

    Args:
        model: A ``ModelConfig``-shaped instance exposing ``request_timeout``
            and ``model_copy``.
        request_timeout: Pipeline-level default. ``None`` is a no-op.

    Returns:
        The original model when no change is needed, or a shallow copy with
        the timeout applied.
    """
    if request_timeout is not None and model.request_timeout is None:
        return model.model_copy(update={"request_timeout": request_timeout})
    return model


def stamp_retry_policy(model: Any, retry_policy: Any | None) -> Any:
    """Stamp a pipeline-level ``retry_policy`` onto ``model`` if unset.

    Args:
        model: A ``ModelConfig``-shaped instance exposing ``retry_policy``
            and ``model_copy``.
        retry_policy: Pipeline-level default. ``None`` is a no-op.

    Returns:
        The original model when no change is needed, or a shallow copy with
        the policy applied.
    """
    if retry_policy is not None and model.retry_policy is None:
        return model.model_copy(update={"retry_policy": retry_policy})
    return model


def stamp_pipeline_defaults(
    model: Any,
    *,
    request_timeout: float | None,
    retry_policy: Any | None,
) -> Any:
    """Apply both pipeline-default stamps to ``model``.

    Equivalent to ``stamp_retry_policy(stamp_request_timeout(model, t), p)``,
    matching the order both call sites historically used. Returns the
    original instance when neither field needs stamping.

    Args:
        model: A ``ModelConfig``-shaped instance.
        request_timeout: Pipeline-level timeout default. ``None`` is a no-op.
        retry_policy: Pipeline-level retry default. ``None`` is a no-op.

    Returns:
        The original model, or a copy with the unset fields populated.
    """
    return stamp_retry_policy(stamp_request_timeout(model, request_timeout), retry_policy)
