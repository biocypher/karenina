"""Conditional ground truth for scenario-dependent verification thresholds.

Allows a VerifiedField's ground truth value (and optionally its verification
primitive) to vary based on a prior scenario node's parsed result. Used when
the "correct" threshold depends on what a preceding node detected.
"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class GroundTruthCase(BaseModel):
    """One branch of a conditional ground truth.

    Args:
        value: The ground truth value for this case.
        verify_with: Optional verification primitive override. When provided,
            replaces the field's default primitive for this case. Accepts a
            primitive instance at authoring time; serialized to dict with
            ``type`` key for storage.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any
    verify_with: Any = None


class ConditionalGroundTruth(BaseModel):
    """Ground truth that varies based on scenario state.

    Resolves a dot-path against scenario node_results, then selects
    the matching GroundTruthCase. Used exclusively in scenario contexts;
    in non-scenario contexts the ``default`` case is used.

    Args:
        source: Dot-path into scenario context
            (e.g., ``"node_results.adversarial.parsed.behavior"``).
        cases: Mapping from resolved source value (as string) to the
            GroundTruthCase that should apply.
        default: Fallback case when the resolved value is not in cases
            or when running outside a scenario context.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    cases: dict[str, GroundTruthCase]
    default: GroundTruthCase

    def serialize(self) -> dict[str, Any]:
        """Serialize for storage in VerificationMeta.ground_truth.

        Adds a ``__conditional__`` marker so ``_compute_field_results()``
        can detect conditional ground truth from a plain dict. Primitive
        instances in cases are serialized with the ``type`` key convention
        used by the primitive registry.

        Returns:
            Dict suitable for JSON storage, with ``__conditional__: True``.
        """
        data = self.model_dump(mode="json")
        data["__conditional__"] = True

        # Serialize primitive instances from LIVE model (not from model_dump output)
        for case_key, case_obj in self.cases.items():
            data["cases"][case_key]["verify_with"] = _serialize_primitive(case_obj.verify_with)
        data["default"]["verify_with"] = _serialize_primitive(self.default.verify_with)

        return data


def _serialize_primitive(prim: Any) -> dict[str, Any] | None:
    """Serialize a primitive instance or pass through an already-serialized dict.

    Args:
        prim: A VerificationPrimitive instance, an already-serialized dict,
            or None.

    Returns:
        Serialized dict with ``type`` key, or None.
    """
    if prim is None:
        return None
    if isinstance(prim, dict):
        # Already serialized (e.g., from model_dump). Ensure type key exists.
        result: dict[str, Any] = prim
        return result if "type" in result else None
    # Live primitive instance: serialize like VerifiedField does
    prim_data: dict[str, Any] = prim.model_dump(mode="json")
    prim_data["type"] = type(prim).__name__
    return prim_data


def _resolve_dot_path(path: str, context: dict[str, Any]) -> Any:
    """Resolve a dot-separated path against a nested dict.

    Args:
        path: Dot-separated key path (e.g., ``"node_results.ask.parsed.field"``).
        context: The nested dict to resolve against.

    Returns:
        The resolved value, or None if any key is missing.
    """
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def resolve_conditional(
    cgt_data: dict[str, Any],
    context: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any] | None]:
    """Resolve a serialized ConditionalGroundTruth against scenario context.

    Args:
        cgt_data: Serialized conditional ground truth dict (with
            ``__conditional__: True``).
        context: Scenario context dict with ``"node_results"`` key,
            or None if not in a scenario.

    Returns:
        Tuple of ``(ground_truth_value, verify_with_data)``. The
        ``verify_with_data`` is a serialized primitive dict (or None
        if the case does not override the primitive).
    """
    default = cgt_data.get("default", {})

    if context is None:
        return default.get("value"), default.get("verify_with")

    source_value = _resolve_dot_path(cgt_data["source"], context)

    if source_value is None:
        logger.debug(
            "Conditional source %r resolved to None; using default",
            cgt_data["source"],
        )
        return default.get("value"), default.get("verify_with")

    case = cgt_data.get("cases", {}).get(str(source_value))
    if case is None:
        logger.debug(
            "Conditional source %r=%r has no matching case; using default",
            cgt_data["source"],
            source_value,
        )
        return default.get("value"), default.get("verify_with")

    return case.get("value"), case.get("verify_with")
