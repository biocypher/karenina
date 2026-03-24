"""Rubric serialization/deserialization utilities for database storage.

This module provides helper functions for converting Rubric objects to/from
JSON-serializable dictionary format for database storage.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def serialize_rubric_to_dict(
    llm_traits: list[Any] | None = None,
    regex_traits: list[Any] | None = None,
    callable_traits: list[Any] | None = None,
    metric_traits: list[Any] | None = None,
    agentic_traits: list[Any] | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Serialize rubric traits to dictionary format for database storage.

    Args:
        llm_traits: List of LLMRubricTrait objects
        regex_traits: List of RegexRubricTrait objects
        callable_traits: List of CallableRubricTrait objects
        metric_traits: List of MetricRubricTrait objects
        agentic_traits: List of AgenticRubricTrait objects

    Returns:
        Dictionary with serialized traits, or None if all trait lists are empty
    """
    llm_traits = llm_traits or []
    regex_traits = regex_traits or []
    callable_traits = callable_traits or []
    metric_traits = metric_traits or []
    agentic_traits = agentic_traits or []

    if not (llm_traits or regex_traits or callable_traits or metric_traits or agentic_traits):
        return None

    return {
        "llm_traits": [t.model_dump() for t in llm_traits],
        "regex_traits": [t.model_dump() for t in regex_traits],
        "callable_traits": [t.model_dump() for t in callable_traits],
        "metric_traits": [t.model_dump() for t in metric_traits],
        "agentic_traits": [t.model_dump() for t in agentic_traits],
    }


def serialize_rubric(rubric: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Serialize a Rubric object to dictionary format.

    Args:
        rubric: Rubric object with llm_traits, regex_traits, callable_traits, metric_traits

    Returns:
        Dictionary with serialized traits, or None if rubric is empty/None
    """
    if rubric is None:
        return None

    return serialize_rubric_to_dict(
        llm_traits=rubric.llm_traits,
        regex_traits=rubric.regex_traits,
        callable_traits=rubric.callable_traits,
        metric_traits=rubric.metric_traits,
        agentic_traits=getattr(rubric, "agentic_traits", None),
    )


def _deserialize_llm_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single LLMRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        LLMRubricTrait instance
    """
    from ..schemas.entities import LLMRubricTrait

    kind = trait_data.get("kind", "score")
    return LLMRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        summary=trait_data.get("summary"),
        kind=kind,
        higher_is_better=trait_data.get("higher_is_better", True),
        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
        classes=trait_data.get("classes"),
        deep_judgment_enabled=trait_data.get("deep_judgment_enabled", False),
        deep_judgment_excerpt_enabled=trait_data.get("deep_judgment_excerpt_enabled", False),
        deep_judgment_max_excerpts=trait_data.get("deep_judgment_max_excerpts"),
        deep_judgment_fuzzy_match_threshold=trait_data.get("deep_judgment_fuzzy_match_threshold"),
        deep_judgment_excerpt_retry_attempts=trait_data.get("deep_judgment_excerpt_retry_attempts"),
        deep_judgment_search_enabled=trait_data.get("deep_judgment_search_enabled", False),
    )


def _deserialize_regex_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single RegexRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        RegexRubricTrait instance
    """
    from ..schemas.entities import RegexRubricTrait

    return RegexRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        summary=trait_data.get("summary"),
        pattern=trait_data.get("pattern", ".*"),
        higher_is_better=trait_data.get("higher_is_better", True),
        case_sensitive=trait_data.get("case_sensitive", True),
        invert_result=trait_data.get("invert_result", False),
    )


def _deserialize_callable_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single CallableRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        CallableRubricTrait instance
    """
    from ..schemas.entities import CallableRubricTrait

    return CallableRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        summary=trait_data.get("summary"),
        kind=trait_data["kind"],
        callable_code=trait_data["callable_code"],
        higher_is_better=trait_data.get("higher_is_better", True),
        min_score=trait_data.get("min_score"),
        max_score=trait_data.get("max_score"),
        invert_result=trait_data.get("invert_result", False),
        classes=trait_data.get("classes"),
    )


def _deserialize_metric_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single MetricRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        MetricRubricTrait instance
    """
    from ..schemas.entities import MetricRubricTrait

    return MetricRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        summary=trait_data.get("summary"),
        evaluation_mode=trait_data.get("evaluation_mode", "tp_only"),
        metrics=trait_data.get("metrics", []),
        tp_instructions=trait_data.get("tp_instructions", []),
        tn_instructions=trait_data.get("tn_instructions", []),
        repeated_extraction=trait_data.get("repeated_extraction", True),
        higher_is_better=trait_data.get("higher_is_better"),
    )


def _deserialize_agentic_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single AgenticRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        AgenticRubricTrait instance
    """
    from ..schemas.entities.rubric import AgenticRubricTrait

    model_override = trait_data.get("model_override")
    if model_override is not None and isinstance(model_override, dict):
        from ..schemas.config.models import ModelConfig

        model_override = ModelConfig(**model_override)

    return AgenticRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description") or "",
        summary=trait_data.get("summary"),
        kind=trait_data.get("kind", "boolean"),
        higher_is_better=trait_data.get("higher_is_better", True),
        context_mode=trait_data.get("context_mode", "trace_and_workspace"),
        materialize_trace=trait_data.get("materialize_trace", False),
        persist_trace=trait_data.get("persist_trace", False),
        max_turns=trait_data.get("max_turns", 15),
        timeout_seconds=trait_data.get("timeout_seconds", 120),
        min_score=trait_data.get("min_score", 1),
        max_score=trait_data.get("max_score", 5),
        classes=trait_data.get("classes"),
        model_override=model_override,
    )


def deserialize_rubric_from_dict(rubric_data: dict[str, Any] | None) -> Any | None:
    """Deserialize a Rubric object from dictionary format.

    Args:
        rubric_data: Dictionary with serialized traits (from database)

    Returns:
        Rubric instance, or None if rubric_data is None/empty

    Raises:
        ValueError: If rubric_data contains unsupported 'manual_traits' key
    """
    if not rubric_data:
        return None

    from ..schemas.entities import Rubric

    # Check for unsupported old format
    if "manual_traits" in rubric_data:
        raise ValueError(
            "Rubric contains unsupported 'manual_traits'. Please migrate your database using the migration script."
        )

    # Deserialize each trait type
    llm_traits = [_deserialize_llm_trait(t) for t in (rubric_data.get("llm_traits") or rubric_data.get("traits", []))]
    regex_traits = [_deserialize_regex_trait(t) for t in rubric_data.get("regex_traits", [])]
    callable_traits = [_deserialize_callable_trait(t) for t in rubric_data.get("callable_traits", [])]
    metric_traits = [_deserialize_metric_trait(t) for t in rubric_data.get("metric_traits", [])]
    agentic_traits = [_deserialize_agentic_trait(t) for t in rubric_data.get("agentic_traits", [])]

    # Return None if no traits found
    if not (llm_traits or regex_traits or callable_traits or metric_traits or agentic_traits):
        return None

    return Rubric(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=metric_traits,
        agentic_traits=agentic_traits,
    )


def serialize_question_rubric_from_cache(
    rubric_data: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Serialize question rubric from cache format to database format.

    Args:
        rubric_data: Rubric data dict with 'llm_traits', 'regex_traits', etc. keys

    Returns:
        Dictionary with serialized traits for database storage, or None if empty
    """
    if not rubric_data:
        return None

    llm_traits = rubric_data.get("llm_traits", [])
    regex_traits = rubric_data.get("regex_traits", [])
    callable_traits = rubric_data.get("callable_traits", [])
    metric_traits = rubric_data.get("metric_traits", [])
    agentic_traits = rubric_data.get("agentic_traits", [])

    if not (llm_traits or regex_traits or callable_traits or metric_traits or agentic_traits):
        return None

    return serialize_rubric_to_dict(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=metric_traits,
        agentic_traits=agentic_traits,
    )
