"""Rubric serialization/deserialization utilities for database storage.

This module provides helper functions for converting Rubric objects to/from
JSON-serializable dictionary format for database storage.
"""

from typing import Any


def serialize_rubric_to_dict(
    llm_traits: list[Any] | None = None,
    regex_traits: list[Any] | None = None,
    callable_traits: list[Any] | None = None,
    metric_traits: list[Any] | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Serialize rubric traits to dictionary format for database storage.

    Args:
        llm_traits: List of LLMRubricTrait objects
        regex_traits: List of RegexTrait objects
        callable_traits: List of CallableTrait objects
        metric_traits: List of MetricRubricTrait objects

    Returns:
        Dictionary with serialized traits, or None if all trait lists are empty
    """
    llm_traits = llm_traits or []
    regex_traits = regex_traits or []
    callable_traits = callable_traits or []
    metric_traits = metric_traits or []

    if not (llm_traits or regex_traits or callable_traits or metric_traits):
        return None

    return {
        "traits": [t.model_dump() for t in llm_traits],
        "regex_traits": [t.model_dump() for t in regex_traits],
        "callable_traits": [t.model_dump() for t in callable_traits],
        "metric_traits": [t.model_dump() for t in metric_traits],
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
    )


def _deserialize_llm_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single LLMRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        LLMRubricTrait instance
    """
    from ..schemas.domain import LLMRubricTrait

    kind = trait_data.get("kind", "score")
    return LLMRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        kind=kind,
        higher_is_better=trait_data.get("higher_is_better", True),
        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
        deep_judgment_enabled=trait_data.get("deep_judgment_enabled", False),
        deep_judgment_excerpt_enabled=trait_data.get("deep_judgment_excerpt_enabled", False),
        deep_judgment_max_excerpts=trait_data.get("deep_judgment_max_excerpts"),
        deep_judgment_fuzzy_match_threshold=trait_data.get("deep_judgment_fuzzy_match_threshold"),
        deep_judgment_excerpt_retry_attempts=trait_data.get("deep_judgment_excerpt_retry_attempts"),
        deep_judgment_search_enabled=trait_data.get("deep_judgment_search_enabled", False),
    )


def _deserialize_regex_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single RegexTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        RegexTrait instance
    """
    from ..schemas.domain import RegexTrait

    return RegexTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        pattern=trait_data.get("pattern", ".*"),
        higher_is_better=trait_data.get("higher_is_better", True),
        case_sensitive=trait_data.get("case_sensitive", True),
        invert_result=trait_data.get("invert_result", False),
    )


def _deserialize_callable_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single CallableTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        CallableTrait instance
    """
    from ..schemas.domain import CallableTrait

    return CallableTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        kind=trait_data["kind"],
        callable_code=trait_data["callable_code"],
        higher_is_better=trait_data.get("higher_is_better", True),
        min_score=trait_data.get("min_score"),
        max_score=trait_data.get("max_score"),
        invert_result=trait_data.get("invert_result", False),
    )


def _deserialize_metric_trait(trait_data: dict[str, Any]) -> Any:
    """Deserialize a single MetricRubricTrait from dictionary data.

    Args:
        trait_data: Dictionary with trait data

    Returns:
        MetricRubricTrait instance
    """
    from ..schemas.domain import MetricRubricTrait

    return MetricRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        evaluation_mode=trait_data.get("evaluation_mode", "tp_only"),
        metrics=trait_data.get("metrics", []),
        tp_instructions=trait_data.get("tp_instructions", []),
        tn_instructions=trait_data.get("tn_instructions", []),
        repeated_extraction=trait_data.get("repeated_extraction", True),
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

    from ..schemas.domain import Rubric

    # Check for unsupported old format
    if "manual_traits" in rubric_data:
        raise ValueError(
            "Rubric contains unsupported 'manual_traits'. Please migrate your database using the migration script."
        )

    # Deserialize each trait type
    llm_traits = [_deserialize_llm_trait(t) for t in rubric_data.get("traits", [])]
    regex_traits = [_deserialize_regex_trait(t) for t in rubric_data.get("regex_traits", [])]
    callable_traits = [_deserialize_callable_trait(t) for t in rubric_data.get("callable_traits", [])]
    metric_traits = [_deserialize_metric_trait(t) for t in rubric_data.get("metric_traits", [])]

    # Return None if no traits found
    if not (llm_traits or regex_traits or callable_traits or metric_traits):
        return None

    return Rubric(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=metric_traits,
    )


def serialize_question_rubric_from_cache(
    rubric_data: dict[str, Any] | list[Any] | None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Serialize question rubric from cache format to database format.

    The benchmark cache may store rubrics in different formats:
    - Dict format: {"llm_traits": [...], "regex_traits": [...], ...}
    - Legacy list format: [trait1, trait2, ...]

    Args:
        rubric_data: Rubric data from benchmark cache

    Returns:
        Dictionary with serialized traits for database storage, or None if empty
    """
    if not rubric_data:
        return None

    from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait

    if isinstance(rubric_data, dict):
        # Cache format: dict with llm_traits, regex_traits, etc.
        llm_traits = rubric_data.get("llm_traits", [])
        regex_traits = rubric_data.get("regex_traits", [])
        callable_traits = rubric_data.get("callable_traits", [])
        metric_traits = rubric_data.get("metric_traits", [])

        if not (llm_traits or regex_traits or callable_traits or metric_traits):
            return None

        return serialize_rubric_to_dict(
            llm_traits=llm_traits,
            regex_traits=regex_traits,
            callable_traits=callable_traits,
            metric_traits=metric_traits,
        )

    elif isinstance(rubric_data, list) and len(rubric_data) > 0:
        # Legacy format: flat list of trait objects
        llm_traits = [t for t in rubric_data if isinstance(t, LLMRubricTrait)]
        regex_traits = [t for t in rubric_data if isinstance(t, RegexTrait)]
        callable_traits = [t for t in rubric_data if isinstance(t, CallableTrait)]
        metric_traits = [t for t in rubric_data if isinstance(t, MetricRubricTrait)]

        return serialize_rubric_to_dict(
            llm_traits=llm_traits,
            regex_traits=regex_traits,
            callable_traits=callable_traits,
            metric_traits=metric_traits,
        )

    return None
