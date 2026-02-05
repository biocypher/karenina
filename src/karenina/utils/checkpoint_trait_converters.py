"""Rubric trait conversion utilities for JSON-LD checkpoint format.

This module handles the bidirectional conversion between internal rubric trait
representations (LLMRubricTrait, RegexTrait, CallableTrait, MetricRubricTrait)
and the schema.org Rating format used in JSON-LD checkpoints.
"""

import base64
import json
import logging
from typing import Literal, cast

from ..schemas.checkpoint import (
    JsonLdCheckpoint,
    SchemaOrgPropertyValue,
    SchemaOrgRating,
)
from ..schemas.entities import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait
from ..schemas.entities.rubric import TraitKind

logger = logging.getLogger(__name__)


def convert_rubric_trait_to_rating(
    trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait, rubric_type: str = "global"
) -> SchemaOrgRating:
    """
    Convert an internal trait to a schema.org Rating.

    Args:
        trait: The trait to convert (LLM, regex, callable, or metric)
        rubric_type: Either 'global' or 'question-specific'

    Returns:
        A SchemaOrgRating object
    """
    # Handle MetricRubricTrait
    if isinstance(trait, MetricRubricTrait):
        return _convert_metric_trait_to_rating(trait, rubric_type)

    # Handle RegexTrait (always boolean)
    if isinstance(trait, RegexTrait):
        return _convert_regex_trait_to_rating(trait, rubric_type)

    # Handle CallableTrait (can be boolean or score)
    if isinstance(trait, CallableTrait):
        return _convert_callable_trait_to_rating(trait, rubric_type)

    # Handle LLMRubricTrait
    return _convert_llm_trait_to_rating(trait, rubric_type)


def _convert_metric_trait_to_rating(trait: MetricRubricTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert MetricRubricTrait to SchemaOrgRating."""
    # Store metric trait configuration in additionalProperty
    # Note: Store arrays directly, not as JSON strings - Pydantic will serialize correctly
    additional_props = [
        SchemaOrgPropertyValue(name="metrics", value=trait.metrics),
        SchemaOrgPropertyValue(name="repeated_extraction", value=trait.repeated_extraction),
        SchemaOrgPropertyValue(name="evaluation_mode", value=trait.evaluation_mode),
    ]

    # Add instruction lists (only if non-empty)
    if trait.tp_instructions:
        additional_props.append(SchemaOrgPropertyValue(name="tp_instructions", value=trait.tp_instructions))
    if trait.tn_instructions:
        additional_props.append(SchemaOrgPropertyValue(name="tn_instructions", value=trait.tn_instructions))

    return SchemaOrgRating(
        name=trait.name,
        description=trait.description,
        bestRating=1.0,  # Metrics are in 0-1 range
        worstRating=0.0,
        additionalType="GlobalMetricRubricTrait" if rubric_type == "global" else "QuestionSpecificMetricRubricTrait",
        additionalProperty=additional_props,
    )


def _convert_regex_trait_to_rating(trait: RegexTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert RegexTrait to SchemaOrgRating."""
    # Store regex configuration in additionalProperty
    additional_props = [
        SchemaOrgPropertyValue(name="pattern", value=trait.pattern),
        SchemaOrgPropertyValue(name="case_sensitive", value=trait.case_sensitive),
        SchemaOrgPropertyValue(name="invert_result", value=trait.invert_result),
        SchemaOrgPropertyValue(name="higher_is_better", value=trait.higher_is_better),
    ]

    return SchemaOrgRating(
        name=trait.name,
        description=trait.description,
        bestRating=1,
        worstRating=0,
        additionalType="GlobalRegexTrait" if rubric_type == "global" else "QuestionSpecificRegexTrait",
        additionalProperty=additional_props,
    )


def _convert_callable_trait_to_rating(trait: CallableTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert CallableTrait to SchemaOrgRating."""
    # Store callable code and metadata in additionalProperty
    additional_props = [
        SchemaOrgPropertyValue(name="callable_code", value=base64.b64encode(trait.callable_code).decode("utf-8")),
        SchemaOrgPropertyValue(name="kind", value=trait.kind),
        SchemaOrgPropertyValue(name="invert_result", value=trait.invert_result),
        SchemaOrgPropertyValue(name="higher_is_better", value=trait.higher_is_better),
    ]

    # Add score fields if score-based
    if trait.kind == "score":
        if trait.min_score is not None:
            additional_props.append(SchemaOrgPropertyValue(name="min_score", value=trait.min_score))
        if trait.max_score is not None:
            additional_props.append(SchemaOrgPropertyValue(name="max_score", value=trait.max_score))

    # Determine best/worst rating based on kind
    if trait.kind == "boolean":
        best_rating = 1.0
        worst_rating = 0.0
    else:  # score
        best_rating = float(trait.max_score) if trait.max_score is not None else 5.0
        worst_rating = float(trait.min_score) if trait.min_score is not None else 1.0

    return SchemaOrgRating(
        name=trait.name,
        description=trait.description,
        bestRating=best_rating,
        worstRating=worst_rating,
        additionalType="GlobalCallableTrait" if rubric_type == "global" else "QuestionSpecificCallableTrait",
        additionalProperty=additional_props,
    )


def _convert_llm_trait_to_rating(trait: LLMRubricTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert LLMRubricTrait to SchemaOrgRating."""
    # Always store deep judgment settings in additionalProperty (in-memory)
    # Will be optionally stripped when saving to file based on save() toggle
    additional_props = [
        SchemaOrgPropertyValue(name="deep_judgment_enabled", value=trait.deep_judgment_enabled),
        SchemaOrgPropertyValue(name="deep_judgment_excerpt_enabled", value=trait.deep_judgment_excerpt_enabled),
        SchemaOrgPropertyValue(name="deep_judgment_search_enabled", value=trait.deep_judgment_search_enabled),
    ]

    # Add optional fields only if not None
    if trait.deep_judgment_max_excerpts is not None:
        additional_props.append(
            SchemaOrgPropertyValue(name="deep_judgment_max_excerpts", value=trait.deep_judgment_max_excerpts)
        )
    if trait.deep_judgment_fuzzy_match_threshold is not None:
        additional_props.append(
            SchemaOrgPropertyValue(
                name="deep_judgment_fuzzy_match_threshold", value=trait.deep_judgment_fuzzy_match_threshold
            )
        )
    if trait.deep_judgment_excerpt_retry_attempts is not None:
        additional_props.append(
            SchemaOrgPropertyValue(
                name="deep_judgment_excerpt_retry_attempts", value=trait.deep_judgment_excerpt_retry_attempts
            )
        )

    # Add directionality field
    additional_props.append(SchemaOrgPropertyValue(name="higher_is_better", value=trait.higher_is_better))

    if trait.kind == "boolean":
        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=1,
            worstRating=0,
            additionalType="GlobalRubricTrait" if rubric_type == "global" else "QuestionSpecificRubricTrait",
            additionalProperty=additional_props,
        )
    elif trait.kind == "literal":
        # Literal kind: store kind and classes in additionalProperty
        # min_score=0, max_score=len(classes)-1 (auto-derived from classes)
        additional_props.append(SchemaOrgPropertyValue(name="kind", value="literal"))
        if trait.classes is not None:
            additional_props.append(SchemaOrgPropertyValue(name="classes", value=trait.classes))

        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=float(trait.max_score) if trait.max_score is not None else 0.0,
            worstRating=float(trait.min_score) if trait.min_score is not None else 0.0,
            additionalType="GlobalLLMRubricTrait" if rubric_type == "global" else "QuestionSpecificLLMRubricTrait",
            additionalProperty=additional_props,
        )
    else:  # score
        min_score = trait.min_score if trait.min_score is not None else 1
        max_score = trait.max_score if trait.max_score is not None else 5

        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=float(max_score),
            worstRating=float(min_score),
            additionalType="GlobalRubricTrait" if rubric_type == "global" else "QuestionSpecificRubricTrait",
            additionalProperty=additional_props,
        )


def convert_rating_to_rubric_trait(
    rating: SchemaOrgRating,
) -> LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait:
    """
    Convert a schema.org Rating back to a rubric trait.

    Args:
        rating: The SchemaOrgRating to convert

    Returns:
        A LLMRubricTrait, RegexTrait, CallableTrait, or MetricRubricTrait object

    Raises:
        ValueError: If the rating has an unrecognized additionalType
    """
    # Check if it's a MetricRubricTrait
    if rating.additionalType in ["GlobalMetricRubricTrait", "QuestionSpecificMetricRubricTrait"]:
        return _convert_rating_to_metric_trait(rating)

    # Handle RegexTrait
    if rating.additionalType in ["GlobalRegexTrait", "QuestionSpecificRegexTrait"]:
        return _convert_rating_to_regex_trait(rating)

    # Handle CallableTrait
    if rating.additionalType in ["GlobalCallableTrait", "QuestionSpecificCallableTrait"]:
        return _convert_rating_to_callable_trait(rating)

    # Handle LLMRubricTrait (default for GlobalRubricTrait, QuestionSpecificRubricTrait,
    # GlobalLLMRubricTrait, QuestionSpecificLLMRubricTrait)
    known_llm_types = [
        "GlobalRubricTrait",
        "QuestionSpecificRubricTrait",
        "GlobalLLMRubricTrait",
        "QuestionSpecificLLMRubricTrait",
    ]
    if rating.additionalType in known_llm_types or rating.additionalType is None:
        return _convert_rating_to_llm_trait(rating)

    msg = f"Unrecognized rubric trait type: '{rating.additionalType}' (trait: '{rating.name}')"
    raise ValueError(msg)


def _convert_rating_to_metric_trait(rating: SchemaOrgRating) -> MetricRubricTrait:
    """Convert SchemaOrgRating to MetricRubricTrait."""
    # Extract configuration from additionalProperty
    metrics = []
    repeated_extraction = True  # Default
    evaluation_mode: Literal["tp_only", "full_matrix"] = "tp_only"  # Default for backward compatibility
    tp_instructions = []
    tn_instructions = []

    if rating.additionalProperty:
        for prop in rating.additionalProperty:
            if prop.name == "metrics":
                try:
                    metrics = json.loads(prop.value)
                except (json.JSONDecodeError, TypeError):
                    metrics = prop.value if isinstance(prop.value, list) else []
            elif prop.name == "repeated_extraction":
                repeated_extraction = prop.value
            elif prop.name == "evaluation_mode":
                # Cast to Literal type for type safety
                evaluation_mode = cast(Literal["tp_only", "full_matrix"], prop.value)
            elif prop.name == "tp_instructions":
                try:
                    tp_instructions = json.loads(prop.value)
                except (json.JSONDecodeError, TypeError):
                    tp_instructions = prop.value if isinstance(prop.value, list) else []
            elif prop.name == "tn_instructions":
                try:
                    tn_instructions = json.loads(prop.value)
                except (json.JSONDecodeError, TypeError):
                    tn_instructions = prop.value if isinstance(prop.value, list) else []

    return MetricRubricTrait(
        name=rating.name,
        description=rating.description,
        evaluation_mode=evaluation_mode,
        metrics=metrics,
        tp_instructions=tp_instructions,
        tn_instructions=tn_instructions,
        repeated_extraction=repeated_extraction,
    )


def _convert_rating_to_regex_trait(rating: SchemaOrgRating) -> RegexTrait:
    """Convert SchemaOrgRating to RegexTrait."""
    # Extract configuration from additionalProperty
    pattern = ""
    case_sensitive = True
    invert_result = False
    higher_is_better = True  # Legacy default

    if rating.additionalProperty:
        for prop in rating.additionalProperty:
            if prop.name == "pattern":
                pattern = prop.value
            elif prop.name == "case_sensitive":
                case_sensitive = prop.value
            elif prop.name == "invert_result":
                invert_result = prop.value
            elif prop.name == "higher_is_better":
                higher_is_better = prop.value

    return RegexTrait(
        name=rating.name,
        description=rating.description,
        pattern=pattern,
        case_sensitive=case_sensitive,
        invert_result=invert_result,
        higher_is_better=higher_is_better,
    )


def _convert_rating_to_callable_trait(rating: SchemaOrgRating) -> CallableTrait:
    """Convert SchemaOrgRating to CallableTrait."""
    # Extract configuration from additionalProperty
    callable_code = b""
    kind: Literal["boolean", "score"] = "boolean"
    invert_result = False
    min_score = None
    max_score = None
    higher_is_better = True  # Legacy default

    if rating.additionalProperty:
        for prop in rating.additionalProperty:
            if prop.name == "callable_code":
                callable_code = base64.b64decode(prop.value)
            elif prop.name == "kind":
                kind = cast(Literal["boolean", "score"], prop.value)
            elif prop.name == "invert_result":
                invert_result = prop.value
            elif prop.name == "min_score":
                min_score = prop.value
            elif prop.name == "max_score":
                max_score = prop.value
            elif prop.name == "higher_is_better":
                higher_is_better = prop.value

    return CallableTrait(
        name=rating.name,
        description=rating.description,
        kind=kind,
        callable_code=callable_code,
        min_score=min_score,
        max_score=max_score,
        invert_result=invert_result,
        higher_is_better=higher_is_better,
    )


def _convert_rating_to_llm_trait(rating: SchemaOrgRating) -> LLMRubricTrait:
    """Convert SchemaOrgRating to LLMRubricTrait."""
    # Extract configuration from additionalProperty
    deep_judgment_enabled = False
    deep_judgment_excerpt_enabled = False
    deep_judgment_search_enabled = False
    deep_judgment_max_excerpts = None
    deep_judgment_fuzzy_match_threshold = None
    deep_judgment_excerpt_retry_attempts = None
    higher_is_better = True  # Legacy default
    kind: TraitKind | None = None  # Explicit kind from property (for literal)
    classes: dict[str, str] | None = None  # Classes for literal kind

    if rating.additionalProperty:
        for prop in rating.additionalProperty:
            if prop.name == "deep_judgment_enabled":
                deep_judgment_enabled = prop.value
            elif prop.name == "deep_judgment_excerpt_enabled":
                deep_judgment_excerpt_enabled = prop.value
            elif prop.name == "deep_judgment_search_enabled":
                deep_judgment_search_enabled = prop.value
            elif prop.name == "deep_judgment_max_excerpts":
                deep_judgment_max_excerpts = prop.value
            elif prop.name == "deep_judgment_fuzzy_match_threshold":
                deep_judgment_fuzzy_match_threshold = prop.value
            elif prop.name == "deep_judgment_excerpt_retry_attempts":
                deep_judgment_excerpt_retry_attempts = prop.value
            elif prop.name == "higher_is_better":
                higher_is_better = prop.value
            elif prop.name == "kind":
                kind = cast(TraitKind, prop.value)
            elif prop.name == "classes":
                # Classes can be stored as dict directly or as JSON string
                if isinstance(prop.value, dict):
                    classes = prop.value
                elif isinstance(prop.value, str):
                    try:
                        classes = json.loads(prop.value)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse classes JSON for trait '%s'", rating.name)
                        classes = None

    # Determine kind: explicit kind takes precedence, then infer from rating range
    if kind is None:
        # Infer kind from rating range (legacy support)
        is_boolean = rating.bestRating == 1 and rating.worstRating == 0
        kind = "boolean" if is_boolean else "score"

    # Build the trait based on kind
    if kind == "literal":
        # Literal kind: min_score/max_score are auto-derived from classes by the model validator
        return LLMRubricTrait(
            name=rating.name,
            description=rating.description,
            kind="literal",
            classes=classes,
            min_score=None,  # Auto-derived by model validator
            max_score=None,  # Auto-derived by model validator
            deep_judgment_enabled=deep_judgment_enabled,
            deep_judgment_excerpt_enabled=deep_judgment_excerpt_enabled,
            deep_judgment_max_excerpts=deep_judgment_max_excerpts,
            deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
            deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
            deep_judgment_search_enabled=deep_judgment_search_enabled,
            higher_is_better=higher_is_better,
        )
    elif kind == "boolean":
        return LLMRubricTrait(
            name=rating.name,
            description=rating.description,
            kind="boolean",
            min_score=None,
            max_score=None,
            classes=None,
            deep_judgment_enabled=deep_judgment_enabled,
            deep_judgment_excerpt_enabled=deep_judgment_excerpt_enabled,
            deep_judgment_max_excerpts=deep_judgment_max_excerpts,
            deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
            deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
            deep_judgment_search_enabled=deep_judgment_search_enabled,
            higher_is_better=higher_is_better,
        )
    else:  # score
        return LLMRubricTrait(
            name=rating.name,
            description=rating.description,
            kind="score",
            min_score=int(rating.worstRating),
            max_score=int(rating.bestRating),
            classes=None,
            deep_judgment_enabled=deep_judgment_enabled,
            deep_judgment_excerpt_enabled=deep_judgment_excerpt_enabled,
            deep_judgment_max_excerpts=deep_judgment_max_excerpts,
            deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
            deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
            deep_judgment_search_enabled=deep_judgment_search_enabled,
            higher_is_better=higher_is_better,
        )


def strip_deep_judgment_config_from_checkpoint(checkpoint: JsonLdCheckpoint) -> None:
    """
    Strip deep judgment configuration from all LLM rubric traits in a checkpoint.

    This modifies the checkpoint in-place, removing deep judgment fields from
    additionalProperty arrays in Rating objects for LLM traits (both global and
    question-specific).

    Args:
        checkpoint: The checkpoint to modify
    """
    # Strip from global rubrics
    if checkpoint.rating:
        for rating in checkpoint.rating:
            if (
                rating.additionalType in ["GlobalRubricTrait", "QuestionSpecificRubricTrait"]
                and rating.additionalProperty
            ):
                # Filter out deep judgment properties
                rating.additionalProperty = [
                    prop for prop in rating.additionalProperty if not prop.name.startswith("deep_judgment_")
                ]
                # If no properties left, set to None
                if not rating.additionalProperty:
                    rating.additionalProperty = None

    # Strip from question-specific rubrics
    for item in checkpoint.dataFeedElement:
        if item.item.rating:
            for rating in item.item.rating:
                if (
                    rating.additionalType in ["QuestionSpecificRubricTrait", "GlobalRubricTrait"]
                    and rating.additionalProperty
                ):
                    # Filter out deep judgment properties
                    rating.additionalProperty = [
                        prop for prop in rating.additionalProperty if not prop.name.startswith("deep_judgment_")
                    ]
                    # If no properties left, set to None
                    if not rating.additionalProperty:
                        rating.additionalProperty = None
