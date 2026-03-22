"""Rubric trait conversion utilities for JSON-LD checkpoint format.

This module handles the bidirectional conversion between internal rubric trait
representations (LLMRubricTrait, RegexRubricTrait, CallableRubricTrait, MetricRubricTrait)
and the schema.org Rating format used in JSON-LD checkpoints.
"""

import base64
import json
import logging
from typing import Any, Literal, cast

from pydantic import BaseModel

from ..schemas.checkpoint import (
    JsonLdCheckpoint,
    SchemaOrgPropertyValue,
    SchemaOrgRating,
)
from ..schemas.entities import CallableRubricTrait, LLMRubricTrait, MetricRubricTrait, RegexRubricTrait
from ..schemas.entities.rubric import AgenticRubricTrait, DynamicRubric, TraitKind

logger = logging.getLogger(__name__)


def convert_rubric_trait_to_rating(
    trait: LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait,
    rubric_type: str = "global",
) -> SchemaOrgRating:
    """
    Convert an internal trait to a schema.org Rating.

    Args:
        trait: The trait to convert (LLM, regex, callable, metric, or agentic)
        rubric_type: Either 'global' or 'question-specific'

    Returns:
        A SchemaOrgRating object
    """
    # Handle AgenticRubricTrait (must be before LLMRubricTrait check)
    if isinstance(trait, AgenticRubricTrait):
        return _convert_agentic_trait_to_rating(trait, rubric_type)

    # Handle MetricRubricTrait
    if isinstance(trait, MetricRubricTrait):
        return _convert_metric_trait_to_rating(trait, rubric_type)

    # Handle RegexRubricTrait (always boolean)
    if isinstance(trait, RegexRubricTrait):
        return _convert_regex_trait_to_rating(trait, rubric_type)

    # Handle CallableRubricTrait (can be boolean or score)
    if isinstance(trait, CallableRubricTrait):
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
        additionalType="karenina:GlobalMetricRubricTrait"
        if rubric_type == "global"
        else "karenina:QuestionSpecificMetricRubricTrait",
        additionalProperty=additional_props,
    )


def _convert_regex_trait_to_rating(trait: RegexRubricTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert RegexRubricTrait to SchemaOrgRating."""
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
        additionalType="karenina:GlobalRegexTrait"
        if rubric_type == "global"
        else "karenina:QuestionSpecificRegexTrait",
        additionalProperty=additional_props,
    )


def _convert_callable_trait_to_rating(trait: CallableRubricTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert CallableRubricTrait to SchemaOrgRating."""
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
        additionalType="karenina:GlobalCallableTrait"
        if rubric_type == "global"
        else "karenina:QuestionSpecificCallableTrait",
        additionalProperty=additional_props,
    )


def _convert_agentic_trait_to_rating(trait: AgenticRubricTrait, rubric_type: str) -> SchemaOrgRating:
    """Convert AgenticRubricTrait to SchemaOrgRating."""
    # Serialize template kind as a dict with JSON Schema; pass string kinds through
    kind_value: Any
    if trait.is_template_kind:
        template_cls = cast(type[BaseModel], trait.kind)
        kind_value = {"type": "template", "schema": template_cls.model_json_schema()}
    else:
        kind_value = trait.kind

    additional_props = [
        SchemaOrgPropertyValue(name="kind", value=kind_value),
        SchemaOrgPropertyValue(name="higher_is_better", value=trait.higher_is_better),
        SchemaOrgPropertyValue(name="context_mode", value=trait.context_mode),
        SchemaOrgPropertyValue(name="materialize_trace", value=trait.materialize_trace),
        SchemaOrgPropertyValue(name="persist_trace", value=trait.persist_trace),
        SchemaOrgPropertyValue(name="max_turns", value=trait.max_turns),
        SchemaOrgPropertyValue(name="timeout_seconds", value=trait.timeout_seconds),
    ]

    if trait.classes is not None:
        additional_props.append(SchemaOrgPropertyValue(name="classes", value=trait.classes))
    if trait.model_override is not None:
        additional_props.append(SchemaOrgPropertyValue(name="model_override", value=trait.model_override.model_dump()))

    # Determine best/worst rating based on kind
    if trait.is_template_kind:
        best_rating, worst_rating = 0.0, 0.0
    elif trait.kind == "boolean":
        best_rating, worst_rating = 1.0, 0.0
    elif trait.kind == "literal" and trait.max_score is not None and trait.min_score is not None:
        best_rating, worst_rating = float(trait.max_score), float(trait.min_score)
    else:
        best_rating = float(trait.max_score) if trait.max_score is not None else 5.0
        worst_rating = float(trait.min_score) if trait.min_score is not None else 1.0

    return SchemaOrgRating(
        name=trait.name,
        description=trait.description,
        bestRating=best_rating,
        worstRating=worst_rating,
        additionalType="karenina:GlobalAgenticRubricTrait"
        if rubric_type == "global"
        else "karenina:QuestionSpecificAgenticRubricTrait",
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
            additionalType="karenina:GlobalRubricTrait"
            if rubric_type == "global"
            else "karenina:QuestionSpecificRubricTrait",
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
            additionalType="karenina:GlobalLLMRubricTrait"
            if rubric_type == "global"
            else "karenina:QuestionSpecificLLMRubricTrait",
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
            additionalType="karenina:GlobalRubricTrait"
            if rubric_type == "global"
            else "karenina:QuestionSpecificRubricTrait",
            additionalProperty=additional_props,
        )


def convert_rating_to_rubric_trait(
    rating: SchemaOrgRating,
) -> LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait:
    """
    Convert a schema.org Rating back to a rubric trait.

    Args:
        rating: The SchemaOrgRating to convert

    Returns:
        A LLMRubricTrait, RegexRubricTrait, CallableRubricTrait, MetricRubricTrait, or AgenticRubricTrait object

    Raises:
        ValueError: If the rating has an unrecognized additionalType
    """
    # Check if it's an AgenticRubricTrait
    if rating.additionalType in [
        "karenina:GlobalAgenticRubricTrait",
        "karenina:QuestionSpecificAgenticRubricTrait",
    ]:
        return _convert_rating_to_agentic_trait(rating)

    # Check if it's a MetricRubricTrait
    if rating.additionalType in ["karenina:GlobalMetricRubricTrait", "karenina:QuestionSpecificMetricRubricTrait"]:
        return _convert_rating_to_metric_trait(rating)

    # Handle RegexRubricTrait
    if rating.additionalType in ["karenina:GlobalRegexTrait", "karenina:QuestionSpecificRegexTrait"]:
        return _convert_rating_to_regex_trait(rating)

    # Handle CallableRubricTrait
    if rating.additionalType in ["karenina:GlobalCallableTrait", "karenina:QuestionSpecificCallableTrait"]:
        return _convert_rating_to_callable_trait(rating)

    # Handle LLMRubricTrait (default for GlobalRubricTrait, QuestionSpecificRubricTrait,
    # GlobalLLMRubricTrait, QuestionSpecificLLMRubricTrait)
    known_llm_types = [
        "karenina:GlobalRubricTrait",
        "karenina:QuestionSpecificRubricTrait",
        "karenina:GlobalLLMRubricTrait",
        "karenina:QuestionSpecificLLMRubricTrait",
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
        summary=None,
        evaluation_mode=evaluation_mode,
        metrics=metrics,
        tp_instructions=tp_instructions,
        tn_instructions=tn_instructions,
        repeated_extraction=repeated_extraction,
    )


def _convert_rating_to_regex_trait(rating: SchemaOrgRating) -> RegexRubricTrait:
    """Convert SchemaOrgRating to RegexRubricTrait."""
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

    return RegexRubricTrait(
        name=rating.name,
        description=rating.description,
        summary=None,
        pattern=pattern,
        case_sensitive=case_sensitive,
        invert_result=invert_result,
        higher_is_better=higher_is_better,
    )


def _convert_rating_to_callable_trait(rating: SchemaOrgRating) -> CallableRubricTrait:
    """Convert SchemaOrgRating to CallableRubricTrait."""
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

    return CallableRubricTrait(
        name=rating.name,
        description=rating.description,
        summary=None,
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
            summary=None,
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
            summary=None,
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
            summary=None,
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


def _convert_rating_to_agentic_trait(rating: SchemaOrgRating) -> AgenticRubricTrait:
    """Convert SchemaOrgRating to AgenticRubricTrait."""
    kind: Any = "boolean"
    higher_is_better: bool | None = True
    context_mode: Literal["workspace_only", "trace_and_workspace", "trace_only"] = "trace_and_workspace"
    materialize_trace = False
    persist_trace = False
    max_turns = 15
    timeout_seconds = 120
    classes: dict[str, str] | None = None
    model_override = None

    if rating.additionalProperty:
        for prop in rating.additionalProperty:
            if prop.name == "kind":
                # Pass value through directly; the field_validator on
                # AgenticRubricTrait handles both string literals and
                # {"type": "template", "schema": ...} dicts.
                kind = prop.value
            elif prop.name == "higher_is_better":
                higher_is_better = prop.value
            elif prop.name == "context_mode":
                context_mode = cast(Literal["workspace_only", "trace_and_workspace", "trace_only"], prop.value)
            elif prop.name == "materialize_trace":
                materialize_trace = prop.value
            elif prop.name == "persist_trace":
                persist_trace = prop.value
            elif prop.name == "max_turns":
                max_turns = prop.value
            elif prop.name == "timeout_seconds":
                timeout_seconds = prop.value
            elif prop.name == "classes":
                if isinstance(prop.value, dict):
                    classes = prop.value
                elif isinstance(prop.value, str):
                    try:
                        classes = json.loads(prop.value)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse classes JSON for agentic trait '%s'", rating.name)
            elif prop.name == "model_override" and isinstance(prop.value, dict):
                from ..schemas.config.models import ModelConfig

                model_override = ModelConfig(**prop.value)

    return AgenticRubricTrait(
        name=rating.name,
        description=rating.description or "",
        summary=None,
        kind=kind,
        higher_is_better=higher_is_better,
        context_mode=context_mode,
        materialize_trace=materialize_trace,
        persist_trace=persist_trace,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        min_score=int(rating.worstRating),
        max_score=int(rating.bestRating),
        classes=classes,
        model_override=model_override,
    )


def convert_dynamic_rubric_to_ratings(
    dynamic_rubric: "DynamicRubric",
    rubric_type: str = "global",
) -> list[SchemaOrgRating]:
    """Convert a DynamicRubric to a list of SchemaOrgRating objects.

    Each trait in the DynamicRubric is serialized using the existing per-type
    converter, then re-tagged with the dynamic rubric ``@type`` discriminator.
    The ``summary`` field is preserved in ``additionalProperty`` so that
    concept presence checking can work after deserialization.

    Args:
        dynamic_rubric: The DynamicRubric to serialize.
        rubric_type: Either 'global' or 'question-specific'.

    Returns:
        List of SchemaOrgRating objects.
    """

    additional_type = (
        "karenina:GlobalDynamicRubricTrait"
        if rubric_type == "global"
        else "karenina:QuestionSpecificDynamicRubricTrait"
    )

    ratings: list[SchemaOrgRating] = []

    for trait in dynamic_rubric._all_traits():
        # Delegate to the existing per-type converter to build the rating
        rating = convert_rubric_trait_to_rating(trait, rubric_type)

        # Re-tag with the dynamic rubric discriminator
        rating.additionalType = additional_type  # type: ignore[assignment]

        # Inject the trait type tag so we know the original type on deserialization
        trait_type_tag = _trait_type_tag(trait)
        props = list(rating.additionalProperty or [])
        props.append(SchemaOrgPropertyValue(name="dynamic_trait_type", value=trait_type_tag))

        # Persist summary (critical for concept presence checking)
        summary = getattr(trait, "summary", None)
        if summary is not None:
            props.append(SchemaOrgPropertyValue(name="summary", value=summary))

        rating.additionalProperty = props
        ratings.append(rating)

    return ratings


def convert_ratings_to_dynamic_rubric(
    ratings: list[SchemaOrgRating],
) -> "DynamicRubric":
    """Convert a list of SchemaOrgRating objects back to a DynamicRubric.

    This is the inverse of :func:`convert_dynamic_rubric_to_ratings`. Each
    rating is dispatched to the appropriate per-type converter based on the
    ``dynamic_trait_type`` property, and the ``summary`` field is restored.

    Args:
        ratings: List of SchemaOrgRating objects with dynamic rubric @type.

    Returns:
        A DynamicRubric with all traits restored.
    """
    llm_traits: list[LLMRubricTrait] = []
    regex_traits: list[RegexRubricTrait] = []
    callable_traits: list[CallableRubricTrait] = []
    metric_traits: list[MetricRubricTrait] = []
    agentic_traits: list[AgenticRubricTrait] = []

    for rating in ratings:
        # Extract dynamic_trait_type and summary from additionalProperty
        trait_type_tag = "llm"
        summary: str | None = None
        clean_props: list[SchemaOrgPropertyValue] = []

        if rating.additionalProperty:
            for prop in rating.additionalProperty:
                if prop.name == "dynamic_trait_type":
                    trait_type_tag = prop.value
                elif prop.name == "summary":
                    summary = prop.value
                else:
                    clean_props.append(prop)

        # Temporarily restore the per-type additionalType so the existing
        # converters can dispatch correctly, then convert.
        original_additional_type = rating.additionalType
        rating.additionalProperty = clean_props or None

        if trait_type_tag == "llm":
            rating.additionalType = "karenina:GlobalLLMRubricTrait"
            llm_t = _convert_rating_to_llm_trait(rating)
            llm_t.summary = summary
            llm_traits.append(llm_t)
        elif trait_type_tag == "regex":
            rating.additionalType = "karenina:GlobalRegexTrait"
            regex_t = _convert_rating_to_regex_trait(rating)
            regex_t.summary = summary
            regex_traits.append(regex_t)
        elif trait_type_tag == "callable":
            rating.additionalType = "karenina:GlobalCallableTrait"
            callable_t = _convert_rating_to_callable_trait(rating)
            callable_t.summary = summary
            callable_traits.append(callable_t)
        elif trait_type_tag == "metric":
            rating.additionalType = "karenina:GlobalMetricRubricTrait"
            metric_t = _convert_rating_to_metric_trait(rating)
            metric_t.summary = summary
            metric_traits.append(metric_t)
        elif trait_type_tag == "agentic":
            rating.additionalType = "karenina:GlobalAgenticRubricTrait"
            agentic_t = _convert_rating_to_agentic_trait(rating)
            agentic_t.summary = summary
            agentic_traits.append(agentic_t)
        else:
            logger.warning(
                "Unknown dynamic_trait_type '%s' for trait '%s'; skipping.",
                trait_type_tag,
                rating.name,
            )

        # Restore original additionalType (defensive)
        rating.additionalType = original_additional_type

    return DynamicRubric(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=metric_traits,
        agentic_traits=agentic_traits,
    )


def _trait_type_tag(
    trait: LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait,
) -> str:
    """Return a short string tag identifying the trait type for serialization."""
    if isinstance(trait, AgenticRubricTrait):
        return "agentic"
    if isinstance(trait, MetricRubricTrait):
        return "metric"
    if isinstance(trait, RegexRubricTrait):
        return "regex"
    if isinstance(trait, CallableRubricTrait):
        return "callable"
    return "llm"


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
                rating.additionalType in ["karenina:GlobalRubricTrait", "karenina:QuestionSpecificRubricTrait"]
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
                    rating.additionalType in ["karenina:QuestionSpecificRubricTrait", "karenina:GlobalRubricTrait"]
                    and rating.additionalProperty
                ):
                    # Filter out deep judgment properties
                    rating.additionalProperty = [
                        prop for prop in rating.additionalProperty if not prop.name.startswith("deep_judgment_")
                    ]
                    # If no properties left, set to None
                    if not rating.additionalProperty:
                        rating.additionalProperty = None
