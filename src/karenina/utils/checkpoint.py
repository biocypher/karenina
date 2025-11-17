"""Benchmark converter utilities for JSON-LD format.

This module provides utilities to convert between internal Python representations
and the JSON-LD format used by the frontend.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, cast

from ..schemas.checkpoint import (
    SCHEMA_ORG_CONTEXT,
    JsonLdCheckpoint,
    SchemaOrgAnswer,
    SchemaOrgDataFeedItem,
    SchemaOrgPropertyValue,
    SchemaOrgQuestion,
    SchemaOrgRating,
    SchemaOrgSoftwareSourceCode,
)
from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait


class BenchmarkConversionError(Exception):
    """Raised when benchmark conversion fails."""

    pass


def generate_question_id(question_text: str) -> str:
    """
    Generate a deterministic ID for a question based on its text.

    Args:
        question_text: The question text

    Returns:
        A URN-formatted question ID
    """
    # Create MD5 hash of the question text
    hash_obj = hashlib.md5(question_text.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()

    # Create a readable prefix from the question
    prefix = question_text.lower().replace(" ", "-").replace("?", "").replace(".", "").replace(",", "")[:50]

    return f"urn:uuid:question-{prefix}-{hash_hex[:8]}"


def generate_template_id(template: str | None) -> str:
    """
    Generate a deterministic ID for an answer template based on its content.

    This is used as part of a composite key system where question identity =
    (question_id + template_id). This allows the same question text to have
    different templates in the same benchmark.

    Args:
        template: The answer template code (Python class definition)

    Returns:
        MD5 hash of the template (32 chars), or "no_template" if template is empty/None
    """
    if not template or template.strip() == "":
        return "no_template"

    # Create MD5 hash of the template
    hash_obj = hashlib.md5(template.strip().encode("utf-8"))
    return hash_obj.hexdigest()[:32]


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
        # Store metric trait configuration in additionalProperty
        additional_props = [
            SchemaOrgPropertyValue(name="metrics", value=json.dumps(trait.metrics)),
            SchemaOrgPropertyValue(name="repeated_extraction", value=trait.repeated_extraction),
            SchemaOrgPropertyValue(name="evaluation_mode", value=trait.evaluation_mode),
        ]

        # Add instruction lists (only if non-empty)
        if trait.tp_instructions:
            additional_props.append(
                SchemaOrgPropertyValue(name="tp_instructions", value=json.dumps(trait.tp_instructions))
            )
        if trait.tn_instructions:
            additional_props.append(
                SchemaOrgPropertyValue(name="tn_instructions", value=json.dumps(trait.tn_instructions))
            )

        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=1.0,  # Metrics are in 0-1 range
            worstRating=0.0,
            additionalType="GlobalMetricRubricTrait"
            if rubric_type == "global"
            else "QuestionSpecificMetricRubricTrait",
            additionalProperty=additional_props,
        )

    # Handle RegexTrait (always boolean)
    if isinstance(trait, RegexTrait):
        # Store regex configuration in additionalProperty
        additional_props = [
            SchemaOrgPropertyValue(name="pattern", value=trait.pattern),
            SchemaOrgPropertyValue(name="case_sensitive", value=trait.case_sensitive),
            SchemaOrgPropertyValue(name="invert_result", value=trait.invert_result),
        ]

        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=1,
            worstRating=0,
            additionalType="GlobalRegexTrait" if rubric_type == "global" else "QuestionSpecificRegexTrait",
            additionalProperty=additional_props,
        )

    # Handle CallableTrait (can be boolean or score)
    if isinstance(trait, CallableTrait):
        # Store callable code and metadata in additionalProperty
        import base64

        additional_props = [
            SchemaOrgPropertyValue(name="callable_code", value=base64.b64encode(trait.callable_code).decode("utf-8")),
            SchemaOrgPropertyValue(name="kind", value=trait.kind),
            SchemaOrgPropertyValue(name="invert_result", value=trait.invert_result),
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

    # Handle LLMRubricTrait
    if trait.kind == "boolean":
        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=1,
            worstRating=0,
            additionalType="GlobalRubricTrait" if rubric_type == "global" else "QuestionSpecificRubricTrait",
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
    """
    # Check if it's a MetricRubricTrait
    if rating.additionalType in ["GlobalMetricRubricTrait", "QuestionSpecificMetricRubricTrait"]:
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
                # Note: fp_instructions and fn_instructions are no longer supported
                # Old checkpoints with these fields will be ignored (backward compatibility)

        return MetricRubricTrait(
            name=rating.name,
            description=rating.description,
            evaluation_mode=evaluation_mode,
            metrics=metrics,
            tp_instructions=tp_instructions,
            tn_instructions=tn_instructions,
            repeated_extraction=repeated_extraction,
        )

    # Handle RegexTrait
    if rating.additionalType in ["GlobalRegexTrait", "QuestionSpecificRegexTrait"]:
        # Extract configuration from additionalProperty
        pattern = ""
        case_sensitive = True
        invert_result = False

        if rating.additionalProperty:
            for prop in rating.additionalProperty:
                if prop.name == "pattern":
                    pattern = prop.value
                elif prop.name == "case_sensitive":
                    case_sensitive = prop.value
                elif prop.name == "invert_result":
                    invert_result = prop.value

        return RegexTrait(
            name=rating.name,
            description=rating.description,
            pattern=pattern,
            case_sensitive=case_sensitive,
            invert_result=invert_result,
        )

    # Handle CallableTrait
    if rating.additionalType in ["GlobalCallableTrait", "QuestionSpecificCallableTrait"]:
        # Extract configuration from additionalProperty
        import base64

        callable_code = b""
        kind: Literal["boolean", "score"] = "boolean"
        invert_result = False
        min_score = None
        max_score = None

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

        return CallableTrait(
            name=rating.name,
            description=rating.description,
            kind=kind,
            callable_code=callable_code,
            min_score=min_score,
            max_score=max_score,
            invert_result=invert_result,
        )

    # Unsupported trait type - raise error (no backward compatibility)
    if rating.additionalType in ["GlobalManualRubricTrait", "QuestionSpecificManualRubricTrait"]:
        raise ValueError(
            f"ManualRubricTrait is no longer supported. Found trait '{rating.name}' with type "
            f"'{rating.additionalType}'. Please migrate your checkpoint using the migration script."
        )

    # Handle LLMRubricTrait
    # Determine if it's a boolean trait (0-1 range)
    is_boolean = rating.bestRating == 1 and rating.worstRating == 0

    return LLMRubricTrait(
        name=rating.name,
        description=rating.description,
        kind="boolean" if is_boolean else "score",
        min_score=None if is_boolean else int(rating.worstRating),
        max_score=None if is_boolean else int(rating.bestRating),
    )


def create_jsonld_benchmark(
    name: str,
    description: str = "",
    version: str = "0.1.0",
    creator: str = "Karenina Benchmarking System",
) -> JsonLdCheckpoint:
    """
    Create a new empty JSON-LD benchmark.

    Args:
        name: Name of the benchmark
        description: Description of the benchmark
        version: Version of the benchmark content
        creator: Creator name or organization

    Returns:
        A new JsonLdCheckpoint object
    """
    timestamp = datetime.now().isoformat()

    checkpoint_dict = {
        "@context": SCHEMA_ORG_CONTEXT,
        "@type": "DataFeed",
        "@id": f"urn:uuid:karenina-checkpoint-{datetime.now().timestamp()}",
        "name": name,
        "description": description or "Benchmark containing questions",
        "version": version,
        "creator": creator,
        "dateCreated": timestamp,
        "dateModified": timestamp,
        "rating": None,
        "dataFeedElement": [],
        "additionalProperty": [
            SchemaOrgPropertyValue(
                name="benchmark_format_version",
                value="3.0.0-jsonld",
            )
        ],
    }
    return JsonLdCheckpoint.model_validate(checkpoint_dict)


def add_question_to_benchmark(
    benchmark: JsonLdCheckpoint,
    question: str,
    raw_answer: str,
    answer_template: str,
    question_id: str | None = None,
    question_rubric_traits: list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait] | None = None,
    finished: bool = False,
    author: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
    custom_metadata: dict[str, Any] | None = None,
    keywords: list[str] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> str:
    """
    Add a question to a JSON-LD benchmark.

    Args:
        benchmark: The benchmark to modify
        question: Question text
        raw_answer: Expected answer text
        answer_template: Python code for the answer template
        question_id: Optional question ID (will be generated if not provided)
        question_rubric_traits: Optional question-specific rubric traits
        finished: Whether the template is finished
        author: Optional author information
        sources: Optional source documents
        custom_metadata: Optional custom metadata
        keywords: Optional keywords list
        few_shot_examples: Optional list of few-shot examples with 'question' and 'answer' keys

    Returns:
        The question ID that was added
    """
    if question_id is None:
        # Generate base ID from question text
        base_id = generate_question_id(question)

        # Make sure ID is unique by adding a counter if needed
        question_id = base_id
        counter = 1

        # Extract existing IDs from benchmark
        existing_ids = set()
        for item in benchmark.dataFeedElement:
            if item.id:
                existing_ids.add(item.id)
            else:
                # Generate ID from question text if no ID set
                existing_ids.add(generate_question_id(item.item.text))

        while question_id in existing_ids:
            question_id = f"{base_id}-{counter}"
            counter += 1

    timestamp = datetime.now().isoformat()

    # Build additional properties
    additional_props = [
        SchemaOrgPropertyValue(name="finished", value=finished),
    ]

    if author:
        additional_props.append(SchemaOrgPropertyValue(name="author", value=json.dumps(author)))

    if sources:
        additional_props.append(SchemaOrgPropertyValue(name="sources", value=json.dumps(sources)))

    if custom_metadata:
        for key, value in custom_metadata.items():
            additional_props.append(SchemaOrgPropertyValue(name=f"custom_{key}", value=value))

    if few_shot_examples:
        additional_props.append(SchemaOrgPropertyValue(name="few_shot_examples", value=json.dumps(few_shot_examples)))

    # Convert question-specific rubric traits to ratings
    ratings = None
    if question_rubric_traits:
        ratings = [convert_rubric_trait_to_rating(trait, "question-specific") for trait in question_rubric_traits]

    # Create the question object
    question_obj = SchemaOrgQuestion(
        text=question,
        acceptedAnswer=SchemaOrgAnswer(text=raw_answer),
        hasPart=SchemaOrgSoftwareSourceCode(
            name=f"{question[:30]}... Answer Template",
            text=answer_template,
        ),
        rating=ratings,
        additionalProperty=additional_props,
    )

    # Create the data feed item with ID
    item_dict = {
        "@id": question_id,
        "dateCreated": timestamp,
        "dateModified": timestamp,
        "item": question_obj,
        "keywords": keywords,
    }
    item = SchemaOrgDataFeedItem.model_validate(item_dict)

    # Add to benchmark
    benchmark.dataFeedElement.append(item)
    benchmark.dateModified = timestamp

    return question_id


def add_global_rubric_to_benchmark(
    benchmark: JsonLdCheckpoint,
    rubric_traits: list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait],
) -> None:
    """
    Add global rubric traits to a benchmark.

    Args:
        benchmark: The benchmark to modify
        rubric_traits: List of rubric traits to add as global rubric
    """
    ratings = [convert_rubric_trait_to_rating(trait, "global") for trait in rubric_traits]
    benchmark.rating = ratings
    benchmark.dateModified = datetime.now().isoformat()


def extract_questions_from_benchmark(
    benchmark: JsonLdCheckpoint,
) -> list[dict[str, Any]]:
    """
    Extract questions from a JSON-LD benchmark.

    Args:
        benchmark: The benchmark to extract from

    Returns:
        List of question dictionaries with id, text, answer, template, and metadata
    """
    questions = []

    for item in benchmark.dataFeedElement:
        question = item.item

        # Extract additional properties
        finished = False
        author = None
        sources = None
        custom_metadata = {}
        few_shot_examples = None

        if question.additionalProperty:
            for prop in question.additionalProperty:
                if prop.name == "finished":
                    finished = prop.value
                elif prop.name == "author":
                    try:
                        author = json.loads(prop.value)
                    except (json.JSONDecodeError, TypeError):
                        author = prop.value
                elif prop.name == "sources":
                    try:
                        sources = json.loads(prop.value)
                    except (json.JSONDecodeError, TypeError):
                        sources = prop.value
                elif prop.name == "few_shot_examples":
                    try:
                        few_shot_examples = json.loads(prop.value)
                    except (json.JSONDecodeError, TypeError):
                        few_shot_examples = prop.value
                elif prop.name.startswith("custom_"):
                    key = prop.name.replace("custom_", "")
                    custom_metadata[key] = prop.value

        # Extract question-specific rubric
        question_rubric = None
        if question.rating:
            question_rubric = [
                convert_rating_to_rubric_trait(rating)
                for rating in question.rating
                if rating.additionalType
                in [
                    "QuestionSpecificRubricTrait",
                    "QuestionSpecificRegexTrait",
                    "QuestionSpecificCallableTrait",
                    "QuestionSpecificMetricRubricTrait",
                ]
            ]

        questions.append(
            {
                "id": item.id or generate_question_id(question.text),
                "question": question.text,
                "raw_answer": question.acceptedAnswer.text,
                "answer_template": question.hasPart.text,
                "date_created": item.dateCreated,
                "date_modified": item.dateModified,
                "finished": finished,
                "author": author,
                "sources": sources,
                "custom_metadata": custom_metadata if custom_metadata else None,
                "question_rubric": question_rubric,
                "keywords": item.keywords,
                "few_shot_examples": few_shot_examples,
            }
        )

    return questions


def extract_global_rubric_from_benchmark(
    benchmark: JsonLdCheckpoint,
) -> list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait] | None:
    """
    Extract global rubric traits from a benchmark.

    Args:
        benchmark: The benchmark to extract from

    Returns:
        List of trait objects or None if no global rubric
    """
    if not benchmark.rating:
        return None

    traits = []
    for rating in benchmark.rating:
        if rating.additionalType in [
            "GlobalRubricTrait",
            "GlobalRegexTrait",
            "GlobalCallableTrait",
            "GlobalMetricRubricTrait",
        ]:
            traits.append(convert_rating_to_rubric_trait(rating))

    return traits if traits else None


def validate_jsonld_benchmark(benchmark: JsonLdCheckpoint) -> tuple[bool, str]:
    """
    Validate a JSON-LD benchmark structure.

    Args:
        benchmark: The benchmark to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields
        if not benchmark.name:
            return False, "Benchmark must have a name"

        if not benchmark.dataFeedElement:
            # Empty benchmark is valid
            pass
        else:
            # Validate each question
            for i, item in enumerate(benchmark.dataFeedElement):
                if not item.item.text:
                    return False, f"Question {i} missing text"

                if not item.item.acceptedAnswer:
                    return False, f"Question {i} missing acceptedAnswer"

                if not item.item.hasPart:
                    return False, f"Question {i} missing answer template"

                # Validate ratings if present
                if item.item.rating:
                    for rating in item.item.rating:
                        if rating.additionalType not in [
                            "GlobalRubricTrait",
                            "QuestionSpecificRubricTrait",
                            "GlobalManualRubricTrait",
                            "QuestionSpecificManualRubricTrait",
                            "GlobalMetricRubricTrait",
                            "QuestionSpecificMetricRubricTrait",
                        ]:
                            return (
                                False,
                                f"Invalid additionalType for rating: {rating.additionalType}",
                            )

        # Validate global ratings if present
        if benchmark.rating:
            for rating in benchmark.rating:
                if rating.additionalType not in [
                    "GlobalRubricTrait",
                    "GlobalManualRubricTrait",
                    "GlobalMetricRubricTrait",
                ]:
                    return (
                        False,
                        f"Dataset-level rating must be a global trait type, got {rating.additionalType}",
                    )

        return True, "Valid benchmark"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
