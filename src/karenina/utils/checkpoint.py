"""Benchmark converter utilities for JSON-LD format.

This module provides utilities to convert between internal Python representations
and the JSON-LD format used by the frontend.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from ..schemas.checkpoint import (
    SCHEMA_ORG_CONTEXT,
    JsonLdCheckpoint,
    SchemaOrgAnswer,
    SchemaOrgDataFeedItem,
    SchemaOrgPropertyValue,
    SchemaOrgQuestion,
    SchemaOrgSoftwareSourceCode,
)
from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait
from .checkpoint_trait_converters import (
    convert_rating_to_rubric_trait,
    convert_rubric_trait_to_rating,
    strip_deep_judgment_config_from_checkpoint,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "BenchmarkConversionError",
    "generate_question_id",
    "generate_template_id",
    "convert_rubric_trait_to_rating",
    "convert_rating_to_rubric_trait",
    "strip_deep_judgment_config_from_checkpoint",
    "create_jsonld_benchmark",
    "add_question_to_benchmark",
    "add_global_rubric_to_benchmark",
    "extract_questions_from_benchmark",
    "extract_global_rubric_from_benchmark",
    "validate_jsonld_benchmark",
]


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
            # Convert ratings to traits (filtering out None for unsupported types)
            traits = [
                trait
                for rating in question.rating
                if rating.additionalType
                in [
                    "QuestionSpecificRubricTrait",
                    "QuestionSpecificRegexTrait",
                    "QuestionSpecificCallableTrait",
                    "QuestionSpecificMetricRubricTrait",
                ]
                and (trait := convert_rating_to_rubric_trait(rating)) is not None
            ]

            # Categorize traits by type to match Rubric schema
            if traits:
                from ..schemas.entities import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait

                llm_traits = [t for t in traits if isinstance(t, LLMRubricTrait)]
                regex_traits = [t for t in traits if isinstance(t, RegexTrait)]
                callable_traits = [t for t in traits if isinstance(t, CallableTrait)]
                metric_traits = [t for t in traits if isinstance(t, MetricRubricTrait)]

                question_rubric = {
                    "llm_traits": llm_traits,
                    "regex_traits": regex_traits,
                    "callable_traits": callable_traits,
                    "metric_traits": metric_traits,
                }

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

    This function extracts traits from two sources:
    1. benchmark.rating - Rating objects with additionalType indicating trait type
    2. benchmark.additionalProperty - Legacy format where traits are stored as JSON strings
       under keys like 'global_regex_rubric_traits', 'global_callable_rubric_traits', etc.

    Args:
        benchmark: The benchmark to extract from

    Returns:
        List of trait objects or None if no global rubric
    """
    traits: list[LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait] = []

    # Extract from rating array (standard format)
    if benchmark.rating:
        for rating in benchmark.rating:
            if rating.additionalType in [
                "GlobalRubricTrait",
                "GlobalRegexTrait",
                "GlobalCallableTrait",
                "GlobalMetricRubricTrait",
            ]:
                trait = convert_rating_to_rubric_trait(rating)
                if trait is not None:
                    traits.append(trait)

    # Extract from additionalProperty (legacy format from GUI checkpoint-converter)
    if benchmark.additionalProperty:
        for prop in benchmark.additionalProperty:
            if prop.name == "global_regex_rubric_traits" and isinstance(prop.value, str):
                try:
                    regex_traits_data = json.loads(prop.value)
                    for trait_data in regex_traits_data:
                        # Normalize legacy "invert" field to "invert_result"
                        if "invert" in trait_data and "invert_result" not in trait_data:
                            trait_data["invert_result"] = trait_data.pop("invert")
                        traits.append(RegexTrait(**trait_data))
                except (json.JSONDecodeError, TypeError):
                    pass  # Invalid JSON, skip

            elif prop.name == "global_callable_rubric_traits" and isinstance(prop.value, str):
                try:
                    callable_traits_data = json.loads(prop.value)
                    for trait_data in callable_traits_data:
                        traits.append(CallableTrait(**trait_data))
                except (json.JSONDecodeError, TypeError):
                    pass  # Invalid JSON, skip

            elif prop.name == "global_metric_rubric_traits" and isinstance(prop.value, str):
                try:
                    metric_traits_data = json.loads(prop.value)
                    for trait_data in metric_traits_data:
                        traits.append(MetricRubricTrait(**trait_data))
                except (json.JSONDecodeError, TypeError):
                    pass  # Invalid JSON, skip

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
                            "GlobalRegexTrait",
                            "QuestionSpecificRegexTrait",
                            "GlobalCallableTrait",
                            "QuestionSpecificCallableTrait",
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
                    "GlobalRegexTrait",
                    "GlobalCallableTrait",
                    "GlobalMetricRubricTrait",
                ]:
                    return (
                        False,
                        f"Dataset-level rating must be a global trait type, got {rating.additionalType}",
                    )

        return True, "Valid benchmark"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
