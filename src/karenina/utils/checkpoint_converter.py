"""Benchmark converter utilities for JSON-LD format.

This module provides utilities to convert between internal Python representations
and the JSON-LD format used by the frontend.
"""

import hashlib
import json
from datetime import datetime
from typing import Any

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
from ..schemas.rubric_class import ManualRubricTrait, RubricTrait


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


def convert_rubric_trait_to_rating(
    trait: RubricTrait | ManualRubricTrait, rubric_type: str = "global"
) -> SchemaOrgRating:
    """
    Convert an internal RubricTrait or ManualRubricTrait to a schema.org Rating.

    Args:
        trait: The RubricTrait or ManualRubricTrait to convert
        rubric_type: Either 'global' or 'question-specific'

    Returns:
        A SchemaOrgRating object
    """
    # Handle ManualRubricTrait (always boolean)
    if isinstance(trait, ManualRubricTrait):
        return SchemaOrgRating(
            name=trait.name,
            description=trait.description,
            bestRating=1,
            worstRating=0,
            additionalType="GlobalManualRubricTrait"
            if rubric_type == "global"
            else "QuestionSpecificManualRubricTrait",
        )

    # Handle RubricTrait
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


def convert_rating_to_rubric_trait(rating: SchemaOrgRating) -> RubricTrait | ManualRubricTrait:
    """
    Convert a schema.org Rating back to a RubricTrait or ManualRubricTrait.

    Args:
        rating: The SchemaOrgRating to convert

    Returns:
        A RubricTrait or ManualRubricTrait object
    """
    # Check if it's a ManualRubricTrait
    if rating.additionalType in ["GlobalManualRubricTrait", "QuestionSpecificManualRubricTrait"]:
        return ManualRubricTrait(
            name=rating.name,
            description=rating.description or "",
            pattern=".*",  # Default pattern that matches everything
            callable_name=None,
            case_sensitive=True,
            invert_result=False,
        )

    # Handle regular RubricTrait
    # Determine if it's a boolean trait (0-1 range)
    is_boolean = rating.bestRating == 1 and rating.worstRating == 0

    return RubricTrait(
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
    question_rubric_traits: list[RubricTrait] | None = None,
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
    rubric_traits: list[RubricTrait | ManualRubricTrait],
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
                if rating.additionalType == "QuestionSpecificRubricTrait"
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
) -> list[RubricTrait | ManualRubricTrait] | None:
    """
    Extract global rubric traits from a benchmark.

    Args:
        benchmark: The benchmark to extract from

    Returns:
        List of RubricTrait or ManualRubricTrait objects or None if no global rubric
    """
    if not benchmark.rating:
        return None

    traits = []
    for rating in benchmark.rating:
        if rating.additionalType in ["GlobalRubricTrait", "GlobalManualRubricTrait"]:
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
                        ]:
                            return (
                                False,
                                f"Invalid additionalType for rating: {rating.additionalType}",
                            )

        # Validate global ratings if present
        if benchmark.rating:
            for rating in benchmark.rating:
                if rating.additionalType != "GlobalRubricTrait":
                    return (
                        False,
                        f"Dataset-level rating must be GlobalRubricTrait, got {rating.additionalType}",
                    )

        return True, "Valid benchmark"

    except Exception as e:
        return False, f"Validation error: {str(e)}"
