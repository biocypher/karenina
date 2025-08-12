"""High-level benchmark management for Karenina benchmarks.

This module provides the main Benchmark class for creating, loading,
saving, and executing benchmarks in JSON-LD format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas.checkpoint import JsonLdCheckpoint
from ..schemas.rubric_class import Rubric, RubricTrait
from ..utils.checkpoint_converter import (
    add_global_rubric_to_benchmark,
    add_question_to_benchmark,
    create_jsonld_benchmark,
    extract_global_rubric_from_benchmark,
    extract_questions_from_benchmark,
    validate_jsonld_benchmark,
)
from .models import FinishedTemplate, VerificationConfig, VerificationResult

# from .verification.validation import validate_answer_template  # TODO: Create this function


class Benchmark:
    """
    Main class for managing Karenina benchmarks in JSON-LD format.

    This class provides a high-level API for:
    - Creating benchmarks manually or automatically
    - Loading/saving JSON-LD benchmark files
    - Running verification with existing execution system
    - Full compatibility with frontend GUI exports
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        creator: str = "Karenina Benchmarking System",
    ):
        """
        Initialize a new checkpoint.

        Args:
            name: Name of the benchmark
            description: Description of the benchmark
            version: Version of the benchmark content
            creator: Creator name or organization
        """
        self._checkpoint = create_jsonld_benchmark(name, description, version, creator)
        self._questions_cache: dict[str, dict[str, Any]] = {}
        self._rebuild_cache()

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        creator: str = "Karenina Benchmarking System",
    ) -> "Benchmark":
        """
        Create a new benchmark (alias for constructor).

        Args:
            name: Name of the benchmark
            description: Description of the benchmark
            version: Version of the benchmark content
            creator: Creator name or organization

        Returns:
            A new Benchmark instance
        """
        return cls(name, description, version, creator)

    @classmethod
    def load(cls, path: Path) -> "Benchmark":
        """
        Load a benchmark from a JSON-LD file.

        Args:
            path: Path to the JSON-LD benchmark file

        Returns:
            A Benchmark instance loaded from the file

        Raises:
            ValueError: If the file is not valid JSON-LD
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Parse into JsonLdCheckpoint model
        try:
            checkpoint_data = JsonLdCheckpoint(**data)
        except Exception as e:
            raise ValueError(f"Invalid JSON-LD benchmark format: {e}") from e

        # Validate structure
        is_valid, error_msg = validate_jsonld_benchmark(checkpoint_data)
        if not is_valid:
            raise ValueError(f"Invalid benchmark: {error_msg}")

        # Create instance and set data
        instance = cls.__new__(cls)
        instance._checkpoint = checkpoint_data
        instance._questions_cache = {}
        instance._rebuild_cache()

        return instance

    def save(self, path: Path) -> None:
        """
        Save the benchmark to a JSON-LD file.

        Args:
            path: Path where to save the benchmark
        """
        path = Path(path)

        # Ensure .jsonld extension
        if path.suffix not in [".jsonld", ".json"]:
            path = path.with_suffix(".jsonld")

        # Update modified timestamp
        self._checkpoint.dateModified = datetime.now().isoformat()

        # Convert to dict for JSON serialization
        benchmark_dict = self._checkpoint.model_dump(by_alias=True, exclude_none=True)

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(benchmark_dict, f, indent=2, ensure_ascii=False)

    def add_question(
        self,
        question: str,
        raw_answer: str,
        answer_template: str | None = None,
        question_id: str | None = None,
        finished: bool = False,
        author: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a question to the benchmark.

        Args:
            question: The question text
            raw_answer: The expected answer text
            answer_template: Optional Python code for answer template (can be added later)
            question_id: Optional question ID (will be generated if not provided)
            finished: Whether the template is finished
            author: Optional author information
            sources: Optional source documents
            custom_metadata: Optional custom metadata

        Returns:
            The question ID
        """
        # If no template provided, create a minimal one
        if answer_template is None:
            answer_template = self._create_default_template(question)

        q_id = add_question_to_benchmark(
            self._checkpoint,
            question,
            raw_answer,
            answer_template,
            question_id,
            None,  # question rubric traits added separately
            finished,
            author,
            sources,
            custom_metadata,
        )

        # Update cache
        self._rebuild_cache()

        return q_id

    def add_answer_template(self, question_id: str, template_code: str) -> None:
        """
        Add or update an answer template for a question.

        Args:
            question_id: The question ID
            template_code: Python code defining the Answer class

        Raises:
            ValueError: If question_id not found or template is invalid
        """
        # TODO: Validate the template when validate_answer_template function exists
        # For now, do basic validation that it's not empty
        if not template_code.strip():
            raise ValueError("Invalid template: Template cannot be empty")

        # Find the question in the checkpoint
        found = False
        for item in self._checkpoint.hasPart:
            if self._get_item_id(item) == question_id:
                item.item.hasPart.text = template_code
                item.dateModified = datetime.now().isoformat()
                found = True
                break

        if not found:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        self._rebuild_cache()

    def add_global_rubric_trait(self, trait: RubricTrait) -> None:
        """
        Add a global rubric trait to the benchmark.

        Args:
            trait: The rubric trait to add
        """
        current_traits = extract_global_rubric_from_benchmark(self._checkpoint) or []
        current_traits.append(trait)
        add_global_rubric_to_benchmark(self._checkpoint, current_traits)

    def add_question_rubric_trait(self, question_id: str, trait: RubricTrait) -> None:
        """
        Add a question-specific rubric trait.

        Args:
            question_id: The question ID
            trait: The rubric trait to add

        Raises:
            ValueError: If question not found
        """
        from ..utils.checkpoint_converter import convert_rubric_trait_to_rating

        # Find the question
        found = False
        for item in self._checkpoint.hasPart:
            if self._get_item_id(item) == question_id:
                rating = convert_rubric_trait_to_rating(trait, "question-specific")
                if item.item.rating is None:
                    item.item.rating = []
                item.item.rating.append(rating)
                item.dateModified = datetime.now().isoformat()
                found = True
                break

        if not found:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        self._rebuild_cache()

    def get_question_ids(self) -> list[str]:
        """
        Get all question IDs in the benchmark.

        Returns:
            List of question IDs
        """
        return list(self._questions_cache.keys())

    def get_question(self, question_id: str) -> dict[str, Any]:
        """
        Get a question by ID.

        Args:
            question_id: The question ID

        Returns:
            Question dictionary with all fields

        Raises:
            ValueError: If question not found
        """
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")
        return self._questions_cache[question_id]

    def get_all_questions(self) -> list[dict[str, Any]]:
        """
        Get all questions in the benchmark.

        Returns:
            List of question dictionaries
        """
        return list(self._questions_cache.values())

    def get_finished_templates(self) -> list[FinishedTemplate]:
        """
        Get all finished templates for verification.

        Returns:
            List of FinishedTemplate objects ready for verification
        """
        templates = []
        for q_id, q_data in self._questions_cache.items():
            if q_data.get("finished", False):
                # Convert question rubric to dict format if present
                question_rubric = None
                if q_data.get("question_rubric"):
                    question_rubric = {"traits": [trait.model_dump() for trait in q_data["question_rubric"]]}

                template = FinishedTemplate(
                    question_id=q_id,
                    question_text=q_data["question"],
                    question_preview=q_data["question"][:100] + "..."
                    if len(q_data["question"]) > 100
                    else q_data["question"],
                    template_code=q_data["answer_template"],
                    last_modified=q_data.get("date_modified", datetime.now().isoformat()),
                    finished=True,
                    question_rubric=question_rubric,
                )
                templates.append(template)
        return templates

    def get_global_rubric(self) -> Rubric | None:
        """
        Get the global rubric from the benchmark.

        Returns:
            Rubric object or None if no global rubric
        """
        traits = extract_global_rubric_from_benchmark(self._checkpoint)
        if traits:
            return Rubric(traits=traits)
        return None

    def set_metadata(self, **metadata: Any) -> None:
        """
        Set benchmark metadata.

        Args:
            **metadata: Metadata fields to update (name, description, version, creator)
        """
        for key, value in metadata.items():
            if hasattr(self._checkpoint, key):
                setattr(self._checkpoint, key, value)
        self._checkpoint.dateModified = datetime.now().isoformat()

    def validate(self) -> tuple[bool, str]:
        """
        Validate the benchmark structure and all templates.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate benchmark structure
        is_valid, error_msg = validate_jsonld_benchmark(self._checkpoint)
        if not is_valid:
            return False, error_msg

        # Validate all templates
        for q_id, q_data in self._questions_cache.items():
            template_code = q_data.get("answer_template")
            if template_code is not None and not template_code.strip():
                return False, f"Invalid template for {q_id}: Template cannot be empty"

        return True, "Benchmark is valid"

    # Integration with existing verification system
    def run_verification(
        self,
        config: VerificationConfig,  # noqa: ARG002
        question_ids: list[str] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Run verification on the benchmark using existing execution system.

        Args:
            config: Verification configuration
            question_ids: Optional list of question IDs to verify (default: all)

        Returns:
            Dictionary mapping question IDs to VerificationResult objects
        """
        # TODO: This function doesn't exist yet - create a placeholder
        # from .verification.orchestrator import run_question_verification

        if question_ids is None:
            question_ids = [q_id for q_id, q in self._questions_cache.items() if q.get("finished", False)]

        results: dict[str, VerificationResult] = {}
        for q_id in question_ids:
            if q_id not in self._questions_cache:
                continue

            q_data = self._questions_cache[q_id]
            template_code = q_data.get("answer_template")

            if not template_code:
                continue

            # Create placeholder verification result
            # TODO: Replace with actual verification call when run_question_verification exists
            result = VerificationResult(
                question_id=q_id,
                success=True,
                question_text=q_data["question"],
                raw_llm_response="Mock response for placeholder",
                answering_model="placeholder",
                parsing_model="placeholder",
                execution_time=0.0,
                timestamp=datetime.now().isoformat(),
            )

            results[q_id] = result

        return results

    # Private helper methods
    def _rebuild_cache(self) -> None:
        """Rebuild the internal questions cache from benchmark data."""
        self._questions_cache = {}
        questions = extract_questions_from_benchmark(self._checkpoint)
        for q in questions:
            self._questions_cache[q["id"]] = q

    def _get_item_id(self, item: Any) -> str:
        """Get the ID for a DataFeedItem."""
        if item.id:
            return str(item.id)
        # Generate from question text if no ID
        from ..utils.checkpoint_converter import generate_question_id

        return generate_question_id(item.item.text)

    def _create_default_template(self, question: str) -> str:
        """Create a minimal default template for a question."""
        return f'''class Answer(BaseAnswer):
    """Answer template for: {question[:50]}..."""

    response: str = Field(description="The answer response")

    def verify(self) -> bool:
        # TODO: Implement verification logic
        return True
'''

    @property
    def jsonld_data(self) -> JsonLdCheckpoint:
        """Get the raw JSON-LD benchmark data."""
        return self._checkpoint
