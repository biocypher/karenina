"""High-level benchmark management for Karenina benchmarks.

This module provides the main Benchmark class for creating, loading,
saving, and executing benchmarks in JSON-LD format.

This is the refactored version that uses the core submodule managers
while maintaining 100% backward compatibility.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ..schemas.checkpoint import SchemaOrgQuestion
    from ..schemas.domain import Question

from ..domain.answers.generator import generate_answer_template, load_answer_templates_from_json
from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric
from ..schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultSet,
)
from .core import (
    BenchmarkBase,
    ExportManager,
    MetadataManager,
    QuestionManager,
    ResultsManager,
    RubricManager,
    TemplateManager,
    VerificationManager,
)
from .core.questions import _NOT_PROVIDED


class Benchmark:
    """
    Main class for managing Karenina benchmarks in JSON-LD format.

    This class provides a high-level API for:
    - Creating benchmarks manually or automatically
    - Loading/saving JSON-LD benchmark files
    - Running verification with existing execution system
    - Full compatibility with frontend GUI exports

    This is a facade that delegates to specialized manager classes for better maintainability.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        creator: str = "Karenina Benchmarking System",
    ):
        """
        Initialize a new benchmark.

        Args:
            name: Name of the benchmark
            description: Description of the benchmark
            version: Version of the benchmark content
            creator: Creator name or organization
        """
        # Initialize the base class
        self._base = BenchmarkBase(name, description, version, creator)

        # Initialize managers
        self._metadata_manager = MetadataManager(self._base)
        self._question_manager = QuestionManager(self._base)
        self._rubric_manager = RubricManager(self._base)
        self._template_manager = TemplateManager(self._base)
        self._results_manager = ResultsManager(self._base)
        self._verification_manager = VerificationManager(self._base, self._rubric_manager)
        # Set the results manager on verification manager for auto-storage
        self._verification_manager._results_manager = self._results_manager
        self._export_manager = ExportManager(self._base, self._template_manager, self._rubric_manager)

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
        # Load using base class
        base = BenchmarkBase.load(path)

        # Create new instance and replace base
        instance = cls.__new__(cls)
        instance._base = base

        # Initialize managers
        instance._metadata_manager = MetadataManager(instance._base)
        instance._question_manager = QuestionManager(instance._base)
        instance._rubric_manager = RubricManager(instance._base)
        instance._template_manager = TemplateManager(instance._base)
        instance._results_manager = ResultsManager(instance._base)
        instance._verification_manager = VerificationManager(instance._base, instance._rubric_manager)
        # Set the results manager on verification manager for auto-storage
        instance._verification_manager._results_manager = instance._results_manager
        instance._export_manager = ExportManager(instance._base, instance._template_manager, instance._rubric_manager)

        return instance

    def save(self, path: Path) -> None:
        """Save the benchmark to a JSON-LD file."""
        self._base.save(path)

    def save_to_db(self, storage: str, checkpoint_path: Path | None = None) -> "Benchmark":
        """Save this benchmark to a database.

        Args:
            storage: Database storage URL (e.g., "sqlite:///example.db")
            checkpoint_path: Optional path to the checkpoint file

        Returns:
            The same benchmark instance (for chaining)
        """
        from typing import cast

        from ..storage import save_benchmark

        # save_benchmark returns just the benchmark when detect_duplicates_only=False (default)
        result = save_benchmark(self, storage, checkpoint_path)
        # Type cast since we know detect_duplicates_only=False returns Benchmark, not tuple
        return cast("Benchmark", result)

    @classmethod
    def load_from_db(cls, benchmark_name: str, storage: str) -> "Benchmark":
        """Load a benchmark from a database.

        Args:
            benchmark_name: Name of the benchmark to load
            storage: Database storage URL

        Returns:
            Loaded Benchmark instance
        """
        from ..storage import load_benchmark

        result = load_benchmark(benchmark_name, storage, load_config=False)
        return result  # type: ignore[return-value]

    # Question management methods - delegate to QuestionManager
    def add_question(
        self,
        question: Union[str, "Question"],
        raw_answer: str | None = None,
        answer_template: str | type | None = None,
        question_id: str | None = None,
        finished: bool | object = _NOT_PROVIDED,
        author: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        custom_metadata: dict[str, Any] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> str:
        """Add a question to the benchmark.

        This method supports three usage patterns:
        1. Traditional kwargs: add_question("What is 2+2?", "4", ...)
        2. Question object: add_question(Question(...), ...)
        3. Answer class: add_question("What is 2+2?", "4", answer_template=AnswerClass)

        Args:
            question: Either the question text (str) or a Question object
            raw_answer: The expected answer text (required if question is str)
            answer_template: Optional Python code (str), Answer class (type), or None
            question_id: Optional question ID (will be generated if not provided)
            finished: Whether the template is finished (auto-set to True if answer_template provided)
            author: Optional author information
            sources: Optional source documents
            custom_metadata: Optional custom metadata
            few_shot_examples: Optional list of few-shot examples with 'question' and 'answer' keys

        Returns:
            The question ID
        """
        return self._question_manager.add_question(
            question,
            raw_answer,
            answer_template,
            question_id,
            finished,
            author,
            sources,
            custom_metadata,
            few_shot_examples,
        )

    def get_question_ids(self) -> list[str]:
        """Get all question IDs in the benchmark."""
        return self._question_manager.get_question_ids()

    def get_question(self, question_id: str) -> dict[str, Any]:
        """Get a question by ID."""
        return self._question_manager.get_question(question_id)

    def get_all_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get all questions in the benchmark.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        return self._question_manager.get_all_questions(ids_only)

    def get_question_as_object(self, question_id: str) -> "Question":
        """Get a question as a Question object."""
        return self._question_manager.get_question_as_object(question_id)

    def get_all_questions_as_objects(self) -> list["Question"]:
        """Get all questions as Question objects."""
        return self._question_manager.get_all_questions_as_objects()

    def add_question_from_object(self, question_obj: "Question", **metadata: Any) -> str:
        """Add a question to the benchmark from a Question object."""
        return self._question_manager.add_question_from_object(question_obj, **metadata)

    def update_question_metadata(self, question_id: str, **metadata: Any) -> None:
        """Update question metadata fields."""
        self._question_manager.update_question_metadata(question_id, **metadata)

    def get_question_metadata(self, question_id: str) -> dict[str, Any]:
        """Get all metadata for a specific question."""
        return self._question_manager.get_question_metadata(question_id)

    def get_question_custom_property(self, question_id: str, name: str) -> Any:
        """Get a custom property from question metadata."""
        return self._question_manager.get_question_custom_property(question_id, name)

    def set_question_custom_property(self, question_id: str, name: str, value: Any) -> None:
        """Set a custom property on question metadata."""
        self._question_manager.set_question_custom_property(question_id, name, value)

    def remove_question_custom_property(self, question_id: str, name: str) -> bool:
        """Remove a custom property from question metadata."""
        return self._question_manager.remove_question_custom_property(question_id, name)

    def get_question_author(self, question_id: str) -> dict[str, Any] | None:
        """Get author information for a question."""
        return self._question_manager.get_question_author(question_id)

    def set_question_author(self, question_id: str, author: dict[str, Any] | None) -> None:
        """Set author information for a question."""
        self._question_manager.set_question_author(question_id, author)

    def get_question_sources(self, question_id: str) -> list[dict[str, Any]] | None:
        """Get source documents for a question."""
        return self._question_manager.get_question_sources(question_id)

    def set_question_sources(self, question_id: str, sources: list[dict[str, Any]] | None) -> None:
        """Set source documents for a question."""
        self._question_manager.set_question_sources(question_id, sources)

    def get_question_timestamps(self, question_id: str) -> dict[str, str]:
        """Get creation and modification timestamps for a question."""
        return self._question_manager.get_question_timestamps(question_id)

    def remove_question(self, question_id: str) -> bool:
        """Remove a specific question from the benchmark."""
        return self._question_manager.remove_question(question_id)

    def clear_questions(self) -> int:
        """Remove all questions from the benchmark."""
        return self._question_manager.clear_questions()

    def add_questions_batch(self, questions_data: list[dict[str, Any]]) -> list[str]:
        """Add multiple questions at once."""
        return self._question_manager.add_questions_batch(questions_data)

    def mark_finished(self, question_id: str) -> None:
        """Mark a question as finished."""
        self._question_manager.mark_finished(question_id)

    def mark_unfinished(self, question_id: str) -> None:
        """Mark a question as unfinished."""
        self._question_manager.mark_unfinished(question_id)

    def mark_finished_batch(self, question_ids: list[str]) -> None:
        """Mark multiple questions as finished."""
        self._question_manager.mark_finished_batch(question_ids)

    def mark_unfinished_batch(self, question_ids: list[str]) -> None:
        """Mark multiple questions as unfinished."""
        self._question_manager.mark_unfinished_batch(question_ids)

    def toggle_finished(self, question_id: str) -> bool:
        """Toggle finished status of a question."""
        return self._question_manager.toggle_finished(question_id)

    def get_unfinished_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get questions that are not marked as finished.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        return self._question_manager.get_unfinished_questions(ids_only)

    def get_finished_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get questions that are marked as finished.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        return self._question_manager.get_finished_questions(ids_only)

    def filter_questions(
        self,
        finished: bool | None = None,
        has_template: bool | None = None,
        has_rubric: bool | None = None,
        author: str | None = None,
        custom_filter: Any = None,
    ) -> list[dict[str, Any]]:
        """Filter questions based on criteria."""
        return self._question_manager.filter_questions(finished, has_template, has_rubric, author, custom_filter)

    def filter_by_metadata(
        self,
        field_path: str,
        value: Any,
        match_mode: str = "exact",
    ) -> list[dict[str, Any]]:
        """Filter questions by a metadata field using dot notation."""
        return self._question_manager.filter_by_metadata(field_path, value, match_mode)

    def filter_by_custom_metadata(
        self,
        match_all: bool = True,
        **criteria: Any,
    ) -> list[dict[str, Any]]:
        """Filter questions by custom metadata fields with AND/OR logic."""
        return self._question_manager.filter_by_custom_metadata(match_all, **criteria)

    def search_questions(
        self,
        query: str | list[str],
        match_all: bool = True,
        fields: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for questions containing the query text (unified search method)."""
        return self._question_manager.search_questions(query, match_all, fields, case_sensitive, regex)

    def get_questions_by_author(self, author: str) -> list[dict[str, Any]]:
        """Get questions created by a specific author."""
        return self._question_manager.get_questions_by_author(author)

    def get_questions_with_rubric(self) -> list[dict[str, Any]]:
        """Get questions that have question-specific rubrics."""
        return self._question_manager.get_questions_with_rubric()

    def count_by_field(
        self,
        field_path: str,
        questions: list[dict[str, Any]] | None = None,
    ) -> dict[Any, int]:
        """Count questions grouped by a field value using dot notation."""
        return self._question_manager.count_by_field(field_path, questions)

    # Template management methods - delegate to TemplateManager
    def add_answer_template(self, question_id: str, template_code: str) -> None:
        """Add or update an answer template for a question."""
        self._template_manager.add_answer_template(question_id, template_code)

    def has_template(self, question_id: str) -> bool:
        """Check if a question has a non-default template."""
        return self._template_manager.has_template(question_id)

    def get_template(self, question_id: str) -> str:
        """Get template code for a question."""
        return self._template_manager.get_template(question_id)

    def update_template(self, question_id: str, template_code: str) -> None:
        """Update existing template."""
        self._template_manager.update_template(question_id, template_code)

    def copy_template(self, from_id: str, to_id: str) -> None:
        """Copy template from one question to another."""
        self._template_manager.copy_template(from_id, to_id)

    def get_finished_templates(self) -> list[FinishedTemplate]:
        """Get all finished templates for verification."""
        return self._template_manager.get_finished_templates()

    def get_missing_templates(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """
        Get questions that don't have non-default templates.

        Args:
            ids_only: If True, return only question IDs. If False (default), return full question objects.

        Returns:
            List of question IDs (if ids_only=True) or list of question dictionaries (if ids_only=False)
        """
        return self._template_manager.get_missing_templates(ids_only)

    def apply_global_template(self, template_code: str) -> list[str]:
        """Apply a template to all questions that don't have one."""
        return self._template_manager.apply_global_template(template_code)

    def validate_templates(self) -> tuple[bool, list[dict[str, str]]]:
        """Validate all templates are valid Python code."""
        return self._template_manager.validate_templates()

    # Template generation methods using LLMs
    def generate_template_for_question(
        self,
        question_id: str,
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        """
        Generate an answer template for a specific question using LLM.

        Args:
            question_id: The question ID to generate template for
            model: The model to use for generation (default: gemini-2.0-flash)
            model_provider: The provider of the model (default: google_genai)
            temperature: The temperature for generation (default: 0)
            interface: The interface to use (default: langchain)
            force_regenerate: If True, regenerate even if template exists (default: False)

        Returns:
            Dict with 'success', 'template_code', 'error', and 'raw_response' keys

        Raises:
            ValueError: If question_id not found
        """
        # Check if question exists
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Check if template already exists and force_regenerate is False
        if not force_regenerate and self.has_template(question_id):
            return {
                "success": True,
                "template_code": self.get_template(question_id),
                "error": "Template already exists (use force_regenerate=True to override)",
                "raw_response": None,
                "skipped": True,
            }

        try:
            question_data = self._questions_cache[question_id]
            question_text = question_data.get("question", "")
            raw_answer = question_data.get("raw_answer", "")

            # Generate template using the generator function
            template_code = generate_answer_template(
                question=question_text,
                raw_answer=raw_answer,
                model=model,
                model_provider=model_provider,
                temperature=temperature,
                interface=interface,
            )

            # Template code is now returned directly from the generator
            # No need to extract code blocks as the new generator returns ready-to-use code

            # If no code blocks found, return error
            if not template_code.strip():
                return {
                    "success": False,
                    "template_code": "",
                    "error": "No valid code blocks found in LLM response",
                    "raw_response": template_code,
                    "skipped": False,
                }

            # Save the template to the benchmark
            self.add_answer_template(question_id, template_code)
            error_msg = None

            return {
                "success": True,
                "template_code": template_code,
                "error": error_msg,
                "raw_response": template_code,
                "skipped": False,
            }

        except Exception as e:
            return {
                "success": False,
                "template_code": "",
                "error": str(e),
                "raw_response": None,
                "skipped": False,
            }

    def generate_templates(
        self,
        question_ids: list[str],
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Generate templates for multiple questions using LLM.

        Args:
            question_ids: List of question IDs to generate templates for
            model: The model to use for generation
            model_provider: The provider of the model
            temperature: The temperature for generation
            interface: The interface to use
            force_regenerate: If True, regenerate even if templates exist
            progress_callback: Optional callback for progress updates (percentage, message)

        Returns:
            Dict mapping question_id to generation result dict

        Raises:
            ValueError: If any question_id not found
        """
        # Validate all question IDs first
        invalid_ids = [qid for qid in question_ids if qid not in self._questions_cache]
        if invalid_ids:
            raise ValueError(f"Questions not found: {invalid_ids}")

        results = {}
        total_questions = len(question_ids)

        for i, question_id in enumerate(question_ids):
            # Update progress
            if progress_callback:
                percentage = (i / total_questions) * 100
                question_text = self._questions_cache[question_id].get("question", "")
                message = f"Processing: {question_text[:50]}..."
                progress_callback(percentage, message)

            # Generate template for this question
            result = self.generate_template_for_question(
                question_id=question_id,
                model=model,
                model_provider=model_provider,
                temperature=temperature,
                interface=interface,
                force_regenerate=force_regenerate,
            )
            results[question_id] = result

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Template generation completed")

        return results

    def generate_all_templates(
        self,
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
        only_missing: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Generate templates for all questions in the benchmark using LLM.

        Args:
            model: The model to use for generation
            model_provider: The provider of the model
            temperature: The temperature for generation
            interface: The interface to use
            force_regenerate: If True, regenerate even if templates exist
            progress_callback: Optional callback for progress updates
            only_missing: If True, only generate for questions without templates

        Returns:
            Dict mapping question_id to generation result dict
        """
        if only_missing and not force_regenerate:
            # Get questions that don't have templates
            from typing import cast

            question_ids = cast(list[str], self.get_missing_templates(ids_only=True))
        else:
            # Get all question IDs
            question_ids = self.get_question_ids()

        if not question_ids:
            return {}

        return self.generate_templates(
            question_ids=question_ids,
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            interface=interface,
            force_regenerate=force_regenerate,
            progress_callback=progress_callback,
        )

    def export_generated_templates(self, file_path: Path) -> None:
        """
        Export all generated templates to a JSON file.

        Args:
            file_path: Path where to save the JSON file

        The exported format is compatible with load_answer_templates_from_json.
        """
        templates_dict = {}

        for question_id in self.get_question_ids():
            if self.has_template(question_id):
                templates_dict[question_id] = self.get_template(question_id)

        with file_path.open("w") as f:
            json.dump(templates_dict, f, indent=2)

    def import_generated_templates(self, file_path: Path, force_overwrite: bool = False) -> dict[str, bool]:
        """
        Import templates from a JSON file generated by export_generated_templates.

        Args:
            file_path: Path to the JSON file to load
            force_overwrite: If True, overwrite existing templates

        Returns:
            Dict mapping question_id to success status (True if imported, False if skipped)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        # Load templates using the generator function
        answer_templates = load_answer_templates_from_json(str(file_path), return_blocks=True)
        if isinstance(answer_templates, tuple):
            _, code_blocks = answer_templates
        else:
            raise ValueError("Unable to load code blocks from JSON file")

        results = {}

        for question_id, template_code in code_blocks.items():
            # Check if question exists in benchmark
            if question_id not in self._questions_cache:
                results[question_id] = False
                continue

            # Check if template already exists
            if not force_overwrite and self.has_template(question_id):
                results[question_id] = False
                continue

            try:
                # Add template to benchmark
                self.add_answer_template(question_id, template_code)
                results[question_id] = True
            except Exception:
                results[question_id] = False

        return results

    # Rubric management methods - delegate to RubricManager
    def add_global_rubric_trait(self, trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait) -> None:
        """Add a global rubric trait to the benchmark."""
        self._rubric_manager.add_global_rubric_trait(trait)

    def add_question_rubric_trait(
        self, question_id: str, trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait
    ) -> None:
        """Add a question-specific rubric trait."""
        self._rubric_manager.add_question_rubric_trait(question_id, trait)

    def set_global_rubric(self, rubric: Rubric) -> None:
        """Set the complete global rubric (replaces existing)."""
        # Clear existing global rubric
        self.clear_global_rubric()
        # Add all traits from the rubric
        for trait in rubric.llm_traits:
            self.add_global_rubric_trait(trait)
        for regex_trait in rubric.regex_traits:
            self.add_global_rubric_trait(regex_trait)
        for callable_trait in rubric.callable_traits:
            self.add_global_rubric_trait(callable_trait)
        for metric_trait in rubric.metric_traits:
            self.add_global_rubric_trait(metric_trait)

    def set_question_rubric(self, question_id: str, rubric: Rubric) -> None:
        """Set the complete question-specific rubric (replaces existing)."""
        # Clear existing question rubric
        self.remove_question_rubric(question_id)
        # Add all traits from the rubric
        for trait in rubric.llm_traits:
            self.add_question_rubric_trait(question_id, trait)
        for regex_trait in rubric.regex_traits:
            self.add_question_rubric_trait(question_id, regex_trait)
        for callable_trait in rubric.callable_traits:
            self.add_question_rubric_trait(question_id, callable_trait)
        for metric_trait in rubric.metric_traits:
            self.add_question_rubric_trait(question_id, metric_trait)

    def get_global_rubric(self) -> Rubric | None:
        """Get the global rubric from the benchmark."""
        return self._rubric_manager.get_global_rubric()

    def clear_global_rubric(self) -> bool:
        """Remove the global rubric."""
        return self._rubric_manager.clear_global_rubric()

    def remove_question_rubric(self, question_id: str) -> bool:
        """Remove question-specific rubric."""
        return self._rubric_manager.remove_question_rubric(question_id)

    def clear_all_rubrics(self) -> int:
        """Remove all rubrics (global and question-specific)."""
        return self._rubric_manager.clear_all_rubrics()

    def validate_rubrics(self) -> tuple[bool, list[str]]:
        """Validate all rubrics are properly configured."""
        return self._rubric_manager.validate_rubrics()

    # Verification methods - delegate to VerificationManager
    def verify_question(
        self,
        question_id: str,
        config: VerificationConfig,
        run_name: str | None = None,
        async_enabled: bool | None = None,
    ) -> dict[str, VerificationResult]:
        """Verify a single question."""
        return self._verification_manager.verify_question(question_id, config, run_name, async_enabled)

    def verify_questions(
        self,
        question_ids: list[str],
        config: VerificationConfig,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """Verify multiple specific questions."""
        return self._verification_manager.verify_questions(
            question_ids, config, run_name, async_enabled, progress_callback
        )

    def verify_filtered(
        self,
        config: VerificationConfig,
        finished: bool | None = True,
        has_template: bool | None = True,
        has_rubric: bool | None = None,
        author: str | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """Verify questions matching specific criteria."""
        return self._verification_manager.verify_filtered(
            config, finished, has_template, has_rubric, author, run_name, async_enabled, progress_callback
        )

    def verify_all_finished(
        self,
        config: VerificationConfig,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """Verify all finished questions in the benchmark."""
        return self._verification_manager.verify_all_finished(config, run_name, async_enabled, progress_callback)

    def verify_custom(
        self,
        question_selector: Callable[[dict[str, Any]], bool],
        config: VerificationConfig,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """Verify questions selected by a custom function."""
        return self._verification_manager.verify_custom(
            question_selector, config, run_name, async_enabled, progress_callback
        )

    def verify_dry_run(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
    ) -> dict[str, bool]:
        """Perform a dry run verification (validate without executing)."""
        return self._verification_manager.verify_dry_run(config, question_ids)

    def run_verification(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> VerificationResultSet:
        """Run verification on the benchmark using existing execution system."""
        return self._verification_manager.run_verification(
            config, question_ids, run_name, async_enabled, progress_callback
        )

    def verify_with_mixed_configs(
        self,
        question_configs: dict[str, VerificationConfig],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, VerificationResult]]:
        """Verify different questions with different configurations."""
        return self._verification_manager.verify_with_mixed_configs(question_configs, progress_callback)

    def verify_comparative(
        self,
        question_ids: list[str],
        configs: list[VerificationConfig],
        run_names: list[str],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, VerificationResult]]:
        """Run same questions with multiple configurations for comparison."""
        return self._verification_manager.verify_comparative(question_ids, configs, run_names, progress_callback)

    def verify_progressive(
        self,
        config: VerificationConfig,
        batch_size: int = 5,
        run_name: str | None = None,
        resume_from: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """Verify questions in batches with ability to resume from interruptions."""
        return self._verification_manager.verify_progressive(
            config, batch_size, run_name, resume_from, progress_callback
        )

    # Results management methods - delegate to ResultsManager
    def store_verification_results(
        self,
        results: VerificationResultSet | dict[str, VerificationResult],
        run_name: str | None = None,
    ) -> None:
        """
        Store verification results in the benchmark metadata.

        Args:
            results: VerificationResultSet or dict mapping result keys to VerificationResult objects
            run_name: Optional run name for tracking
        """
        # Convert VerificationResultSet to dict format for storage
        if isinstance(results, VerificationResultSet):
            # Generate keys for results (needed for storage)
            results_dict = {}
            for i, result in enumerate(results):
                # Create a key similar to the old format
                key = f"{result.metadata.question_id}_{result.metadata.answering_model}_{result.metadata.parsing_model}"
                if result.metadata.answering_replicate is not None:
                    key += f"_rep{result.metadata.answering_replicate}"
                if result.metadata.timestamp:
                    key += f"_{result.metadata.timestamp}"
                # Handle potential duplicates by appending index
                if key in results_dict:
                    key += f"_{i}"
                results_dict[key] = result
            self._results_manager.store_verification_results(results_dict, run_name)
        else:
            self._results_manager.store_verification_results(results, run_name)

    def get_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """Get verification results for specific questions and/or runs."""
        return self._results_manager.get_verification_results(question_ids, run_name)

    def get_verification_history(self, question_id: str | None = None) -> dict[str, dict[str, VerificationResult]]:
        """Get verification history organized by run name."""
        return self._results_manager.get_verification_history(question_id)

    def clear_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """Clear verification results."""
        return self._results_manager.clear_verification_results(question_ids, run_name)

    def export_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str = "json",
        global_rubric: "Rubric | None" = None,
    ) -> str:
        """Export verification results in specified format."""
        return self._results_manager.export_verification_results(question_ids, run_name, format, global_rubric)

    def export_verification_results_to_file(
        self,
        file_path: Path,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str | None = None,
        global_rubric: "Rubric | None" = None,
    ) -> None:
        """Export verification results directly to a file."""
        self._results_manager.export_results_to_file(file_path, question_ids, run_name, format, global_rubric)

    def load_verification_results_from_file(
        self,
        file_path: Path,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """Load verification results from a previously exported file."""
        return self._results_manager.load_results_from_file(file_path, run_name)

    def get_verification_summary(self, run_name: str | None = None) -> dict[str, Any]:
        """Get summary statistics for verification results."""
        return self._results_manager.get_verification_summary(run_name)

    def get_all_run_names(self) -> list[str]:
        """Get all verification run names."""
        return self._results_manager.get_all_run_names()

    def get_results_statistics_by_run(self) -> dict[str, dict[str, Any]]:
        """Get verification statistics for each run."""
        return self._results_manager.get_results_statistics_by_run()

    # Metadata management methods - delegate to MetadataManager
    def get_custom_property(self, name: str) -> Any:
        """Get a custom property from benchmark metadata."""
        return self._metadata_manager.get_custom_property(name)

    def set_custom_property(self, name: str, value: Any) -> None:
        """Set a custom property in benchmark metadata."""
        self._metadata_manager.set_custom_property(name, value)

    def remove_custom_property(self, name: str) -> bool:
        """Remove a custom property from benchmark metadata."""
        return self._metadata_manager.remove_custom_property(name)

    def get_all_custom_properties(self) -> dict[str, Any]:
        """Get all custom properties as a dictionary."""
        return self._metadata_manager.get_all_custom_properties()

    def set_multiple_custom_properties(self, properties: dict[str, Any]) -> None:
        """Set multiple custom properties at once."""
        self._metadata_manager.set_multiple_custom_properties(properties)

    # Export and reporting methods - delegate to ExportManager
    def to_dict(self) -> dict[str, Any]:
        """Export benchmark as a plain dictionary."""
        return self._export_manager.to_dict()

    def to_markdown(self) -> str:
        """Export benchmark as markdown document."""
        return self._export_manager.to_markdown()

    def to_csv(self) -> str:
        """Export questions as CSV format."""
        return self._export_manager.to_csv()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive benchmark statistics."""
        return self._export_manager.get_summary()

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about the benchmark."""
        return self._export_manager.get_statistics()

    def check_readiness(self) -> dict[str, Any]:
        """Comprehensive readiness check for verification."""
        return self._export_manager.check_readiness()

    def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive health/status report."""
        return self._export_manager.get_health_report()

    def clone(self) -> "Benchmark":
        """Create a deep copy of the benchmark."""
        cloned_base = self._export_manager.clone()

        # Create new instance and replace base
        instance = Benchmark.__new__(Benchmark)
        instance._base = cloned_base

        # Initialize managers
        instance._metadata_manager = MetadataManager(instance._base)
        instance._question_manager = QuestionManager(instance._base)
        instance._rubric_manager = RubricManager(instance._base)
        instance._template_manager = TemplateManager(instance._base)
        instance._results_manager = ResultsManager(instance._base)
        instance._verification_manager = VerificationManager(instance._base, instance._rubric_manager)
        # Set the results manager on verification manager for auto-storage
        instance._verification_manager._results_manager = instance._results_manager
        instance._export_manager = ExportManager(instance._base, instance._template_manager, instance._rubric_manager)

        return instance

    def validate(self) -> tuple[bool, str]:
        """Validate the benchmark structure and all templates."""
        from .verification.utils.validation import validate_answer_template

        # Validate benchmark structure
        is_valid, error_msg = self._base.validate()
        if not is_valid:
            return False, error_msg

        # Validate all templates using the verification system (like original)
        for q_id, q_data in self._questions_cache.items():
            template_code = q_data.get("answer_template")
            if template_code is not None:
                is_valid, error_msg_or_none, _ = validate_answer_template(template_code)
                error_msg = error_msg_or_none or "Unknown validation error"
                if not is_valid:
                    return False, f"Invalid template for {q_id}: {error_msg}"

        return True, "Benchmark is valid"

    def set_metadata(self, **metadata: Any) -> None:
        """Set benchmark metadata."""
        self._base.set_metadata(**metadata)

    # Base class property delegation
    @property
    def _checkpoint(self) -> Any:
        """Get the raw JSON-LD checkpoint data (for backward compatibility)."""
        return self._base._checkpoint

    @property
    def _questions_cache(self) -> dict[str, Any]:
        """Get the questions cache (for backward compatibility)."""
        return self._base._questions_cache

    def _get_item_id(self, item: Any) -> str:
        """Get the ID for a DataFeedItem (for backward compatibility)."""
        return self._base._get_item_id(item)

    def _rebuild_cache(self) -> None:
        """Rebuild the internal questions cache (for backward compatibility)."""
        return self._base._rebuild_cache()

    def _get_merged_rubric_for_question(self, question_id: str) -> Rubric | None:
        """Get merged rubric for a question (for backward compatibility)."""
        return self._rubric_manager.get_merged_rubric_for_question(question_id)

    @property
    def jsonld_data(self) -> Any:
        """Get the raw JSON-LD benchmark data."""
        return self._base.jsonld_data

    @property
    def name(self) -> str:
        """Get the benchmark name."""
        return self._base.name

    @name.setter
    def name(self, value: str) -> None:
        """Set the benchmark name."""
        self._base.name = value

    @property
    def description(self) -> str:
        """Get the benchmark description."""
        return self._base.description

    @description.setter
    def description(self, value: str) -> None:
        """Set the benchmark description."""
        self._base.description = value

    @property
    def version(self) -> str:
        """Get the benchmark version."""
        return self._base.version

    @version.setter
    def version(self, value: str) -> None:
        """Set the benchmark version."""
        self._base.version = value

    @property
    def creator(self) -> str:
        """Get the benchmark creator."""
        return self._base.creator

    @creator.setter
    def creator(self, value: str) -> None:
        """Set the benchmark creator."""
        self._base.creator = value

    @property
    def id(self) -> str | None:
        """Get the benchmark ID."""
        return self._base.id

    @id.setter
    def id(self, value: str | None) -> None:
        """Set the benchmark ID."""
        self._base.id = value

    @property
    def created_at(self) -> str:
        """Get the creation timestamp."""
        return self._base.created_at

    @created_at.setter
    def created_at(self, value: str) -> None:
        """Set the creation timestamp."""
        self._base.created_at = value

    @property
    def modified_at(self) -> str:
        """Get the last modification timestamp."""
        return self._base.modified_at

    @modified_at.setter
    def modified_at(self, value: str) -> None:
        """Set the last modification timestamp."""
        self._base.modified_at = value

    @property
    def question_count(self) -> int:
        """Get the total number of questions."""
        return self._base.question_count

    @property
    def finished_count(self) -> int:
        """Get the number of finished questions."""
        return self._base.finished_count

    @property
    def is_empty(self) -> bool:
        """Check if the benchmark has no questions."""
        return self._base.is_empty

    @property
    def is_complete(self) -> bool:
        """Check if all questions have templates and are finished."""
        return self._base.is_complete

    def get_progress(self) -> float:
        """Get completion progress as percentage (0-100)."""
        return self._base.get_progress()

    # Magic methods for better usability
    def __repr__(self) -> str:
        """
        Developer-friendly representation with detailed statistics.

        Shows metadata, content stats, rubric breakdown, sample questions,
        and readiness status in a structured multi-line format.
        """
        lines = ["Benchmark("]

        # === METADATA ===
        lines.append("  === METADATA ===")
        lines.append(f"  Name: {self._base.name}")
        lines.append(f"  Version: {self._base.version}")
        lines.append(f"  Creator: {self._base.creator}")
        lines.append(f"  Created: {self._base.created_at}")
        lines.append(f"  Modified: {self._base.modified_at}")

        # Collect unique keywords
        all_questions_data = self.get_all_questions(ids_only=False)
        assert isinstance(all_questions_data, list)
        unique_keywords: set[str] = set()
        for q in all_questions_data:
            if isinstance(q, dict):
                keywords = q.get("keywords", [])
                if keywords:
                    unique_keywords.update(keywords)

        if unique_keywords:
            keyword_list = sorted(unique_keywords)
            keywords_str = ", ".join(keyword_list)
            lines.append(f"  Keywords: {keywords_str} ({len(keyword_list)} total)")
        else:
            lines.append("  Keywords: none")

        # === CONTENT ===
        lines.append("")
        lines.append("  === CONTENT ===")
        summary = self._export_manager.get_summary()
        progress = summary["progress_percentage"]
        lines.append(
            f"  Questions: {self._base.question_count} total, "
            f"{self._base.finished_count} finished ({progress:.1f}% complete)"
        )
        lines.append(f"  Templates: {summary['has_template_count']}/{self._base.question_count} questions")

        # === RUBRICS ===
        lines.append("")
        lines.append("  === RUBRICS ===")
        global_rubric = self._rubric_manager.get_global_rubric()

        if global_rubric:
            # Count traits by type in global rubric
            llm_count = len(global_rubric.llm_traits)
            regex_count = len(global_rubric.regex_traits)
            metric_count = len(global_rubric.metric_traits)
            callable_count = len(global_rubric.callable_traits)

            total_traits = llm_count + regex_count + metric_count + callable_count
            lines.append(f"  Global Rubric: {total_traits} traits")
            trait_breakdown = []
            if llm_count > 0:
                trait_breakdown.append(f"LLM: {llm_count}")
            if regex_count > 0:
                trait_breakdown.append(f"Regex: {regex_count}")
            if metric_count > 0:
                trait_breakdown.append(f"Metric: {metric_count}")
            if callable_count > 0:
                trait_breakdown.append(f"Callable: {callable_count}")

            if trait_breakdown:
                lines.append(f"    └─ {', '.join(trait_breakdown)}")
        else:
            lines.append("  Global Rubric: none")

        # Question-specific rubrics
        questions_with_rubrics = []
        total_llm = total_regex = total_metric = total_callable = 0

        for q in all_questions_data:
            if not isinstance(q, dict):
                continue
            question_rubric = self._rubric_manager.get_question_rubric(q["id"])
            if question_rubric and len(question_rubric) > 0:
                questions_with_rubrics.append(q["id"])
                # Count traits by type
                for trait in question_rubric:
                    if isinstance(trait, LLMRubricTrait):
                        total_llm += 1
                    elif isinstance(trait, RegexTrait):
                        total_regex += 1
                    elif isinstance(trait, MetricRubricTrait):
                        total_metric += 1
                    elif isinstance(trait, CallableTrait):
                        total_callable += 1

        if questions_with_rubrics:
            lines.append(f"  Question-Specific: {len(questions_with_rubrics)} questions with rubrics")
            trait_breakdown = []
            if total_llm > 0:
                trait_breakdown.append(f"LLM: {total_llm} total")
            if total_regex > 0:
                trait_breakdown.append(f"Regex: {total_regex} total")
            if total_metric > 0:
                trait_breakdown.append(f"Metric: {total_metric} total")
            if total_callable > 0:
                trait_breakdown.append(f"Callable: {total_callable} total")

            if trait_breakdown:
                lines.append(f"    └─ {', '.join(trait_breakdown)}")
        else:
            lines.append("  Question-Specific: none")

        # === QUESTIONS ===
        lines.append("")
        if self._base.question_count == 0:
            lines.append("  === QUESTIONS ===")
            lines.append("  (empty benchmark)")
        else:
            display_count = min(3, len(all_questions_data))
            lines.append(f"  === QUESTIONS (showing {display_count} of {self._base.question_count}) ===")

            for idx, q_item in enumerate(all_questions_data[:display_count], 1):
                if not isinstance(q_item, dict):
                    continue

                # Question text (truncate to 80 chars)
                question_text = q_item.get("question", "")
                if len(question_text) > 80:
                    question_text = question_text[:77] + "..."

                # Keywords
                keywords = q_item.get("keywords", [])
                keywords_str = f" [{', '.join(keywords)}]" if keywords else ""

                lines.append(f"  {idx}. {question_text}{keywords_str}")

                # Raw answer (truncate to 80 chars)
                raw_answer = q_item.get("raw_answer", "")
                if raw_answer:
                    if len(raw_answer) > 80:
                        raw_answer = raw_answer[:77] + "..."
                    lines.append(f"     → {raw_answer}")
                else:
                    lines.append("     → (no answer yet)")

                # Add blank line between questions
                if idx < display_count:
                    lines.append("")

            # Show remaining count
            if self._base.question_count > display_count:
                remaining = self._base.question_count - display_count
                lines.append(f"  ... ({remaining} more)")

        # === READINESS ===
        if not self._base.is_complete:
            lines.append("")
            lines.append("  === READINESS ===")
            readiness = self._export_manager.check_readiness()

            if readiness["ready_for_verification"]:
                lines.append("  Status: Ready for verification")
            else:
                lines.append("  Status: Not ready for verification")
                lines.append("  Issues:")

                if readiness["missing_templates_count"] > 0:
                    lines.append(f"    - {readiness['missing_templates_count']} questions missing templates")

                unfinished_count = int(readiness["unfinished_count"])
                if unfinished_count > 0:
                    lines.append(f"    - {unfinished_count} questions not finished")

        lines.append(")")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (same as repr for developer-friendly output)."""
        return self.__repr__()

    def __len__(self) -> int:
        """Return the number of questions in the benchmark."""
        return len(self._base)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over questions in the benchmark."""
        return iter(self._question_manager)

    def __contains__(self, question_id: str) -> bool:
        """Check if a question ID exists in the benchmark."""
        return question_id in self._base

    def __getitem__(self, key: str | int | slice) -> "SchemaOrgQuestion | list[SchemaOrgQuestion]":
        """
        Get question(s) as SchemaOrgQuestion object(s) using bracket notation.

        Supports:
        - String key: Get question by ID (returns SchemaOrgQuestion)
        - Integer key: Get question by index (returns SchemaOrgQuestion)
        - Slice: Get multiple questions by slice (returns list[SchemaOrgQuestion])

        Args:
            key: Question ID (str), index (int), or slice

        Returns:
            Single SchemaOrgQuestion or list of SchemaOrgQuestion objects

        Raises:
            ValueError: If question ID not found
            IndexError: If index out of range
            TypeError: If key type not supported
        """

        # Handle string keys (question ID)
        if isinstance(key, str):
            question_data = self._base[key]
            return self._convert_to_schema_org_question(question_data)

        # Handle integer keys (index)
        elif isinstance(key, int):
            question_ids = self.get_question_ids()
            original_key = key  # Store original for error message
            if key < 0:
                key += len(question_ids)  # Support negative indexing
            if not 0 <= key < len(question_ids):
                raise IndexError(f"Question index {original_key} out of range (0-{len(question_ids) - 1})")
            question_id = question_ids[key]
            question_data = self._base[question_id]
            return self._convert_to_schema_org_question(question_data)

        # Handle slice objects
        elif isinstance(key, slice):
            question_ids = self.get_question_ids()
            selected_ids = question_ids[key]
            return [self._convert_to_schema_org_question(self._base[qid]) for qid in selected_ids]

        else:
            raise TypeError(f"Invalid key type {type(key)}. Expected str, int, or slice.")

    def _convert_to_schema_org_question(self, question_data: dict[str, Any]) -> "SchemaOrgQuestion":
        """
        Convert internal question dictionary to SchemaOrgQuestion object.

        Args:
            question_data: Internal question dictionary

        Returns:
            SchemaOrgQuestion object
        """
        from ..schemas.checkpoint import (
            SchemaOrgAnswer,
            SchemaOrgPropertyValue,
            SchemaOrgQuestion,
            SchemaOrgSoftwareSourceCode,
        )
        from ..utils.checkpoint import convert_rubric_trait_to_rating

        # Create answer object using model_validate to handle aliased fields
        accepted_answer = SchemaOrgAnswer.model_validate(
            {"@id": f"{question_data['id']}-answer", "text": question_data["raw_answer"]}
        )

        # Create software source code (template) object using model_validate
        has_part = SchemaOrgSoftwareSourceCode.model_validate(
            {
                "@id": f"{question_data['id']}-template",
                "name": f"{question_data['question'][:30]}... Answer Template",
                "text": question_data.get("answer_template", ""),
            }
        )

        # Convert question-specific rubric traits to ratings
        ratings = None
        if question_data.get("question_rubric"):
            ratings = [
                convert_rubric_trait_to_rating(trait, "question-specific") for trait in question_data["question_rubric"]
            ]

        # Create additional properties from custom metadata
        additional_properties = []
        if question_data.get("finished") is not None:
            additional_properties.append(SchemaOrgPropertyValue(name="finished", value=question_data["finished"]))

        if question_data.get("custom_metadata"):
            for key, value in question_data["custom_metadata"].items():
                additional_properties.append(SchemaOrgPropertyValue(name=f"custom_{key}", value=value))

        # Create SchemaOrgQuestion object using model_validate
        return SchemaOrgQuestion.model_validate(
            {
                "@id": question_data["id"],
                "text": question_data["question"],
                "acceptedAnswer": accepted_answer,
                "hasPart": has_part,
                "rating": ratings,
                "additionalProperty": additional_properties if additional_properties else None,
            }
        )

    def __eq__(self, other: object) -> bool:
        """Compare two benchmarks for equality."""
        if not isinstance(other, Benchmark):
            return NotImplemented
        return self._base == other._base
