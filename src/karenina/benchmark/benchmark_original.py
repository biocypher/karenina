"""High-level benchmark management for Karenina benchmarks.

This module provides the main Benchmark class for creating, loading,
saving, and executing benchmarks in JSON-LD format.
"""

import json
from collections.abc import Callable, Iterator
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
from .verification.orchestrator import run_question_verification
from .verification.validation import validate_answer_template


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
        # Validate the template using the verification system
        is_valid, error_msg, _ = validate_answer_template(template_code)
        if not is_valid:
            raise ValueError(f"Invalid template: {error_msg}")

        # Find the question in the checkpoint
        found = False
        for item in self._checkpoint.dataFeedElement:
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
        for item in self._checkpoint.dataFeedElement:
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

        # Validate all templates using the verification system
        for q_id, q_data in self._questions_cache.items():
            template_code = q_data.get("answer_template")
            if template_code is not None:
                is_valid, error_msg_or_none, _ = validate_answer_template(template_code)
                error_msg = error_msg_or_none or "Unknown validation error"
                if not is_valid:
                    return False, f"Invalid template for {q_id}: {error_msg}"

        return True, "Benchmark is valid"

    # Integration with existing verification system
    def verify_question(
        self,
        question_id: str,
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify a single question.

        Args:
            question_id: The question ID to verify
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking

        Returns:
            Dictionary mapping result keys to VerificationResult objects

        Raises:
            ValueError: If question not found or not ready for verification
        """
        return self.run_verification(
            config=config,
            question_ids=[question_id],
            run_name=run_name,
            job_id=job_id,
        )

    def verify_questions(
        self,
        question_ids: list[str],
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify multiple specific questions.

        Args:
            question_ids: List of question IDs to verify
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        return self.run_verification(
            config=config,
            question_ids=question_ids,
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_filtered(
        self,
        config: VerificationConfig,
        finished: bool | None = True,
        has_template: bool | None = True,
        has_rubric: bool | None = None,
        author: str | None = None,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify questions matching specific criteria.

        Args:
            config: Verification configuration
            finished: Filter by finished status (True/False/None for all)
            has_template: Filter by template existence (True/False/None for all)
            has_rubric: Filter by rubric existence (True/False/None for all)
            author: Filter by author name (None for all)
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        # Get filtered questions
        filtered_questions = self.filter_questions(
            finished=finished,
            has_template=has_template,
            has_rubric=has_rubric,
            author=author,
        )

        question_ids = [q["id"] for q in filtered_questions]

        return self.run_verification(
            config=config,
            question_ids=question_ids,
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_all_finished(
        self,
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify all finished questions in the benchmark.

        Args:
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        return self.run_verification(
            config=config,
            question_ids=None,  # This defaults to all finished questions
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_custom(
        self,
        question_selector: Callable[[dict[str, Any]], bool],
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify questions selected by a custom function.

        Args:
            question_selector: Function that takes question data and returns bool
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        # Select questions using the custom selector
        selected_questions = []
        for q_data in self._questions_cache.values():
            if question_selector(q_data):
                selected_questions.append(q_data["id"])

        return self.run_verification(
            config=config,
            question_ids=selected_questions,
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_dry_run(
        self,
        config: VerificationConfig,  # noqa: ARG002
        question_ids: list[str] | None = None,
    ) -> dict[str, bool]:
        """
        Perform a dry run verification (validate without executing).

        Args:
            config: Verification configuration to validate
            question_ids: Optional list of question IDs (default: all finished)

        Returns:
            Dictionary mapping question IDs to readiness status (True/False)
        """
        # If no question IDs provided, use all finished questions
        if question_ids is None:
            question_ids = [q_id for q_id, q in self._questions_cache.items() if q.get("finished", False)]

        results = {}

        for q_id in question_ids:
            try:
                # Check if question exists
                if q_id not in self._questions_cache:
                    results[q_id] = False
                    continue

                q_data = self._questions_cache[q_id]
                template_code = q_data.get("answer_template")

                # Check if template exists
                if not template_code:
                    results[q_id] = False
                    continue

                # Validate template
                is_valid, _, _ = validate_answer_template(template_code)
                results[q_id] = is_valid

            except Exception:
                results[q_id] = False

        return results

    def run_verification(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Run verification on the benchmark using existing execution system.

        Args:
            config: Verification configuration
            question_ids: Optional list of question IDs to verify (default: all finished)
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping question IDs to VerificationResult objects
        """
        # If no question IDs provided, verify all finished questions
        if question_ids is None:
            question_ids = [q_id for q_id, q in self._questions_cache.items() if q.get("finished", False)]

        # Validate that all requested questions exist and are ready
        for q_id in question_ids:
            if q_id not in self._questions_cache:
                raise ValueError(f"Question not found: {q_id}")

            q_data = self._questions_cache[q_id]
            template_code = q_data.get("answer_template")

            if not template_code:
                raise ValueError(f"Question {q_id} has no answer template")

            # Validate template
            is_valid, error_msg, _ = validate_answer_template(template_code)
            if not is_valid:
                raise ValueError(f"Invalid template for question {q_id}: {error_msg}")

        results: dict[str, VerificationResult] = {}
        total_questions = len(question_ids)

        for i, q_id in enumerate(question_ids):
            q_data = self._questions_cache[q_id]
            template_code = q_data["answer_template"]

            # Update progress
            if progress_callback:
                progress = (i / total_questions) * 100
                progress_callback(progress, f"Verifying question {q_id}")

            # Merge global and question-specific rubrics
            rubric = self._get_merged_rubric_for_question(q_id)

            try:
                # Run verification for this question using the orchestrator
                question_results = run_question_verification(
                    question_id=q_id,
                    question_text=q_data["question"],
                    template_code=template_code,
                    config=config,
                    rubric=rubric,
                )

                # Store all results from this question (may be multiple due to model combinations)
                results.update(question_results)

            except Exception as e:
                # Create error result for this question
                error_result = VerificationResult(
                    question_id=q_id,
                    success=False,
                    error=f"Verification failed: {str(e)}",
                    question_text=q_data["question"],
                    raw_llm_response="",
                    answering_model="unknown",
                    parsing_model="unknown",
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    run_name=run_name,
                    job_id=job_id,
                )
                results[q_id] = error_result

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Verification complete")

        # Store results in benchmark metadata if we have a run name
        if run_name and results:
            self.store_verification_results(results, run_name)

        return results

    def _get_merged_rubric_for_question(self, question_id: str) -> Rubric | None:
        """
        Get merged rubric for a question (global + question-specific traits).

        Args:
            question_id: The question ID

        Returns:
            Merged rubric with both global and question-specific traits, or None if no rubrics
        """
        # Get global rubric traits
        global_traits = extract_global_rubric_from_benchmark(self._checkpoint) or []

        # Get question-specific rubric traits
        question_traits = []
        if question_id in self._questions_cache:
            q_data = self._questions_cache[question_id]
            question_rubric = q_data.get("question_rubric")
            if question_rubric:
                question_traits = question_rubric

        # Merge traits (question-specific traits override global ones with same name)
        merged_traits = list(global_traits)  # Start with global traits

        # Add question-specific traits, replacing any with the same name
        for q_trait in question_traits:
            # Remove any global trait with the same name
            merged_traits = [t for t in merged_traits if t.name != q_trait.name]
            # Add the question-specific trait
            merged_traits.append(q_trait)

        # Return merged rubric if we have any traits
        if merged_traits:
            return Rubric(traits=merged_traits)

        return None

    # Verification result storage and querying
    def store_verification_results(
        self,
        results: dict[str, VerificationResult],
        run_name: str | None = None,
    ) -> None:
        """
        Store verification results in the benchmark metadata.

        Args:
            results: Dictionary of verification results to store
            run_name: Optional run name for organizing results
        """
        # Store results in benchmark custom properties
        results_data = {}
        for result_key, result in results.items():
            results_data[result_key] = result.model_dump()

        # Create a timestamp-based key if no run name provided
        if run_name is None:
            run_name = f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store in benchmark custom properties
        verification_results_key = f"verification_results_{run_name}"
        self.set_custom_property(verification_results_key, results_data)

    def get_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Get verification results for specific questions and/or runs.

        Args:
            question_ids: Optional list of question IDs to filter by
            run_name: Optional run name to filter by

        Returns:
            Dictionary of verification results
        """
        all_results = {}

        # Get all verification result properties
        all_props = self.get_all_custom_properties()

        for prop_name, prop_value in all_props.items():
            if prop_name.startswith("verification_results_"):
                # Check if this matches the run name filter
                if run_name is not None:
                    expected_key = f"verification_results_{run_name}"
                    if prop_name != expected_key:
                        continue

                # Parse the stored results
                try:
                    stored_results = prop_value
                    if isinstance(stored_results, dict):
                        for result_key, result_data in stored_results.items():
                            # Reconstruct VerificationResult object
                            if isinstance(result_data, dict):
                                verification_result = VerificationResult(**result_data)

                                # Check if this matches question ID filter
                                if question_ids is not None and verification_result.question_id not in question_ids:
                                    continue

                                all_results[result_key] = verification_result
                except Exception:
                    # Skip malformed result data
                    continue

        return all_results

    def get_verification_history(self, question_id: str | None = None) -> dict[str, dict[str, VerificationResult]]:
        """
        Get verification history organized by run name.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping run names to their results
        """
        history = {}

        # Get all verification result properties
        all_props = self.get_all_custom_properties()

        for prop_name, prop_value in all_props.items():
            if prop_name.startswith("verification_results_"):
                run_name = prop_name[len("verification_results_") :]

                try:
                    stored_results = prop_value
                    if isinstance(stored_results, dict):
                        run_results = {}
                        for result_key, result_data in stored_results.items():
                            if isinstance(result_data, dict):
                                verification_result = VerificationResult(**result_data)

                                # Check if this matches question ID filter
                                if question_id is not None and verification_result.question_id != question_id:
                                    continue

                                run_results[result_key] = verification_result

                        if run_results:  # Only add if we have results
                            history[run_name] = run_results
                except Exception:
                    # Skip malformed result data
                    continue

        return history

    def clear_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """
        Clear verification results.

        Args:
            question_ids: Optional list of question IDs to clear (None = all)
            run_name: Optional run name to clear (None = all runs)

        Returns:
            Number of result entries that were cleared
        """
        cleared_count = 0

        # Get all verification result properties
        all_props = self.get_all_custom_properties()
        props_to_remove = []
        props_to_update = {}

        for prop_name, prop_value in all_props.items():
            if prop_name.startswith("verification_results_"):
                # Check if this matches the run name filter
                if run_name is not None:
                    expected_key = f"verification_results_{run_name}"
                    if prop_name != expected_key:
                        continue

                try:
                    stored_results = prop_value
                    if isinstance(stored_results, dict):
                        updated_results = {}

                        for result_key, result_data in stored_results.items():
                            if isinstance(result_data, dict):
                                verification_result = VerificationResult(**result_data)

                                # Check if this should be cleared
                                should_clear = False
                                if question_ids is None:
                                    should_clear = True  # Clear all
                                elif verification_result.question_id in question_ids:
                                    should_clear = True

                                if should_clear:
                                    cleared_count += 1
                                else:
                                    updated_results[result_key] = result_data

                        # Update or remove the property
                        if not updated_results:
                            props_to_remove.append(prop_name)
                        else:
                            props_to_update[prop_name] = updated_results
                except Exception:
                    # Skip malformed result data
                    continue

        # Apply the changes
        for prop_name in props_to_remove:
            self.remove_custom_property(prop_name)

        for prop_name, updated_value in props_to_update.items():
            self.set_custom_property(prop_name, updated_value)

        return cleared_count

    def export_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str = "json",
    ) -> str:
        """
        Export verification results in specified format.

        Args:
            question_ids: Optional list of question IDs to export
            run_name: Optional run name to export
            format: Export format ("json" or "csv")

        Returns:
            Exported data as string

        Raises:
            ValueError: If format is not supported
        """
        results = self.get_verification_results(question_ids, run_name)

        if format.lower() == "json":
            # Convert to JSON
            export_data = {}
            for result_key, result in results.items():
                export_data[result_key] = result.model_dump()
            return json.dumps(export_data, indent=2, ensure_ascii=False)

        elif format.lower() == "csv":
            # Convert to CSV
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            # Header
            if results:
                # Get field names from first result
                sample_result = next(iter(results.values()))
                fieldnames = list(sample_result.model_dump().keys())
                writer.writerow(["result_key"] + fieldnames)

                # Data rows
                for result_key, result in results.items():
                    row_data = [result_key]
                    result_dict = result.model_dump()
                    for field in fieldnames:
                        value = result_dict.get(field, "")
                        # Convert complex objects to string
                        if isinstance(value, dict | list):
                            value = json.dumps(value)
                        row_data.append(str(value))
                    writer.writerow(row_data)

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {format}. Supported formats: json, csv")

    # Advanced verification methods
    def verify_with_mixed_configs(
        self,
        question_configs: dict[str, VerificationConfig],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, VerificationResult]]:
        """
        Verify different questions with different configurations.

        Args:
            question_configs: Dictionary mapping question IDs to their configurations
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping question IDs to their verification results
        """
        all_results = {}
        total_questions = len(question_configs)

        for i, (question_id, config) in enumerate(question_configs.items()):
            # Update progress
            if progress_callback:
                progress = (i / total_questions) * 100
                progress_callback(progress, f"Verifying question {question_id} with custom config")

            try:
                question_results = self.verify_question(
                    question_id=question_id,
                    config=config,
                    run_name=f"mixed_config_{question_id}",
                )
                all_results[question_id] = question_results
            except Exception as e:
                # Create error result
                error_result = VerificationResult(
                    question_id=question_id,
                    success=False,
                    error=f"Mixed config verification failed: {str(e)}",
                    question_text=self._questions_cache.get(question_id, {}).get("question", ""),
                    raw_llm_response="",
                    answering_model="unknown",
                    parsing_model="unknown",
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                )
                all_results[question_id] = {f"{question_id}_error": error_result}

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Mixed config verification complete")

        return all_results

    def verify_comparative(
        self,
        question_ids: list[str],
        configs: list[VerificationConfig],
        run_names: list[str],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, VerificationResult]]:
        """
        Run same questions with multiple configurations for comparison.

        Args:
            question_ids: List of question IDs to verify
            configs: List of configurations to test
            run_names: List of run names for each configuration
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping run names to their results
        """
        if len(configs) != len(run_names):
            raise ValueError("Number of configs must match number of run names")

        all_results = {}
        total_runs = len(configs)

        for i, (config, run_name) in enumerate(zip(configs, run_names, strict=False)):
            # Update progress
            if progress_callback:
                progress = (i / total_runs) * 100
                progress_callback(progress, f"Running comparative verification: {run_name}")

            try:
                run_results = self.verify_questions(
                    question_ids=question_ids,
                    config=config,
                    run_name=run_name,
                )
                all_results[run_name] = run_results
            except Exception as e:
                # Create error results for this run
                error_results = {}
                for q_id in question_ids:
                    error_result = VerificationResult(
                        question_id=q_id,
                        success=False,
                        error=f"Comparative verification failed: {str(e)}",
                        question_text=self._questions_cache.get(q_id, {}).get("question", ""),
                        raw_llm_response="",
                        answering_model="unknown",
                        parsing_model="unknown",
                        execution_time=0.0,
                        timestamp=datetime.now().isoformat(),
                        run_name=run_name,
                    )
                    error_results[f"{q_id}_error"] = error_result
                all_results[run_name] = error_results

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Comparative verification complete")

        return all_results

    def verify_progressive(
        self,
        config: VerificationConfig,
        batch_size: int = 5,
        run_name: str | None = None,
        resume_from: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify questions in batches with ability to resume from interruptions.

        Args:
            config: Verification configuration
            batch_size: Number of questions to verify per batch
            run_name: Optional run name for tracking
            resume_from: Optional question ID to resume from
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        # Get all finished questions
        finished_questions = [q_id for q_id, q in self._questions_cache.items() if q.get("finished", False)]

        # If resuming, find the starting point
        start_index = 0
        if resume_from and resume_from in finished_questions:
            start_index = finished_questions.index(resume_from)

        # Process in batches
        all_results = {}
        remaining_questions = finished_questions[start_index:]
        total_batches = (len(remaining_questions) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_questions))
            batch_questions = remaining_questions[start_idx:end_idx]

            # Update progress
            if progress_callback:
                batch_progress = (batch_num / total_batches) * 100
                progress_callback(batch_progress, f"Processing batch {batch_num + 1}/{total_batches}")

            try:
                batch_results = self.verify_questions(
                    question_ids=batch_questions,
                    config=config,
                    run_name=f"{run_name}_batch_{batch_num + 1}" if run_name else None,
                )
                all_results.update(batch_results)

                # Save intermediate results after each batch
                if run_name:
                    self.store_verification_results(batch_results, f"{run_name}_partial")

            except Exception as e:
                # Log the error but continue with remaining batches
                if progress_callback:
                    progress_callback(
                        (batch_num / total_batches) * 100, f"Batch {batch_num + 1} failed: {str(e)[:50]}..."
                    )
                continue

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Progressive verification complete")

        return all_results

    def get_verification_summary(self, run_name: str | None = None) -> dict[str, Any]:
        """
        Get summary statistics for verification results.

        Args:
            run_name: Optional run name to filter by

        Returns:
            Dictionary with verification statistics
        """
        results = self.get_verification_results(run_name=run_name)

        if not results:
            return {
                "total_results": 0,
                "successful_count": 0,
                "failed_count": 0,
                "success_rate": 0.0,
                "unique_questions": 0,
                "average_execution_time": 0.0,
                "model_combinations": 0,
            }

        successful_count = sum(1 for r in results.values() if r.success)
        failed_count = len(results) - successful_count
        unique_questions = len({r.question_id for r in results.values()})
        total_execution_time = sum(r.execution_time for r in results.values() if r.execution_time)
        average_execution_time = total_execution_time / len(results) if results else 0.0

        # Count unique model combinations
        model_combinations = len({f"{r.answering_model}:{r.parsing_model}" for r in results.values()})

        return {
            "total_results": len(results),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "success_rate": (successful_count / len(results)) * 100 if results else 0.0,
            "unique_questions": unique_questions,
            "average_execution_time": round(average_execution_time, 2),
            "total_execution_time": round(total_execution_time, 2),
            "model_combinations": model_combinations,
        }

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

    def _update_question_property(self, question_id: str, property_name: str, value: Any) -> None:
        """Update a property in the underlying JSON-LD structure."""
        # Find the DataFeedItem for this question
        for item in self._checkpoint.dataFeedElement:
            if self._get_item_id(item) == question_id:
                # Update dateModified
                item.dateModified = datetime.now().isoformat()

                # Find or create the property in additionalProperty
                if not hasattr(item.item, "additionalProperty") or item.item.additionalProperty is None:
                    from ..schemas.checkpoint import SchemaOrgPropertyValue

                    item.item.additionalProperty = []

                # Find existing property or create new one
                prop_found = False
                for prop in item.item.additionalProperty:
                    if prop.name == property_name:
                        prop.value = value
                        prop_found = True
                        break

                if not prop_found:
                    from ..schemas.checkpoint import SchemaOrgPropertyValue

                    new_prop = SchemaOrgPropertyValue(name=property_name, value=value)
                    item.item.additionalProperty.append(new_prop)

                # Update main benchmark dateModified
                self._checkpoint.dateModified = datetime.now().isoformat()
                break

    @property
    def jsonld_data(self) -> JsonLdCheckpoint:
        """Get the raw JSON-LD benchmark data."""
        return self._checkpoint

    # Property accessors for common attributes
    @property
    def name(self) -> str:
        """Get the benchmark name."""
        return self._checkpoint.name

    @name.setter
    def name(self, value: str) -> None:
        """Set the benchmark name."""
        self._checkpoint.name = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def description(self) -> str:
        """Get the benchmark description."""
        return self._checkpoint.description or ""

    @description.setter
    def description(self, value: str) -> None:
        """Set the benchmark description."""
        self._checkpoint.description = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def version(self) -> str:
        """Get the benchmark version."""
        return self._checkpoint.version or "0.1.0"

    @version.setter
    def version(self, value: str) -> None:
        """Set the benchmark version."""
        self._checkpoint.version = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def creator(self) -> str:
        """Get the benchmark creator."""
        return self._checkpoint.creator or "Unknown"

    @creator.setter
    def creator(self, value: str) -> None:
        """Set the benchmark creator."""
        self._checkpoint.creator = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def id(self) -> str | None:
        """Get the benchmark ID."""
        return self._checkpoint.id

    @id.setter
    def id(self, value: str | None) -> None:
        """Set the benchmark ID."""
        self._checkpoint.id = value
        self._checkpoint.dateModified = datetime.now().isoformat()

    @property
    def created_at(self) -> str:
        """Get the creation timestamp."""
        return self._checkpoint.dateCreated

    @created_at.setter
    def created_at(self, value: str) -> None:
        """Set the creation timestamp."""
        self._checkpoint.dateCreated = value

    @property
    def modified_at(self) -> str:
        """Get the last modification timestamp."""
        return self._checkpoint.dateModified

    @modified_at.setter
    def modified_at(self, value: str) -> None:
        """Set the last modification timestamp."""
        self._checkpoint.dateModified = value

    @property
    def question_count(self) -> int:
        """Get the total number of questions."""
        return len(self._questions_cache)

    @property
    def finished_count(self) -> int:
        """Get the number of finished questions."""
        return sum(1 for q in self._questions_cache.values() if q.get("finished", False))

    @property
    def is_empty(self) -> bool:
        """Check if the benchmark has no questions."""
        return len(self._questions_cache) == 0

    @property
    def is_complete(self) -> bool:
        """Check if all questions have templates and are finished."""
        if self.is_empty:
            return False
        return all(q.get("answer_template") and q.get("finished", False) for q in self._questions_cache.values())

    # Additional metadata methods
    def get_custom_property(self, name: str) -> Any:
        """
        Get a custom property from benchmark metadata.

        Args:
            name: Property name

        Returns:
            Property value or None if not found
        """
        if not self._checkpoint.additionalProperty:
            return None

        for prop in self._checkpoint.additionalProperty:
            if prop.name == name:
                return prop.value
        return None

    def set_custom_property(self, name: str, value: Any) -> None:
        """
        Set a custom property in benchmark metadata.

        Args:
            name: Property name
            value: Property value
        """
        from ..schemas.checkpoint import SchemaOrgPropertyValue

        if not self._checkpoint.additionalProperty:
            self._checkpoint.additionalProperty = []

        # Update existing property or create new one
        for prop in self._checkpoint.additionalProperty:
            if prop.name == name:
                prop.value = value
                self._checkpoint.dateModified = datetime.now().isoformat()
                return

        # Create new property
        new_prop = SchemaOrgPropertyValue(name=name, value=value)
        self._checkpoint.additionalProperty.append(new_prop)
        self._checkpoint.dateModified = datetime.now().isoformat()

    def remove_custom_property(self, name: str) -> bool:
        """
        Remove a custom property from benchmark metadata.

        Args:
            name: Property name

        Returns:
            True if property was found and removed, False otherwise
        """
        if not self._checkpoint.additionalProperty:
            return False

        for i, prop in enumerate(self._checkpoint.additionalProperty):
            if prop.name == name:
                del self._checkpoint.additionalProperty[i]
                self._checkpoint.dateModified = datetime.now().isoformat()
                return True
        return False

    def get_all_custom_properties(self) -> dict[str, Any]:
        """
        Get all custom properties as a dictionary.

        Returns:
            Dictionary of property name -> value pairs
        """
        if not self._checkpoint.additionalProperty:
            return {}

        return {prop.name: prop.value for prop in self._checkpoint.additionalProperty}

    def set_multiple_custom_properties(self, properties: dict[str, Any]) -> None:
        """
        Set multiple custom properties at once.

        Args:
            properties: Dictionary of property name -> value pairs
        """
        for name, value in properties.items():
            self.set_custom_property(name, value)

    # Question metadata management
    def get_question_metadata(self, question_id: str) -> dict[str, Any]:
        """
        Get all metadata for a specific question.

        Args:
            question_id: The question ID

        Returns:
            Dictionary containing all question metadata

        Raises:
            ValueError: If question not found
        """
        question_data = self.get_question(question_id)
        return {
            "id": question_data["id"],
            "question": question_data["question"],
            "raw_answer": question_data["raw_answer"],
            "date_created": question_data["date_created"],
            "date_modified": question_data["date_modified"],
            "finished": question_data.get("finished", False),
            "author": question_data.get("author"),
            "sources": question_data.get("sources"),
            "custom_metadata": question_data.get("custom_metadata", {}),
            "has_template": self.has_template(question_id),
            "has_rubric": bool(question_data.get("question_rubric")),
        }

    def update_question_metadata(self, question_id: str, **metadata: Any) -> None:
        """
        Update question metadata fields.

        Args:
            question_id: The question ID
            **metadata: Metadata fields to update (question, raw_answer, author, sources, etc.)

        Raises:
            ValueError: If question not found
        """
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        question_data = self._questions_cache[question_id]

        # Update basic fields
        if "question" in metadata:
            question_data["question"] = metadata["question"]
            # Update underlying JSON-LD structure
            for item in self._checkpoint.dataFeedElement:
                if self._get_item_id(item) == question_id:
                    item.item.text = metadata["question"]
                    break

        if "raw_answer" in metadata:
            question_data["raw_answer"] = metadata["raw_answer"]
            # Update underlying JSON-LD structure
            for item in self._checkpoint.dataFeedElement:
                if self._get_item_id(item) == question_id:
                    item.item.acceptedAnswer.text = metadata["raw_answer"]
                    break

        # Update additional properties
        for key in ["author", "sources"]:
            if key in metadata:
                question_data[key] = metadata[key]
                self._update_question_property(question_id, key, metadata[key])

        # Update custom metadata
        if "custom_metadata" in metadata:
            custom_meta = metadata["custom_metadata"] or {}
            question_data["custom_metadata"] = custom_meta
            # Update each custom field as a separate property
            for custom_key, custom_value in custom_meta.items():
                self._update_question_property(question_id, f"custom_{custom_key}", custom_value)

        # Update modification timestamp
        question_data["date_modified"] = datetime.now().isoformat()
        for item in self._checkpoint.dataFeedElement:
            if self._get_item_id(item) == question_id:
                item.dateModified = question_data["date_modified"]
                break

    def get_question_custom_property(self, question_id: str, name: str) -> Any:
        """
        Get a custom property from question metadata.

        Args:
            question_id: The question ID
            name: Property name

        Returns:
            Property value or None if not found

        Raises:
            ValueError: If question not found
        """
        question_data = self.get_question(question_id)
        custom_metadata = question_data.get("custom_metadata", {})
        return custom_metadata.get(name)

    def set_question_custom_property(self, question_id: str, name: str, value: Any) -> None:
        """
        Set a custom property on question metadata.

        Args:
            question_id: The question ID
            name: Property name
            value: Property value

        Raises:
            ValueError: If question not found
        """
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        question_data = self._questions_cache[question_id]
        if not question_data.get("custom_metadata"):
            question_data["custom_metadata"] = {}
        question_data["custom_metadata"][name] = value

        # Update underlying JSON-LD structure
        self._update_question_property(question_id, f"custom_{name}", value)

    def remove_question_custom_property(self, question_id: str, name: str) -> bool:
        """
        Remove a custom property from question metadata.

        Args:
            question_id: The question ID
            name: Property name

        Returns:
            True if property was found and removed, False otherwise

        Raises:
            ValueError: If question not found
        """
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        # Update cache
        question_data = self._questions_cache[question_id]
        custom_metadata = question_data.get("custom_metadata", {})

        if name not in custom_metadata:
            return False

        del custom_metadata[name]
        question_data["custom_metadata"] = custom_metadata if custom_metadata else None

        # Remove from underlying JSON-LD structure
        for item in self._checkpoint.dataFeedElement:
            if self._get_item_id(item) == question_id:
                if item.item.additionalProperty:
                    for i, prop in enumerate(item.item.additionalProperty):
                        if prop.name == f"custom_{name}":
                            del item.item.additionalProperty[i]
                            item.dateModified = datetime.now().isoformat()
                            return True
                break

        return False

    def get_question_author(self, question_id: str) -> dict[str, Any] | None:
        """Get author information for a question."""
        question_data = self.get_question(question_id)
        return question_data.get("author")

    def set_question_author(self, question_id: str, author: dict[str, Any] | None) -> None:
        """Set author information for a question."""
        self.update_question_metadata(question_id, author=author)

    def get_question_sources(self, question_id: str) -> list[dict[str, Any]] | None:
        """Get source documents for a question."""
        question_data = self.get_question(question_id)
        return question_data.get("sources")

    def set_question_sources(self, question_id: str, sources: list[dict[str, Any]] | None) -> None:
        """Set source documents for a question."""
        self.update_question_metadata(question_id, sources=sources)

    def get_question_timestamps(self, question_id: str) -> dict[str, str]:
        """
        Get creation and modification timestamps for a question.

        Args:
            question_id: The question ID

        Returns:
            Dictionary with 'created' and 'modified' timestamp strings

        Raises:
            ValueError: If question not found
        """
        question_data = self.get_question(question_id)
        return {"created": question_data["date_created"], "modified": question_data["date_modified"]}

    # Magic methods for better usability
    def __str__(self) -> str:
        """Human-readable string representation."""
        progress = self.get_progress()
        return f"Benchmark '{self.name}' ({self.question_count} questions, {progress:.1f}% complete)"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Benchmark(name='{self.name}', "
            f"version='{self.version}', "
            f"questions={self.question_count}, "
            f"finished={self.finished_count})"
        )

    def __len__(self) -> int:
        """Return the number of questions in the benchmark."""
        return len(self._questions_cache)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over questions in the benchmark."""
        return iter(self._questions_cache.values())

    def __contains__(self, question_id: str) -> bool:
        """Check if a question ID exists in the benchmark."""
        return question_id in self._questions_cache

    def __getitem__(self, question_id: str) -> dict[str, Any]:
        """Get a question by ID using bracket notation."""
        return self.get_question(question_id)

    # Statistics and summary methods
    def get_progress(self) -> float:
        """Get completion progress as percentage (0-100)."""
        if self.question_count == 0:
            return 0.0
        return (self.finished_count / self.question_count) * 100

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive benchmark statistics."""
        has_template_count = sum(1 for q_id, q in self._questions_cache.items() if self.has_template(q_id))
        has_rubric_count = sum(1 for q in self._questions_cache.values() if q.get("question_rubric"))

        global_rubric = self.get_global_rubric()

        return {
            "name": self.name,
            "version": self.version,
            "creator": self.creator,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "question_count": float(self.question_count),
            "finished_count": self.finished_count,
            "has_template_count": has_template_count,
            "has_rubric_count": has_rubric_count,
            "progress_percentage": self.get_progress(),
            "is_complete": self.is_complete,
            "has_global_rubric": global_rubric is not None,
            "global_rubric_traits": len(global_rubric.traits) if global_rubric else 0,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about the benchmark."""
        templates = [
            q.get("answer_template", "") for q_id, q in self._questions_cache.items() if self.has_template(q_id)
        ]

        avg_template_length = 0
        if templates:
            avg_template_length = int(sum(len(t) for t in templates) / len(templates))

        return {
            **self.get_summary(),
            "avg_template_length": round(avg_template_length, 1),
            "min_template_length": min(len(t) for t in templates) if templates else 0,
            "max_template_length": max(len(t) for t in templates) if templates else 0,
            "questions_with_custom_metadata": sum(
                1
                for q in self._questions_cache.values()
                if q.get("custom_metadata") or q.get("author") or q.get("sources")
            ),
        }

    def get_missing_templates(self) -> list[str]:
        """Get list of question IDs that don't have non-default templates."""
        return [q_id for q_id in self._questions_cache if not self.has_template(q_id)]

    def get_unfinished_questions(self) -> list[str]:
        """Get list of question IDs that are not marked as finished."""
        return [q_id for q_id, q_data in self._questions_cache.items() if not q_data.get("finished", False)]

    def filter_questions(
        self,
        finished: bool | None = None,
        has_template: bool | None = None,
        has_rubric: bool | None = None,
        author: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filter questions based on criteria.

        Args:
            finished: Filter by finished status (True/False/None for all)
            has_template: Filter by template existence (True/False/None for all)
            has_rubric: Filter by rubric existence (True/False/None for all)
            author: Filter by author name (None for all)

        Returns:
            List of question dictionaries matching criteria
        """
        results = []

        for q_id, q_data in self._questions_cache.items():
            # Check finished status
            if finished is not None and q_data.get("finished", False) != finished:
                continue

            # Check template existence (non-default templates only)
            if has_template is not None:
                has_tmpl = self.has_template(q_id)
                if has_tmpl != has_template:
                    continue

            # Check rubric existence
            if has_rubric is not None:
                has_rub = bool(q_data.get("question_rubric"))
                if has_rub != has_rubric:
                    continue

            # Check author
            if author is not None:
                q_author = q_data.get("author", {}).get("name", "") if q_data.get("author") else ""
                if author.lower() not in q_author.lower():
                    continue

            results.append(q_data)

        return results

    def search_questions(self, query: str) -> list[dict[str, Any]]:
        """
        Search for questions containing the query text.

        Args:
            query: Text to search for in question text

        Returns:
            List of question dictionaries containing the query
        """
        query_lower = query.lower()
        return [q_data for q_data in self._questions_cache.values() if query_lower in q_data["question"].lower()]

    def get_questions_by_author(self, author: str) -> list[dict[str, Any]]:
        """Get questions created by a specific author."""
        return self.filter_questions(author=author)

    def get_questions_with_rubric(self) -> list[dict[str, Any]]:
        """Get questions that have question-specific rubrics."""
        return self.filter_questions(has_rubric=True)

    # Bulk operations
    def add_questions_batch(self, questions_data: list[dict[str, Any]]) -> list[str]:
        """
        Add multiple questions at once.

        Args:
            questions_data: List of dictionaries with question data.
                           Each dict should have 'question', 'raw_answer' keys
                           and optionally 'answer_template', 'finished', etc.

        Returns:
            List of question IDs that were created
        """
        question_ids = []
        for data in questions_data:
            q_id = self.add_question(
                question=data["question"],
                raw_answer=data["raw_answer"],
                answer_template=data.get("answer_template"),
                question_id=data.get("question_id"),
                finished=data.get("finished", False),
                author=data.get("author"),
                sources=data.get("sources"),
                custom_metadata=data.get("custom_metadata"),
            )
            question_ids.append(q_id)
        return question_ids

    def mark_finished_batch(self, question_ids: list[str]) -> None:
        """Mark multiple questions as finished."""
        for q_id in question_ids:
            if q_id in self._questions_cache:
                # Update cache
                self._questions_cache[q_id]["finished"] = True
                # Update underlying JSON-LD structure
                self._update_question_property(q_id, "finished", True)

    def mark_unfinished_batch(self, question_ids: list[str]) -> None:
        """Mark multiple questions as unfinished."""
        for q_id in question_ids:
            if q_id in self._questions_cache:
                # Update cache
                self._questions_cache[q_id]["finished"] = False
                # Update underlying JSON-LD structure
                self._update_question_property(q_id, "finished", False)

    def apply_global_template(self, template_code: str) -> list[str]:
        """
        Apply a template to all questions that don't have one.

        Args:
            template_code: The template code to apply

        Returns:
            List of question IDs that received the template
        """
        updated_ids = []
        for q_id in self._questions_cache:
            if not self.has_template(q_id):
                self.add_answer_template(q_id, template_code)
                updated_ids.append(q_id)
        return updated_ids

    # Clear and remove methods
    def remove_question(self, question_id: str) -> bool:
        """
        Remove a specific question from the benchmark.

        Args:
            question_id: The question ID to remove

        Returns:
            True if question was removed, False if not found
        """
        # Remove from cache
        if question_id not in self._questions_cache:
            return False

        del self._questions_cache[question_id]

        # Remove from checkpoint data
        items_to_remove = []
        for i, item in enumerate(self._checkpoint.dataFeedElement):
            if self._get_item_id(item) == question_id:
                items_to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(items_to_remove):
            del self._checkpoint.dataFeedElement[i]

        self._checkpoint.dateModified = datetime.now().isoformat()
        return True

    def clear_questions(self) -> int:
        """
        Remove all questions from the benchmark.

        Returns:
            Number of questions that were removed
        """
        count = len(self._questions_cache)
        self._questions_cache.clear()
        self._checkpoint.dataFeedElement.clear()
        self._checkpoint.dateModified = datetime.now().isoformat()
        return count

    def clear_global_rubric(self) -> bool:
        """
        Remove the global rubric.

        Returns:
            True if global rubric was removed, False if none existed
        """
        if self._checkpoint.rating:
            self._checkpoint.rating = None
            self._checkpoint.dateModified = datetime.now().isoformat()
            return True
        return False

    def remove_question_rubric(self, question_id: str) -> bool:
        """
        Remove question-specific rubric.

        Args:
            question_id: The question ID

        Returns:
            True if rubric was removed, False if not found
        """
        for item in self._checkpoint.dataFeedElement:
            if self._get_item_id(item) == question_id and item.item.rating:
                item.item.rating = None
                item.dateModified = datetime.now().isoformat()
                self._checkpoint.dateModified = datetime.now().isoformat()
                self._rebuild_cache()
                return True
        return False

    def clear_all_rubrics(self) -> int:
        """
        Remove all rubrics (global and question-specific).

        Returns:
            Number of rubrics that were removed
        """
        count = 0

        # Clear global rubric
        if self.clear_global_rubric():
            count += 1

        # Clear question-specific rubrics
        for item in self._checkpoint.dataFeedElement:
            if item.item.rating:
                item.item.rating = None
                item.dateModified = datetime.now().isoformat()
                count += 1

        if count > 0:
            self._checkpoint.dateModified = datetime.now().isoformat()
            self._rebuild_cache()

        return count

    # Enhanced template management
    def _is_default_template(self, template: str, question: str) -> bool:
        """Check if a template is the auto-generated default."""
        if not template:
            return False
        # Check if it matches the default template pattern
        expected_default = self._create_default_template(question)
        return template.strip() == expected_default.strip()

    def has_template(self, question_id: str) -> bool:
        """Check if a question has a non-default template."""
        if question_id not in self._questions_cache:
            return False

        template = self._questions_cache[question_id].get("answer_template")
        if not template:
            return False

        # Check if it's just the default template
        question_text = self._questions_cache[question_id].get("question", "")
        return not self._is_default_template(template, question_text)

    def get_template(self, question_id: str) -> str:
        """
        Get template code for a question.

        Args:
            question_id: The question ID

        Returns:
            Template code string

        Raises:
            ValueError: If question not found or has no template
        """
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        template = self._questions_cache[question_id].get("answer_template")
        if not template:
            raise ValueError(f"Question {question_id} has no template")

        # Check if it's just the default template
        question_text = self._questions_cache[question_id].get("question", "")
        if self._is_default_template(template, question_text):
            raise ValueError(f"Question {question_id} has no template")

        return str(template)

    def update_template(self, question_id: str, template_code: str) -> None:
        """Update existing template (alias for add_answer_template)."""
        self.add_answer_template(question_id, template_code)

    def copy_template(self, from_id: str, to_id: str) -> None:
        """
        Copy template from one question to another.

        Args:
            from_id: Source question ID
            to_id: Destination question ID

        Raises:
            ValueError: If source question not found or has no template
        """
        template = self.get_template(from_id)
        self.add_answer_template(to_id, template)

    # Status management
    def mark_finished(self, question_id: str) -> None:
        """Mark a question as finished."""
        if question_id in self._questions_cache:
            # Update cache
            self._questions_cache[question_id]["finished"] = True
            # Update underlying JSON-LD structure
            self._update_question_property(question_id, "finished", True)

    def mark_unfinished(self, question_id: str) -> None:
        """Mark a question as unfinished."""
        if question_id in self._questions_cache:
            # Update cache
            self._questions_cache[question_id]["finished"] = False
            # Update underlying JSON-LD structure
            self._update_question_property(question_id, "finished", False)

    def toggle_finished(self, question_id: str) -> bool:
        """
        Toggle finished status of a question.

        Args:
            question_id: The question ID

        Returns:
            New finished status (True/False)

        Raises:
            ValueError: If question not found
        """
        if question_id not in self._questions_cache:
            raise ValueError(f"Question not found: {question_id}")

        current_status = self._questions_cache[question_id].get("finished", False)
        new_status = not current_status
        # Update cache
        self._questions_cache[question_id]["finished"] = new_status
        # Update underlying JSON-LD structure
        self._update_question_property(question_id, "finished", new_status)
        return new_status

    # Export methods
    def to_dict(self) -> dict[str, Any]:
        """Export benchmark as a plain dictionary."""
        return {
            "metadata": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "creator": self.creator,
                "created_at": self.created_at,
                "modified_at": self.modified_at,
            },
            "statistics": self.get_summary(),
            "questions": list(self._questions_cache.values()),
            "global_rubric": (global_rubric.model_dump() if (global_rubric := self.get_global_rubric()) else None),
        }

    def to_markdown(self) -> str:
        """Export benchmark as markdown document."""
        lines = []

        # Header
        lines.append(f"# {self.name}")
        lines.append("")
        if self.description:
            lines.append(self.description)
            lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append(f"- **Version**: {self.version}")
        lines.append(f"- **Creator**: {self.creator}")
        lines.append(f"- **Created**: {self.created_at}")
        lines.append(f"- **Modified**: {self.modified_at}")
        lines.append("")

        # Statistics
        stats = self.get_summary()
        lines.append("## Statistics")
        lines.append(f"- **Questions**: {stats['question_count']}")
        lines.append(f"- **Finished**: {stats['finished_count']}")
        lines.append(f"- **Progress**: {stats['progress_percentage']:.1f}%")
        lines.append(f"- **Has Templates**: {stats['has_template_count']}")
        lines.append("")

        # Global rubric
        global_rubric = self.get_global_rubric()
        if global_rubric:
            lines.append("## Global Rubric")
            for trait in global_rubric.traits:
                lines.append(f"- **{trait.name}**: {trait.description}")
            lines.append("")

        # Questions
        lines.append("## Questions")
        for i, q_data in enumerate(self._questions_cache.values(), 1):
            status = "✅" if q_data.get("finished", False) else "❌"
            template_status = "📝" if q_data.get("answer_template") else "❌"

            lines.append(f"### {i}. {q_data['question']}")
            lines.append(f"**Status**: {status} | **Template**: {template_status}")
            lines.append("")

            if q_data.get("raw_answer"):
                lines.append(f"**Expected Answer**: {q_data['raw_answer']}")
                lines.append("")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export questions as CSV format."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            ["Question ID", "Question", "Raw Answer", "Has Template", "Finished", "Author", "Created", "Modified"]
        )

        # Data rows
        for q_id, q_data in self._questions_cache.items():
            author = ""
            if q_data.get("author"):
                author = q_data["author"].get("name", "")

            writer.writerow(
                [
                    q_id,
                    q_data["question"],
                    q_data.get("raw_answer", ""),
                    "Yes" if q_data.get("answer_template") else "No",
                    "Yes" if q_data.get("finished", False) else "No",
                    author,
                    q_data.get("date_created", ""),
                    q_data.get("date_modified", ""),
                ]
            )

        return output.getvalue()

    # Comparison and utility methods
    def clone(self) -> "Benchmark":
        """Create a deep copy of the benchmark."""
        # Create a temporary file to save/load from
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            temp_path = Path(f.name)

        try:
            self.save(temp_path)
            cloned = Benchmark.load(temp_path)
            return cloned
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def __eq__(self, other: object) -> bool:
        """Compare two benchmarks for equality."""
        if not isinstance(other, Benchmark):
            return NotImplemented

        # Compare basic metadata
        if self.name != other.name or self.version != other.version or self.question_count != other.question_count:
            return False

        # Compare questions
        return self._questions_cache == other._questions_cache

    # Enhanced validation and health checks
    def validate_templates(self) -> tuple[bool, list[dict[str, str]]]:
        """
        Validate all templates are valid Python code.

        Returns:
            Tuple of (all_valid, list_of_errors)
            Each error dict has 'question_id', 'error' keys
        """
        import ast

        errors = []

        for q_id, q_data in self._questions_cache.items():
            template = q_data.get("answer_template")
            if not template:
                continue

            try:
                # Try to parse as valid Python syntax
                ast.parse(template)
            except SyntaxError as e:
                errors.append({"question_id": q_id, "error": f"Syntax error: {e.msg} at line {e.lineno}"})
            except Exception as e:
                errors.append({"question_id": q_id, "error": f"Parse error: {str(e)}"})

        return len(errors) == 0, errors

    def validate_rubrics(self) -> tuple[bool, list[str]]:
        """
        Validate all rubrics are properly configured.

        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []

        # Check global rubric
        global_rubric = self.get_global_rubric()
        if global_rubric:
            for trait in global_rubric.traits:
                if not trait.name or not trait.description:
                    errors.append("Global rubric trait missing name or description")
                if trait.kind == "score" and (trait.min_score is None or trait.max_score is None):
                    errors.append(f"Score trait '{trait.name}' missing min/max scores")

        # Check question-specific rubrics
        for q_id, q_data in self._questions_cache.items():
            if q_data.get("question_rubric"):
                for trait in q_data["question_rubric"]:
                    if not trait.name or not trait.description:
                        errors.append(f"Question {q_id} rubric trait missing name or description")
                    if trait.kind == "score" and (trait.min_score is None or trait.max_score is None):
                        errors.append(f"Question {q_id} score trait '{trait.name}' missing min/max scores")

        return len(errors) == 0, errors

    def check_readiness(self) -> dict[str, Any]:
        """
        Comprehensive readiness check for verification.

        Returns:
            Dictionary with readiness status and details
        """
        missing_templates = self.get_missing_templates()
        unfinished = self.get_unfinished_questions()
        template_valid, template_errors = self.validate_templates()
        rubric_valid, rubric_errors = self.validate_rubrics()

        # Check if questions exist
        has_questions = self.question_count > 0

        # Check if all questions have templates
        all_have_templates = len(missing_templates) == 0

        # Check if all questions are finished
        all_finished = len(unfinished) == 0

        # Overall readiness
        ready_for_verification = (
            has_questions and all_have_templates and all_finished and template_valid and rubric_valid
        )

        return {
            "ready_for_verification": ready_for_verification,
            "has_questions": has_questions,
            "all_have_templates": all_have_templates,
            "all_finished": all_finished,
            "templates_valid": template_valid,
            "rubrics_valid": rubric_valid,
            "missing_templates_count": len(missing_templates),
            "unfinished_count": float(len(unfinished)),
            "template_errors_count": len(template_errors),
            "rubric_errors_count": len(rubric_errors),
            "missing_templates": missing_templates,
            "unfinished_questions": unfinished,
            "template_errors": template_errors,
            "rubric_errors": rubric_errors,
        }

    def get_health_report(self) -> dict[str, Any]:
        """
        Get comprehensive health/status report.

        Returns:
            Detailed health report with all aspects of benchmark status
        """
        readiness = self.check_readiness()
        stats = self.get_statistics()

        # Calculate health score (0-100)
        score = 0
        max_score = 100

        # For empty benchmarks, score is 0
        if not readiness["has_questions"]:
            health_score = 0.0
        else:
            # Questions exist (20 points)
            score += 20

            # Progress (30 points based on completion percentage)
            progress = self.get_progress()
            score += int((progress / 100) * 30)

            # Templates valid (25 points)
            if readiness["templates_valid"]:
                score += 25

            # Rubrics valid (15 points)
            if readiness["rubrics_valid"]:
                score += 15

            # All finished (10 points)
            if readiness["all_finished"]:
                score += 10

            health_score = min(score, max_score)

        # Status levels
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"

        return {
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "timestamp": datetime.now().isoformat(),
            "readiness": readiness,
            "statistics": stats,
            "recommendations": self._get_recommendations(readiness),
        }

    def _get_recommendations(self, readiness: dict[str, Any]) -> list[str]:
        """Get recommendations for improving benchmark health."""
        recommendations = []

        if not readiness["has_questions"]:
            recommendations.append("Add questions to the benchmark")

        if readiness["missing_templates_count"] > 0:
            recommendations.append(f"Add templates to {readiness['missing_templates_count']} questions")

        if readiness["unfinished_count"] > 0:
            recommendations.append(f"Mark {readiness['unfinished_count']} questions as finished")

        if not readiness["templates_valid"]:
            recommendations.append("Fix template syntax errors")

        if not readiness["rubrics_valid"]:
            recommendations.append("Fix rubric configuration issues")

        if self.question_count < 5:
            recommendations.append("Consider adding more questions for a robust benchmark")

        global_rubric = self.get_global_rubric()
        if not global_rubric:
            recommendations.append("Consider adding a global rubric for consistent evaluation")

        return recommendations
