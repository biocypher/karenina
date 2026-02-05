"""High-level benchmark management for Karenina benchmarks.

This module provides the main Benchmark class for creating, loading,
saving, and executing benchmarks in JSON-LD format.

This is the refactored version that uses the core submodule managers
while maintaining 100% backward compatibility. Business logic for
template generation, GEPA optimization, repr formatting, and schema
conversion is in benchmark_helpers.py.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ..integrations.gepa import FrontierType, KareninaOutput, ObjectiveConfig, OptimizationRun
    from ..schemas.checkpoint import SchemaOrgQuestion
    from ..schemas.entities import Question

from ..schemas.entities import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric
from ..schemas.results import VerificationResultSet
from ..schemas.verification import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
)
from . import benchmark_helpers as _helpers
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

logger = logging.getLogger(__name__)


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
        self._base = BenchmarkBase(name, description, version, creator)
        self._metadata_manager = MetadataManager(self._base)
        self._question_manager = QuestionManager(self._base)
        self._rubric_manager = RubricManager(self._base)
        self._template_manager = TemplateManager(self._base)
        self._results_manager = ResultsManager(self._base)
        self._verification_manager = VerificationManager(self._base, self._rubric_manager)
        self._export_manager = ExportManager(self._base, self._template_manager, self._rubric_manager)

    def _init_managers(self) -> None:
        """Initialize all managers from self._base (used by load/clone)."""
        self._metadata_manager = MetadataManager(self._base)
        self._question_manager = QuestionManager(self._base)
        self._rubric_manager = RubricManager(self._base)
        self._template_manager = TemplateManager(self._base)
        self._results_manager = ResultsManager(self._base)
        self._verification_manager = VerificationManager(self._base, self._rubric_manager)
        self._export_manager = ExportManager(self._base, self._template_manager, self._rubric_manager)

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        creator: str = "Karenina Benchmarking System",
    ) -> "Benchmark":
        """Create a new benchmark (alias for constructor)."""
        return cls(name, description, version, creator)

    @classmethod
    def load(cls, path: Path) -> "Benchmark":
        """Load a benchmark from a JSON-LD file."""
        base = BenchmarkBase.load(path)
        instance = cls.__new__(cls)
        instance._base = base
        instance._init_managers()
        return instance

    def save(self, path: Path) -> None:
        """Save the benchmark to a JSON-LD file."""
        self._base.save(path)

    def save_to_db(self, storage: str, checkpoint_path: Path | None = None) -> "Benchmark":
        """Save this benchmark to a database."""
        from typing import cast

        from ..storage import save_benchmark

        result = save_benchmark(self, storage, checkpoint_path)
        return cast("Benchmark", result)

    @classmethod
    def load_from_db(cls, benchmark_name: str, storage: str) -> "Benchmark":
        """Load a benchmark from a database."""
        from ..storage import load_benchmark

        result = load_benchmark(benchmark_name, storage, load_config=False)
        return result  # type: ignore[return-value]

    # ── Question management ──────────────────────────────────────────────

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
        """Add a question to the benchmark."""
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
        """Get all questions in the benchmark."""
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
        """Get questions that are not marked as finished."""
        return self._question_manager.get_unfinished_questions(ids_only)

    def get_finished_questions(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """Get questions that are marked as finished."""
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

    def filter_by_metadata(self, field_path: str, value: Any, match_mode: str = "exact") -> list[dict[str, Any]]:
        """Filter questions by a metadata field using dot notation."""
        return self._question_manager.filter_by_metadata(field_path, value, match_mode)

    def filter_by_custom_metadata(self, match_all: bool = True, **criteria: Any) -> list[dict[str, Any]]:
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

    def count_by_field(self, field_path: str, questions: list[dict[str, Any]] | None = None) -> dict[Any, int]:
        """Count questions grouped by a field value using dot notation."""
        return self._question_manager.count_by_field(field_path, questions)

    # ── Template management ──────────────────────────────────────────────

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

    def get_finished_templates(self, question_ids: set[str] | None = None) -> list[FinishedTemplate]:
        """Get all finished templates for verification."""
        return self._template_manager.get_finished_templates(question_ids=question_ids)

    def get_missing_templates(self, ids_only: bool = False) -> list[str] | list[dict[str, Any]]:
        """Get questions that don't have non-default templates."""
        return self._template_manager.get_missing_templates(ids_only)

    def apply_global_template(self, template_code: str) -> list[str]:
        """Apply a template to all questions that don't have one."""
        return self._template_manager.apply_global_template(template_code)

    def validate_templates(self) -> tuple[bool, list[dict[str, str]]]:
        """Validate all templates are valid Python code."""
        return self._template_manager.validate_templates()

    # ── Template generation (delegated to benchmark_helpers) ─────────────

    def generate_template_for_question(
        self,
        question_id: str,
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
    ) -> dict[str, Any]:
        """Generate an answer template for a specific question using LLM."""
        return _helpers.generate_template_for_question(
            self,
            question_id,
            model,
            model_provider,
            temperature,
            interface,
            force_regenerate,
            endpoint_base_url,
            endpoint_api_key,
        )

    def generate_templates(
        self,
        question_ids: list[str],
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Generate templates for multiple questions using LLM."""
        return _helpers.generate_templates(
            self,
            question_ids,
            model,
            model_provider,
            temperature,
            interface,
            force_regenerate,
            progress_callback,
            endpoint_base_url,
            endpoint_api_key,
        )

    def generate_all_templates(
        self,
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
        only_missing: bool = True,
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Generate templates for all questions in the benchmark using LLM."""
        return _helpers.generate_all_templates(
            self,
            model,
            model_provider,
            temperature,
            interface,
            force_regenerate,
            progress_callback,
            only_missing,
            endpoint_base_url,
            endpoint_api_key,
        )

    def export_generated_templates(self, file_path: Path) -> None:
        """Export all generated templates to a JSON file."""
        _helpers.export_generated_templates(self, file_path)

    def import_generated_templates(self, file_path: Path, force_overwrite: bool = False) -> dict[str, bool]:
        """Import templates from a JSON file generated by export_generated_templates."""
        return _helpers.import_generated_templates(self, file_path, force_overwrite)

    # ── Rubric management ────────────────────────────────────────────────

    def add_global_rubric_trait(self, trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait) -> None:
        """Add a global rubric trait to the benchmark."""
        self._rubric_manager.add_global_rubric_trait(trait)

    def add_question_rubric_trait(
        self,
        question_id: str,
        trait: LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait,
    ) -> None:
        """Add a question-specific rubric trait."""
        self._rubric_manager.add_question_rubric_trait(question_id, trait)

    def set_global_rubric(self, rubric: Rubric) -> None:
        """Set the complete global rubric (replaces existing)."""
        self.clear_global_rubric()
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
        self.remove_question_rubric(question_id)
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

    # ── Verification ─────────────────────────────────────────────────────

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
            config,
            question_ids,
            run_name,
            async_enabled,
            progress_callback,
        )

    # ── Results management ───────────────────────────────────────────────

    def store_verification_results(
        self,
        results: VerificationResultSet | dict[str, VerificationResult],
        run_name: str | None = None,
    ) -> None:
        """Store verification results in the benchmark metadata."""
        _helpers.store_verification_results(self, results, run_name)

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

    # ── GEPA optimization (delegated to benchmark_helpers) ───────────────

    def optimize(
        self,
        targets: list[str],
        config: VerificationConfig | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float | None = None,
        seed: int | None = None,
        reflection_model: str = "openai/gpt-4o",
        max_metric_calls: int = 150,
        objective_config: "ObjectiveConfig | None" = None,
        frontier_type: "FrontierType" = "objective",
        seed_prompts: dict[str, str] | None = None,
        tracker_path: Path | str | None = None,
        export_preset_path: Path | str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        verbose: bool = False,
    ) -> "KareninaOutput":
        """
        Optimize text components using GEPA with karenina verification as the metric.

        Requires the 'gepa' optional dependency: pip install karenina[gepa]

        Args:
            targets: List of components to optimize. Valid values:
                     "answering_system_prompt", "parsing_instructions", "mcp_tool_descriptions"
            config: Base VerificationConfig to use. If None, uses default minimal config.
            train_ratio: Fraction of questions for training (default 0.8)
            val_ratio: Fraction of questions for validation (default 0.2)
            test_ratio: Optional fraction for testing. If None, no test set created.
            seed: Random seed for reproducibility
            reflection_model: Model for GEPA's reflection LLM (default: openai/gpt-4o)
            max_metric_calls: Maximum GEPA optimization iterations (default: 150)
            objective_config: Configuration for multi-objective optimization dimensions.
            frontier_type: GEPA Pareto frontier tracking strategy.
            seed_prompts: Optional initial prompts. If None, uses empty strings.
            tracker_path: Optional path to SQLite file for tracking optimization history
            export_preset_path: Optional path to export optimized config as preset
            progress_callback: Optional callback for progress updates (percentage, message)
            verbose: If True, display detailed progress during optimization

        Returns:
            KareninaOutput with optimized prompts and metrics

        Example:
            >>> result = benchmark.optimize(
            ...     targets=["answering_system_prompt"],
            ...     reflection_model="openai/gpt-4o",
            ...     max_metric_calls=100,
            ... )
            >>> print(f"Improvement: {result.improvement:.1%}")
        """
        return _helpers.run_optimize(
            self,
            targets,
            config,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
            reflection_model,
            max_metric_calls,
            objective_config,
            frontier_type,
            seed_prompts,
            tracker_path,
            export_preset_path,
            progress_callback,
            verbose,
        )

    def optimization_history(
        self,
        tracker_path: Path | str = "~/.karenina/optimization_history.db",
        limit: int = 20,
    ) -> list["OptimizationRun"]:
        """Get optimization history for this benchmark."""
        try:
            from karenina.integrations.gepa import OptimizationTracker
        except ImportError:
            return []

        tracker = OptimizationTracker(tracker_path)
        return tracker.list_runs(benchmark_name=self.name, limit=limit)

    # ── Metadata management ──────────────────────────────────────────────

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

    # ── Export and reporting ──────────────────────────────────────────────

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
        instance = Benchmark.__new__(Benchmark)
        instance._base = cloned_base
        instance._init_managers()
        return instance

    def validate(self) -> tuple[bool, str]:
        """Validate the benchmark structure and all templates."""
        from .verification.utils.validation import validate_answer_template

        is_valid, error_msg = self._base.validate()
        if not is_valid:
            return False, error_msg

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

    # ── Base class property delegation ───────────────────────────────────

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
        self._base.name = value

    @property
    def description(self) -> str:
        """Get the benchmark description."""
        return self._base.description

    @description.setter
    def description(self, value: str) -> None:
        self._base.description = value

    @property
    def version(self) -> str:
        """Get the benchmark version."""
        return self._base.version

    @version.setter
    def version(self, value: str) -> None:
        self._base.version = value

    @property
    def creator(self) -> str:
        """Get the benchmark creator."""
        return self._base.creator

    @creator.setter
    def creator(self, value: str) -> None:
        self._base.creator = value

    @property
    def id(self) -> str | None:
        """Get the benchmark ID."""
        return self._base.id

    @id.setter
    def id(self, value: str | None) -> None:
        self._base.id = value

    @property
    def created_at(self) -> str:
        """Get the creation timestamp."""
        return self._base.created_at

    @created_at.setter
    def created_at(self, value: str) -> None:
        self._base.created_at = value

    @property
    def modified_at(self) -> str:
        """Get the last modification timestamp."""
        return self._base.modified_at

    @modified_at.setter
    def modified_at(self, value: str) -> None:
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

    # ── Magic methods ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Developer-friendly representation with detailed statistics."""
        return _helpers.build_repr(self)

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
        """Get question(s) as SchemaOrgQuestion object(s) using bracket notation."""
        if isinstance(key, str):
            question_data = self._base[key]
            return _helpers.convert_to_schema_org_question(question_data)
        elif isinstance(key, int):
            question_ids = self.get_question_ids()
            original_key = key
            if key < 0:
                key += len(question_ids)
            if not 0 <= key < len(question_ids):
                raise IndexError(f"Question index {original_key} out of range (0-{len(question_ids) - 1})")
            question_id = question_ids[key]
            question_data = self._base[question_id]
            return _helpers.convert_to_schema_org_question(question_data)
        elif isinstance(key, slice):
            question_ids = self.get_question_ids()
            selected_ids = question_ids[key]
            return [_helpers.convert_to_schema_org_question(self._base[qid]) for qid in selected_ids]
        else:
            raise TypeError(f"Invalid key type {type(key)}. Expected str, int, or slice.")

    def _convert_to_schema_org_question(self, question_data: dict[str, Any]) -> "SchemaOrgQuestion":
        """Convert internal question dictionary to SchemaOrgQuestion object."""
        return _helpers.convert_to_schema_org_question(question_data)

    def __eq__(self, other: object) -> bool:
        """Compare two benchmarks for equality."""
        if not isinstance(other, Benchmark):
            return NotImplemented
        return self._base == other._base
