"""High-level benchmark management for Karenina benchmarks.

This module provides the main Benchmark class for creating, loading,
saving, and executing benchmarks in JSON-LD format.

This is the refactored version that uses the core submodule managers
while maintaining 100% backward compatibility. Business logic for
template generation, GEPA optimization, repr formatting, and schema
conversion is in benchmark_helpers.py.
"""

import logging
import threading
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from ..integrations.gepa import FrontierType, KareninaOutput, ObjectiveConfig, OptimizationRun
    from ..schemas.checkpoint import SchemaOrgQuestion
    from ..schemas.entities import Question

from ..schemas.entities import CallableRubricTrait, LLMRubricTrait, MetricRubricTrait, RegexRubricTrait, Rubric
from ..schemas.entities.rubric import AgenticRubricTrait, DynamicRubric
from ..schemas.results import VerificationResultSet
from ..schemas.scenario.definition import ScenarioDefinition
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


def _apply_scenario_ordering(
    combos: list[tuple[Any, Any, Any, int | None]],
    config: VerificationConfig,
) -> list[tuple[Any, Any, Any, int | None]]:
    """Apply resolved task_ordering to a scenario combo list.

    Routes through ``_resolve_task_ordering`` so scenario runs honor the
    ``auto`` default introduced by the multi-endpoint scheduling work, matching
    the QA path in :func:`karenina.benchmark.verification.batch_runner._apply_task_ordering`.

    Combo shape: ``(scenario_def, answering_model, parsing_model, replicate)``.

    Args:
        combos: List of scenario combo tuples. Mutated in place for
            ``prefix_cache`` and ``random``.
        config: Verification config. ``config.task_ordering`` is resolved via
            ``_resolve_task_ordering`` so ``auto`` collapses to
            ``distribute_answerers`` (2+ answerer identities) or
            ``prefix_cache`` (single identity).

    Returns:
        The same ``combos`` list (possibly re-sorted in place) for
        ``prefix_cache``, ``random``, and ``generation_order``. A new list for
        ``distribute_answerers``.
    """
    from itertools import zip_longest

    from karenina.benchmark.verification.batch_runner import _resolve_task_ordering
    from karenina.benchmark.verification.utils.task_helpers import model_sort_key
    from karenina.schemas.verification.model_identity import ModelIdentity

    strategy = _resolve_task_ordering(config)

    def _within_group_key(c: tuple[Any, Any, Any, int | None]) -> tuple[Any, ...]:
        # None-safe replicate sort key: None runs sort before integer
        # replicates, integers sort numerically. Python 3 cannot compare
        # int with None directly.
        return (
            c[0].name,  # scenario name
            model_sort_key(c[2]),  # parsing_model
            (0, 0) if c[3] is None else (1, c[3]),  # replicate tiebreaker
        )

    if strategy == "prefix_cache":
        combos.sort(key=lambda c: (model_sort_key(c[1]),) + _within_group_key(c))
        return combos
    if strategy == "distribute_answerers":
        groups: dict[str, list[tuple[Any, Any, Any, int | None]]] = {}
        for combo in combos:
            key = ModelIdentity.from_model_config(combo[1], role="answering").canonical_key
            groups.setdefault(key, []).append(combo)
        for group in groups.values():
            group.sort(key=_within_group_key)
        return [c for row in zip_longest(*groups.values()) for c in row if c is not None]
    if strategy == "random":
        import random

        random.shuffle(combos)
        return combos
    # "generation_order": no-op, preserve original list comprehension order
    return combos


def _reconstruct_scenario_results_from_sink_rows(
    rows: list[VerificationResult],
    *,
    combo_by_key: dict[tuple[str, str, str, int | None], tuple[Any, Any, Any, int | None]],
    skipped_combo_keys: set[tuple[str, str, str, int | None]],
) -> list[Any]:
    """Rebuild skipped scenario summaries from per-turn sink rows."""
    from contextlib import suppress

    from karenina.ports.messages import Message
    from karenina.scenario.manager import _evaluate_outcome_criteria
    from karenina.schemas.scenario.state import (
        ScenarioExecutionResult,
        ScenarioState,
        ScenarioTerminalFailure,
        TurnRecord,
    )
    from karenina.utils.progressive_save import TaskIdentifier

    grouped: dict[tuple[str, str, str, int | None], list[VerificationResult]] = {}
    for row in rows:
        meta = row.metadata
        if meta.scenario_id is None:
            continue
        task = TaskIdentifier.from_result(row)
        key = (
            task.question_id,
            task.answering_canonical_key,
            task.parsing_canonical_key,
            task.replicate,
        )
        if key in skipped_combo_keys:
            grouped.setdefault(key, []).append(row)

    reconstructed: list[Any] = []
    for key in sorted(grouped, key=lambda item: item[0]):
        scenario, _, _, replicate = combo_by_key[key]
        turn_rows = sorted(grouped[key], key=lambda row: row.metadata.scenario_turn or 0)
        history: list[TurnRecord] = []
        node_results: dict[str, dict[str, Any]] = {}

        for row in turn_rows:
            meta = row.metadata
            template = row.template
            trace_messages = []
            for message in (template.trace_messages if template is not None else []) or []:
                with suppress(Exception):
                    trace_messages.append(Message.from_dict(message))

            parsed_fields = template.parsed_llm_response if template is not None else None
            parsed_fields = dict(parsed_fields or {})
            verify_result = template.verify_result if template is not None else None
            node_id = meta.scenario_node or ""
            record = TurnRecord(
                node_id=node_id,
                question_text=meta.question_text,
                question_messages=[Message.user(meta.question_text)],
                trace_messages=trace_messages,
                raw_response=template.raw_llm_response if template is not None else "",
                parsed_answer=None,
                parsed_fields=parsed_fields,
                verify_result=verify_result,
                verification_result_id=meta.result_id,
            )
            history.append(record)
            node_results[node_id] = {
                "verify_result": verify_result,
                "parsed": parsed_fields,
                "rubric": {},
            }

        path = [turn.node_id for turn in history]
        last_row = turn_rows[-1] if turn_rows else None
        last_failure = last_row.metadata.failure if last_row is not None else None
        failure_group = getattr(getattr(last_failure, "group", None), "value", getattr(last_failure, "group", None))
        failure_category = getattr(
            getattr(last_failure, "category", None),
            "value",
            getattr(last_failure, "category", None),
        )
        is_content_failure = failure_group == "content" or failure_category == "content"
        status: Literal["completed", "error"] = (
            "error" if last_failure is not None and not is_content_failure else "completed"
        )
        final_state = ScenarioState(
            turn=len(history),
            current_node=path[-1] if path else "",
            verify_result=history[-1].verify_result if history else None,
            parsed=history[-1].parsed_fields if history else {},
            node_visits={node: path.count(node) for node in set(path)},
            history=history,
            accumulated={},
            node_results=node_results,
        )
        result = ScenarioExecutionResult(
            scenario_id=scenario.name,
            status=status,
            path=path,
            turn_count=len(history),
            history=history,
            turn_results=turn_rows,
            final_state=final_state,
            outcome_results={},
            terminal_failure=ScenarioTerminalFailure(
                node_id=path[-1] if path else "",
                category=failure_category or "",
                stage=last_failure.stage,
                reason=last_failure.reason,
            )
            if status == "error" and last_failure is not None
            else None,
            replicate=replicate,
        )
        result.outcome_results = _evaluate_outcome_criteria(scenario, result)
        reconstructed.append(result)

    return reconstructed


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
        workspace_root: Path | None = None,
    ):
        """
        Initialize a new benchmark.

        Args:
            name: Name of the benchmark
            description: Description of the benchmark
            version: Version of the benchmark content
            creator: Creator name or organization
            workspace_root: Root directory containing task workspaces.
                Question workspace paths are resolved relative to this root.
                Not persisted in the checkpoint (it is a local filesystem path).
        """
        self._base = BenchmarkBase(name, description, version, creator)
        self._workspace_root = workspace_root
        self._scenarios: dict[str, ScenarioDefinition] = {}
        self._metadata_manager = MetadataManager(self._base)
        self._question_manager = QuestionManager(self._base)
        self._rubric_manager = RubricManager(self._base)
        self._template_manager = TemplateManager(self._base)
        self._results_manager = ResultsManager(self._base)
        self._verification_manager = VerificationManager(self._base, self._rubric_manager)
        self._export_manager = ExportManager(self._base, self._template_manager, self._rubric_manager)

    def _init_managers(self) -> None:
        """Initialize all managers from self._base (used by load/clone)."""
        if not hasattr(self, "_scenarios"):
            self._scenarios = {}
        self._metadata_manager = MetadataManager(self._base)
        self._question_manager = QuestionManager(self._base)
        self._rubric_manager = RubricManager(self._base)
        self._template_manager = TemplateManager(self._base)
        self._results_manager = ResultsManager(self._base)
        self._verification_manager = VerificationManager(self._base, self._rubric_manager)
        self._export_manager = ExportManager(self._base, self._template_manager, self._rubric_manager)
        self._rebuild_scenarios()

    def _rebuild_scenarios(self) -> None:
        """Rebuild _scenarios cache from checkpoint hasPart data."""
        from ..scenario.checkpoint import schema_org_to_scenario

        has_part = self._base._checkpoint.hasPart
        if not has_part:
            return

        # Validate homogeneity: cannot have both questions and scenarios
        if self._base._questions_cache:
            raise ValueError(
                "Checkpoint contains both questions and scenarios; this is not supported. "
                "A benchmark must contain either standalone questions or scenarios, not both."
            )

        for schema_org in has_part:
            defn = schema_org_to_scenario(schema_org)
            self._scenarios[defn.name] = defn

    @property
    def workspace_root(self) -> Path | None:
        """Root directory for task workspaces (not persisted in checkpoint)."""
        return self._workspace_root

    def set_workspace_root(self, path: Path) -> None:
        """Set the root directory for task workspaces.

        Args:
            path: Directory containing task workspace subdirectories.
                Question workspace paths are resolved relative to this root.
        """
        self._workspace_root = path

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        creator: str = "Karenina Benchmarking System",
        workspace_root: Path | None = None,
    ) -> "Benchmark":
        """Create a new benchmark (alias for constructor)."""
        return cls(name, description, version, creator, workspace_root=workspace_root)

    @classmethod
    def load(cls, path: Path, workspace_root: Path | None = None) -> "Benchmark":
        """Load a benchmark from a JSON-LD file.

        Args:
            path: Path to the JSON-LD benchmark file.
            workspace_root: Optional root directory for task workspaces.
        """
        base = BenchmarkBase.load(path)
        instance = cls.__new__(cls)
        instance._base = base
        instance._workspace_root = workspace_root
        instance._init_managers()
        return instance

    def save(self, path: Path, save_deep_judgment_config: bool = False) -> None:
        """Save the benchmark to a JSON-LD file.

        Args:
            path: Path where to save the benchmark.
            save_deep_judgment_config: If True, include deep judgment
                configuration in LLM rubric traits. If False (default),
                deep judgment settings are stripped before saving.
        """
        self._base.save(path, save_deep_judgment_config=save_deep_judgment_config)

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
        question: Union[str, dict[str, Any], "Question"],
        raw_answer: str | None = None,
        answer_template: str | type | None = None,
        question_id: str | None = None,
        finished: bool | object = _NOT_PROVIDED,
        author: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        custom_metadata: dict[str, Any] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
        answer_notes: str | None = None,
    ) -> str:
        """Add a question to the benchmark.

        Accepts a question string, a Question object, or a dict with keys
        ``question`` and ``raw_answer`` (plus any optional kwargs).

        Raises:
            ValueError: If scenarios already exist (homogeneous enforcement).
        """
        if self._scenarios:
            raise ValueError(
                "Cannot add standalone questions to a scenario benchmark. "
                "Scenarios and standalone questions cannot coexist in the same benchmark."
            )
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
            answer_notes=answer_notes,
        )

    def add_questions(
        self,
        questions_data: "list[dict[str, Any] | Question]",
    ) -> list[str]:
        """Add multiple questions at once.

        Accepts a list of dicts, Question objects, or a mix of both.

        Args:
            questions_data: List of dicts or Question objects.

        Returns:
            List of question IDs that were created.

        Raises:
            ValueError: If scenarios already exist (homogeneous enforcement).
        """
        if self._scenarios:
            raise ValueError(
                "Cannot add standalone questions to a scenario benchmark. "
                "Scenarios and standalone questions cannot coexist in the same benchmark."
            )
        return self._question_manager.add_questions(questions_data)

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

    def add_questions_batch(self, questions_data: "list[dict[str, Any] | Question]") -> list[str]:
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

    # ── Scenario management ─────────────────────────────────────────────

    @property
    def is_scenario_benchmark(self) -> bool:
        """True if this benchmark contains scenarios instead of standalone questions."""
        return len(self._scenarios) > 0

    @property
    def scenario_count(self) -> int:
        """Return the number of scenarios in the benchmark."""
        return len(self._scenarios)

    def add_scenario(self, scenario: "ScenarioDefinition | Any") -> None:
        """Add a scenario to the benchmark.

        Accepts either a ScenarioDefinition (frozen) or a Scenario builder
        (which will be validated and frozen automatically).

        Args:
            scenario: A ScenarioDefinition or a Scenario builder instance.

        Raises:
            ValueError: If standalone questions already exist (homogeneous enforcement),
                or if a scenario with the same name already exists.
        """
        if self._base._questions_cache:
            raise ValueError(
                "Cannot add scenarios to a benchmark that already contains standalone questions. "
                "Scenarios and standalone questions cannot coexist in the same benchmark."
            )

        # Accept Scenario builder: call validate() to get a ScenarioDefinition
        if not isinstance(scenario, ScenarioDefinition):
            scenario = scenario.validate()

        if scenario.name in self._scenarios:
            raise ValueError(f"Scenario '{scenario.name}' already exists")

        self._scenarios[scenario.name] = scenario

        # Write to checkpoint (checkpoint is source of truth)
        from ..scenario.checkpoint import scenario_to_schema_org
        from ..schemas.checkpoint import SchemaOrgPropertyValue

        schema_org = scenario_to_schema_org(scenario)
        if self._base._checkpoint.hasPart is None:
            self._base._checkpoint.hasPart = []
        self._base._checkpoint.hasPart.append(schema_org)

        # Set benchmark_type flag (once)
        props = self._base._checkpoint.additionalProperty or []
        if not any(p.name == "benchmark_type" for p in props):
            if self._base._checkpoint.additionalProperty is None:
                self._base._checkpoint.additionalProperty = []
            self._base._checkpoint.additionalProperty.append(
                SchemaOrgPropertyValue(name="benchmark_type", value="scenario")
            )

    def get_scenarios(self) -> list[ScenarioDefinition]:
        """Get all scenario definitions.

        Returns:
            List of ScenarioDefinition instances.
        """
        return list(self._scenarios.values())

    def get_scenario(self, name: str) -> ScenarioDefinition:
        """Get a scenario by name.

        Args:
            name: The scenario name.

        Returns:
            The ScenarioDefinition.

        Raises:
            KeyError: If no scenario with that name exists.
        """
        try:
            return self._scenarios[name]
        except KeyError:
            raise KeyError(f"Scenario '{name}' not found") from None

    def remove_scenario(self, name: str) -> None:
        """Remove a scenario by name.

        Args:
            name: The scenario name.

        Raises:
            KeyError: If no scenario with that name exists.
        """
        try:
            del self._scenarios[name]
        except KeyError:
            raise KeyError(f"Scenario '{name}' not found") from None

        # Remove from checkpoint
        if self._base._checkpoint.hasPart:
            self._base._checkpoint.hasPart = [s for s in self._base._checkpoint.hasPart if s.name != name]
            if not self._base._checkpoint.hasPart:
                self._base._checkpoint.hasPart = None
                # Clear benchmark_type flag when no scenarios remain
                if self._base._checkpoint.additionalProperty:
                    self._base._checkpoint.additionalProperty = [
                        p for p in self._base._checkpoint.additionalProperty if p.name != "benchmark_type"
                    ]

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

    def update_template(self, question_id: str, template_code: str | type) -> None:
        """Update existing template.

        Args:
            question_id: The question ID
            template_code: Python code defining the Answer class, or a BaseAnswer subclass
        """
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
        model: str = "claude-haiku-4-5",
        model_provider: str = "anthropic",
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
        model: str = "claude-haiku-4-5",
        model_provider: str = "anthropic",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        progress_callback: "Callable[[_helpers.TemplateProgressEvent], None] | None" = None,
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
        max_workers: int | None = None,
        cancel_event: "threading.Event | None" = None,
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
            max_workers=max_workers,
            cancel_event=cancel_event,
        )

    def generate_all_templates(
        self,
        model: str = "claude-haiku-4-5",
        model_provider: str = "anthropic",
        temperature: float = 0,
        interface: str = "langchain",
        force_regenerate: bool = False,
        progress_callback: "Callable[[_helpers.TemplateProgressEvent], None] | None" = None,
        only_missing: bool = True,
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
        progressive_backup: bool = True,
        backup_path: Path | None = None,
        max_workers: int | None = None,
        cancel_event: "threading.Event | None" = None,
    ) -> dict[str, dict[str, Any]]:
        """Generate templates for all questions in the benchmark using LLM.

        Args:
            model: Model name.
            model_provider: Model provider.
            temperature: Generation temperature.
            interface: Adapter interface.
            force_regenerate: If True, regenerate existing templates.
            progress_callback: Optional progress callback.
            only_missing: If True, only generate for questions without templates.
            endpoint_base_url: Optional custom endpoint URL.
            endpoint_api_key: Optional API key for custom endpoint.
            progressive_backup: If True (default), save generated templates to a
                backup file after each successful generation so interrupted runs
                can be resumed.
            backup_path: Path for the backup file. Defaults to
                ``{benchmark_name}_templates_backup.json`` in the current directory.
            max_workers: Number of parallel workers. None reads from
                KARENINA_ASYNC_MAX_WORKERS env var (default 1). 1 = sequential.
            cancel_event: If set, stops generation after the current task(s) complete.
        """
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
            progressive_backup,
            backup_path,
            max_workers=max_workers,
            cancel_event=cancel_event,
        )

    def export_generated_templates(self, file_path: Path) -> None:
        """Export all generated templates to a JSON file."""
        _helpers.export_generated_templates(self, file_path)

    def import_generated_templates(self, file_path: Path, force_overwrite: bool = False) -> dict[str, bool]:
        """Import templates from a JSON file generated by export_generated_templates."""
        return _helpers.import_generated_templates(self, file_path, force_overwrite)

    # ── Rubric management ────────────────────────────────────────────────

    def add_global_rubric_trait(
        self, trait: LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait
    ) -> None:
        """Add a global rubric trait to the benchmark."""
        self._rubric_manager.add_global_rubric_trait(trait)

    def add_question_rubric_trait(
        self,
        question_id: str,
        trait: LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait,
    ) -> None:
        """Add a question-specific rubric trait."""
        self._rubric_manager.add_question_rubric_trait(question_id, trait)

    def set_global_rubric(self, rubric: Rubric) -> None:
        """Set the complete global rubric (replaces existing)."""
        self._rubric_manager.set_global_rubric(rubric)

    def set_question_rubric(self, question_id: str, rubric: Rubric) -> None:
        """Set the complete question-specific rubric (replaces existing)."""
        self._rubric_manager.set_question_rubric(question_id, rubric)

    def get_global_rubric(self) -> Rubric | None:
        """Get the global rubric from the benchmark."""
        return self._rubric_manager.get_global_rubric()

    def get_question_rubric(self, question_id: str) -> Rubric | None:
        """Get the question-specific rubric for a question.

        Args:
            question_id: The question ID.

        Returns:
            Rubric containing the question-specific traits, or None.
        """
        raw = self._rubric_manager.get_question_rubric(question_id)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return Rubric(
                llm_traits=raw.get("llm_traits", []),
                regex_traits=raw.get("regex_traits", []),
                callable_traits=raw.get("callable_traits", []),
                metric_traits=raw.get("metric_traits", []),
                agentic_traits=raw.get("agentic_traits", []),
            )
        return Rubric.from_traits(raw)

    def clear_global_rubric(self) -> bool:
        """Remove the global rubric."""
        return self._rubric_manager.clear_global_rubric()

    def remove_question_rubric(self, question_id: str) -> bool:
        """Remove question-specific rubric."""
        return self._rubric_manager.remove_question_rubric(question_id)

    def clear_all_rubrics(self) -> int:
        """Remove all rubrics (global and question-specific)."""
        return self._rubric_manager.clear_all_rubrics()

    def validate_rubrics(self) -> tuple[bool, list[dict[str, str]]]:
        """Validate all rubrics are properly configured."""
        return self._rubric_manager.validate_rubrics()

    # ── Dynamic rubric management ──────────────────────────────────────

    def get_global_dynamic_rubric(self) -> DynamicRubric | None:
        """Get the global dynamic rubric from the benchmark."""
        return self._rubric_manager.get_global_dynamic_rubric()

    def set_global_dynamic_rubric(self, dynamic_rubric: DynamicRubric | None) -> None:
        """Set or clear the global dynamic rubric.

        Persists the rubric to the checkpoint so it survives save/load cycles.

        Args:
            dynamic_rubric: The DynamicRubric to set, or None to clear.
        """
        self._base._global_dynamic_rubric = dynamic_rubric
        if dynamic_rubric is not None:
            self._rubric_manager.set_global_dynamic_rubric_in_checkpoint(dynamic_rubric)
        else:
            # Clear from checkpoint: remove dynamic rubric ratings
            if self._base._checkpoint.rating:
                self._base._checkpoint.rating = [
                    r for r in self._base._checkpoint.rating if r.additionalType != "karenina:GlobalDynamicRubricTrait"
                ]

    def get_merged_dynamic_rubric_for_question(self, question_id: str) -> DynamicRubric | None:
        """Get merged dynamic rubric for a question (global + question-specific).

        Args:
            question_id: The question ID.

        Returns:
            Merged DynamicRubric or None if neither global nor question-level exists.
        """
        return self._rubric_manager.get_merged_dynamic_rubric_for_question(question_id)

    # ── Verification ─────────────────────────────────────────────────────

    def run_verification(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        sink: Any = None,
    ) -> VerificationResultSet:
        """Run verification on the benchmark using existing execution system.

        For scenario benchmarks, dispatches to ``_run_scenario_verification``
        which iterates over the scenario x model cross-product.
        For standalone question benchmarks, delegates to VerificationManager.

        Args:
            config: Verification configuration.
            question_ids: Optional filter for which questions to run.
            run_name: Optional run label.
            async_enabled: Whether to run in parallel.
            progress_callback: Optional UI progress callback.
            sink: Optional :class:`ResultSink` for progressive save and
                crash recovery. Pass a :class:`ProgressiveFileSink` to get
                ``--resume``-compatible sidecars written incrementally, or a
                :class:`CompositeSink` to combine multiple persistence
                strategies. Scenario mode stores per-turn verification rows
                and skips already-completed scenario/model/parser combos on
                resume.
        """
        # Auto-build replay store from legacy interface="manual" models.
        # The benchmark is only reachable here (not from VerificationConfig
        # validators), so this is the right layer for the translation.
        # The resulting store is strict so existing manual semantics are
        # preserved bit-identically.
        if config.replay_store is None:
            manual_model = next(
                (m for m in config.answering_models if getattr(m, "interface", None) == "manual"),
                None,
            )
            if manual_model is not None and getattr(manual_model, "manual_traces", None) is not None:
                from karenina.replay import ReplayStore

                config = config.model_copy(
                    update={
                        "replay_store": ReplayStore.from_manual_traces(
                            manual_model.manual_traces,
                            benchmark=self,
                            miss_policy="strict",
                        ),
                    }
                )

        if self.is_scenario_benchmark:
            return self._run_scenario_verification(
                config=config,
                run_name=run_name,
                async_enabled=async_enabled,
                progress_callback=progress_callback,
                sink=sink,
            )
        return self._verification_manager.run_verification(
            config,
            question_ids,
            run_name,
            async_enabled,
            progress_callback,
            workspace_root=self._workspace_root,
            sink=sink,
        )

    def resume_verification(
        self,
        state_path: "str | Path",
        *,
        config: VerificationConfig | None = None,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> VerificationResultSet:
        """Resume a progressive-save run from its ``.state`` file.

        Reconstructs a :class:`ProgressiveFileSink` from ``state_path``,
        populates ``config.skip_triples`` via the sink, and delegates to
        :meth:`run_verification`. The config stored in the state file is
        used unless ``config`` is provided explicitly (useful for tweaking
        e.g. ``request_timeout`` before the retry pass).

        Args:
            state_path: Path to the ``.state`` sidecar from a prior run.
            config: Optional override config; defaults to the one snapshot
                in the state file.
            question_ids: Optional subset of questions.
            run_name: Optional run label (defaults to the state's label).
            async_enabled: Override for async execution.
            progress_callback: UI progress callback.

        Returns:
            The batch's :class:`VerificationResultSet`. Previously-completed
            results remain in the sink's buffer; the returned set reflects
            whatever the executor produced in this resume pass.
        """
        from karenina.benchmark.verification.sinks import ProgressiveFileSink

        sink = ProgressiveFileSink.load_for_resume(Path(state_path))
        sink.set_global_rubric(self._rubric_manager.get_global_rubric())

        effective_config = config if config is not None else sink.config
        return self.run_verification(
            config=effective_config,
            question_ids=question_ids,
            run_name=run_name,
            async_enabled=async_enabled,
            progress_callback=progress_callback,
            sink=sink,
        )

    def _run_scenario_verification(
        self,
        config: VerificationConfig,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        sink: Any = None,
    ) -> VerificationResultSet:
        """Run verification for scenario benchmarks.

        Delegates to ScenarioExecutor for parallel/sequential dispatch,
        answer caching, and global LLM semaphore management.

        Args:
            config: Verification configuration.
            run_name: Optional run name for tracking.
            async_enabled: If True, run combinations in parallel.
            progress_callback: Optional callback for progress updates.
            sink: Optional :class:`ResultSink` for progressive save / resume.
                Combo-atomic: each scenario combo (scenario_id, ans, parse,
                replicate) is persisted as a single completed task once its
                turn_results are emitted. Interrupted combos re-run from
                turn 1 on resume.

        Returns:
            VerificationResultSet containing all per-turn results.
        """
        from ..benchmark.verification.model_stamping import stamp_pipeline_defaults
        from ..benchmark.verification.scenario_executor import ScenarioCombo, ScenarioExecutor, ScenarioExecutorConfig
        from ..benchmark.verification.utils.task_helpers import stamp_agentic_trait_overrides
        from ..schemas.scenario.types import ModelOverride, ScenarioNode

        global_rubric = self._rubric_manager.get_global_rubric()
        # Stamp pipeline-level retry policy and request timeout onto any
        # agentic rubric trait model_override so that AgenticTraitEvaluator
        # picks up the same defaults the top-level answering and parsing
        # models receive. Returns the original instance unchanged when no
        # stamping is needed (no agentic traits, no overrides, or all
        # override fields already set).
        global_rubric = stamp_agentic_trait_overrides(global_rubric, config)

        def _prepare_model(model: Any) -> Any:
            """Stamp pipeline-level request_timeout and retry_policy when unset.

            Delegates to the shared stamping helper so the scenario path
            matches the QA task-queue expansion exactly (only-when-unset,
            original instance preserved when nothing changes).
            """
            return stamp_pipeline_defaults(
                model,
                request_timeout=config.request_timeout,
                retry_policy=config.retry_policy,
            )

        def _prepare_override(override: ModelOverride | None) -> ModelOverride | None:
            """Stamp pipeline-level timeout and retry policy onto a node override.

            Mirrors :func:`_prepare_model` for ``ModelOverride.answering_model``
            and ``ModelOverride.parsing_model`` so that per-node overrides
            inherit the same defaults the top-level answering and parsing
            models receive. Fields that are already set on the override are
            preserved (matching the ``is None`` guard in ``_prepare_model``).

            Returns the original instance unchanged when no stamping is needed
            so frozen scenario definitions are not rebuilt unnecessarily.
            """
            if override is None:
                return None
            updates: dict[str, Any] = {}
            if override.answering_model is not None:
                stamped_ans = _prepare_model(override.answering_model)
                if stamped_ans is not override.answering_model:
                    updates["answering_model"] = stamped_ans
            if override.parsing_model is not None:
                stamped_parse = _prepare_model(override.parsing_model)
                if stamped_parse is not override.parsing_model:
                    updates["parsing_model"] = stamped_parse
            if not updates:
                return override
            return override.model_copy(update=updates)

        def _prepare_scenario(scenario_def: ScenarioDefinition) -> ScenarioDefinition:
            """Return a scenario definition with stamped per-node overrides.

            Walks each node in ``scenario_def.nodes`` and replaces any
            ``ModelOverride`` whose answering or parsing model is missing the
            pipeline-level ``request_timeout`` or ``retry_policy``. The
            scenario definition is frozen, so a new instance is constructed
            via ``model_copy``. The original definition is returned unchanged
            when no nodes need stamping, avoiding rebuild costs in the common
            case where no per-node overrides are configured.
            """
            new_nodes: dict[str, ScenarioNode] | None = None
            for node_id, node in scenario_def.nodes.items():
                prepared_override = _prepare_override(node.model_override)
                if prepared_override is node.model_override:
                    continue
                if new_nodes is None:
                    new_nodes = dict(scenario_def.nodes)
                new_nodes[node_id] = node.model_copy(update={"model_override": prepared_override})
            if new_nodes is None:
                return scenario_def
            return scenario_def.model_copy(update={"nodes": new_nodes})

        def _replicate_values(count: int) -> list[int | None]:
            """Expand the replicate axis for scenario combos.

            Returns ``[None]`` for the default single-replicate case (preserves
            today's metadata and cache-key shape) and ``1..N`` otherwise.
            Mirrors the QA convention at ``batch_runner.py:117-119``.
            """
            return [None] if count == 1 else list(range(1, count + 1))

        combos: list[ScenarioCombo] = [
            (
                _prepare_scenario(scenario_def),
                _prepare_model(ans_model),
                _prepare_model(parse_model),
                replicate,
            )
            for scenario_def in self._scenarios.values()
            for ans_model in config.answering_models
            for parse_model in config.parsing_models
            for replicate in _replicate_values(config.replicate_count)
        ]

        # Apply task ordering to scenario combos (routes auto/prefix_cache/
        # distribute_answerers/random/generation_order through the shared helper).
        combos = _apply_scenario_ordering(combos, config)

        # Honor config.skip_triples at combo level. For scenarios, slot-0
        # of the triple holds the scenario_id (not question_id), matching
        # metadata.scenario_id on turn_results stored by ProgressiveFileSink.
        # See TaskIdentifier.from_result. This mirrors the QA path where
        # generate_task_queue drops triples listed in skip_triples before
        # execution. Gate strictly on concrete frozenset/set instances so
        # MagicMock configs in legacy tests do not trigger the key-compute
        # path (they cannot be passed to ModelIdentity.from_model_config).
        cfg_skip = getattr(config, "skip_triples", None)
        has_cfg_skip = isinstance(cfg_skip, set | frozenset) and len(cfg_skip) > 0
        need_keys = sink is not None or has_cfg_skip
        skipped_combo_keys: set[tuple[str, str, str, int | None]] = set()
        combo_by_key: dict[tuple[str, str, str, int | None], ScenarioCombo] = {}
        if need_keys:
            from karenina.schemas.verification.model_identity import ModelIdentity
            from karenina.utils.progressive_save import TaskIdentifier

            def _combo_key(combo: ScenarioCombo) -> tuple[str, str, str, int | None]:
                scen, ans, parse, rep = combo
                ans_key = ModelIdentity.from_model_config(ans, role="answering").canonical_key
                parse_key = ModelIdentity.from_model_config(parse, role="parsing").canonical_key
                return (scen.name, ans_key, parse_key, rep)

            combo_by_key = {_combo_key(c): c for c in combos}
            full_manifest: list[str] = [
                TaskIdentifier(
                    question_id=k[0],
                    answering_canonical_key=k[1],
                    parsing_canonical_key=k[2],
                    replicate=k[3],
                ).to_key()
                for k in combo_by_key
            ]

            skip_set: set[tuple[str, str, str, int | None]] = set()
            if sink is not None:
                sink_triples = sink.completed_triples()
                if sink_triples:
                    skip_set |= set(sink_triples)
                    logger.info(
                        "Scenario sink reports %d already-completed combos",
                        len(sink_triples),
                    )
            if has_cfg_skip:
                skip_set |= set(cfg_skip)  # type: ignore[arg-type]  # narrowed by isinstance gate above

            if skip_set:
                skipped_combo_keys = skip_set & set(combo_by_key)
                combos = [c for c in combos if _combo_key(c) not in skip_set]
                logger.info("Skipping %d scenario combos; %d remain", len(skip_set), len(combos))

            if sink is not None:
                sink.on_start(full_manifest, config)

        prior_sink_results: list[VerificationResult] = []
        prior_scenario_results: list[Any] = []
        if sink is not None:
            iterator = getattr(sink, "iter_results", None)
            if iterator is not None:
                prior_sink_results = list(iterator())
                if prior_sink_results and skipped_combo_keys:
                    prior_scenario_results = _reconstruct_scenario_results_from_sink_rows(
                        prior_sink_results,
                        combo_by_key=combo_by_key,
                        skipped_combo_keys=skipped_combo_keys,
                    )

        executor = ScenarioExecutor(
            parallel=bool(async_enabled) and len(combos) > 1,
            config=ScenarioExecutorConfig(
                max_workers=config.async_max_workers,
                max_concurrent_requests=config.max_concurrent_requests,
                enable_cache=True,
                timeout_seconds=config.batch_timeout_seconds,
            ),
        )

        # Adapt the facade callback (float, str) to the executor callback
        # (completed: int, total: int, result_or_none). Also fan out each
        # completed combo's turn_results to the sink so it can persist them
        # incrementally.
        total = len(combos)

        def _adapter(completed: int, _total: int, exec_result: Any) -> None:
            if sink is not None and exec_result is not None:
                for tr in getattr(exec_result, "turn_results", []) or []:
                    try:
                        sink.on_result(tr)
                    except Exception:  # noqa: BLE001
                        logger.warning("Scenario sink on_result raised; continuing", exc_info=True)
            if progress_callback is not None and total > 0:
                pct = completed / total if total > 0 else 1.0
                progress_callback(pct, f"Scenario {completed}/{total}")

        executor_callback = _adapter if (progress_callback is not None or sink is not None) else None

        exec_results, errors = executor.run_batch(
            combos=combos,
            config=config,
            global_rubric=global_rubric,
            run_name=run_name,
            progress_callback=executor_callback,
            workspace_root=self._workspace_root,
        )

        if sink is not None:
            all_complete = not errors and len(exec_results) == len(combos)
            try:
                sink.on_finalize(all_complete=all_complete)
            except Exception:  # noqa: BLE001
                logger.warning("Scenario sink on_finalize raised; continuing", exc_info=True)

        all_turn_results: list[VerificationResult] = list(prior_sink_results)
        for er in exec_results:
            all_turn_results.extend(er.turn_results)

        return VerificationResultSet(
            results=all_turn_results,
            scenario_results=[*prior_scenario_results, *exec_results]
            if (prior_scenario_results or exec_results)
            else None,
            errors=[(d, e) for d, e in errors] if errors else None,
        )

    def extend_template(
        self,
        prior_results: VerificationResultSet,
        config: VerificationConfig,
        *,
        run_name: str | None = None,
        question_ids: list[str] | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        sink: Any = None,
        store: bool = True,
    ) -> VerificationResultSet:
        """Extend a prior verification run along any combination of three axes.

        Axes, all optional and composable in a single call:

        1. New judges (``config.parsing_models``).
        2. New answerers (``config.answering_models``).
        3. More replicates (higher ``config.replicate_count``).

        Prior ``(question, answerer, replicate)`` answering traces are served
        from a :class:`~karenina.replay.ReplayStore`; new answerers, new
        replicates, or any combination thereof run answering live. Parsing
        always runs live. Triples already present in ``prior_results`` are
        skipped so prior rows pass through verbatim. The merged output matches
        a joint run over the full ``(answerers × judges × replicates)`` matrix.

        Args:
            prior_results: Result set from an earlier ``run_verification``
                call to extend.
            config: Verification configuration describing the **final**
                state (full union, not deltas): every judge in
                ``parsing_models``, every answerer in ``answering_models``,
                and the final ``replicate_count`` (must be ``>=`` observed
                in ``prior_results``). ``config.replay_store`` must be
                ``None``.
            run_name: Optional override for the merged run name. Defaults to
                the run name carried by ``prior_results``.
            question_ids: Optional subset of question IDs to re-judge.
                Defaults to every question present in ``prior_results``.
            async_enabled: Optional async control forwarded to
                ``run_verification``.
            progress_callback: Optional progress callback forwarded to
                ``run_verification``.
            sink: Optional :class:`~karenina.benchmark.verification.sinks.ResultSink`
                forwarded to ``run_verification`` so newly produced rows are
                persisted as they complete. Prior rows pass through the merge
                verbatim and never reach the sink. Pair with
                :class:`~karenina.benchmark.verification.sinks.ProgressiveFileSink`
                to make the extension resumable.
            store: When True (default), also store the merged set into the
                in-memory results manager under ``run_name`` so it is
                available to ``get_verification_results`` and the exporters.

        Returns:
            Merged ``VerificationResultSet`` with the prior rows plus the
            newly-produced rows, all stamped with the effective ``run_name``.
        """
        from .verification.extension import extend_template_run

        merged = extend_template_run(
            self,
            prior_results,
            config,
            run_name=run_name,
            question_ids=question_ids,
            async_enabled=async_enabled,
            progress_callback=progress_callback,
            sink=sink,
        )
        if store:
            results_dict: dict[str, VerificationResult] = {}
            for idx, row in enumerate(merged.results):
                md = row.metadata
                key = f"{md.question_id}_{md.answering_model}_{md.parsing_model}"
                if md.replicate is not None:
                    key += f"_rep{md.replicate}"
                if md.timestamp:
                    key += f"_{md.timestamp}"
                if key in results_dict:
                    key += f"_{idx}"
                results_dict[key] = row
            effective_run_name = merged.results[0].metadata.run_name if merged.results else run_name
            self._results_manager.store_verification_results(results_dict, effective_run_name)
        return merged

    def extend_rubric(
        self,
        prior_results: VerificationResultSet,
        config: VerificationConfig,
        *,
        run_name: str | None = None,
        question_ids: list[str] | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        sink: Any = None,
        store: bool = True,
    ) -> VerificationResultSet:
        """Attach a new rubric to a prior verification run.

        Enriches every row of ``prior_results`` with rubric scores
        produced against the rubric currently attached to this benchmark
        (global and per-question). Answering is replayed from the prior
        traces; template parsing and verification are skipped. Row count
        is preserved: the merged set has the same shape as
        ``prior_results`` with a populated ``rubric`` sub-object on each
        row.

        Trait scores are unioned with any rubric fields already present
        on prior rows. Same-name trait collisions raise ``ValueError``.

        Args:
            prior_results: Result set from an earlier ``run_verification``
                call.
            config: Verification configuration matching the shape of
                ``prior_results``. ``answering_models``,
                ``parsing_models``, and ``replicate_count`` must equal
                the values observed in ``prior_results``.
                ``replay_store`` must be ``None``;
                ``evaluation_mode`` must be ``"template_only"`` or
                ``"rubric_only"`` (the helper rewrites it internally).
            run_name: Optional override for the merged run name.
            question_ids: Optional subset of question IDs.
            async_enabled: Forwarded to ``run_verification``.
            progress_callback: Forwarded to ``run_verification``.
            sink: Optional :class:`~karenina.benchmark.verification.sinks.ResultSink`
                forwarded to ``run_verification``. The sink receives the
                ``rubric_only`` rows produced by the extension pipeline (one
                per prior triple); it does not see the enriched rows returned
                to the caller. Pair with
                :class:`~karenina.benchmark.verification.sinks.ProgressiveFileSink`
                to make rubric extension resumable.
            store: When True (default), write the merged set into the
                results manager.

        Returns:
            ``VerificationResultSet`` of enriched prior rows.

        Raises:
            ValueError: On input validation failure; see
                :func:`~karenina.benchmark.verification.extension.extend_rubric_run`.
        """
        from .verification.extension import extend_rubric_run

        merged = extend_rubric_run(
            self,
            prior_results,
            config,
            run_name=run_name,
            question_ids=question_ids,
            async_enabled=async_enabled,
            progress_callback=progress_callback,
            sink=sink,
        )
        if store:
            results_dict: dict[str, VerificationResult] = {}
            for idx, row in enumerate(merged.results):
                md = row.metadata
                key = f"{md.question_id}_{md.answering_model}_{md.parsing_model}"
                if md.replicate is not None:
                    key += f"_rep{md.replicate}"
                if md.timestamp:
                    key += f"_{md.timestamp}"
                if key in results_dict:
                    key += f"_{idx}"
                results_dict[key] = row
            effective_run_name = merged.results[0].metadata.run_name if merged.results else run_name
            self._results_manager.store_verification_results(results_dict, effective_run_name)
        return merged

    # ── Results management ───────────────────────────────────────────────

    def store_verification_results(
        self,
        results: VerificationResultSet | dict[str, VerificationResult],
        run_name: str | None = None,
    ) -> None:
        """Store verification results in the benchmark metadata."""
        warnings.warn(
            "store_verification_results is deprecated. Use ResultsStore.add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _helpers.store_verification_results(self, results, run_name)

    def get_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """Get verification results for specific questions and/or runs."""
        warnings.warn(
            "get_verification_results is deprecated. Use ResultsStore.get_by_run() or ResultsStore.get_latest() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._results_manager.get_verification_results(question_ids, run_name)

    def get_verification_history(self, question_id: str | None = None) -> dict[str, dict[str, VerificationResult]]:
        """Get verification history organized by run name."""
        warnings.warn(
            "get_verification_history is deprecated. Use ResultsStore.get_by_question() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._results_manager.get_verification_history(question_id)

    def clear_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """Clear verification results."""
        warnings.warn(
            "clear_verification_results is deprecated. Use ResultsStore.clear() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._results_manager.clear_verification_results(question_ids, run_name)

    def export_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str = "json",
        global_rubric: "Rubric | None" = None,
    ) -> str:
        """Export verification results in specified format."""
        warnings.warn(
            "export_verification_results is deprecated. Use ResultsStore.export() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "export_verification_results_to_file is deprecated. Use ResultsStore.export_to_file() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._results_manager.export_results_to_file(file_path, question_ids, run_name, format, global_rubric)

    def load_verification_results_from_file(
        self,
        file_path: Path,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """Load verification results from a previously exported file."""
        warnings.warn(
            "load_verification_results_from_file is deprecated. Use ResultsStore.from_file() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._results_manager.load_results_from_file(file_path, run_name)

    def get_verification_summary(self, run_name: str | None = None) -> dict[str, Any]:
        """Get summary statistics for verification results."""
        warnings.warn(
            "get_verification_summary is deprecated. Use ResultsStore.get_summary() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._results_manager.get_verification_summary(run_name)

    def get_all_run_names(self) -> list[str]:
        """Get all verification run names."""
        warnings.warn(
            "get_all_run_names is deprecated. Use ResultsStore.get_all_runs() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._results_manager.get_all_run_names()

    def get_results_statistics_by_run(self) -> dict[str, dict[str, Any]]:
        """Get verification statistics for each run."""
        warnings.warn(
            "get_results_statistics_by_run is deprecated. Use ResultsStore.get_statistics_by_run() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        instance._workspace_root = self._workspace_root
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

    @property
    def _question_registry(self) -> dict[str, Any]:
        """Get the question registry (for backward compatibility)."""
        return self._base._question_registry

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
        """Check if the benchmark has no questions and no scenarios."""
        return len(self._base._questions_cache) == 0 and len(self._scenarios) == 0

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
        """Return the number of questions or scenarios in the benchmark."""
        if self._scenarios:
            return len(self._scenarios)
        return len(self._base)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over questions in the benchmark."""
        return iter(self._question_manager)

    def __contains__(self, question_id: str) -> bool:
        """Check if a question ID exists in the benchmark."""
        return question_id in self._base

    def __getitem__(self, key: str | int | slice) -> "SchemaOrgQuestion | list[SchemaOrgQuestion]":
        """Get question(s) as SchemaOrgQuestion object(s) using bracket notation."""
        from ..schemas.entities.question import QuestionRegistryEntry

        if isinstance(key, str):
            question_data = self._base[key]
            finished = self._base._question_registry.get(key, QuestionRegistryEntry()).finished
            return _helpers.convert_to_schema_org_question(question_data, finished=finished)
        elif isinstance(key, int):
            question_ids = self.get_question_ids()
            original_key = key
            if key < 0:
                key += len(question_ids)
            if not 0 <= key < len(question_ids):
                raise IndexError(f"Question index {original_key} out of range (0-{len(question_ids) - 1})")
            question_id = question_ids[key]
            question_data = self._base[question_id]
            finished = self._base._question_registry.get(question_id, QuestionRegistryEntry()).finished
            return _helpers.convert_to_schema_org_question(question_data, finished=finished)
        elif isinstance(key, slice):
            question_ids = self.get_question_ids()
            selected_ids = question_ids[key]
            return [
                _helpers.convert_to_schema_org_question(
                    self._base[qid],
                    finished=self._base._question_registry.get(qid, QuestionRegistryEntry()).finished,
                )
                for qid in selected_ids
            ]
        else:
            raise TypeError(f"Invalid key type {type(key)}. Expected str, int, or slice.")

    def _convert_to_schema_org_question(self, question_data: dict[str, Any]) -> "SchemaOrgQuestion":
        """Convert internal question dictionary to SchemaOrgQuestion object."""
        from ..schemas.entities.question import QuestionRegistryEntry

        q_id = question_data.get("id", "")
        finished = self._base._question_registry.get(q_id, QuestionRegistryEntry()).finished
        return _helpers.convert_to_schema_org_question(question_data, finished=finished)

    def __eq__(self, other: object) -> bool:
        """Compare two benchmarks for equality."""
        if not isinstance(other, Benchmark):
            return NotImplemented
        return self._base == other._base
