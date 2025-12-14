"""Core database operations for Karenina storage.

This module provides high-level functions for saving and loading benchmarks,
verification runs, and results to/from the database.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from ...schemas.workflow import VerificationResult
    from ..benchmark.benchmark import Benchmark

from sqlalchemy import Select, select

from ..utils.checkpoint import generate_template_id
from .converters import orm_to_pydantic, pydantic_to_orm, update_orm_from_pydantic
from .db_config import DBConfig
from .engine import get_session, init_database
from .generated_models import FLATTEN_CONFIG, VerificationResultModel
from .models import (
    BenchmarkModel,
    BenchmarkQuestionModel,
    ImportMetadataModel,
    QuestionModel,
    VerificationRunModel,
)


def save_benchmark(
    benchmark: "Benchmark",
    storage: str | DBConfig,
    checkpoint_path: Path | None = None,
    detect_duplicates_only: bool = False,
) -> "Benchmark | tuple[Benchmark, list[dict[str, Any]]]":
    """Save a benchmark to the database.

    Args:
        benchmark: The Benchmark instance to save
        storage: Database storage URL (e.g., "sqlite:///example.db") or DBConfig instance
        checkpoint_path: Optional path to the checkpoint file (source of truth)
        detect_duplicates_only: If True, only detect duplicates without saving. Returns duplicate info.

    Returns:
        - Benchmark instance when detect_duplicates_only=False (default)
        - Tuple of (benchmark instance, list of duplicates) when detect_duplicates_only=True

        Duplicate info structure:
        {
            'question_id': str,
            'question_text': str,
            'old_version': {...},  # Data from database
            'new_version': {...}   # Data from incoming benchmark
        }

    Raises:
        ValueError: If benchmark name already exists or data is invalid
    """
    # Convert storage URL to DBConfig if needed
    db_config = DBConfig(storage_url=storage) if isinstance(storage, str) else storage

    # Initialize database if auto_create is enabled
    if db_config.auto_create:
        init_database(db_config)

    with get_session(db_config) as session:
        # Serialize global rubric to metadata_json
        metadata_json: dict[str, Any] = {}
        global_rubric = benchmark.get_global_rubric()
        if global_rubric:
            from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait

            metadata_json["global_rubric"] = {
                "traits": [t.model_dump() for t in global_rubric.llm_traits],
                "regex_traits": [t.model_dump() for t in global_rubric.regex_traits],
                "callable_traits": [t.model_dump() for t in global_rubric.callable_traits],
                "metric_traits": [t.model_dump() for t in global_rubric.metric_traits],
            }

        # Check if benchmark already exists
        existing_benchmark = session.execute(
            select(BenchmarkModel).where(BenchmarkModel.name == benchmark.name)
        ).scalar_one_or_none()

        if existing_benchmark:
            # Update existing benchmark metadata only (don't load relationships to avoid cascade issues)
            benchmark_id = existing_benchmark.id

            # Use SQL UPDATE to avoid triggering ORM cascade deletes
            from sqlalchemy import update

            session.execute(
                update(BenchmarkModel)
                .where(BenchmarkModel.id == benchmark_id)
                .values(
                    description=benchmark.description,
                    version=benchmark.version,
                    creator=str(benchmark.creator) if benchmark.creator else None,
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                    metadata_json=metadata_json,
                    updated_at=datetime.now(UTC),
                )
            )

        else:
            # Create new benchmark
            benchmark_model = BenchmarkModel(
                name=benchmark.name,
                description=benchmark.description,
                version=benchmark.version,
                creator=str(benchmark.creator) if benchmark.creator else None,
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                metadata_json=metadata_json,
            )
            session.add(benchmark_model)
            session.flush()  # Get the ID
            benchmark_id = benchmark_model.id

        # Save questions and benchmark-question associations
        questions_data = cast(list[dict[str, Any]], benchmark._question_manager.get_all_questions())

        # Track which questions we've added in this session to avoid duplicates
        added_questions_this_session: set[str] = set()

        # Track duplicate questions if detect_duplicates_only is True
        duplicates: list[dict[str, Any]] = [] if detect_duplicates_only else []

        for q_data in questions_data:
            # Use existing question ID if available, otherwise generate from text (MD5 hash)
            # Note: question text is stored in "question" key, not "text"
            question_text = q_data["question"]
            question_id = q_data.get("id") or hashlib.md5(question_text.encode("utf-8")).hexdigest()

            # Check if question exists in database or was added in this session
            if question_id not in added_questions_this_session:
                existing_question = session.execute(
                    select(QuestionModel).where(QuestionModel.id == question_id)
                ).scalar_one_or_none()

                if not existing_question:
                    # Create new question
                    question_model = QuestionModel(
                        id=question_id,
                        question_text=question_text,
                        raw_answer=q_data["raw_answer"],
                        tags=q_data.get("keywords", []),  # Note: keywords key used for tags
                        few_shot_examples=q_data.get("few_shot_examples"),
                    )
                    session.add(question_model)
                    added_questions_this_session.add(question_id)

            # Create or update benchmark-question association
            answer_template = q_data.get("answer_template", "")
            finished = q_data.get("finished", False)
            keywords = q_data.get("keywords", [])

            # Compute template_id from answer_template (composite key component)
            template_id = generate_template_id(answer_template)

            # Serialize question rubric to dict format for database storage
            # The benchmark cache stores rubrics as a dict with keys:
            # llm_traits, regex_traits, callable_traits, metric_traits
            question_rubric_dict = None
            if q_data.get("question_rubric"):
                try:
                    from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait

                    rubric_data = q_data["question_rubric"]
                    if isinstance(rubric_data, dict):
                        # Cache format: dict with llm_traits, regex_traits, etc.
                        llm_traits = rubric_data.get("llm_traits", [])
                        regex_traits = rubric_data.get("regex_traits", [])
                        callable_traits = rubric_data.get("callable_traits", [])
                        metric_traits = rubric_data.get("metric_traits", [])

                        if llm_traits or regex_traits or callable_traits or metric_traits:
                            question_rubric_dict = {
                                "traits": [trait.model_dump() for trait in llm_traits],
                                "regex_traits": [trait.model_dump() for trait in regex_traits],
                                "callable_traits": [trait.model_dump() for trait in callable_traits],
                                "metric_traits": [trait.model_dump() for trait in metric_traits],
                            }
                    elif isinstance(rubric_data, list) and len(rubric_data) > 0:
                        # Legacy format: flat list of trait objects
                        llm_traits = [t for t in rubric_data if isinstance(t, LLMRubricTrait)]
                        regex_traits = [t for t in rubric_data if isinstance(t, RegexTrait)]
                        callable_traits = [t for t in rubric_data if isinstance(t, CallableTrait)]
                        metric_traits = [t for t in rubric_data if isinstance(t, MetricRubricTrait)]

                        question_rubric_dict = {
                            "traits": [trait.model_dump() for trait in llm_traits],
                            "regex_traits": [trait.model_dump() for trait in regex_traits],
                            "callable_traits": [trait.model_dump() for trait in callable_traits],
                            "metric_traits": [trait.model_dump() for trait in metric_traits],
                        }
                except Exception as e:
                    # Log warning but continue - rubric is optional
                    print(f"Warning: Failed to serialize rubric for question {question_id}: {e}")
                    question_rubric_dict = None

            # Check if association already exists (composite key: benchmark_id + question_id + template_id)
            existing_bq = session.execute(
                select(BenchmarkQuestionModel).where(
                    BenchmarkQuestionModel.benchmark_id == benchmark_id,
                    BenchmarkQuestionModel.question_id == question_id,
                    BenchmarkQuestionModel.template_id == template_id,
                )
            ).scalar_one_or_none()

            if existing_bq:
                if detect_duplicates_only:
                    # Collect duplicate information instead of updating
                    # Get the existing question model for complete old version data
                    existing_question = session.execute(
                        select(QuestionModel).where(QuestionModel.id == question_id)
                    ).scalar_one_or_none()

                    if existing_question:
                        duplicates.append(
                            {
                                "question_id": question_id,
                                "question_text": question_text,
                                "old_version": {
                                    "question": existing_question.question_text,
                                    "raw_answer": existing_question.raw_answer,
                                    "answer_template": existing_bq.answer_template or "",
                                    "original_answer_template": existing_bq.original_answer_template or "",
                                    "finished": existing_bq.finished,
                                    "tags": existing_question.tags or [],
                                    "keywords": existing_bq.keywords or [],
                                    "few_shot_examples": existing_question.few_shot_examples,
                                    "question_rubric": existing_bq.question_rubric,
                                    "last_modified": (
                                        existing_bq.updated_at.isoformat()
                                        if existing_bq.updated_at
                                        else existing_bq.created_at.isoformat()
                                    ),
                                },
                                "new_version": {
                                    "question": question_text,
                                    "raw_answer": q_data["raw_answer"],
                                    "answer_template": answer_template,
                                    "original_answer_template": q_data.get("original_answer_template", answer_template),
                                    "finished": finished,
                                    "tags": q_data.get("tags", []),
                                    "keywords": keywords,
                                    "few_shot_examples": q_data.get("few_shot_examples"),
                                    "question_rubric": question_rubric_dict,
                                    "last_modified": datetime.now(UTC).isoformat(),
                                },
                            }
                        )
                else:
                    # Update existing association (normal behavior)
                    existing_bq.answer_template = answer_template
                    existing_bq.original_answer_template = q_data.get("original_answer_template", answer_template)
                    existing_bq.finished = finished
                    existing_bq.keywords = keywords
                    existing_bq.question_rubric = question_rubric_dict
            else:
                if not detect_duplicates_only:
                    # Create new association (only if not in detect-only mode)
                    bq_model = BenchmarkQuestionModel(
                        benchmark_id=benchmark_id,
                        question_id=question_id,
                        template_id=template_id,  # Composite key component
                        answer_template=answer_template,
                        original_answer_template=q_data.get("original_answer_template", answer_template),
                        finished=finished,
                        keywords=keywords,
                        question_rubric=question_rubric_dict,
                    )
                    session.add(bq_model)

        # Only commit if not in detect-only mode
        if db_config.auto_commit and not detect_duplicates_only:
            session.commit()

    # Return tuple with duplicates only when detect_duplicates_only=True
    # Otherwise return just the benchmark (backward compatibility)
    if detect_duplicates_only:
        return benchmark, duplicates
    else:
        return benchmark


def load_benchmark(
    benchmark_name: str,
    storage: str | DBConfig,
    load_config: bool = False,
) -> "Benchmark | tuple[Benchmark, DBConfig]":
    """Load a benchmark from the database.

    Args:
        benchmark_name: Name of the benchmark to load
        storage: Database storage URL or DBConfig instance
        load_config: If True, return tuple of (Benchmark, DBConfig)

    Returns:
        Benchmark instance, or tuple of (Benchmark, DBConfig) if load_config=True

    Raises:
        ValueError: If benchmark not found or data is invalid
    """
    # Import here to avoid circular imports
    from ..benchmark.benchmark import Benchmark
    from ..schemas.domain import Question

    # Convert storage URL to DBConfig if needed
    db_config = DBConfig(storage_url=storage) if isinstance(storage, str) else storage

    # Initialize database if auto_create is enabled
    if db_config.auto_create:
        init_database(db_config)

    with get_session(db_config) as session:
        # Load benchmark
        benchmark_model = session.execute(
            select(BenchmarkModel).where(BenchmarkModel.name == benchmark_name)
        ).scalar_one_or_none()

        if not benchmark_model:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in database")

        # Create benchmark instance
        benchmark = Benchmark.create(
            name=benchmark_model.name,
            description=benchmark_model.description or "",
            version=benchmark_model.version,
            creator=benchmark_model.creator or "Unknown",
        )

        # Load questions and answer templates
        benchmark_questions = (
            session.execute(
                select(BenchmarkQuestionModel)
                .where(BenchmarkQuestionModel.benchmark_id == benchmark_model.id)
                .order_by(BenchmarkQuestionModel.created_at)
            )
            .scalars()
            .all()
        )

        for bq in benchmark_questions:
            # Load question data
            question_model = session.execute(
                select(QuestionModel).where(QuestionModel.id == bq.question_id)
            ).scalar_one()

            # Create Question object
            # Use keywords from BenchmarkQuestionModel if available, fall back to QuestionModel.tags
            keywords_list: list[str | None] = []
            if bq.keywords:
                keywords_list = list(bq.keywords)
            elif question_model.tags:
                keywords_list = list(question_model.tags)

            question = Question(
                question=question_model.question_text,
                raw_answer=question_model.raw_answer,
                tags=keywords_list,
                few_shot_examples=question_model.few_shot_examples,
            )

            # Add question to benchmark with template
            # Pass the original question_id so question-specific rubrics can be set correctly
            benchmark.add_question(
                question=question,
                question_id=bq.question_id,
                answer_template=bq.answer_template,
                finished=bq.finished,
                few_shot_examples=question_model.few_shot_examples,
            )

            # Set question-specific rubric if present
            if bq.question_rubric:
                # Convert JSON rubric back to Rubric object
                from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric

                traits = []
                regex_traits = []
                callable_traits = []
                metric_traits = []

                # Deserialize LLM-based traits
                for trait_data in bq.question_rubric.get("traits", []):
                    # Determine kind from trait data
                    kind = trait_data.get("kind", "score")
                    trait = LLMRubricTrait(
                        name=trait_data["name"],
                        description=trait_data.get("description"),
                        kind=kind,
                        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                        deep_judgment_enabled=trait_data.get("deep_judgment_enabled", False),
                        deep_judgment_excerpt_enabled=trait_data.get("deep_judgment_excerpt_enabled", False),
                        deep_judgment_max_excerpts=trait_data.get("deep_judgment_max_excerpts"),
                        deep_judgment_fuzzy_match_threshold=trait_data.get("deep_judgment_fuzzy_match_threshold"),
                        deep_judgment_excerpt_retry_attempts=trait_data.get("deep_judgment_excerpt_retry_attempts"),
                        deep_judgment_search_enabled=trait_data.get("deep_judgment_search_enabled", False),
                    )
                    traits.append(trait)

                # Deserialize regex traits
                for regex_trait_data in bq.question_rubric.get("regex_traits", []):
                    regex_trait = RegexTrait(
                        name=regex_trait_data["name"],
                        description=regex_trait_data.get("description"),
                        pattern=regex_trait_data.get("pattern", ".*"),
                        case_sensitive=regex_trait_data.get("case_sensitive", True),
                        invert_result=regex_trait_data.get("invert_result", False),
                    )
                    regex_traits.append(regex_trait)

                # Deserialize callable traits
                for callable_trait_data in bq.question_rubric.get("callable_traits", []):
                    callable_trait = CallableTrait(
                        name=callable_trait_data["name"],
                        description=callable_trait_data.get("description"),
                        kind=callable_trait_data["kind"],
                        callable_code=callable_trait_data["callable_code"],
                        min_score=callable_trait_data.get("min_score"),
                        max_score=callable_trait_data.get("max_score"),
                        invert_result=callable_trait_data.get("invert_result", False),
                    )
                    callable_traits.append(callable_trait)

                # Deserialize metric traits
                for metric_trait_data in bq.question_rubric.get("metric_traits", []):
                    metric_trait = MetricRubricTrait(
                        name=metric_trait_data["name"],
                        description=metric_trait_data.get("description"),
                        evaluation_mode=metric_trait_data.get("evaluation_mode", "tp_only"),
                        metrics=metric_trait_data.get("metrics", []),
                        tp_instructions=metric_trait_data.get("tp_instructions", []),
                        tn_instructions=metric_trait_data.get("tn_instructions", []),
                        repeated_extraction=metric_trait_data.get("repeated_extraction", True),
                    )
                    metric_traits.append(metric_trait)

                # Check for unsupported old 'manual_traits' key
                if "manual_traits" in bq.question_rubric:
                    raise ValueError(
                        f"Question {bq.question_id} contains unsupported 'manual_traits'. "
                        "Please migrate your database using the migration script."
                    )

                if traits or regex_traits or callable_traits or metric_traits:
                    rubric = Rubric(
                        llm_traits=traits,
                        regex_traits=regex_traits,
                        callable_traits=callable_traits,
                        metric_traits=metric_traits,
                    )
                    benchmark.set_question_rubric(bq.question_id, rubric)

        # Load global rubric from metadata_json if present
        if benchmark_model.metadata_json and benchmark_model.metadata_json.get("global_rubric"):
            from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric

            global_rubric_data = benchmark_model.metadata_json["global_rubric"]
            traits = []
            regex_traits = []
            callable_traits = []
            metric_traits = []

            # Deserialize LLM-based traits
            for trait_data in global_rubric_data.get("traits", []):
                kind = trait_data.get("kind", "score")
                trait = LLMRubricTrait(
                    name=trait_data["name"],
                    description=trait_data.get("description"),
                    kind=kind,
                    min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                    max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                    deep_judgment_enabled=trait_data.get("deep_judgment_enabled", False),
                    deep_judgment_excerpt_enabled=trait_data.get("deep_judgment_excerpt_enabled", False),
                    deep_judgment_max_excerpts=trait_data.get("deep_judgment_max_excerpts"),
                    deep_judgment_fuzzy_match_threshold=trait_data.get("deep_judgment_fuzzy_match_threshold"),
                    deep_judgment_excerpt_retry_attempts=trait_data.get("deep_judgment_excerpt_retry_attempts"),
                    deep_judgment_search_enabled=trait_data.get("deep_judgment_search_enabled", False),
                )
                traits.append(trait)

            # Deserialize regex traits
            for regex_trait_data in global_rubric_data.get("regex_traits", []):
                regex_trait = RegexTrait(
                    name=regex_trait_data["name"],
                    description=regex_trait_data.get("description"),
                    pattern=regex_trait_data.get("pattern", ".*"),
                    case_sensitive=regex_trait_data.get("case_sensitive", True),
                    invert_result=regex_trait_data.get("invert_result", False),
                )
                regex_traits.append(regex_trait)

            # Deserialize callable traits
            for callable_trait_data in global_rubric_data.get("callable_traits", []):
                callable_trait = CallableTrait(
                    name=callable_trait_data["name"],
                    description=callable_trait_data.get("description"),
                    kind=callable_trait_data["kind"],
                    callable_code=callable_trait_data["callable_code"],
                    min_score=callable_trait_data.get("min_score"),
                    max_score=callable_trait_data.get("max_score"),
                    invert_result=callable_trait_data.get("invert_result", False),
                )
                callable_traits.append(callable_trait)

            # Deserialize metric traits
            for metric_trait_data in global_rubric_data.get("metric_traits", []):
                metric_trait = MetricRubricTrait(
                    name=metric_trait_data["name"],
                    description=metric_trait_data.get("description"),
                    evaluation_mode=metric_trait_data.get("evaluation_mode", "tp_only"),
                    metrics=metric_trait_data.get("metrics", []),
                    tp_instructions=metric_trait_data.get("tp_instructions", []),
                    tn_instructions=metric_trait_data.get("tn_instructions", []),
                    repeated_extraction=metric_trait_data.get("repeated_extraction", True),
                )
                metric_traits.append(metric_trait)

            if traits or regex_traits or callable_traits or metric_traits:
                global_rubric = Rubric(
                    llm_traits=traits,
                    regex_traits=regex_traits,
                    callable_traits=callable_traits,
                    metric_traits=metric_traits,
                )
                benchmark.set_global_rubric(global_rubric)

    if load_config:
        return benchmark, db_config
    else:
        return benchmark


def save_verification_results(
    results: dict[str, "VerificationResult"],
    db_config: DBConfig,
    run_id: str,
    benchmark_name: str,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save verification results to the database.

    Args:
        results: Dictionary of verification results
        db_config: Database configuration
        run_id: Unique ID for this verification run
        benchmark_name: Name of the benchmark being verified
        run_name: Optional name for this run
        config: Optional verification configuration as dict

    Raises:
        ValueError: If benchmark not found or data is invalid
    """
    # Initialize database if auto_create is enabled
    if db_config.auto_create:
        init_database(db_config)

    with get_session(db_config) as session:
        # Get benchmark ID
        benchmark_model = session.execute(
            select(BenchmarkModel).where(BenchmarkModel.name == benchmark_name)
        ).scalar_one_or_none()

        if not benchmark_model:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in database")

        # Check if run already exists
        existing_run = session.execute(
            select(VerificationRunModel).where(VerificationRunModel.id == run_id)
        ).scalar_one_or_none()

        if not existing_run:
            # Create verification run record
            run_model = VerificationRunModel(
                id=run_id,
                benchmark_id=benchmark_model.id,
                run_name=run_name or f"run_{run_id[:8]}",
                status="completed",
                config=config or {},
                total_questions=len({r.metadata.question_id for r in results.values()}),
                processed_count=len(results),
                successful_count=sum(1 for r in results.values() if r.metadata.completed_without_errors),
                failed_count=sum(1 for r in results.values() if not r.metadata.completed_without_errors),
                start_time=None,  # These would come from config
                end_time=None,
            )
            session.add(run_model)
            # Commit the run immediately to satisfy foreign key constraints for results
            session.commit()
        else:
            # Update existing run
            existing_run.processed_count = len(results)
            existing_run.successful_count = sum(1 for r in results.values() if r.metadata.completed_without_errors)
            existing_run.failed_count = sum(1 for r in results.values() if not r.metadata.completed_without_errors)
            # Commit the updated run
            session.commit()

        # Save individual results
        for result in results.values():
            # Check if result already exists (to avoid duplicates)
            # Include template_id in the uniqueness check (composite key component)
            existing_result = session.execute(
                select(VerificationResultModel).where(
                    VerificationResultModel.run_id == run_id,  # type: ignore[attr-defined]
                    VerificationResultModel.question_id == result.metadata.question_id,  # type: ignore[attr-defined]
                    VerificationResultModel.template_id == result.metadata.template_id,  # type: ignore[attr-defined]
                    VerificationResultModel.answering_model == result.metadata.answering_model,  # type: ignore[attr-defined]
                    VerificationResultModel.parsing_model == result.metadata.parsing_model,  # type: ignore[attr-defined]
                    VerificationResultModel.answering_replicate == result.metadata.answering_replicate,  # type: ignore[attr-defined]
                    VerificationResultModel.parsing_replicate == result.metadata.parsing_replicate,  # type: ignore[attr-defined]
                )
            ).scalar_one_or_none()

            if existing_result:
                # Update existing result
                _update_result_model(existing_result, result)
            else:
                # Create new result
                result_model = _create_result_model(run_id, result)
                session.add(result_model)

        if db_config.auto_commit:
            session.commit()


def load_verification_results(
    db_config: DBConfig,
    run_name: str | None = None,
    run_id: str | None = None,
    benchmark_name: str | None = None,
    question_ids: list[str] | None = None,
    question_id: str | None = None,
    answering_model: str | None = None,
    limit: int | None = None,
    as_dict: bool = True,
) -> dict[str, "VerificationResult"] | list[dict[str, Any]]:
    """Load verification results from the database.

    Args:
        db_config: Database configuration
        run_name: Optional run name to filter by
        run_id: Optional run ID to filter by
        benchmark_name: Optional benchmark name to filter by
        question_ids: Optional list of question IDs to filter by
        question_id: Optional single question ID to filter by
        answering_model: Optional answering model to filter by
        limit: Optional limit on number of results
        as_dict: If True, return dict of VerificationResult keyed by question_id_index.
                 If False, return list of dicts with full data including id and run_id.

    Returns:
        Either a dictionary of VerificationResult objects (as_dict=True)
        or a list of dict representations with metadata (as_dict=False)

    Raises:
        ValueError: If no results found
    """
    with get_session(db_config) as session:
        # Build query
        query: Select[tuple[VerificationResultModel]] = select(VerificationResultModel)  # type: ignore[valid-type]

        # Apply filters
        needs_run_join = False

        if run_id:
            query = query.where(VerificationResultModel.run_id == run_id)  # type: ignore[attr-defined]

        if run_name:
            needs_run_join = True
            query = query.join(VerificationRunModel, VerificationResultModel.run_id == VerificationRunModel.id).where(  # type: ignore[attr-defined]
                VerificationRunModel.run_name == run_name
            )

        if benchmark_name:
            if not needs_run_join:
                query = query.join(VerificationRunModel, VerificationResultModel.run_id == VerificationRunModel.id)  # type: ignore[attr-defined]
            query = query.join(BenchmarkModel, VerificationRunModel.benchmark_id == BenchmarkModel.id).where(
                BenchmarkModel.name == benchmark_name
            )

        if question_ids:
            query = query.where(VerificationResultModel.question_id.in_(question_ids))  # type: ignore[attr-defined]

        if question_id:
            query = query.where(VerificationResultModel.question_id == question_id)  # type: ignore[attr-defined]

        if answering_model:
            query = query.where(VerificationResultModel.metadata_answering_model == answering_model)  # type: ignore[attr-defined]

        # Apply limit if specified
        if limit:
            query = query.limit(limit)

        # Execute query
        results_models = session.execute(query).scalars().all()

        if as_dict:
            # Convert to VerificationResult objects (original behavior)
            results_dict: dict[str, Any] = {}
            for i, result_model in enumerate(results_models):
                result_key = f"{result_model.question_id}_{i}"  # type: ignore[attr-defined]
                results_dict[result_key] = _model_to_verification_result(result_model)
            return results_dict
        else:
            # Return list of dicts with full data
            results_list: list[dict[str, Any]] = []
            for result_model in results_models:
                vr = _model_to_verification_result(result_model)
                result_dict = vr.model_dump()
                result_dict["id"] = result_model.id  # type: ignore[attr-defined]
                result_dict["run_id"] = result_model.run_id  # type: ignore[attr-defined]
                results_list.append(result_dict)
            return results_list


def _create_result_model(run_id: str, result: "VerificationResult") -> VerificationResultModel:  # type: ignore[valid-type]
    """Convert VerificationResult to VerificationResultModel using auto-converter.

    This function uses the auto-generated schema and converters to
    automatically map all fields from the nested Pydantic model to
    the flat SQLAlchemy model.
    """
    # Use auto-converter to flatten Pydantic model to ORM
    # The question_id is needed for the foreign key relationship
    orm_obj = pydantic_to_orm(
        result,
        VerificationResultModel,
        FLATTEN_CONFIG,
        extra_values={
            "run_id": run_id,
            "question_id": result.metadata.question_id,
        },
    )
    return orm_obj


def _update_result_model(model: VerificationResultModel, result: "VerificationResult") -> None:  # type: ignore[valid-type]
    """Update an existing VerificationResultModel with new data using auto-converter.

    This function uses the auto-generated schema and converters to
    automatically update all fields from the nested Pydantic model.
    """
    # Use auto-converter to update ORM model from Pydantic
    # Exclude id, run_id, question_id, and created_at to preserve identity
    update_orm_from_pydantic(
        model,
        result,
        FLATTEN_CONFIG,
        exclude_fields={"id", "run_id", "question_id", "created_at"},
    )


def _model_to_verification_result(model: VerificationResultModel) -> "VerificationResult":  # type: ignore[valid-type]
    """Convert VerificationResultModel to VerificationResult using auto-converter.

    This function uses the auto-generated schema and converters to
    automatically reconstruct the nested Pydantic model from the
    flat SQLAlchemy model.
    """
    from karenina.schemas.workflow import VerificationResult

    # Use auto-converter to reconstruct nested Pydantic model from flat ORM
    return orm_to_pydantic(model, VerificationResult, FLATTEN_CONFIG)


def import_verification_results(
    json_data: dict[str, Any],
    db_config: DBConfig,
    benchmark_name: str,
    run_name: str | None = None,
    source_filename: str | None = None,
) -> tuple[str, int, int]:
    """Import verification results from JSON export format.

    Supports:
    - v2.0 format: {format_version: "2.0", metadata, shared_data, results}
    - Legacy unified format: {metadata, results}
    - Legacy array format: [result1, result2, ...]

    Args:
        json_data: Parsed JSON data from export file
        db_config: Database configuration
        benchmark_name: Target benchmark name (must exist in database)
        run_name: Optional name for the import run (auto-generated if not provided)
        source_filename: Optional source filename for audit

    Returns:
        Tuple of (run_id, imported_count, skipped_count)

    Raises:
        ValueError: If benchmark not found or JSON format unrecognized
    """
    from karenina.schemas.workflow import VerificationResult

    # Initialize database if auto_create is enabled
    if db_config.auto_create:
        init_database(db_config)

    # Detect format version
    format_version = json_data.get("format_version", "1.0")

    # Extract results and metadata based on format
    if format_version == "2.0":
        results_list = json_data.get("results", [])
        metadata = json_data.get("metadata", {})
        shared_data = json_data.get("shared_data", {})
    elif "metadata" in json_data and "results" in json_data:
        # Legacy unified format
        results_list = json_data.get("results", [])
        metadata = json_data.get("metadata", {})
        shared_data = {}
        format_version = "1.0"
    elif isinstance(json_data, list):
        # Legacy array format
        results_list = json_data
        metadata = {}
        shared_data = {}
        format_version = "legacy"
    else:
        raise ValueError(
            "Unrecognized JSON format. Expected v2.0 format with 'format_version' key, "
            "legacy unified format with 'metadata' and 'results' keys, "
            "or legacy array format."
        )

    if not results_list:
        raise ValueError("No results found in JSON data")

    # Generate run_id
    run_id = str(uuid4())

    with get_session(db_config) as session:
        # Verify benchmark exists
        benchmark_model = session.execute(
            select(BenchmarkModel).where(BenchmarkModel.name == benchmark_name)
        ).scalar_one_or_none()

        if not benchmark_model:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in database")

        # Create verification run record
        auto_run_name = run_name or f"import_{run_id[:8]}"
        run_model = VerificationRunModel(
            id=run_id,
            benchmark_id=benchmark_model.id,
            run_name=auto_run_name,
            status="completed",
            config=metadata.get("verification_config", {}),
            total_questions=len(
                {r.get("metadata", {}).get("question_id") for r in results_list if isinstance(r, dict)}
            ),
            processed_count=len(results_list),
            successful_count=0,  # Will be updated below
            failed_count=0,  # Will be updated below
            start_time=None,
            end_time=datetime.now(UTC),
        )
        session.add(run_model)
        session.commit()  # Commit run to satisfy foreign key constraints

        # Import results
        imported_count = 0
        skipped_count = 0
        successful_count = 0
        failed_count = 0

        for result_data in results_list:
            try:
                # Parse result with Pydantic validation
                result = VerificationResult.model_validate(result_data)

                # Create ORM model using auto-converter
                result_model = _create_result_model(run_id, result)
                session.add(result_model)

                imported_count += 1
                if result.metadata.completed_without_errors:
                    successful_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                # Log warning but continue with other results
                print(f"Warning: Failed to import result: {e}")
                skipped_count += 1

        # Update run counts
        run_model.successful_count = successful_count
        run_model.failed_count = failed_count

        # Create import metadata record for audit
        import_metadata = ImportMetadataModel(
            run_id=run_id,
            import_source="json_file",
            source_format_version=format_version,
            source_filename=source_filename,
            source_job_id=metadata.get("job_id"),
            source_export_timestamp=metadata.get("timestamp"),
            source_karenina_version=metadata.get("karenina_version"),
            results_count=imported_count,
            shared_rubric_definition=shared_data.get("rubric"),
        )
        session.add(import_metadata)

        if db_config.auto_commit:
            session.commit()

    return run_id, imported_count, skipped_count
