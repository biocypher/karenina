"""Core database operations for Karenina storage.

This module provides high-level functions for saving and loading benchmarks,
verification runs, and results to/from the database.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ...schemas.workflow import VerificationResult
    from ..benchmark.benchmark import Benchmark

from sqlalchemy import select

from ..utils.checkpoint import generate_template_id
from .db_config import DBConfig
from .engine import get_session, init_database
from .models import (
    BenchmarkModel,
    BenchmarkQuestionModel,
    QuestionModel,
    VerificationResultModel,
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
                metadata_json={},
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
            # The benchmark cache stores rubrics as list of Pydantic RubricTrait/ManualRubricTrait objects,
            # but the database expects a JSON dict format with separate 'traits' and 'manual_traits' lists
            question_rubric_dict = None
            if q_data.get("question_rubric"):
                try:
                    from ..schemas.domain import ManualRubricTrait, RubricTrait

                    rubric_traits = q_data["question_rubric"]
                    if isinstance(rubric_traits, list) and len(rubric_traits) > 0:
                        # Separate traits by type
                        llm_traits = [t for t in rubric_traits if isinstance(t, RubricTrait)]
                        manual_traits = [t for t in rubric_traits if isinstance(t, ManualRubricTrait)]

                        question_rubric_dict = {
                            "traits": [trait.model_dump() for trait in llm_traits],
                            "manual_traits": [trait.model_dump() for trait in manual_traits],
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
            benchmark.add_question(
                question=question,
                answer_template=bq.answer_template,
                finished=bq.finished,
                few_shot_examples=question_model.few_shot_examples,
            )

            # Set question-specific rubric if present
            if bq.question_rubric:
                # Convert JSON rubric back to Rubric object
                from ..schemas.domain import ManualRubricTrait, Rubric, RubricTrait

                traits = []
                manual_traits = []

                # Deserialize LLM-based traits
                for trait_data in bq.question_rubric.get("traits", []):
                    # Determine kind from trait data
                    kind = trait_data.get("kind", "score")
                    trait = RubricTrait(
                        name=trait_data["name"],
                        description=trait_data.get("description"),
                        kind=kind,
                        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                    )
                    traits.append(trait)

                # Deserialize manual (regex) traits
                for manual_trait_data in bq.question_rubric.get("manual_traits", []):
                    manual_trait = ManualRubricTrait(
                        name=manual_trait_data["name"],
                        description=manual_trait_data.get("description"),
                        pattern=manual_trait_data.get("pattern"),
                        callable_name=manual_trait_data.get("callable_name"),
                        case_sensitive=manual_trait_data.get("case_sensitive", True),
                        invert_result=manual_trait_data.get("invert_result", False),
                    )
                    manual_traits.append(manual_trait)

                if traits or manual_traits:
                    rubric = Rubric(traits=traits, manual_traits=manual_traits)
                    benchmark.set_question_rubric(bq.question_id, rubric)

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
                total_questions=len({r.question_id for r in results.values()}),
                processed_count=len(results),
                successful_count=sum(1 for r in results.values() if r.completed_without_errors),
                failed_count=sum(1 for r in results.values() if not r.completed_without_errors),
                start_time=None,  # These would come from config
                end_time=None,
            )
            session.add(run_model)
            # Commit the run immediately to satisfy foreign key constraints for results
            session.commit()
        else:
            # Update existing run
            existing_run.processed_count = len(results)
            existing_run.successful_count = sum(1 for r in results.values() if r.completed_without_errors)
            existing_run.failed_count = sum(1 for r in results.values() if not r.completed_without_errors)
            # Commit the updated run
            session.commit()

        # Save individual results
        for result in results.values():
            # Check if result already exists (to avoid duplicates)
            # Include template_id in the uniqueness check (composite key component)
            existing_result = session.execute(
                select(VerificationResultModel).where(
                    VerificationResultModel.run_id == run_id,
                    VerificationResultModel.question_id == result.question_id,
                    VerificationResultModel.template_id == result.template_id,
                    VerificationResultModel.answering_model == result.answering_model,
                    VerificationResultModel.parsing_model == result.parsing_model,
                    VerificationResultModel.answering_replicate == result.answering_replicate,
                    VerificationResultModel.parsing_replicate == result.parsing_replicate,
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
    benchmark_name: str | None = None,
    question_ids: list[str] | None = None,
) -> dict[str, "VerificationResult"]:
    """Load verification results from the database.

    Args:
        db_config: Database configuration
        run_name: Optional run name to filter by
        benchmark_name: Optional benchmark name to filter by
        question_ids: Optional list of question IDs to filter by

    Returns:
        Dictionary of verification results

    Raises:
        ValueError: If no results found
    """
    # Import here to avoid circular imports

    with get_session(db_config) as session:
        # Build query
        query = select(VerificationResultModel)

        # Apply filters
        if run_name:
            # Join with runs table to filter by run_name
            query = query.join(VerificationRunModel, VerificationResultModel.run_id == VerificationRunModel.id).where(
                VerificationRunModel.run_name == run_name
            )

        if benchmark_name:
            # Additional join to filter by benchmark
            if not run_name:
                query = query.join(VerificationRunModel, VerificationResultModel.run_id == VerificationRunModel.id)
            query = query.join(BenchmarkModel, VerificationRunModel.benchmark_id == BenchmarkModel.id).where(
                BenchmarkModel.name == benchmark_name
            )

        if question_ids:
            query = query.where(VerificationResultModel.question_id.in_(question_ids))

        # Execute query
        results_models = session.execute(query).scalars().all()

        # Convert to VerificationResult objects
        results = {}
        for i, result_model in enumerate(results_models):
            result_key = f"{result_model.question_id}_{i}"
            results[result_key] = _model_to_verification_result(result_model)

        return results


def _create_result_model(run_id: str, result: "VerificationResult") -> VerificationResultModel:
    """Convert VerificationResult to VerificationResultModel."""
    return VerificationResultModel(
        run_id=run_id,
        question_id=result.question_id,
        template_id=result.template_id,
        completed_without_errors=result.completed_without_errors,
        error=result.error,
        question_text=result.question_text,
        raw_llm_response=result.raw_llm_response,
        parsed_gt_response=result.parsed_gt_response,
        parsed_llm_response=result.parsed_llm_response,
        template_verification_performed=result.template_verification_performed,
        verify_result=result.verify_result,
        verify_granular_result=result.verify_granular_result,
        rubric_evaluation_performed=result.rubric_evaluation_performed,
        verify_rubric=result.verify_rubric,
        evaluation_rubric=result.evaluation_rubric,
        keywords=result.keywords,
        answering_model=result.answering_model,
        parsing_model=result.parsing_model,
        answering_system_prompt=result.answering_system_prompt,
        parsing_system_prompt=result.parsing_system_prompt,
        execution_time=result.execution_time,
        timestamp=result.timestamp,
        job_id=result.job_id,
        answering_replicate=result.answering_replicate,
        parsing_replicate=result.parsing_replicate,
        embedding_check_performed=result.embedding_check_performed,
        embedding_similarity_score=result.embedding_similarity_score,
        embedding_override_applied=result.embedding_override_applied,
        embedding_model_used=result.embedding_model_used,
        regex_validations_performed=result.regex_validations_performed,
        regex_validation_results=result.regex_validation_results,
        regex_validation_details=result.regex_validation_details,
        regex_overall_success=result.regex_overall_success,
        regex_extraction_results=result.regex_extraction_results,
        recursion_limit_reached=result.recursion_limit_reached,
        answering_mcp_servers=result.answering_mcp_servers,
        abstention_check_performed=result.abstention_check_performed,
        abstention_detected=result.abstention_detected,
        abstention_override_applied=result.abstention_override_applied,
        abstention_reasoning=result.abstention_reasoning,
        # Deep-judgment fields
        deep_judgment_enabled=result.deep_judgment_enabled,
        deep_judgment_performed=result.deep_judgment_performed,
        extracted_excerpts=result.extracted_excerpts,
        attribute_reasoning=result.attribute_reasoning,
        deep_judgment_stages_completed=result.deep_judgment_stages_completed,
        deep_judgment_model_calls=result.deep_judgment_model_calls,
        deep_judgment_excerpt_retry_count=result.deep_judgment_excerpt_retry_count,
        attributes_without_excerpts=result.attributes_without_excerpts,
        # Search-enhanced deep-judgment fields
        deep_judgment_search_enabled=result.deep_judgment_search_enabled,
        hallucination_risk_assessment=result.hallucination_risk_assessment,
        # Metric trait fields
        metric_trait_confusion_lists=result.metric_trait_confusion_lists,
        metric_trait_metrics=result.metric_trait_metrics,
        # LLM usage tracking fields
        usage_metadata=result.usage_metadata,
        agent_metrics=result.agent_metrics,
    )


def _update_result_model(model: VerificationResultModel, result: "VerificationResult") -> None:
    """Update an existing VerificationResultModel with new data."""
    model.completed_without_errors = result.completed_without_errors
    model.error = result.error
    model.question_text = result.question_text
    model.raw_llm_response = result.raw_llm_response
    model.parsed_gt_response = result.parsed_gt_response
    model.parsed_llm_response = result.parsed_llm_response
    model.template_verification_performed = result.template_verification_performed
    model.verify_result = result.verify_result
    model.verify_granular_result = result.verify_granular_result
    model.rubric_evaluation_performed = result.rubric_evaluation_performed
    model.verify_rubric = result.verify_rubric
    model.evaluation_rubric = result.evaluation_rubric
    model.keywords = result.keywords
    model.execution_time = result.execution_time
    model.timestamp = result.timestamp
    model.embedding_check_performed = result.embedding_check_performed
    model.embedding_similarity_score = result.embedding_similarity_score
    model.embedding_override_applied = result.embedding_override_applied
    model.embedding_model_used = result.embedding_model_used
    model.regex_validations_performed = result.regex_validations_performed
    model.regex_validation_results = result.regex_validation_results
    model.regex_validation_details = result.regex_validation_details
    model.regex_overall_success = result.regex_overall_success
    model.regex_extraction_results = result.regex_extraction_results
    model.recursion_limit_reached = result.recursion_limit_reached
    model.answering_mcp_servers = result.answering_mcp_servers
    model.abstention_check_performed = result.abstention_check_performed
    model.abstention_detected = result.abstention_detected
    model.abstention_override_applied = result.abstention_override_applied
    model.abstention_reasoning = result.abstention_reasoning
    # Deep-judgment fields
    model.deep_judgment_enabled = result.deep_judgment_enabled
    model.deep_judgment_performed = result.deep_judgment_performed
    model.extracted_excerpts = result.extracted_excerpts
    model.attribute_reasoning = result.attribute_reasoning
    model.deep_judgment_stages_completed = result.deep_judgment_stages_completed
    model.deep_judgment_model_calls = result.deep_judgment_model_calls
    model.deep_judgment_excerpt_retry_count = result.deep_judgment_excerpt_retry_count
    model.attributes_without_excerpts = result.attributes_without_excerpts
    # Search-enhanced deep-judgment fields
    model.deep_judgment_search_enabled = result.deep_judgment_search_enabled
    model.hallucination_risk_assessment = result.hallucination_risk_assessment
    # Metric trait fields
    model.metric_trait_confusion_lists = result.metric_trait_confusion_lists
    model.metric_trait_metrics = result.metric_trait_metrics
    # LLM usage tracking fields
    model.usage_metadata = result.usage_metadata
    model.agent_metrics = result.agent_metrics


def _model_to_verification_result(model: VerificationResultModel) -> "VerificationResult":
    """Convert VerificationResultModel to VerificationResult."""
    from karenina.schemas.workflow import VerificationResult

    return VerificationResult(
        question_id=model.question_id,
        template_id=model.template_id,
        completed_without_errors=model.completed_without_errors,
        error=model.error,
        question_text=model.question_text,
        raw_llm_response=model.raw_llm_response,
        parsed_gt_response=model.parsed_gt_response,
        parsed_llm_response=model.parsed_llm_response,
        template_verification_performed=model.template_verification_performed,
        verify_result=model.verify_result,
        verify_granular_result=model.verify_granular_result,
        rubric_evaluation_performed=model.rubric_evaluation_performed,
        verify_rubric=model.verify_rubric,
        evaluation_rubric=model.evaluation_rubric,
        keywords=model.keywords,
        answering_model=model.answering_model,
        parsing_model=model.parsing_model,
        answering_system_prompt=model.answering_system_prompt,
        parsing_system_prompt=model.parsing_system_prompt,
        execution_time=model.execution_time,
        timestamp=model.timestamp,
        run_name=model.run.run_name if model.run else None,
        job_id=model.job_id,
        answering_replicate=model.answering_replicate,
        parsing_replicate=model.parsing_replicate,
        embedding_check_performed=model.embedding_check_performed,
        embedding_similarity_score=model.embedding_similarity_score,
        embedding_override_applied=model.embedding_override_applied,
        embedding_model_used=model.embedding_model_used,
        regex_validations_performed=model.regex_validations_performed,
        regex_validation_results=model.regex_validation_results,
        regex_validation_details=model.regex_validation_details,
        regex_overall_success=model.regex_overall_success,
        regex_extraction_results=model.regex_extraction_results,
        recursion_limit_reached=model.recursion_limit_reached,
        answering_mcp_servers=model.answering_mcp_servers,
        abstention_check_performed=model.abstention_check_performed,
        abstention_detected=model.abstention_detected,
        abstention_override_applied=model.abstention_override_applied,
        abstention_reasoning=model.abstention_reasoning,
        # Deep-judgment fields
        deep_judgment_enabled=model.deep_judgment_enabled,
        deep_judgment_performed=model.deep_judgment_performed,
        extracted_excerpts=model.extracted_excerpts,
        attribute_reasoning=model.attribute_reasoning,
        deep_judgment_stages_completed=model.deep_judgment_stages_completed,
        deep_judgment_model_calls=model.deep_judgment_model_calls,
        deep_judgment_excerpt_retry_count=model.deep_judgment_excerpt_retry_count,
        attributes_without_excerpts=model.attributes_without_excerpts,
        # Search-enhanced deep-judgment fields
        deep_judgment_search_enabled=model.deep_judgment_search_enabled,
        hallucination_risk_assessment=model.hallucination_risk_assessment,
        # Metric trait fields
        metric_trait_confusion_lists=model.metric_trait_confusion_lists,
        metric_trait_metrics=model.metric_trait_metrics,
        # LLM usage tracking fields
        usage_metadata=model.usage_metadata,
        agent_metrics=model.agent_metrics,
    )
