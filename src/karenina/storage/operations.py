"""Core database operations for Karenina storage.

This module provides high-level functions for saving and loading benchmarks,
verification runs, and results to/from the database.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..benchmark.benchmark import Benchmark
    from ..benchmark.models import VerificationResult

from sqlalchemy import select

from .db_config import DBConfig
from .engine import get_session, init_database
from .models import (
    BenchmarkModel,
    BenchmarkQuestionModel,
    QuestionModel,
    VerificationResultModel,
    VerificationRunModel,
)


def save_benchmark(benchmark: "Benchmark", storage: str | DBConfig, checkpoint_path: Path | None = None) -> "Benchmark":
    """Save a benchmark to the database.

    Args:
        benchmark: The Benchmark instance to save
        storage: Database storage URL (e.g., "sqlite:///example.db") or DBConfig instance
        checkpoint_path: Optional path to the checkpoint file (source of truth)

    Returns:
        The same benchmark instance (for chaining)

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
        questions_data = benchmark._question_manager.get_all_questions()

        for q_data in questions_data:
            # Generate question ID from text (MD5 hash)
            # Note: question text is stored in "question" key, not "text"
            question_text = q_data["question"]
            question_id = hashlib.md5(question_text.encode("utf-8")).hexdigest()

            # Check if question exists
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

            # Create or update benchmark-question association
            answer_template = q_data.get("answer_template", "")
            finished = q_data.get("finished", False)
            keywords = q_data.get("keywords", [])

            # Check if association already exists
            existing_bq = session.execute(
                select(BenchmarkQuestionModel).where(
                    BenchmarkQuestionModel.benchmark_id == benchmark_id,
                    BenchmarkQuestionModel.question_id == question_id,
                )
            ).scalar_one_or_none()

            if existing_bq:
                # Update existing association
                existing_bq.answer_template = answer_template
                existing_bq.original_answer_template = q_data.get("original_answer_template", answer_template)
                existing_bq.finished = finished
                existing_bq.keywords = keywords
                existing_bq.question_rubric = q_data.get("question_rubric")
            else:
                # Create new association
                bq_model = BenchmarkQuestionModel(
                    benchmark_id=benchmark_id,
                    question_id=question_id,
                    answer_template=answer_template,
                    original_answer_template=q_data.get("original_answer_template", answer_template),
                    finished=finished,
                    keywords=keywords,
                    question_rubric=q_data.get("question_rubric"),
                )
                session.add(bq_model)

        if db_config.auto_commit:
            session.commit()

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
    from ..schemas.question_class import Question

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
            # Convert tags list to proper type for Question
            tags_list: list[str | None] = list(question_model.tags) if question_model.tags else []
            question = Question(
                question=question_model.question_text,
                raw_answer=question_model.raw_answer,
                tags=tags_list,
                few_shot_examples=question_model.few_shot_examples,
            )

            # Add question to benchmark with template
            benchmark.add_question(
                question=question,
                answer_template=bq.answer_template,
                finished=bq.finished,
                few_shot_examples=question_model.few_shot_examples,
            )

            # Note: Keywords are stored in benchmark_questions table but not restored
            # because there's no set_keywords method on Benchmark class.
            # They're available via get_question() method.

            # Set question-specific rubric if present
            if bq.question_rubric:
                # Convert JSON rubric back to Rubric object
                from ..schemas.rubric_class import Rubric, RubricTrait

                traits = []
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

                if traits:
                    rubric = Rubric(traits=traits)
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
                successful_count=sum(1 for r in results.values() if r.success),
                failed_count=sum(1 for r in results.values() if not r.success),
                start_time=None,  # These would come from config
                end_time=None,
            )
            session.add(run_model)
        else:
            # Update existing run
            existing_run.processed_count = len(results)
            existing_run.successful_count = sum(1 for r in results.values() if r.success)
            existing_run.failed_count = sum(1 for r in results.values() if not r.success)

        # Save individual results
        for result in results.values():
            # Check if result already exists (to avoid duplicates)
            existing_result = session.execute(
                select(VerificationResultModel).where(
                    VerificationResultModel.run_id == run_id,
                    VerificationResultModel.question_id == result.question_id,
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
        success=result.success,
        error=result.error,
        question_text=result.question_text,
        raw_llm_response=result.raw_llm_response,
        parsed_gt_response=result.parsed_gt_response,
        parsed_llm_response=result.parsed_llm_response,
        verify_result=result.verify_result,
        verify_granular_result=result.verify_granular_result,
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
    )


def _update_result_model(model: VerificationResultModel, result: "VerificationResult") -> None:
    """Update an existing VerificationResultModel with new data."""
    model.success = result.success
    model.error = result.error
    model.question_text = result.question_text
    model.raw_llm_response = result.raw_llm_response
    model.parsed_gt_response = result.parsed_gt_response
    model.parsed_llm_response = result.parsed_llm_response
    model.verify_result = result.verify_result
    model.verify_granular_result = result.verify_granular_result
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


def _model_to_verification_result(model: VerificationResultModel) -> "VerificationResult":
    """Convert VerificationResultModel to VerificationResult."""
    from ..benchmark.models import VerificationResult

    return VerificationResult(
        question_id=model.question_id,
        success=model.success,
        error=model.error,
        question_text=model.question_text,
        raw_llm_response=model.raw_llm_response,
        parsed_gt_response=model.parsed_gt_response,
        parsed_llm_response=model.parsed_llm_response,
        verify_result=model.verify_result,
        verify_granular_result=model.verify_granular_result,
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
    )
