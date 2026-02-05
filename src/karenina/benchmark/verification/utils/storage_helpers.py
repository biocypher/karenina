"""Storage helper functions for verification.

This module provides database auto-save functionality for verification results.
"""

import logging
from typing import Any

from ....schemas.verification import FinishedTemplate, VerificationResult

logger = logging.getLogger(__name__)


def auto_save_results(
    results: dict[str, VerificationResult],
    templates: list[FinishedTemplate],
    storage_url: str,
    benchmark_name: str,
    run_name: str,
    config_dict: dict[str, Any],
    run_id: str,
) -> None:
    """Auto-save verification results to database.

    Args:
        results: Dictionary of verification results
        templates: List of templates that were verified
        storage_url: Database URL for storage
        benchmark_name: Name of the benchmark
        run_name: Name of this verification run
        config_dict: Configuration dictionary for this run
        run_id: Unique identifier for this run
    """
    try:
        from ....benchmark import Benchmark
        from ....storage import DBConfig, get_benchmark_summary, save_benchmark, save_verification_results

        # Create database config
        db_config = DBConfig(storage_url=storage_url)

        # Check if benchmark already exists
        existing_benchmarks = get_benchmark_summary(db_config, benchmark_name=benchmark_name)

        if not existing_benchmarks:
            # Benchmark doesn't exist, create it
            logger.info(f"Creating new benchmark '{benchmark_name}' in database")
            benchmark = Benchmark.create(
                name=benchmark_name,
                description=f"Auto-created for verification run: {run_name}",
                version="1.0.0",
            )

            # Add questions from templates
            for template in templates:
                # Add question using text format to ensure question_id is preserved
                benchmark.add_question(
                    question=template.question_text,
                    raw_answer="[Placeholder - see template]",
                    answer_template=template.template_code,
                    question_id=template.question_id,  # Explicitly set question_id to match template
                )

            # Save benchmark to database
            save_benchmark(benchmark, db_config)

        # Save verification results
        save_verification_results(
            results=results,
            db_config=db_config,
            run_id=run_id,
            benchmark_name=benchmark_name,
            run_name=run_name,
            config=config_dict,
        )

        logger.info(f"Auto-saved {len(results)} results to {storage_url} (benchmark: {benchmark_name})")

    except Exception as e:
        logger.error(f"Auto-save failed: {e}")
        # Don't raise - auto-save failure shouldn't fail the verification
