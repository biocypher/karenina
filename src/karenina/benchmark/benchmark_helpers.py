"""Extracted helper functions for the Benchmark facade.

These functions contain the business logic that was previously embedded
in the Benchmark class methods. The Benchmark class now delegates to
these functions, keeping the facade thin.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..integrations.gepa import FrontierType, KareninaOutput, ObjectiveConfig
    from ..schemas.checkpoint import SchemaOrgQuestion
    from ..schemas.workflow import VerificationConfig, VerificationResult, VerificationResultSet
    from .benchmark import Benchmark

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def generate_template_for_question(
    benchmark: Benchmark,
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
    from .authoring.answers.generator import generate_answer_template

    if question_id not in benchmark._questions_cache:
        raise ValueError(f"Question not found: {question_id}")

    if not force_regenerate and benchmark.has_template(question_id):
        return {
            "success": True,
            "template_code": benchmark.get_template(question_id),
            "error": "Template already exists (use force_regenerate=True to override)",
            "raw_response": None,
            "skipped": True,
        }

    try:
        question_data = benchmark._questions_cache[question_id]
        question_text = question_data.get("question", "")
        raw_answer = question_data.get("raw_answer", "")

        template_code = generate_answer_template(
            question=question_text,
            raw_answer=raw_answer,
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            interface=interface,
            endpoint_base_url=endpoint_base_url,
            endpoint_api_key=endpoint_api_key,
        )

        if not template_code.strip():
            return {
                "success": False,
                "template_code": "",
                "error": "No valid code blocks found in LLM response",
                "raw_response": template_code,
                "skipped": False,
            }

        benchmark.add_answer_template(question_id, template_code)

        return {
            "success": True,
            "template_code": template_code,
            "error": None,
            "raw_response": template_code,
            "skipped": False,
        }

    except Exception as e:
        logger.warning("Template generation failed for question %s: %s", question_id, e, exc_info=True)
        return {
            "success": False,
            "template_code": "",
            "error": str(e),
            "raw_response": None,
            "skipped": False,
        }


def generate_templates(
    benchmark: Benchmark,
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
    invalid_ids = [qid for qid in question_ids if qid not in benchmark._questions_cache]
    if invalid_ids:
        raise ValueError(f"Questions not found: {invalid_ids}")

    results = {}
    total_questions = len(question_ids)

    for i, question_id in enumerate(question_ids):
        if progress_callback:
            percentage = (i / total_questions) * 100
            question_text = benchmark._questions_cache[question_id].get("question", "")
            message = f"Processing: {question_text[:50]}..."
            progress_callback(percentage, message)

        result = generate_template_for_question(
            benchmark,
            question_id=question_id,
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            interface=interface,
            force_regenerate=force_regenerate,
            endpoint_base_url=endpoint_base_url,
            endpoint_api_key=endpoint_api_key,
        )
        results[question_id] = result

    if progress_callback:
        progress_callback(100.0, "Template generation completed")

    return results


def generate_all_templates(
    benchmark: Benchmark,
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
    if only_missing and not force_regenerate:
        from typing import cast

        question_ids = cast(list[str], benchmark.get_missing_templates(ids_only=True))
    else:
        question_ids = benchmark.get_question_ids()

    if not question_ids:
        return {}

    return generate_templates(
        benchmark,
        question_ids=question_ids,
        model=model,
        model_provider=model_provider,
        temperature=temperature,
        interface=interface,
        force_regenerate=force_regenerate,
        progress_callback=progress_callback,
        endpoint_base_url=endpoint_base_url,
        endpoint_api_key=endpoint_api_key,
    )


def export_generated_templates(benchmark: Benchmark, file_path: Path) -> None:
    """Export all generated templates to a JSON file."""
    templates_dict = {}

    for question_id in benchmark.get_question_ids():
        if benchmark.has_template(question_id):
            templates_dict[question_id] = benchmark.get_template(question_id)

    with file_path.open("w") as f:
        json.dump(templates_dict, f, indent=2)


def import_generated_templates(benchmark: Benchmark, file_path: Path, force_overwrite: bool = False) -> dict[str, bool]:
    """Import templates from a JSON file generated by export_generated_templates."""
    from .authoring.answers.generator import load_answer_templates_from_json

    answer_templates = load_answer_templates_from_json(str(file_path), return_blocks=True)
    if isinstance(answer_templates, tuple):
        _, code_blocks = answer_templates
    else:
        raise ValueError("Unable to load code blocks from JSON file")

    results = {}

    for question_id, template_code in code_blocks.items():
        if question_id not in benchmark._questions_cache:
            results[question_id] = False
            continue

        if not force_overwrite and benchmark.has_template(question_id):
            results[question_id] = False
            continue

        try:
            benchmark.add_answer_template(question_id, template_code)
            results[question_id] = True
        except Exception:
            logger.warning("Failed to import template for question %s", question_id, exc_info=True)
            results[question_id] = False

    return results


# ---------------------------------------------------------------------------
# GEPA optimization
# ---------------------------------------------------------------------------


def run_optimize(
    benchmark: Benchmark,
    targets: list[str],
    config: VerificationConfig | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float | None = None,
    seed: int | None = None,
    reflection_model: str = "openai/gpt-4o",
    max_metric_calls: int = 150,
    objective_config: ObjectiveConfig | None = None,
    frontier_type: FrontierType = "objective",
    seed_prompts: dict[str, str] | None = None,
    tracker_path: Path | str | None = None,
    export_preset_path: Path | str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    verbose: bool = False,
) -> KareninaOutput:
    """Run GEPA optimization on the benchmark."""
    try:
        from karenina.integrations.gepa import (
            GEPA_AVAILABLE,
            KareninaAdapter,
            KareninaOutput,
            ObjectiveConfig,
            OptimizationRun,
            OptimizationTarget,
            OptimizationTracker,
            VerboseLogger,
            export_to_preset,
            split_benchmark,
        )
    except ImportError as e:
        raise ImportError("GEPA integration components not available. Install with: pip install karenina[gepa]") from e

    if not GEPA_AVAILABLE:
        raise ImportError("gepa package is required for optimization. Install with: pip install gepa")

    try:
        import gepa
    except ImportError as e:
        raise ImportError("gepa package is required for optimization. Install with: pip install gepa") from e

    # Validate targets
    valid_targets = {t.value for t in OptimizationTarget}
    for target in targets:
        if target not in valid_targets:
            raise ValueError(f"Invalid target '{target}'. Valid targets: {valid_targets}")

    opt_targets = [OptimizationTarget(t) for t in targets]

    # Create default config if not provided
    if config is None:
        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        config = VerificationConfig(
            answering_models=[ModelConfig(id="answerer-gpt4o", model_name="gpt-4o", model_provider="openai")],
            parsing_models=[ModelConfig(id="parser-gpt4o-mini", model_name="gpt-4o-mini", model_provider="openai")],
            evaluation_mode="template_only",
        )

    # Split benchmark
    split = split_benchmark(
        benchmark,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    if progress_callback:
        progress_callback(5.0, f"Split benchmark: {len(split.train)} train, {len(split.val)} val")

    # Create adapter with multi-objective config
    if objective_config is None:
        objective_config = ObjectiveConfig()
    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=config,
        targets=opt_targets,
        objective_config=objective_config,
    )

    # Prepare seed candidate
    seed_candidate = seed_prompts or {}
    for target in targets:
        if target not in seed_candidate:
            seed_candidate[target] = ""

    if progress_callback:
        progress_callback(10.0, "Starting GEPA optimization...")

    # Create verbose logger if enabled
    verbose_logger: VerboseLogger | None = None
    if verbose:
        verbose_logger = VerboseLogger(max_iterations=max_metric_calls)

    # Run GEPA optimization
    result: Any = gepa.optimize(  # type: ignore[attr-defined]
        seed_candidate=seed_candidate,
        trainset=split.train,
        valset=split.val,
        adapter=adapter,
        reflection_lm=reflection_model,
        max_metric_calls=max_metric_calls,
        frontier_type=frontier_type,
        logger=verbose_logger,
        display_progress_bar=verbose,
    )

    if verbose_logger:
        verbose_logger.print_summary()

    # Build output from GEPAResult
    optimized_prompts = result.best_candidate if hasattr(result, "best_candidate") else {}
    best_idx = result.best_idx if hasattr(result, "best_idx") else 0
    val_score = result.val_aggregate_scores[best_idx] if result.val_aggregate_scores else 0.0
    baseline_score = result.val_aggregate_scores[0] if result.val_aggregate_scores else 0.0
    train_score = val_score

    improvement = (val_score - baseline_score) / baseline_score if baseline_score > 0 else val_score

    # Run test set evaluation if available
    test_score: float | None = None
    if split.test:
        if progress_callback:
            progress_callback(90.0, "Evaluating on test set...")

        test_results = adapter.evaluate(split.test, optimized_prompts, capture_traces=False)
        test_score = sum(test_results.scores) / len(test_results.scores) if test_results.scores else 0.0

    output = KareninaOutput(
        answering_system_prompt=optimized_prompts.get("answering_system_prompt"),
        parsing_instructions=optimized_prompts.get("parsing_instructions"),
        mcp_tool_descriptions={k[9:]: v for k, v in optimized_prompts.items() if k.startswith("mcp_tool_")} or None,
        train_score=train_score,
        val_score=val_score,
        test_score=test_score,
        improvement=improvement,
    )

    # Track results if path provided
    if tracker_path:
        tracker = OptimizationTracker(tracker_path)
        run = OptimizationRun(
            benchmark_name=benchmark.name,
            targets=targets,
            seed_prompts=seed_prompts or {},
            optimized_prompts=optimized_prompts,
            train_score=train_score,
            val_score=val_score,
            test_score=test_score,
            improvement=improvement,
            reflection_model=reflection_model,
            metric_calls=max_metric_calls,
            best_generation=getattr(result, "best_generation", 0),
            total_generations=getattr(result, "total_generations", 0),
        )
        tracker.log_run(run)

    # Export preset if path provided
    if export_preset_path:
        export_to_preset(
            optimized_prompts,
            config,
            Path(export_preset_path),
            opt_targets,
        )

    if progress_callback:
        progress_callback(100.0, f"Optimization complete. Improvement: {improvement:.1%}")

    return output


# ---------------------------------------------------------------------------
# Results storage
# ---------------------------------------------------------------------------


def store_verification_results(
    benchmark: Benchmark,
    results: VerificationResultSet | dict[str, VerificationResult],
    run_name: str | None = None,
) -> None:
    """Store verification results in the benchmark metadata."""
    from ..schemas.workflow import VerificationResultSet

    if isinstance(results, VerificationResultSet):
        results_dict: dict[str, Any] = {}
        for i, result in enumerate(results):
            key = f"{result.metadata.question_id}_{result.metadata.answering_model}_{result.metadata.parsing_model}"
            if result.metadata.replicate is not None:
                key += f"_rep{result.metadata.replicate}"
            if result.metadata.timestamp:
                key += f"_{result.metadata.timestamp}"
            if key in results_dict:
                key += f"_{i}"
            results_dict[key] = result
        benchmark._results_manager.store_verification_results(results_dict, run_name)
    else:
        benchmark._results_manager.store_verification_results(results, run_name)


# ---------------------------------------------------------------------------
# __repr__ formatting
# ---------------------------------------------------------------------------


def build_repr(benchmark: Benchmark) -> str:
    """Build the developer-friendly repr string for a Benchmark."""
    from ..schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait

    lines = ["Benchmark("]

    # === METADATA ===
    lines.append("  === METADATA ===")
    lines.append(f"  Name: {benchmark._base.name}")
    lines.append(f"  Version: {benchmark._base.version}")
    lines.append(f"  Creator: {benchmark._base.creator}")
    lines.append(f"  Created: {benchmark._base.created_at}")
    lines.append(f"  Modified: {benchmark._base.modified_at}")

    # Collect unique keywords
    all_questions_data = benchmark.get_all_questions(ids_only=False)
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
    summary = benchmark._export_manager.get_summary()
    progress = summary["progress_percentage"]
    lines.append(
        f"  Questions: {benchmark._base.question_count} total, "
        f"{benchmark._base.finished_count} finished ({progress:.1f}% complete)"
    )
    lines.append(f"  Templates: {summary['has_template_count']}/{benchmark._base.question_count} questions")

    # === RUBRICS ===
    lines.append("")
    lines.append("  === RUBRICS ===")
    global_rubric = benchmark._rubric_manager.get_global_rubric()

    if global_rubric:
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
        question_rubric = benchmark._rubric_manager.get_question_rubric(q["id"])
        if question_rubric and len(question_rubric) > 0:
            questions_with_rubrics.append(q["id"])
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
    if benchmark._base.question_count == 0:
        lines.append("  === QUESTIONS ===")
        lines.append("  (empty benchmark)")
    else:
        display_count = min(3, len(all_questions_data))
        lines.append(f"  === QUESTIONS (showing {display_count} of {benchmark._base.question_count}) ===")

        for idx, q_item in enumerate(all_questions_data[:display_count], 1):
            if not isinstance(q_item, dict):
                continue

            question_text = q_item.get("question", "")
            if len(question_text) > 80:
                question_text = question_text[:77] + "..."

            keywords = q_item.get("keywords", [])
            keywords_str = f" [{', '.join(keywords)}]" if keywords else ""

            lines.append(f"  {idx}. {question_text}{keywords_str}")

            raw_answer = q_item.get("raw_answer", "")
            if raw_answer:
                if len(raw_answer) > 80:
                    raw_answer = raw_answer[:77] + "..."
                lines.append(f"     → {raw_answer}")
            else:
                lines.append("     → (no answer yet)")

            if idx < display_count:
                lines.append("")

        if benchmark._base.question_count > display_count:
            remaining = benchmark._base.question_count - display_count
            lines.append(f"  ... ({remaining} more)")

    # === READINESS ===
    if not benchmark._base.is_complete:
        lines.append("")
        lines.append("  === READINESS ===")
        readiness = benchmark._export_manager.check_readiness()

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


# ---------------------------------------------------------------------------
# SchemaOrgQuestion conversion
# ---------------------------------------------------------------------------


def convert_to_schema_org_question(question_data: dict[str, Any]) -> SchemaOrgQuestion:
    """Convert internal question dictionary to SchemaOrgQuestion object."""
    from ..schemas.checkpoint import (
        SchemaOrgAnswer,
        SchemaOrgPropertyValue,
        SchemaOrgQuestion,
        SchemaOrgSoftwareSourceCode,
    )
    from ..utils.checkpoint import convert_rubric_trait_to_rating

    accepted_answer = SchemaOrgAnswer.model_validate(
        {"@id": f"{question_data['id']}-answer", "text": question_data["raw_answer"]}
    )

    has_part = SchemaOrgSoftwareSourceCode.model_validate(
        {
            "@id": f"{question_data['id']}-template",
            "name": f"{question_data['question'][:30]}... Answer Template",
            "text": question_data.get("answer_template", ""),
        }
    )

    ratings = None
    if question_data.get("question_rubric"):
        ratings = [
            convert_rubric_trait_to_rating(trait, "question-specific") for trait in question_data["question_rubric"]
        ]

    additional_properties = []
    if question_data.get("finished") is not None:
        additional_properties.append(SchemaOrgPropertyValue(name="finished", value=question_data["finished"]))

    if question_data.get("custom_metadata"):
        for key, value in question_data["custom_metadata"].items():
            additional_properties.append(SchemaOrgPropertyValue(name=f"custom_{key}", value=value))

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
