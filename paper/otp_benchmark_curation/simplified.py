"""Turn an expert-authored question table into a draft Karenina benchmark.

Run from the Karenina repository root:

    uv run python -m paper.otp_benchmark_curation.simplified \
        --source /path/to/open_targets_questions.xlsx

The full source table is processed by default. Pass ``--limit 1`` for a small
live check. The source spreadsheet remains outside the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from karenina.benchmark import Benchmark, TemplateProgressEvent, export_curation_workbook
from karenina.benchmark.authoring.questions import extract_questions_from_file
from paper.common.bootstrap import bootstrap
from paper.config import BENCHMARK_CURATION_OUTPUT_DIR

DEFAULT_MODEL = "claude-opus-4-6"


def progress(event: TemplateProgressEvent) -> None:
    """Print concise progress from Karenina's bulk template generator."""

    if event.event == "job_started":
        print(f"Drafting {event.total_count} answer templates")
    elif event.event == "task_completed":
        print(f"[{event.processed_count}/{event.total_count}] drafted {event.question_id}")
    elif event.event == "task_failed":
        print(f"[{event.processed_count}/{event.total_count}] failed {event.question_id}: {event.error}")
    elif event.event == "job_completed":
        print(f"Drafting complete: {event.successful_count} succeeded, {event.failed_count} failed")


def build_draft_benchmark(
    source: Path,
    output_dir: Path,
    *,
    sheet_name: str | None = None,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    """Import source rows, draft templates, and save unfinished review files."""

    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source table not found: {source}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    existing_names = {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
    completed_outputs = {"draft_benchmark.jsonld", "curation_review.xlsx"}
    if existing_names & completed_outputs or existing_names - {"template_backup.json"}:
        raise FileExistsError(
            f"Output directory contains completed or unrelated artifacts: {output_dir}. "
            "Choose another path so previous model-generated drafts remain inspectable."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if "template_backup.json" in existing_names:
        print(f"Resuming template drafting from {output_dir / 'template_backup.json'}")

    # ## 1. Import the expert-authored table
    # This Karenina function accepts Excel, CSV, and TSV files. The
    # question, reference answer, area, subcategory, source ID, and complexity
    # columns become Question data and metadata.
    questions = extract_questions_from_file(
        file_path=str(source),
        question_column="Question",
        answer_column="Answer",
        sheet_name=sheet_name,
        keywords_columns=[
            {"column": "Area", "separator": ","},
            {"column": "Subcategories", "separator": ","},
        ],
        custom_metadata_columns=["id", "Area", "Subcategories", "Complexity"],
    )
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError("The source table contains no usable question and answer rows")
    print(f"[1] imported {len(questions)} expert-authored question rows")

    # ## 2. Create unfinished benchmark items
    # A successful LLM draft is not curator approval. The items therefore stay
    # unfinished until a domain expert reviews them in Karenina's interface.
    benchmark = Benchmark.create(name="Open Targets Platform benchmark")
    question_ids = benchmark.add_questions(questions, finished=False)
    print(f"[2] created {len(question_ids)} unfinished benchmark items")

    # ## 3. Draft answer templates with the LLM
    # For each item Karenina makes three sequential drafting
    # calls: a field plan, a reference-value specification, and judge-facing
    # instructions. Progressive backup makes an interrupted run resumable.
    print(f"[3] calling {model} to draft answer templates")
    results = benchmark.generate_all_templates(
        model=model,
        model_provider="anthropic",
        temperature=0.0,
        interface="langchain",
        progressive_backup=True,
        backup_path=output_dir / "template_backup.json",
        progress_callback=progress,
        max_workers=1,
    )
    failed = {
        question_id: result.get("error") or "unknown drafting failure"
        for question_id, result in results.items()
        if not result.get("success")
    }

    # ## 4. Validate and save the draft
    # Template validation is deterministic. It checks generated Python
    # structure but deliberately does not mark items as approved.
    all_valid, validation_errors = benchmark.validate_templates()
    if not all_valid:
        raise RuntimeError(f"Generated template validation failed: {validation_errors}")
    checkpoint = output_dir / "draft_benchmark.jsonld"
    benchmark.save(checkpoint)
    workbook = export_curation_workbook(benchmark, output_dir / "curation_review.xlsx")

    # ## 5. Hand the draft to the curator
    # The workbook exposes reference values and judge instructions for review.
    # Approval remains a human action in Karenina's graphical interface.
    if benchmark.get_finished_questions(ids_only=True):
        raise RuntimeError("Freshly drafted items were incorrectly marked finished")
    print(f"[4] saved unfinished benchmark: {checkpoint}")
    print(f"[5] saved curation workbook: {workbook.output_path}")
    print(f"[5] fields for expert review: {workbook.field_count}")
    if failed:
        print(f"[5] drafting failures requiring regeneration: {len(failed)}")
    return checkpoint, workbook.output_path


def main() -> None:
    """Run the spreadsheet-to-draft-benchmark walkthrough."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Private Excel or CSV source table")
    parser.add_argument("--sheet-name", help="Excel sheet containing the question table")
    parser.add_argument("--limit", type=int, help="Optional row limit, for example 1 for a live check")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_CURATION_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    build_draft_benchmark(
        args.source,
        args.output_dir,
        sheet_name=args.sheet_name,
        limit=args.limit,
        model=args.model,
    )


if __name__ == "__main__":
    main()
