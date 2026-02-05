"""
Verify command for running benchmark verifications.

This module implements the main 'karenina verify' command.
Config building is in verify_config.py, output/display helpers in verify_output.py.
"""

import time
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from karenina.benchmark import Benchmark
from karenina.benchmark.verification.batch_runner import generate_task_queue
from karenina.schemas import FinishedTemplate, VerificationConfig, VerificationResultSet
from karenina.utils.progressive_save import ProgressiveSaveManager, TaskIdentifier, generate_task_manifest

from .utils import cli_error, filter_templates_by_indices, parse_question_indices, validate_output_path
from .verify_config import build_config_non_interactive
from .verify_output import (
    display_summary,
    export_results,
    prompt_for_output_interactive,
    run_verification_with_progress,
    validate_output_and_prompt,
)

console = Console()


def _handle_resume_mode(
    resume: Path,
) -> tuple[ProgressiveSaveManager, VerificationConfig, str, Path, str, set[str]]:
    """
    Handle resume mode - load everything from state file.

    Returns:
        Tuple of (progressive_manager, config, benchmark_path, output, output_format, pending_task_ids)

    Raises:
        typer.Exit: If state file not found, load error, or all tasks completed
    """
    if not resume.exists():
        cli_error(f"State file not found: {resume}")

    console.print(f"[cyan]Loading resume state from {resume}...[/cyan]")
    try:
        progressive_manager = ProgressiveSaveManager.load_for_resume(resume)
        config = progressive_manager.config
        benchmark_path = progressive_manager.benchmark_path
        output = progressive_manager.output_path
        output_format = validate_output_path(output)

        console.print(
            f"[green]\u2713 Resuming: {progressive_manager.completed_count}/{progressive_manager.total_tasks} "
            f"tasks already completed[/green]"
        )

        pending_task_ids = progressive_manager.get_pending_task_ids()
        if not pending_task_ids:
            console.print("[yellow]All tasks already completed. Nothing to resume.[/yellow]")
            raise typer.Exit(code=0)

        return progressive_manager, config, benchmark_path, output, output_format, pending_task_ids

    except typer.Exit:
        raise
    except Exception as e:
        cli_error(f"loading resume state: {e}", e)


def _load_benchmark(benchmark_path: str) -> Benchmark:
    """
    Load benchmark from path.

    Returns:
        Loaded Benchmark instance

    Raises:
        typer.Exit: If loading fails
    """
    console.print("[cyan]Loading benchmark...[/cyan]")
    try:
        benchmark = Benchmark.load(Path(benchmark_path))
    except (FileNotFoundError, ValidationError) as e:
        cli_error(f"loading benchmark: {e}", e)

    console.print(f"[green]\u2713 Loaded benchmark: {benchmark.name or 'Unnamed'}[/green]")
    console.print(f"  Total questions: {len(benchmark.get_all_questions())}")
    return benchmark


def _filter_templates(
    all_templates: list[FinishedTemplate],
    selected_question_indices: list[int] | None,
    questions: str | None,
) -> list[FinishedTemplate]:
    """
    Filter templates based on index-based criteria.

    Returns:
        Filtered list of templates

    Raises:
        typer.Exit: If no templates remain after filtering
    """
    if not all_templates:
        console.print("[yellow]Warning: No finished templates found in benchmark[/yellow]")
        raise typer.Exit(code=1)

    templates = all_templates
    if selected_question_indices is not None:
        templates = filter_templates_by_indices(all_templates, selected_question_indices)
        console.print(f"[dim]Using {len(templates)} question(s) from interactive selection[/dim]")
    elif questions:
        try:
            indices = parse_question_indices(questions, len(all_templates))
            templates = filter_templates_by_indices(all_templates, indices)
            console.print(f"[dim]Filtered to {len(templates)} question(s) by indices[/dim]")
        except ValueError as e:
            cli_error(f"parsing question indices: {e}", e)

    if not templates:
        cli_error("No templates to verify after filtering")

    return templates


def _initialize_progressive_save(
    output: Path,
    config: VerificationConfig,
    benchmark_path: str,
    templates: list[FinishedTemplate],
    benchmark: Benchmark,
) -> ProgressiveSaveManager:
    """
    Initialize progressive save manager.

    Returns:
        Initialized ProgressiveSaveManager
    """
    task_queue = generate_task_queue(
        templates=templates,
        config=config,
        global_rubric=benchmark.get_global_rubric(),
        run_name="cli-verification",
    )
    task_manifest = generate_task_manifest(task_queue)

    progressive_manager = ProgressiveSaveManager(
        output, config, benchmark_path, global_rubric=benchmark.get_global_rubric()
    )
    progressive_manager.initialize(task_manifest)
    console.print(f"[green]\u2713 Progressive save enabled: {progressive_manager.tmp_path}[/green]")
    return progressive_manager


def _filter_templates_for_resume(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    benchmark: Benchmark,
    pending_task_ids: set[str],
) -> list[FinishedTemplate]:
    """
    Filter templates to only those with pending tasks (for resume mode).

    Returns:
        Filtered list of templates
    """
    task_queue = generate_task_queue(
        templates=templates,
        config=config,
        global_rubric=benchmark.get_global_rubric(),
        run_name="cli-verification",
    )

    pending_question_ids = set()
    for task in task_queue:
        task_id = TaskIdentifier.from_task_dict(task).to_key()
        if task_id in pending_task_ids:
            pending_question_ids.add(task["question_id"])

    original_count = len(templates)
    templates = [t for t in templates if t.question_id in pending_question_ids]
    if len(templates) < original_count:
        console.print(f"[dim]Filtered to {len(templates)} questions with pending tasks[/dim]")

    return templates


def verify(
    benchmark_path: Annotated[
        str | None, typer.Argument(help="Path to benchmark JSON-LD file (not required with --resume)")
    ] = None,
    # Configuration sources (priority: interactive > CLI+preset > preset > defaults)
    preset: Annotated[Path | None, typer.Option(help="Path to preset configuration")] = None,
    interactive: Annotated[bool, typer.Option("--interactive", help="Interactive configuration mode")] = False,
    _mode: Annotated[str, typer.Option("--mode", help="Interactive mode (basic or advanced)")] = "basic",
    # Output and filtering
    output: Annotated[Path | None, typer.Option(help="Output file (.json or .csv)")] = None,
    questions: Annotated[str | None, typer.Option(help="Question indices (e.g., '0,1,2' or '5-10')")] = None,
    question_ids: Annotated[str | None, typer.Option(help="Comma-separated question IDs")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show progress bar")] = False,
    # Model configuration (single answering + single parsing model)
    answering_model: Annotated[str | None, typer.Option(help="Answering model name (required without preset)")] = None,
    answering_provider: Annotated[
        str | None, typer.Option(help="Answering model provider for langchain (required without preset)")
    ] = None,
    answering_id: Annotated[str, typer.Option(help="Answering model ID")] = "answering-1",
    parsing_model: Annotated[str | None, typer.Option(help="Parsing model name (required without preset)")] = None,
    parsing_provider: Annotated[
        str | None, typer.Option(help="Parsing model provider for langchain (required without preset)")
    ] = None,
    parsing_id: Annotated[str, typer.Option(help="Parsing model ID")] = "parsing-1",
    temperature: Annotated[float | None, typer.Option(help="Model temperature (0.0-2.0)")] = None,
    interface: Annotated[
        str | None, typer.Option(help="Model interface: langchain/openrouter/openai_endpoint (required without preset)")
    ] = None,
    # General settings
    replicate_count: Annotated[int | None, typer.Option(help="Number of replicates per verification")] = None,
    # Feature flags
    abstention: Annotated[bool, typer.Option("--abstention", help="Enable abstention detection")] = False,
    sufficiency: Annotated[bool, typer.Option("--sufficiency", help="Enable trace sufficiency detection")] = False,
    embedding_check: Annotated[bool, typer.Option("--embedding-check", help="Enable embedding check")] = False,
    deep_judgment: Annotated[bool, typer.Option("--deep-judgment", help="Enable deep judgment for templates")] = False,
    # Deep judgment rubric settings
    deep_judgment_rubric_mode: Annotated[
        str, typer.Option(help="Deep judgment mode for rubrics (disabled/enable_all/use_checkpoint/custom)")
    ] = "disabled",
    deep_judgment_rubric_excerpts: Annotated[
        bool,
        typer.Option("--deep-judgment-rubric-excerpts", help="Enable excerpts for rubric traits (enable_all mode)"),
    ] = True,
    deep_judgment_rubric_max_excerpts: Annotated[
        int, typer.Option(help="Max excerpts per rubric trait (enable_all mode)")
    ] = 7,
    deep_judgment_rubric_fuzzy_threshold: Annotated[
        float, typer.Option(help="Fuzzy match threshold for rubric excerpts (0.0-1.0)")
    ] = 0.80,
    deep_judgment_rubric_retry_attempts: Annotated[
        int, typer.Option(help="Retry attempts for rubric excerpt extraction")
    ] = 2,
    deep_judgment_rubric_search: Annotated[
        bool, typer.Option("--deep-judgment-rubric-search", help="Enable search validation for rubric excerpts")
    ] = False,
    deep_judgment_rubric_search_tool: Annotated[
        str, typer.Option(help="Search tool for rubric hallucination detection")
    ] = "tavily",
    deep_judgment_rubric_config: Annotated[
        Path | None, typer.Option(help="Path to custom rubric deep judgment config JSON (custom mode)")
    ] = None,
    # MCP agent trace filtering
    use_full_trace_for_template: Annotated[
        bool,
        typer.Option(
            "--use-full-trace-for-template/--use-final-message-for-template",
            help="Use full MCP agent trace (True) or only final AI message (False) for template parsing",
        ),
    ] = False,
    use_full_trace_for_rubric: Annotated[
        bool,
        typer.Option(
            "--use-full-trace-for-rubric/--use-final-message-for-rubric",
            help="Use full MCP agent trace (True) or only final AI message (False) for rubric evaluation",
        ),
    ] = True,
    # Advanced settings
    evaluation_mode: Annotated[
        str, typer.Option(help="Evaluation mode (template_only/template_and_rubric/rubric_only)")
    ] = "template_only",
    embedding_threshold: Annotated[float, typer.Option(help="Embedding similarity threshold (0.0-1.0)")] = 0.85,
    embedding_model: Annotated[str, typer.Option(help="Embedding model name")] = "all-MiniLM-L6-v2",
    no_async: Annotated[bool, typer.Option("--no-async", help="Disable async execution")] = False,
    async_workers: Annotated[
        int | None, typer.Option(help="Number of async workers (default: 2 or KARENINA_ASYNC_MAX_WORKERS)")
    ] = None,
    # Manual trace support
    manual_traces: Annotated[
        Path | None,
        typer.Option(
            "--manual-traces",
            help="JSON file with manual traces (question_hash: trace_string mapping). Required when --interface manual.",
        ),
    ] = None,
    # Progressive save and resume support
    progressive_save: Annotated[
        bool,
        typer.Option(
            "--progressive-save",
            help="Enable incremental saving to .tmp and .state files for crash recovery",
        ),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Resume from a .state file (loads config from state, ignores other config options)",
        ),
    ] = None,
) -> None:
    """
    Run verification on a benchmark.

    Examples:
        # With preset only
        karenina verify checkpoint.jsonld --preset default.json --questions 0,1

        # Override preset model
        karenina verify checkpoint.jsonld --preset default.json --answering-model gpt-4o

        # CLI arguments only (no preset)
        karenina verify checkpoint.jsonld --answering-model gpt-4.1-mini --parsing-model gpt-4.1-mini --interface langchain --answering-provider openai --parsing-provider openai

        # With feature flags
        karenina verify checkpoint.jsonld --preset default.json --abstention --deep-judgment

        # With rubric evaluation
        karenina verify checkpoint.jsonld --preset default.json --evaluation-mode template_and_rubric

        # Interactive mode
        karenina verify checkpoint.jsonld --interactive --mode basic

        # With output and progress
        karenina verify checkpoint.jsonld --preset default.json --output results.csv --verbose

        # With manual traces
        karenina verify checkpoint.jsonld --interface manual --manual-traces traces/my_traces.json --parsing-model gpt-4.1-mini --parsing-provider openai
    """
    try:
        # Convert no_async to async_execution
        async_execution = not no_async

        # Step 1: Validate output path and prompt if needed
        output, output_format = validate_output_and_prompt(output, progressive_save, bool(resume), interactive)

        # Validate benchmark_path is provided unless resuming
        if not resume and not benchmark_path:
            cli_error("BENCHMARK_PATH is required (unless using --resume)")

        # Initialize state variables
        progressive_manager: ProgressiveSaveManager | None = None
        pending_task_ids: set[str] | None = None
        config: VerificationConfig | None = None
        selected_question_indices: list[int] | None = None
        show_progress_bar_interactive: bool | None = None

        # Step 2: Handle resume mode or build fresh config
        if resume:
            progressive_manager, config, benchmark_path, output, output_format, pending_task_ids = _handle_resume_mode(
                resume
            )

        # Step 3: Load benchmark
        if benchmark_path is None:
            cli_error("benchmark_path must be set (not resuming or resume should have set it)")
        benchmark = _load_benchmark(benchmark_path)

        # Set global rubric on progressive manager if resuming
        if resume and progressive_manager:
            progressive_manager.set_global_rubric(benchmark.get_global_rubric())

        # Step 4: Build config (skip if resuming - config already loaded)
        if resume:
            console.print("[dim]Using configuration from resume state[/dim]")
        elif interactive:
            from .interactive import build_config_interactively

            config, selected_question_indices, show_progress_bar_interactive, progressive_save_interactive = (
                build_config_interactively(benchmark, mode=_mode)
            )
            if not progressive_save:
                progressive_save = progressive_save_interactive
        else:
            config, progressive_save = build_config_non_interactive(
                preset=preset,
                interface=interface,
                answering_model=answering_model,
                answering_provider=answering_provider,
                answering_id=answering_id,
                parsing_model=parsing_model,
                parsing_provider=parsing_provider,
                parsing_id=parsing_id,
                temperature=temperature,
                replicate_count=replicate_count,
                abstention=abstention,
                sufficiency=sufficiency,
                embedding_check=embedding_check,
                deep_judgment=deep_judgment,
                deep_judgment_rubric_mode=deep_judgment_rubric_mode,
                deep_judgment_rubric_excerpts=deep_judgment_rubric_excerpts,
                deep_judgment_rubric_max_excerpts=deep_judgment_rubric_max_excerpts,
                deep_judgment_rubric_fuzzy_threshold=deep_judgment_rubric_fuzzy_threshold,
                deep_judgment_rubric_retry_attempts=deep_judgment_rubric_retry_attempts,
                deep_judgment_rubric_search=deep_judgment_rubric_search,
                deep_judgment_rubric_search_tool=deep_judgment_rubric_search_tool,
                deep_judgment_rubric_config=deep_judgment_rubric_config,
                use_full_trace_for_template=use_full_trace_for_template,
                use_full_trace_for_rubric=use_full_trace_for_rubric,
                evaluation_mode=evaluation_mode,
                embedding_threshold=embedding_threshold,
                embedding_model=embedding_model,
                async_execution=async_execution,
                async_workers=async_workers,
                manual_traces=manual_traces,
                benchmark=benchmark,
                progressive_save=progressive_save,
            )

        # Validate config was built successfully
        if config is None:
            cli_error("Failed to build verification configuration")

        # Step 5: Get and filter templates
        ids_filter = None
        if question_ids:
            ids_filter = {id.strip() for id in question_ids.split(",")}
        all_templates = benchmark.get_finished_templates(question_ids=ids_filter)
        if ids_filter:
            console.print(f"[dim]Filtered to {len(all_templates)} question(s) by IDs[/dim]")
        templates = _filter_templates(all_templates, selected_question_indices, questions)

        # Step 6: Prompt for output in interactive mode (if not already specified)
        if not output and interactive:
            output, output_format, progressive_save = prompt_for_output_interactive(progressive_save)

        # Step 7: Initialize progressive save (if enabled and not resuming)
        if progressive_save and not resume:
            if output is None:
                cli_error("output path is required for progressive save")
            progressive_manager = _initialize_progressive_save(output, config, benchmark_path, templates, benchmark)

        # Step 8: Filter templates for resume (only pending tasks)
        if resume and pending_task_ids:
            templates = _filter_templates_for_resume(templates, config, benchmark, pending_task_ids)

        # Step 9: Run verification
        console.print("\n[bold cyan]Starting verification...[/bold cyan]")
        console.print(f"  Questions: {len(templates)}")
        console.print(f"  Answering models: {len(config.answering_models)}")
        console.print(f"  Parsing models: {len(config.parsing_models)}")
        console.print(f"  Replicates: {config.replicate_count}")
        console.print(
            f"  Async: {'enabled' if config.async_enabled else 'disabled'} ({config.async_max_workers} workers)"
        )
        if progressive_manager:
            console.print("  Progressive save: enabled")
        console.print()

        start_time = time.time()
        show_progress = show_progress_bar_interactive if show_progress_bar_interactive is not None else verbose

        results = run_verification_with_progress(templates, config, benchmark, progressive_manager, show_progress)

        end_time = time.time()
        duration = end_time - start_time

        # Step 10: Get final results
        final_results: VerificationResultSet
        if progressive_manager:
            final_results = progressive_manager.get_result_set()
            console.print(f"[dim]Total results (including resumed): {len(final_results)}[/dim]")
        else:
            final_results = results

        # Step 11: Export results (if output configured)
        if output and output_format:
            export_results(
                output, output_format, final_results, config, benchmark, start_time, end_time, progressive_manager
            )

        # Step 12: Display summary
        display_summary(final_results, duration)

        raise typer.Exit(code=0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        if progressive_manager is not None and output is not None:
            console.print(f"[dim]To resume: karenina verify --resume {output}[/dim]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e
