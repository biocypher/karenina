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
from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.schemas import FinishedTemplate, VerificationConfig, VerificationResultSet
from karenina.schemas.verification.config import (
    DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD,
    DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS,
    DEFAULT_RUBRIC_MAX_EXCERPTS,
)

from .utils import cli_error, filter_templates_by_indices, parse_question_indices, validate_output_path
from .verify_config import build_config_non_interactive, validate_cli_config_requirements
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
) -> tuple[ProgressiveFileSink, VerificationConfig, str, Path, str]:
    """
    Handle resume mode: load everything from the state file.

    Returns:
        Tuple of (sink, config, benchmark_path, output, output_format)

    Raises:
        typer.Exit: If the state file is missing, invalid, or already complete.
    """
    if not resume.exists():
        cli_error(f"State file not found: {resume}")

    console.print(f"[cyan]Loading resume state from {resume}...[/cyan]")
    try:
        sink = ProgressiveFileSink.load_for_resume(resume)
        config = sink.config
        benchmark_path = sink.benchmark_path
        output = sink.output_path
        output_format = validate_output_path(output)

        console.print(
            f"[green]\u2713 Resuming: {sink.completed_count}/{sink.total_tasks} tasks already completed[/green]"
        )

        if sink.completed_count >= sink.total_tasks and sink.total_tasks > 0:
            console.print("[yellow]All tasks already completed. Nothing to resume.[/yellow]")
            raise typer.Exit(code=0)

        return sink, config, benchmark_path, output, output_format

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
    benchmark: Benchmark,
) -> ProgressiveFileSink:
    """Build a fresh :class:`ProgressiveFileSink` for a new run.

    The batch runner calls ``on_start`` with the task manifest once the
    executor is about to start, so no explicit manifest seeding is needed
    here; this keeps the CLI loosely coupled to task-queue details.
    """
    sink = ProgressiveFileSink(
        output_path=output,
        config=config,
        benchmark_path=benchmark_path,
        global_rubric=benchmark.get_global_rubric(),
    )
    console.print(f"[green]\u2713 Progressive save enabled: {sink.jsonl_path}[/green]")
    return sink


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
    # Feature flags (tri-state: None = use preset/default, True = enable, False = disable)
    abstention: Annotated[
        bool | None, typer.Option("--abstention/--no-abstention", help="Enable/disable abstention detection")
    ] = None,
    sufficiency: Annotated[
        bool | None, typer.Option("--sufficiency/--no-sufficiency", help="Enable/disable trace sufficiency detection")
    ] = None,
    embedding_check: Annotated[
        bool | None, typer.Option("--embedding-check/--no-embedding-check", help="Enable/disable embedding check")
    ] = None,
    deep_judgment: Annotated[
        bool | None,
        typer.Option("--deep-judgment/--no-deep-judgment", help="Enable/disable deep judgment for templates"),
    ] = None,
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
    ] = DEFAULT_RUBRIC_MAX_EXCERPTS,
    deep_judgment_rubric_fuzzy_threshold: Annotated[
        float, typer.Option(help="Fuzzy match threshold for rubric excerpts (0.0-1.0)")
    ] = DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD,
    deep_judgment_rubric_retry_attempts: Annotated[
        int, typer.Option(help="Retry attempts for rubric excerpt extraction")
    ] = DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS,
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
    embedding_threshold: Annotated[float | None, typer.Option(help="Embedding similarity threshold (0.0-1.0)")] = None,
    embedding_model: Annotated[str | None, typer.Option(help="Embedding model name")] = None,
    no_async: Annotated[bool, typer.Option("--no-async", help="Disable async execution")] = False,
    async_workers: Annotated[
        int | None, typer.Option(help="Number of async workers (default: 2 or KARENINA_ASYNC_MAX_WORKERS)")
    ] = None,
    workspace_output_mode: Annotated[
        str,
        typer.Option(help="Workspace sidecar capture mode (none/full/produced)"),
    ] = "none",
    workspace_output_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for captured workspace sidecars. Defaults to OUTPUT.parent/workspaces for JSON."),
    ] = None,
    workspace_output_exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--workspace-output-exclude",
            help="Additional fnmatch-style pattern to exclude from workspace capture. Repeatable.",
        ),
    ] = None,
    # Manual trace support
    manual_traces: Annotated[
        Path | None,
        typer.Option(
            "--manual-traces",
            help="JSON file with manual traces (question_hash: trace_string mapping). Required when --interface manual.",
        ),
    ] = None,
    # Replay store
    replay: Annotated[
        Path | None,
        typer.Option(
            "--replay",
            help=(
                "Path to a replay store JSON (built by VerificationResultSet.to_replay_store). "
                "When set, the pipeline short-circuits to the canned traces on matching keys "
                "and runs live otherwise."
            ),
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
        output, output_format = validate_output_and_prompt(output, progressive_save)

        # Validate benchmark_path is provided unless resuming
        if not resume and not benchmark_path:
            cli_error("BENCHMARK_PATH is required (unless using --resume)")

        # Initialize state variables
        progressive_sink: ProgressiveFileSink | None = None
        config: VerificationConfig | None = None
        selected_question_indices: list[int] | None = None
        show_progress_bar_interactive: bool | None = None

        # Step 2: Early config validation (before benchmark load, so config errors are not masked)
        if not resume and not interactive and not preset:
            validation_errors = validate_cli_config_requirements(
                interface, answering_model, parsing_model, answering_provider, parsing_provider, manual_traces
            )
            if validation_errors:
                console.print("[red]Configuration errors:[/red]")
                for error in validation_errors:
                    console.print(f"  [red]• {error}[/red]")
                console.print("\n[dim]Run with --interactive for guided configuration[/dim]")
                raise typer.Exit(code=1)

        # Step 3: Handle resume mode or build fresh config
        if resume:
            progressive_sink, config, benchmark_path, output, output_format = _handle_resume_mode(resume)

        # Step 3: Load benchmark
        if benchmark_path is None:
            cli_error("benchmark_path must be set (not resuming or resume should have set it)")
        benchmark = _load_benchmark(benchmark_path)

        # Set global rubric on progressive sink if resuming
        if resume and progressive_sink:
            progressive_sink.set_global_rubric(benchmark.get_global_rubric())

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
                workspace_output_mode=workspace_output_mode,
                workspace_output_dir=workspace_output_dir,
                workspace_output_exclude_patterns=workspace_output_exclude,
                manual_traces=manual_traces,
                benchmark=benchmark,
                progressive_save=progressive_save,
            )

        # Validate config was built successfully
        if config is None:
            cli_error("Failed to build verification configuration")

        # Attach a ReplayStore if --replay was supplied. Done after the
        # config is built so the file load happens in one place and is
        # easy to monkeypatch in tests.
        if replay is not None:
            from karenina.replay import ReplayStore

            console.print(f"[dim]Loading replay store from {replay}[/dim]")
            config = config.model_copy(update={"replay_store": ReplayStore.load(replay)})

        if config.workspace_output_mode != "none" and config.workspace_output_dir is None:
            if output is None or output_format != "json":
                cli_error("--workspace-output-dir is required when workspace capture is enabled without JSON output")
            config = config.model_copy(update={"workspace_output_dir": output.parent / "workspaces"})

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
            progressive_sink = _initialize_progressive_save(output, config, benchmark_path, benchmark)

        # Step 8: Run verification. Triple-level skip via skip_triples (honored
        # inside batch_runner) replaces the old per-question filter, so no
        # pre-filtering of templates is needed on resume.
        console.print("\n[bold cyan]Starting verification...[/bold cyan]")
        console.print(f"  Questions: {len(templates)}")
        console.print(f"  Answering models: {len(config.answering_models)}")
        console.print(f"  Parsing models: {len(config.parsing_models)}")
        console.print(f"  Replicates: {config.replicate_count}")
        console.print(
            f"  Async: {'enabled' if config.async_enabled else 'disabled'} ({config.async_max_workers} workers)"
        )
        if progressive_sink:
            console.print("  Progressive save: enabled")
        console.print()

        start_time = time.time()
        show_progress = show_progress_bar_interactive if show_progress_bar_interactive is not None else verbose

        results = run_verification_with_progress(templates, config, benchmark, progressive_sink, show_progress)

        end_time = time.time()
        duration = end_time - start_time

        # Step 9: Combine with previously-persisted results when resuming.
        # The sink's buffer is the authoritative post-run superset.
        final_results: VerificationResultSet
        if progressive_sink:
            final_results = progressive_sink.get_result_set()
            console.print(f"[dim]Total results (including resumed): {len(final_results)}[/dim]")
        else:
            final_results = results

        # Step 10: Export results (if output configured). ProgressiveFileSink
        # already wrote a JSON export at on_finalize(all_complete=True);
        # re-export here so CSV output and the non-progressive path are
        # handled uniformly.
        if output and output_format:
            export_results(output, output_format, final_results, config, benchmark, start_time, end_time)

        # Step 11: Display summary
        display_summary(final_results, duration)

        raise typer.Exit(code=0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        if progressive_sink is not None and output is not None:
            console.print(f"[dim]To resume: karenina verify --resume {progressive_sink.state_path}[/dim]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e
