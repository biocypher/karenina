"""
Verify command for running benchmark verifications.

This module implements the main 'karenina verify' command.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt

from karenina.benchmark import Benchmark
from karenina.benchmark.verification.batch_runner import generate_task_queue, run_verification_batch
from karenina.benchmark.verification.results_exporter import (
    export_verification_results_csv,
    export_verification_results_json,
)
from karenina.schemas import FinishedTemplate, VerificationConfig, VerificationResult, VerificationResultSet

from .progressive_save import ProgressiveSaveManager, TaskIdentifier, generate_task_manifest
from .utils import (
    create_export_job,
    filter_templates_by_ids,
    filter_templates_by_indices,
    get_preset_path,
    get_traces_path,
    load_manual_traces_from_file,
    parse_question_indices,
    validate_output_path,
)

console = Console()


def _build_config_from_cli_args(
    answering_model: str | None,
    answering_provider: str | None,
    answering_id: str,
    parsing_model: str | None,
    parsing_provider: str | None,
    parsing_id: str,
    temperature: float,
    interface: str | None,
    replicate_count: int,
    abstention: bool,
    sufficiency: bool,
    embedding_check: bool,
    deep_judgment: bool,
    deep_judgment_rubric_mode: str,
    deep_judgment_rubric_excerpts: bool,
    deep_judgment_rubric_max_excerpts: int,
    deep_judgment_rubric_fuzzy_threshold: float,
    deep_judgment_rubric_retry_attempts: int,
    deep_judgment_rubric_search: bool,
    deep_judgment_rubric_search_tool: str,
    deep_judgment_rubric_config: Path | None,
    use_full_trace_for_template: bool,
    use_full_trace_for_rubric: bool,
    evaluation_mode: str,
    embedding_threshold: float,
    embedding_model: str,
    async_execution: bool,
    async_workers: int | None,
    preset_config: VerificationConfig | None = None,
    manual_traces_obj: object | None = None,
) -> VerificationConfig:
    """
    Build VerificationConfig respecting hierarchy: CLI > preset > env > defaults.

    Args:
        CLI argument values (all optional)
        preset_config: Optional preset configuration to use as base

    Returns:
        VerificationConfig with CLI overrides applied
    """
    from karenina.schemas.workflow.models import ModelConfig

    # Start with preset if provided, otherwise create new config
    config_dict = preset_config.model_dump() if preset_config else {}

    # Override replicate_count only if no preset OR if explicitly different from default
    # (This preserves preset value unless user explicitly passes --replicate-count)
    if not preset_config or replicate_count != 1:
        config_dict["replicate_count"] = replicate_count

    # Override feature flags (always override since they have defaults)
    config_dict["abstention_enabled"] = abstention
    config_dict["sufficiency_enabled"] = sufficiency
    config_dict["embedding_check_enabled"] = embedding_check
    config_dict["deep_judgment_enabled"] = deep_judgment

    # Override deep judgment rubric settings (always override since they have defaults)
    config_dict["deep_judgment_rubric_mode"] = deep_judgment_rubric_mode
    config_dict["deep_judgment_rubric_global_excerpts"] = deep_judgment_rubric_excerpts
    config_dict["deep_judgment_rubric_max_excerpts_default"] = deep_judgment_rubric_max_excerpts
    config_dict["deep_judgment_rubric_fuzzy_match_threshold_default"] = deep_judgment_rubric_fuzzy_threshold
    config_dict["deep_judgment_rubric_excerpt_retry_attempts_default"] = deep_judgment_rubric_retry_attempts
    config_dict["deep_judgment_rubric_search_enabled"] = deep_judgment_rubric_search
    config_dict["deep_judgment_rubric_search_tool"] = deep_judgment_rubric_search_tool

    # Load custom config JSON if provided (for custom mode)
    if deep_judgment_rubric_config is not None:
        import json

        try:
            with open(deep_judgment_rubric_config) as f:
                custom_config = json.load(f)
            config_dict["deep_judgment_rubric_config"] = custom_config
        except Exception as e:
            raise ValueError(f"Failed to load custom rubric config from {deep_judgment_rubric_config}: {e}") from e

    # Override MCP trace filtering settings (always override since they have defaults)
    config_dict["use_full_trace_for_template"] = use_full_trace_for_template
    config_dict["use_full_trace_for_rubric"] = use_full_trace_for_rubric

    # Override advanced settings (always override since they have defaults)
    config_dict["evaluation_mode"] = evaluation_mode
    # Set rubric_enabled based on evaluation_mode
    config_dict["rubric_enabled"] = evaluation_mode in ["template_and_rubric", "rubric_only"]
    config_dict["embedding_similarity_threshold"] = embedding_threshold
    config_dict["embedding_model_name"] = embedding_model
    config_dict["async_enabled"] = async_execution
    if async_workers is not None:
        config_dict["async_max_workers"] = async_workers
    # else: let VerificationConfig read from KARENINA_ASYNC_MAX_WORKERS or use default

    # Handle model configuration
    # If ANY optional model CLI arg is provided, we override that model completely
    # Note: answering_id, parsing_id, and temperature now have defaults so always count as provided
    answering_has_cli_args = any(
        [
            answering_model is not None,
            answering_provider is not None,
            interface is not None,
        ]
    )

    parsing_has_cli_args = any(
        [
            parsing_model is not None,
            parsing_provider is not None,
            interface is not None,
        ]
    )

    if answering_has_cli_args:
        # Build answering model from CLI args
        # Use preset values as defaults if available
        if preset_config and preset_config.answering_models:
            base_model = preset_config.answering_models[0].model_dump()
        else:
            # No preset - start with minimal required fields
            # For manual interface, only interface and manual_traces are required
            if interface == "manual":
                base_model = {
                    "interface": "manual",
                    # id, model_name, model_provider will be set by ModelConfig defaults
                }
            else:
                # Validation ensures model_name and provider are provided when no preset
                base_model = {
                    "model_name": answering_model or "gpt-4.1-mini",
                    "model_provider": answering_provider or "openai",
                    "interface": interface or "langchain",
                    "temperature": temperature,
                    "id": answering_id,
                }

        # Apply CLI overrides if preset was used
        if preset_config:
            if answering_model is not None:
                base_model["model_name"] = answering_model
            if answering_provider is not None:
                base_model["model_provider"] = answering_provider
            base_model["id"] = answering_id
            base_model["temperature"] = temperature
            if interface is not None:
                base_model["interface"] = interface

        # Create ModelConfig
        if base_model.get("interface") == "manual":
            if manual_traces_obj is None:
                raise ValueError("manual_traces_obj is None but interface is manual")
            # Create ModelConfig directly with required parameters for manual interface
            model_config = ModelConfig(interface="manual", manual_traces=manual_traces_obj)
            config_dict["answering_models"] = [model_config]
        else:
            config_dict["answering_models"] = [ModelConfig(**base_model)]

    if parsing_has_cli_args:
        # Build parsing model from CLI args
        if preset_config and preset_config.parsing_models:
            base_model = preset_config.parsing_models[0].model_dump()
        else:
            # No preset - start with minimal required fields
            # Validation ensures model_name and provider are provided when no preset
            # Note: Parsing model should NOT use manual interface (only answering model can)
            parsing_interface = interface if interface != "manual" else "langchain"
            base_model = {
                "model_name": parsing_model or "gpt-4.1-mini",
                "model_provider": parsing_provider or "openai",
                "interface": parsing_interface,
                "temperature": temperature,
                "id": parsing_id,
            }

        # Apply CLI overrides if preset was used
        if preset_config:
            if parsing_model is not None:
                base_model["model_name"] = parsing_model
            if parsing_provider is not None:
                base_model["model_provider"] = parsing_provider
            base_model["id"] = parsing_id
            base_model["temperature"] = temperature
            if interface is not None and interface != "manual":
                base_model["interface"] = interface

        config_dict["parsing_models"] = [ModelConfig(**base_model)]

    return VerificationConfig(**config_dict)


def _validate_output_and_prompt(
    output: Path | None,
    progressive_save: bool,
    resume: bool,
    interactive: bool,
) -> tuple[Path | None, str | None]:
    """
    Validate output path and prompt for one if needed.

    Returns:
        Tuple of (output_path, output_format) - both may be None if no output wanted
    """
    output_format = None

    # Validate output path upfront (fail fast)
    if output:
        try:
            output_format = validate_output_path(output)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1) from e

    # Validate progressive save requires output
    if progressive_save and not output:
        console.print("[red]Error: --progressive-save requires --output to be specified[/red]")
        raise typer.Exit(code=1)

    # Prompt for output file if not specified (and not resuming or interactive)
    if not output and not resume and not interactive:
        suggested = f"verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        console.print("[yellow]No output file specified.[/yellow]")
        console.print(f"[dim]Suggested: {suggested}[/dim]")

        user_input = typer.prompt("Output file path", default=suggested)
        output = Path(user_input)

        # Validate the prompted output path
        try:
            output_format = validate_output_path(output)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1) from e

    return output, output_format


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
        console.print(f"[red]Error: State file not found: {resume}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Loading resume state from {resume}...[/cyan]")
    try:
        progressive_manager = ProgressiveSaveManager.load_for_resume(resume)
        config = progressive_manager.config
        benchmark_path = progressive_manager.benchmark_path
        output = progressive_manager.output_path
        output_format = validate_output_path(output)

        console.print(
            f"[green]✓ Resuming: {progressive_manager.completed_count}/{progressive_manager.total_tasks} "
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
        console.print(f"[red]Error loading resume state: {e}[/red]")
        raise typer.Exit(code=1) from e


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
    except Exception as e:
        console.print(f"[red]Error loading benchmark: {e}[/red]")
        raise typer.Exit(code=1) from e

    console.print(f"[green]✓ Loaded benchmark: {benchmark.name or 'Unnamed'}[/green]")
    console.print(f"  Total questions: {len(benchmark.get_all_questions())}")
    return benchmark


def _validate_cli_config_requirements(
    interface: str | None,
    answering_model: str | None,
    parsing_model: str | None,
    answering_provider: str | None,
    parsing_provider: str | None,
    manual_traces: Path | None,
) -> list[str]:
    """
    Validate required CLI parameters when no preset is used.

    Returns:
        List of validation error messages (empty if valid)
    """
    validation_errors = []

    if not interface:
        validation_errors.append(
            "--interface is required when not using a preset (langchain/openrouter/openai_endpoint)"
        )

    if not answering_model:
        validation_errors.append("--answering-model is required when not using a preset")

    if not parsing_model:
        validation_errors.append("--parsing-model is required when not using a preset")

    # Validate interface-specific requirements
    if interface == "langchain":
        if not answering_provider:
            validation_errors.append(
                "--answering-provider is required for langchain interface (e.g., openai, anthropic, google)"
            )
        if not parsing_provider:
            validation_errors.append("--parsing-provider is required for langchain interface")
    elif interface == "openai_endpoint":
        console.print(
            "[yellow]Note: OpenAI Endpoint interface requires endpoint_base_url. "
            "Use interactive mode or a preset for this interface.[/yellow]"
        )
        validation_errors.append(
            "OpenAI Endpoint interface is not supported via CLI args alone. Use --interactive or --preset."
        )
    elif interface == "manual":
        if not manual_traces:
            validation_errors.append(
                "--manual-traces is required when --interface manual. "
                "Provide a JSON file mapping question hashes to answer traces."
            )

    # Additional validation for manual traces
    if manual_traces and not manual_traces.exists():
        validation_errors.append(f"Manual traces file not found: {manual_traces}")

    return validation_errors


def _load_manual_traces(manual_traces: Path, benchmark: Benchmark) -> object:
    """
    Load manual traces from file.

    Returns:
        ManualTraces object

    Raises:
        typer.Exit: If loading fails
    """
    console.print(f"[cyan]Loading manual traces from {manual_traces}...[/cyan]")
    try:
        # Resolve trace file path
        trace_file = get_traces_path(manual_traces)

        # Load traces and create ManualTraces object
        manual_traces_obj = load_manual_traces_from_file(trace_file, benchmark)

        # Verify traces loaded and report to user
        from karenina.adapters.manual import get_manual_trace_count

        trace_count = get_manual_trace_count()
        console.print(f"[green]✓ Loaded {trace_count} manual trace(s)[/green]")
        return manual_traces_obj

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error loading manual traces: {e}[/red]")
        raise typer.Exit(code=1) from e


def _validate_deep_judgment_rubric_settings(
    deep_judgment_rubric_mode: str,
    deep_judgment_rubric_fuzzy_threshold: float,
    deep_judgment_rubric_config: Path | None,
) -> None:
    """
    Validate deep judgment rubric settings.

    Raises:
        typer.Exit: If validation fails
    """
    valid_modes = ["disabled", "enable_all", "use_checkpoint", "custom"]
    if deep_judgment_rubric_mode not in valid_modes:
        console.print(f"[red]Error: Invalid deep_judgment_rubric_mode '{deep_judgment_rubric_mode}'[/red]")
        console.print(f"[red]Valid modes: {', '.join(valid_modes)}[/red]")
        raise typer.Exit(code=1)

    if not 0.0 <= deep_judgment_rubric_fuzzy_threshold <= 1.0:
        console.print("[red]Error: deep_judgment_rubric_fuzzy_threshold must be between 0.0 and 1.0[/red]")
        raise typer.Exit(code=1)

    if deep_judgment_rubric_mode == "use_checkpoint":
        console.print(
            "[yellow]Note: use_checkpoint mode requires deep judgment settings to be saved in the checkpoint file.[/yellow]"
        )
        console.print(
            "[yellow]If the checkpoint doesn't have deep judgment config, traits will be disabled by default.[/yellow]"
        )

    if deep_judgment_rubric_mode == "custom" and not deep_judgment_rubric_config:
        console.print("[red]Error: custom mode requires --deep-judgment-rubric-config to be specified[/red]")
        raise typer.Exit(code=1)


def _filter_templates(
    all_templates: list[FinishedTemplate],
    selected_question_indices: list[int] | None,
    questions: str | None,
    question_ids: str | None,
) -> list[FinishedTemplate]:
    """
    Filter templates based on provided criteria.

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
            console.print(f"[red]Error parsing question indices: {e}[/red]")
            raise typer.Exit(code=1) from e
    elif question_ids:
        ids = [id.strip() for id in question_ids.split(",")]
        templates = filter_templates_by_ids(all_templates, ids)
        console.print(f"[dim]Filtered to {len(templates)} question(s) by IDs[/dim]")

    if not templates:
        console.print("[red]Error: No templates to verify after filtering[/red]")
        raise typer.Exit(code=1)

    return templates


def _prompt_for_output_interactive(
    progressive_save: bool,
) -> tuple[Path | None, str | None, bool]:
    """
    Prompt for output file in interactive mode.

    Returns:
        Tuple of (output_path, output_format, progressive_save_updated)
    """
    if progressive_save:
        console.print("\n[yellow]Progressive save requires an output file.[/yellow]")
        save_to_file = True
    else:
        save_to_file = Confirm.ask("\nWould you like to save the verification results to a file?", default=True)

    if save_to_file:
        format_choice = Prompt.ask("Output format", choices=["json", "csv"], default="json")
        default_filename = f"verification_results.{format_choice}"
        output_filename = Prompt.ask("Output filename", default=default_filename)

        output = Path(output_filename)

        try:
            output_format = validate_output_path(output)
            console.print(f"[green]✓ Results will be saved to: {output}[/green]")
            return output, output_format, progressive_save
        except ValueError as e:
            console.print(f"[yellow]Warning: {e}. Results will not be saved to file.[/yellow]")
            if progressive_save:
                console.print("[yellow]Disabling progressive save due to invalid output path.[/yellow]")
            return None, None, False

    return None, None, progressive_save


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
    console.print(f"[green]✓ Progressive save enabled: {progressive_manager.tmp_path}[/green]")
    return progressive_manager


def _run_verification_with_progress(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    benchmark: Benchmark,
    progressive_manager: ProgressiveSaveManager | None,
    show_progress: bool,
) -> VerificationResultSet:
    """
    Run verification with optional progress bar display.

    Returns:
        VerificationResultSet containing all results
    """
    if show_progress:
        total_verifications = (
            len(templates) * len(config.answering_models) * len(config.parsing_models) * config.replicate_count
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            progress_task: TaskID = progress.add_task("Verifying questions...", total=total_verifications)

            def progress_callback(current: int, _total: int, result: VerificationResult | None = None) -> None:
                """Update progress bar on each verification completion."""
                progress.update(progress_task, completed=current)
                if result and result.verify_result is not None:
                    status = "✓" if result.verify_result else "✗"
                    progress.update(progress_task, description=f"Verifying questions... {status}")

                if progressive_manager and result and result.metadata.timestamp:
                    progressive_manager.add_result(result)

            results = run_verification_batch(
                templates=templates,
                config=config,
                run_name="cli-verification",
                global_rubric=benchmark.get_global_rubric(),
                progress_callback=progress_callback,
            )
    else:

        def simple_progress_callback(_current: int, _total: int, result: VerificationResult | None = None) -> None:
            """Save result incrementally (no progress bar display)."""
            if progressive_manager and result and result.metadata.timestamp:
                progressive_manager.add_result(result)

        results = run_verification_batch(
            templates=templates,
            config=config,
            run_name="cli-verification",
            global_rubric=benchmark.get_global_rubric(),
            progress_callback=simple_progress_callback if progressive_manager else None,
        )

    return results


def _export_results(
    output: Path,
    output_format: str,
    final_results: VerificationResultSet,
    config: VerificationConfig,
    benchmark: Benchmark,
    start_time: float,
    end_time: float,
    progressive_manager: ProgressiveSaveManager | None,
) -> None:
    """
    Export verification results to file.
    """
    console.print(f"\n[cyan]Exporting results to {output}...[/cyan]")

    job = create_export_job(final_results, config, "cli-verification", start_time, end_time)

    global_rubric = benchmark.get_global_rubric()
    if output_format == "json":
        export_data = export_verification_results_json(job, final_results, global_rubric)
    else:  # csv
        export_data = export_verification_results_csv(job, final_results, global_rubric)

    output.write_text(export_data)
    console.print(f"[green]✓ Results exported to {output}[/green]")

    if progressive_manager:
        progressive_manager.finalize()
        console.print("[dim]Progressive save files cleaned up[/dim]")


def _display_summary(
    final_results: VerificationResultSet,
    duration: float,
) -> None:
    """
    Display verification summary.
    """
    total = len(final_results)
    passed = sum(1 for r in final_results if r.template and r.template.verify_result)
    execution_errors = sum(1 for r in final_results if not r.metadata.completed_without_errors)
    failures = total - passed
    not_passed = failures - execution_errors

    console.print("\n[bold]Verification Summary:[/bold]")
    console.print(f"  Total: {total}")
    console.print(f"  Passed: [green]{passed}[/green]")
    if failures > 0:
        console.print(f"  Failures: [red]{failures}[/red]")
        if execution_errors > 0:
            console.print(f"    Execution Errors: [yellow]{execution_errors}[/yellow]")
        if not_passed > 0:
            console.print(f"    Not Passed: [red]{not_passed}[/red]")
    else:
        console.print("  Failures: [green]0[/green]")
    console.print(f"  Duration: {duration:.2f}s")

    if failures == 0:
        console.print("\n[green]✓ Verification complete![/green]")
    else:
        console.print(f"\n[yellow]⚠ Verification complete with {failures} failure(s)[/yellow]")


def _build_config_non_interactive(
    preset: Path | None,
    interface: str | None,
    answering_model: str | None,
    answering_provider: str | None,
    answering_id: str,
    parsing_model: str | None,
    parsing_provider: str | None,
    parsing_id: str,
    temperature: float,
    replicate_count: int,
    abstention: bool,
    sufficiency: bool,
    embedding_check: bool,
    deep_judgment: bool,
    deep_judgment_rubric_mode: str,
    deep_judgment_rubric_excerpts: bool,
    deep_judgment_rubric_max_excerpts: int,
    deep_judgment_rubric_fuzzy_threshold: float,
    deep_judgment_rubric_retry_attempts: int,
    deep_judgment_rubric_search: bool,
    deep_judgment_rubric_search_tool: str,
    deep_judgment_rubric_config: Path | None,
    use_full_trace_for_template: bool,
    use_full_trace_for_rubric: bool,
    evaluation_mode: str,
    embedding_threshold: float,
    embedding_model: str,
    async_execution: bool,
    async_workers: int | None,
    manual_traces: Path | None,
    benchmark: Benchmark,
    progressive_save: bool,
) -> tuple[VerificationConfig, bool]:
    """
    Build verification config from CLI args and/or preset (non-interactive mode).

    Returns:
        Tuple of (config, progressive_save_updated)

    Raises:
        typer.Exit: If configuration errors occur
    """
    # Load preset if provided
    preset_config = None
    if preset:
        console.print("[cyan]Loading preset...[/cyan]")
        try:
            preset_path = get_preset_path(str(preset))
            preset_config = VerificationConfig.from_preset(preset_path)
            console.print(f"[green]✓ Loaded preset from: {preset_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error loading preset: {e}[/red]")
            raise typer.Exit(code=1) from e

    # Validate required parameters when no preset is used
    if not preset_config:
        validation_errors = _validate_cli_config_requirements(
            interface, answering_model, parsing_model, answering_provider, parsing_provider, manual_traces
        )
        if validation_errors:
            console.print("[red]Configuration errors:[/red]")
            for error in validation_errors:
                console.print(f"  [red]• {error}[/red]")
            console.print("\n[dim]Run with --interactive for guided configuration[/dim]")
            raise typer.Exit(code=1)

    # Load manual traces if provided
    manual_traces_obj = None
    if manual_traces:
        manual_traces_obj = _load_manual_traces(manual_traces, benchmark)

    # Build config with CLI overrides
    try:
        config = _build_config_from_cli_args(
            answering_model=answering_model,
            answering_provider=answering_provider,
            answering_id=answering_id,
            parsing_model=parsing_model,
            parsing_provider=parsing_provider,
            parsing_id=parsing_id,
            temperature=temperature,
            interface=interface,
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
            preset_config=preset_config,
            manual_traces_obj=manual_traces_obj,
        )

        # Show what we're using
        if preset_config:
            has_cli_overrides = any(
                [
                    answering_model,
                    answering_provider,
                    parsing_model,
                    parsing_provider,
                    temperature,
                    interface,
                    replicate_count,
                    abstention,
                    embedding_check,
                    deep_judgment,
                    evaluation_mode,
                    embedding_threshold,
                    embedding_model,
                    async_execution,
                    async_workers,
                ]
            )
            if has_cli_overrides:
                console.print("[dim]CLI arguments will override preset values[/dim]")
        else:
            console.print("[dim]Building configuration from CLI arguments[/dim]")

    except Exception as e:
        console.print(f"[red]Error building configuration: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Validate deep judgment rubric settings
    _validate_deep_judgment_rubric_settings(
        deep_judgment_rubric_mode, deep_judgment_rubric_fuzzy_threshold, deep_judgment_rubric_config
    )

    return config, progressive_save


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
    temperature: Annotated[float, typer.Option(help="Model temperature (0.0-2.0)")] = 0.1,
    interface: Annotated[
        str | None, typer.Option(help="Model interface: langchain/openrouter/openai_endpoint (required without preset)")
    ] = None,
    # General settings
    replicate_count: Annotated[int, typer.Option(help="Number of replicates per verification")] = 1,
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
        output, output_format = _validate_output_and_prompt(output, progressive_save, bool(resume), interactive)

        # Validate benchmark_path is provided unless resuming
        if not resume and not benchmark_path:
            console.print("[red]Error: BENCHMARK_PATH is required (unless using --resume)[/red]")
            raise typer.Exit(code=1)

        # Initialize state variables
        progressive_manager: ProgressiveSaveManager | None = None
        pending_task_ids: set[str] | None = None
        config: VerificationConfig
        selected_question_indices: list[int] | None = None
        show_progress_bar_interactive: bool | None = None

        # Step 2: Handle resume mode or build fresh config
        if resume:
            progressive_manager, config, benchmark_path, output, output_format, pending_task_ids = _handle_resume_mode(
                resume
            )

        # Step 3: Load benchmark
        assert benchmark_path is not None, "benchmark_path should be set"
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
            config, progressive_save = _build_config_non_interactive(
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

        # Step 5: Get and filter templates
        all_templates = benchmark.get_finished_templates()
        templates = _filter_templates(all_templates, selected_question_indices, questions, question_ids)

        # Step 6: Prompt for output in interactive mode (if not already specified)
        if not output and interactive:
            output, output_format, progressive_save = _prompt_for_output_interactive(progressive_save)

        # Step 7: Initialize progressive save (if enabled and not resuming)
        if progressive_save and not resume:
            assert output is not None, "output should be set for progressive save"
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

        results = _run_verification_with_progress(templates, config, benchmark, progressive_manager, show_progress)

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
            _export_results(
                output, output_format, final_results, config, benchmark, start_time, end_time, progressive_manager
            )

        # Step 12: Display summary
        _display_summary(final_results, duration)

        # Force exit to cleanup lingering resources (HTTP clients, MCP connections, etc.)
        import sys

        time.sleep(0.5)
        sys.exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e
