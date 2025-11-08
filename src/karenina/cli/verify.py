"""
Verify command for running benchmark verifications.

This module implements the main 'karenina verify' command.
"""

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from karenina.benchmark import Benchmark
from karenina.benchmark.exporter import export_verification_results_csv, export_verification_results_json
from karenina.benchmark.verification.batch_runner import run_verification_batch
from karenina.schemas import VerificationConfig, VerificationResult

from .utils import (
    create_export_job,
    filter_templates_by_ids,
    filter_templates_by_indices,
    get_preset_path,
    parse_question_indices,
    validate_output_path,
)

console = Console()


def _build_config_from_cli_args(
    answering_model: str | None,
    answering_provider: str | None,
    answering_id: str | None,
    parsing_model: str | None,
    parsing_provider: str | None,
    parsing_id: str | None,
    temperature: float | None,
    interface: str | None,
    replicate_count: int | None,
    rubric: bool | None,
    abstention: bool | None,
    embedding_check: bool | None,
    deep_judgment: bool | None,
    evaluation_mode: str | None,
    embedding_threshold: float | None,
    embedding_model: str | None,
    async_execution: bool | None,
    async_workers: int | None,
    preset_config: VerificationConfig | None = None,
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

    # Helper to check if CLI arg should override
    def should_override(cli_value: Any) -> bool:
        return cli_value is not None

    # Override general settings
    if should_override(replicate_count):
        config_dict["replicate_count"] = replicate_count

    # Override feature flags
    if should_override(rubric):
        config_dict["rubric_enabled"] = rubric
    if should_override(abstention):
        config_dict["abstention_enabled"] = abstention
    if should_override(embedding_check):
        config_dict["embedding_check_enabled"] = embedding_check
    if should_override(deep_judgment):
        config_dict["deep_judgment_enabled"] = deep_judgment

    # Override advanced settings
    if should_override(evaluation_mode):
        config_dict["evaluation_mode"] = evaluation_mode
    if should_override(embedding_threshold):
        config_dict["embedding_similarity_threshold"] = embedding_threshold
    if should_override(embedding_model):
        config_dict["embedding_model_name"] = embedding_model
    if should_override(async_execution):
        config_dict["async_execution_enabled"] = async_execution
    if should_override(async_workers):
        config_dict["async_max_workers"] = async_workers

    # Handle model configuration
    # If ANY model CLI arg is provided, we override that model completely
    answering_has_cli_args = any(
        [
            answering_model is not None,
            answering_provider is not None,
            answering_id is not None,
            temperature is not None,
            interface is not None,
        ]
    )

    parsing_has_cli_args = any(
        [
            parsing_model is not None,
            parsing_provider is not None,
            parsing_id is not None,
            temperature is not None,
            interface is not None,
        ]
    )

    if answering_has_cli_args:
        # Build answering model from CLI args
        # Use preset values as defaults if available
        if preset_config and preset_config.answering_models:
            base_model = preset_config.answering_models[0].model_dump()
        else:
            base_model = {
                "model_name": "gpt-4.1-mini",
                "model_provider": "openai",
                "interface": "langchain",
                "temperature": 0.1,
                "id": "answering-1",
            }

        if answering_model is not None:
            base_model["model_name"] = answering_model
        if answering_provider is not None:
            base_model["model_provider"] = answering_provider
        if answering_id is not None:
            base_model["id"] = answering_id
        if temperature is not None:
            base_model["temperature"] = temperature
        if interface is not None:
            base_model["interface"] = interface

        config_dict["answering_models"] = [ModelConfig(**base_model)]

    if parsing_has_cli_args:
        # Build parsing model from CLI args
        if preset_config and preset_config.parsing_models:
            base_model = preset_config.parsing_models[0].model_dump()
        else:
            base_model = {
                "model_name": "gpt-4.1-mini",
                "model_provider": "openai",
                "interface": "langchain",
                "temperature": 0.1,
                "id": "parsing-1",
            }

        if parsing_model is not None:
            base_model["model_name"] = parsing_model
        if parsing_provider is not None:
            base_model["model_provider"] = parsing_provider
        if parsing_id is not None:
            base_model["id"] = parsing_id
        if temperature is not None:
            base_model["temperature"] = temperature
        if interface is not None:
            base_model["interface"] = interface

        config_dict["parsing_models"] = [ModelConfig(**base_model)]

    # If no preset and no CLI model args, use defaults
    if not preset_config and not answering_has_cli_args:
        config_dict["answering_models"] = [
            ModelConfig(
                id="answering-1",
                model_name="gpt-4.1-mini",
                model_provider="openai",
                interface="langchain",
                temperature=0.1,
            )
        ]

    if not preset_config and not parsing_has_cli_args:
        config_dict["parsing_models"] = [
            ModelConfig(
                id="parsing-1",
                model_name="gpt-4.1-mini",
                model_provider="openai",
                interface="langchain",
                temperature=0.1,
            )
        ]

    # Ensure minimum required fields
    if "replicate_count" not in config_dict:
        config_dict["replicate_count"] = 1

    return VerificationConfig(**config_dict)


def verify(
    benchmark_path: Annotated[str, typer.Argument(help="Path to benchmark JSON-LD file")],
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
    answering_model: Annotated[str | None, typer.Option(help="Answering model name")] = None,
    answering_provider: Annotated[str | None, typer.Option(help="Answering model provider (for langchain)")] = None,
    answering_id: Annotated[str | None, typer.Option(help="Answering model ID")] = None,
    parsing_model: Annotated[str | None, typer.Option(help="Parsing model name")] = None,
    parsing_provider: Annotated[str | None, typer.Option(help="Parsing model provider (for langchain)")] = None,
    parsing_id: Annotated[str | None, typer.Option(help="Parsing model ID")] = None,
    temperature: Annotated[float | None, typer.Option(help="Model temperature (0.0-2.0)")] = None,
    interface: Annotated[
        str | None, typer.Option(help="Model interface (langchain/openrouter/openai_endpoint)")
    ] = None,
    # General settings
    replicate_count: Annotated[int | None, typer.Option(help="Number of replicates per verification")] = None,
    # Feature flags
    rubric: Annotated[bool | None, typer.Option("--rubric/--no-rubric", help="Enable rubric evaluation")] = None,
    abstention: Annotated[
        bool | None, typer.Option("--abstention/--no-abstention", help="Enable abstention detection")
    ] = None,
    embedding_check: Annotated[
        bool | None, typer.Option("--embedding-check/--no-embedding-check", help="Enable embedding check")
    ] = None,
    deep_judgment: Annotated[
        bool | None, typer.Option("--deep-judgment/--no-deep-judgment", help="Enable deep judgment")
    ] = None,
    # Advanced settings
    evaluation_mode: Annotated[
        str | None, typer.Option(help="Evaluation mode (template_only/template_and_rubric/rubric_only)")
    ] = None,
    embedding_threshold: Annotated[float | None, typer.Option(help="Embedding similarity threshold (0.0-1.0)")] = None,
    embedding_model: Annotated[str | None, typer.Option(help="Embedding model name")] = None,
    async_execution: Annotated[bool | None, typer.Option("--async/--no-async", help="Enable async execution")] = None,
    async_workers: Annotated[int | None, typer.Option(help="Number of async workers")] = None,
) -> None:
    """
    Run verification on a benchmark.

    Examples:
        # With preset only
        karenina verify checkpoint.jsonld --preset default.json --questions 0,1

        # Override preset model
        karenina verify checkpoint.jsonld --preset default.json --answering-model gpt-4o

        # CLI arguments only (no preset)
        karenina verify checkpoint.jsonld --answering-model gpt-4.1-mini --parsing-model gpt-4.1-mini

        # With feature flags
        karenina verify checkpoint.jsonld --answering-model gpt-4o --rubric --deep-judgment

        # Interactive mode
        karenina verify checkpoint.jsonld --interactive --mode basic

        # With output and progress
        karenina verify checkpoint.jsonld --preset default.json --output results.csv --verbose
    """
    try:
        # Step 1: Validate output path upfront (fail fast)
        output_format = None
        if output:
            try:
                output_format = validate_output_path(output)
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(code=1) from e

        # Step 2: Load benchmark
        console.print("[cyan]Loading benchmark...[/cyan]")
        try:
            benchmark = Benchmark.load(Path(benchmark_path))
        except Exception as e:
            console.print(f"[red]Error loading benchmark: {e}[/red]")
            raise typer.Exit(code=1) from e

        console.print(f"[green]✓ Loaded benchmark: {benchmark.name or 'Unnamed'}[/green]")
        console.print(f"  Total questions: {len(benchmark.get_all_questions())}")

        # Step 3: Load or build config
        # Priority: interactive > preset+CLI > CLI only
        selected_question_indices = None
        if interactive:
            from .interactive import build_config_interactively

            config, selected_question_indices = build_config_interactively(benchmark, mode=_mode)
        else:
            # Load preset if provided, otherwise use None (will build from CLI args/defaults)
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

            # Build config with CLI overrides (or from scratch if no preset)
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
                    rubric=rubric,
                    abstention=abstention,
                    embedding_check=embedding_check,
                    deep_judgment=deep_judgment,
                    evaluation_mode=evaluation_mode,
                    embedding_threshold=embedding_threshold,
                    embedding_model=embedding_model,
                    async_execution=async_execution,
                    async_workers=async_workers,
                    preset_config=preset_config,
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
                            rubric is not None,
                            abstention is not None,
                            embedding_check is not None,
                            deep_judgment is not None,
                            evaluation_mode,
                            embedding_threshold,
                            embedding_model,
                            async_execution is not None,
                            async_workers,
                        ]
                    )
                    if has_cli_overrides:
                        console.print("[dim]CLI arguments will override preset values[/dim]")
                else:
                    console.print("[dim]Using default configuration with CLI overrides[/dim]")

            except Exception as e:
                console.print(f"[red]Error building configuration: {e}[/red]")
                raise typer.Exit(code=1) from e

        # Step 4: Get and filter templates
        all_templates = benchmark.get_finished_templates()

        if not all_templates:
            console.print("[yellow]Warning: No finished templates found in benchmark[/yellow]")
            raise typer.Exit(code=1)

        # Filter templates
        templates = all_templates
        if selected_question_indices is not None:
            # Use indices from interactive mode
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

        # Step 5: Run verification
        console.print("\n[bold cyan]Starting verification...[/bold cyan]")
        console.print(f"  Questions: {len(templates)}")
        console.print(f"  Answering models: {len(config.answering_models)}")
        console.print(f"  Parsing models: {len(config.parsing_models)}")
        console.print(f"  Replicates: {config.replicate_count}")
        console.print()

        start_time = time.time()

        # Run verification with optional progress bar
        if verbose:
            # Calculate total verifications
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
                task = progress.add_task("Verifying questions...", total=total_verifications)

                def progress_callback(current: int, _total: int, result: VerificationResult | None = None) -> None:
                    """Update progress bar on each verification completion."""
                    progress.update(task, completed=current)
                    if result and result.verify_result is not None:
                        # Add pass/fail indicator to description
                        status = "✓" if result.verify_result else "✗"
                        progress.update(task, description=f"Verifying questions... {status}")

                results = run_verification_batch(
                    templates=templates,
                    config=config,
                    run_name="cli-verification",
                    global_rubric=benchmark.get_global_rubric(),
                    progress_callback=progress_callback,
                )
        else:
            results = run_verification_batch(
                templates=templates,
                config=config,
                run_name="cli-verification",
                global_rubric=benchmark.get_global_rubric(),
                progress_callback=None,
            )

        end_time = time.time()
        duration = end_time - start_time

        # Step 6: Export or display results
        if output:
            console.print(f"\n[cyan]Exporting results to {output}...[/cyan]")

            # Create job for export (comprehensive backend format)
            job = create_export_job(results, config, "cli-verification", start_time, end_time)

            # Export using backend exporter (includes all fields)
            if output_format == "json":
                export_data = export_verification_results_json(job, results)
            else:  # csv
                export_data = export_verification_results_csv(job, results, benchmark.get_global_rubric())

            # Write to file
            output.write_text(export_data)
            console.print(f"[green]✓ Results exported to {output}[/green]")

        # Display summary
        total = len(results)
        passed = sum(1 for r in results.values() if r.verify_result)
        failed = total - passed
        errors = sum(1 for r in results.values() if not r.completed_without_errors)

        console.print("\n[bold]Verification Summary:[/bold]")
        console.print(f"  Total: {total}")
        console.print(f"  Passed: [green]{passed}[/green]")
        console.print(f"  Failed: [red]{failed}[/red]")
        if errors > 0:
            console.print(f"  Errors: [yellow]{errors}[/yellow]")
        console.print(f"  Duration: {duration:.2f}s")

        console.print("\n[green]✓ Verification complete![/green]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e
