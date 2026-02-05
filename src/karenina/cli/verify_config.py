"""
Configuration building helpers for the verify command.

Handles config construction from CLI args, presets, and manual traces.
"""

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from karenina.adapters.manual import load_manual_traces_from_file
from karenina.benchmark import Benchmark
from karenina.schemas import VerificationConfig

from .utils import cli_error, get_preset_path, get_traces_path

console = Console()


def build_config_from_cli_args(
    answering_model: str | None,
    answering_provider: str | None,
    answering_id: str,
    parsing_model: str | None,
    parsing_provider: str | None,
    parsing_id: str,
    temperature: float | None,
    interface: str | None,
    replicate_count: int | None,
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

    Handles CLI-specific concerns (file loading for deep_judgment_rubric_config)
    then delegates to VerificationConfig.from_overrides() for config construction.

    Args:
        CLI argument values (all optional)
        preset_config: Optional preset configuration to use as base

    Returns:
        VerificationConfig with CLI overrides applied
    """
    # CLI-specific: load custom rubric config JSON if a file path was provided
    rubric_config_dict = None
    if deep_judgment_rubric_config is not None:
        import json

        try:
            with open(deep_judgment_rubric_config) as f:
                rubric_config_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to load custom rubric config from {deep_judgment_rubric_config}: {e}") from e

    # Resolve interface for answering vs parsing:
    # Parsing model should NOT use manual interface (only answering model can)
    answering_interface = interface
    parsing_interface = interface if interface != "manual" else "langchain"

    return VerificationConfig.from_overrides(
        base=preset_config,
        # Model configuration
        answering_model=answering_model,
        answering_provider=answering_provider,
        answering_id=answering_id,
        answering_interface=answering_interface,
        parsing_model=parsing_model,
        parsing_provider=parsing_provider,
        parsing_id=parsing_id,
        parsing_interface=parsing_interface,
        temperature=temperature,
        manual_traces=manual_traces_obj,
        # Execution settings
        replicate_count=replicate_count,
        # Feature flags
        abstention=abstention,
        sufficiency=sufficiency,
        embedding_check=embedding_check,
        deep_judgment=deep_judgment,
        # Evaluation settings
        evaluation_mode=evaluation_mode,
        embedding_threshold=embedding_threshold,
        embedding_model=embedding_model,
        async_execution=async_execution,
        async_workers=async_workers,
        # Trace filtering
        use_full_trace_for_template=use_full_trace_for_template,
        use_full_trace_for_rubric=use_full_trace_for_rubric,
        # Deep judgment rubric settings
        deep_judgment_rubric_mode=deep_judgment_rubric_mode,
        deep_judgment_rubric_excerpts=deep_judgment_rubric_excerpts,
        deep_judgment_rubric_max_excerpts=deep_judgment_rubric_max_excerpts,
        deep_judgment_rubric_fuzzy_threshold=deep_judgment_rubric_fuzzy_threshold,
        deep_judgment_rubric_retry_attempts=deep_judgment_rubric_retry_attempts,
        deep_judgment_rubric_search=deep_judgment_rubric_search,
        deep_judgment_rubric_search_tool=deep_judgment_rubric_search_tool,
        deep_judgment_rubric_config=rubric_config_dict,
    )


def validate_cli_config_requirements(
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


def validate_deep_judgment_rubric_settings(
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
        cli_error(
            f"Invalid deep_judgment_rubric_mode '{deep_judgment_rubric_mode}'. Valid modes: {', '.join(valid_modes)}"
        )

    if not 0.0 <= deep_judgment_rubric_fuzzy_threshold <= 1.0:
        cli_error("deep_judgment_rubric_fuzzy_threshold must be between 0.0 and 1.0")

    if deep_judgment_rubric_mode == "use_checkpoint":
        console.print(
            "[yellow]Note: use_checkpoint mode requires deep judgment settings to be saved in the checkpoint file.[/yellow]"
        )
        console.print(
            "[yellow]If the checkpoint doesn't have deep judgment config, traits will be disabled by default.[/yellow]"
        )

    if deep_judgment_rubric_mode == "custom" and not deep_judgment_rubric_config:
        cli_error("custom mode requires --deep-judgment-rubric-config to be specified")


def load_manual_traces(manual_traces: Path, benchmark: Benchmark) -> object:
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
        cli_error(str(e), e)
    except Exception as e:
        cli_error(f"loading manual traces: {e}", e)


def build_config_non_interactive(
    preset: Path | None,
    interface: str | None,
    answering_model: str | None,
    answering_provider: str | None,
    answering_id: str,
    parsing_model: str | None,
    parsing_provider: str | None,
    parsing_id: str,
    temperature: float | None,
    replicate_count: int | None,
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
        except (FileNotFoundError, ValueError, ValidationError) as e:
            cli_error(f"loading preset: {e}", e)

    # Validate required parameters when no preset is used
    if not preset_config:
        validation_errors = validate_cli_config_requirements(
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
        manual_traces_obj = load_manual_traces(manual_traces, benchmark)

    # Build config with CLI overrides
    try:
        config = build_config_from_cli_args(
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
        cli_error(f"building configuration: {e}", e)

    # Validate deep judgment rubric settings
    validate_deep_judgment_rubric_settings(
        deep_judgment_rubric_mode, deep_judgment_rubric_fuzzy_threshold, deep_judgment_rubric_config
    )

    return config, progressive_save
