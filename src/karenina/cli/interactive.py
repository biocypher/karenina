"""
Interactive configuration builder.

This module implements the interactive configuration builder with two modes:
- Basic mode: Essential parameters only
- Advanced mode: All available parameters
"""

from typing import Any

import typer
from pydantic import SecretStr
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from karenina.benchmark import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.workflow.models import (
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENAI_ENDPOINT,
    INTERFACE_OPENROUTER,
)
from karenina.schemas.workflow.verification import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
)

from .utils import parse_question_indices

console = Console()


def build_config_interactively(benchmark: Benchmark, mode: str = "basic") -> tuple[VerificationConfig, list[int]]:
    """
    Build VerificationConfig interactively through prompts.

    Args:
        benchmark: Loaded benchmark for question display
        mode: "basic" or "advanced"

    Returns:
        Tuple of (VerificationConfig object, list of selected question indices)

    Raises:
        ValueError: If mode is invalid or user provides invalid input
    """
    if mode not in ["basic", "advanced"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'basic' or 'advanced'")

    console.print("\n[bold cyan]Interactive Verification Configuration[/bold cyan]")
    console.print(f"Mode: [yellow]{mode}[/yellow]\n")

    # Get all finished templates
    templates = benchmark.get_finished_templates()

    if not templates:
        console.print("[red]Error: No finished templates found in benchmark[/red]")
        raise typer.Exit(code=1)

    # Step 1: Display questions and select subset
    console.print("[bold]Step 1: Question Selection[/bold]")
    _display_questions_table(templates)

    # Prompt for question selection
    console.print("[dim]Examples: 'all', '0,1,2', '5-10'[/dim]")
    question_selection = Prompt.ask("Select questions", default="all")

    # Parse selection
    if question_selection.lower() == "all":
        selected_indices = list(range(len(templates)))
    else:
        try:
            selected_indices = parse_question_indices(question_selection, len(templates))
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1) from e

    console.print(f"[green]✓ Selected {len(selected_indices)} question(s)[/green]\n")

    # Step 2: Replicate count
    console.print("[bold]Step 2: Replicate Count[/bold]")
    replicate_count_str = Prompt.ask("Number of replicates per verification", default="1")

    try:
        replicate_count = int(replicate_count_str)
        if replicate_count < 1:
            raise ValueError("Replicate count must be at least 1")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e

    console.print(f"[green]✓ Replicate count: {replicate_count}[/green]\n")

    # Step 3: Feature flags
    console.print("[bold]Step 3: Feature Configuration[/bold]")

    rubric_enabled = Confirm.ask("Enable rubric evaluation?", default=False)
    abstention_enabled = Confirm.ask("Enable abstention detection?", default=False)

    # Embedding check
    embedding_check_enabled = Confirm.ask("Enable embedding check?", default=False)
    embedding_check_model = "all-MiniLM-L6-v2"
    embedding_check_threshold = 0.85

    if embedding_check_enabled:
        embedding_check_model = Prompt.ask("Embedding model", default="all-MiniLM-L6-v2")
        threshold_str = Prompt.ask("Embedding similarity threshold (0.0-1.0)", default="0.85")
        try:
            embedding_check_threshold = float(threshold_str)
            if not 0.0 <= embedding_check_threshold <= 1.0:
                raise ValueError("Threshold must be between 0.0 and 1.0")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1) from e

    deep_judgment_enabled = Confirm.ask("Enable deep judgment?", default=False)

    console.print("[green]✓ Features configured[/green]\n")

    # Step 4: Collect answering models
    console.print("[bold]Step 4: Answering Models[/bold]")
    answering_models = []

    while True:
        model = _prompt_for_model("answering")
        answering_models.append(model)
        console.print(f"[green]✓ Added answering model: {model.model_name}[/green]\n")

        if not Confirm.ask("Add another answering model?", default=False):
            break

    # Step 5: Collect parsing models
    console.print("[bold]Step 5: Parsing Models[/bold]")
    parsing_models = []

    while True:
        model = _prompt_for_model("parsing")
        parsing_models.append(model)
        console.print(f"[green]✓ Added parsing model: {model.model_name}[/green]\n")

        if not Confirm.ask("Add another parsing model?", default=False):
            break

    # Build VerificationConfig
    config = VerificationConfig(
        answering_models=answering_models,
        parsing_models=parsing_models,
        replicate_count=replicate_count,
        rubric_enabled=rubric_enabled,
        abstention_enabled=abstention_enabled,
        embedding_check_enabled=embedding_check_enabled,
        embedding_check_model=embedding_check_model,
        embedding_check_threshold=embedding_check_threshold,
        deep_judgment_enabled=deep_judgment_enabled,
    )

    # Display summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Answering models: {len(answering_models)}")
    console.print(f"  Parsing models: {len(parsing_models)}")
    console.print(f"  Replicates: {replicate_count}")
    console.print(f"  Rubric: {'enabled' if rubric_enabled else 'disabled'}")
    console.print(f"  Abstention: {'enabled' if abstention_enabled else 'disabled'}")
    console.print(f"  Embedding check: {'enabled' if embedding_check_enabled else 'disabled'}")
    console.print(f"  Deep judgment: {'enabled' if deep_judgment_enabled else 'disabled'}")

    # Step 6: Optionally save as preset
    if Confirm.ask("\nSave this configuration as a preset?", default=False):
        preset_name = Prompt.ask("Preset name")
        preset_description = Prompt.ask("Description (optional)", default="")

        try:
            preset_info = config.save_preset(
                name=preset_name,
                description=preset_description if preset_description else None,
            )
            console.print(f"[green]✓ Preset saved to: {preset_info['filepath']}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving preset: {e}[/red]")

    console.print("\n[green]✓ Configuration complete![/green]\n")

    return config, selected_indices


def _display_questions_table(templates: list[Any]) -> None:
    """
    Display a table of questions with indices.

    Args:
        templates: List of FinishedTemplate objects
    """
    table = Table(title="Available Questions", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Question ID", style="yellow")
    table.add_column("Question", style="white")

    for i, template in enumerate(templates):
        question_id = template.question_id
        # Truncate question text if too long
        question_text = (
            template.question_text[:80] + "..." if len(template.question_text) > 80 else template.question_text
        )
        table.add_row(str(i), question_id, question_text)

    console.print(table)
    console.print(f"\n[dim]Total: {len(templates)} question(s)[/dim]\n")


def _prompt_for_model(model_type: str) -> ModelConfig:
    """
    Prompt user for model configuration details.

    Args:
        model_type: Type of model ("answering" or "parsing")

    Returns:
        Configured ModelConfig instance
    """
    console.print(f"\n[cyan]Configure {model_type} model:[/cyan]")

    # Prompt for interface type
    interface_choices = [INTERFACE_LANGCHAIN, INTERFACE_OPENROUTER, INTERFACE_OPENAI_ENDPOINT, INTERFACE_MANUAL]
    interface_choice_str = "/".join(interface_choices)

    interface = Prompt.ask(
        f"Interface type ({interface_choice_str})",
        choices=interface_choices,
        default=INTERFACE_LANGCHAIN,
    )

    # Prompt for model details
    model_id = Prompt.ask("Model ID", default=f"{model_type}-model-1")
    model_name = Prompt.ask("Model name", default="gpt-4.1-mini")

    # Provider (only for langchain)
    model_provider = None
    if interface == INTERFACE_LANGCHAIN:
        model_provider = Prompt.ask("Model provider", default="openai")

    # Temperature
    temperature_str = Prompt.ask("Temperature (0.0-2.0)", default="0.1")
    try:
        temperature = float(temperature_str)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Max retries
    max_retries_str = Prompt.ask("Max retries", default="2")
    try:
        max_retries = int(max_retries_str)
        if max_retries < 0:
            raise ValueError("Max retries must be non-negative")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e

    # System prompt
    default_system_prompt = (
        DEFAULT_ANSWERING_SYSTEM_PROMPT if model_type == "answering" else DEFAULT_PARSING_SYSTEM_PROMPT
    )

    use_custom_prompt = Confirm.ask("Use custom system prompt?", default=False)
    system_prompt = default_system_prompt

    if use_custom_prompt:
        system_prompt = Prompt.ask("System prompt", default=default_system_prompt)

    # Endpoint configuration (only for openai_endpoint)
    endpoint_base_url = None
    endpoint_api_key = None

    if interface == INTERFACE_OPENAI_ENDPOINT:
        endpoint_base_url = Prompt.ask("Endpoint base URL")
        endpoint_api_key_str = Prompt.ask("API key", password=True)
        endpoint_api_key = SecretStr(endpoint_api_key_str)

    # Manual interface handling
    if interface == INTERFACE_MANUAL:
        console.print("[yellow]Note: Manual interface requires manual_traces to be set programmatically[/yellow]")
        console.print("[yellow]This will be configured later in the code[/yellow]")

    # Build ModelConfig
    config_dict: dict[str, Any] = {
        "id": model_id,
        "model_name": model_name,
        "temperature": temperature,
        "interface": interface,
        "system_prompt": system_prompt,
        "max_retries": max_retries,
    }

    if model_provider:
        config_dict["model_provider"] = model_provider

    if endpoint_base_url:
        config_dict["endpoint_base_url"] = endpoint_base_url

    if endpoint_api_key:
        config_dict["endpoint_api_key"] = endpoint_api_key

    # Skip validation for manual interface as it requires manual_traces
    if interface == INTERFACE_MANUAL:
        console.print("[yellow]Manual interface selected - validation skipped[/yellow]")

    return ModelConfig(**config_dict)
