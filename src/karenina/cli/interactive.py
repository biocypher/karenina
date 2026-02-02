"""
Interactive configuration builder.

This module implements the interactive configuration builder with two modes:
- Basic mode: Essential parameters only
- Advanced mode: All available parameters
"""

from typing import Any, Literal, cast

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

from .utils import cli_error, parse_question_indices

console = Console()


def _prompt_float_range(prompt_text: str, min_val: float, max_val: float, default: str) -> float:
    """Prompt for a float value within a range, exiting on invalid input."""
    value_str = Prompt.ask(prompt_text, default=default)
    try:
        value = float(value_str)
        if not min_val <= value <= max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")
    except ValueError as e:
        cli_error(str(e), e)
    return value


def _prompt_int_min(prompt_text: str, min_val: int, default: str) -> int:
    """Prompt for an integer value with a minimum, exiting on invalid input."""
    value_str = Prompt.ask(prompt_text, default=default)
    try:
        value = int(value_str)
        if value < min_val:
            if min_val == 0:
                raise ValueError("Value must be non-negative")
            else:
                raise ValueError(f"Value must be at least {min_val}")
    except ValueError as e:
        cli_error(str(e), e)
    return value


def build_config_interactively(
    benchmark: Benchmark, mode: str = "basic"
) -> tuple[VerificationConfig, list[int], bool, bool]:
    """
    Build VerificationConfig interactively through prompts.

    Args:
        benchmark: Loaded benchmark for question display
        mode: "basic" or "advanced"

    Returns:
        Tuple of (VerificationConfig object, list of selected question indices, show_progress_bar, progressive_save_enabled)

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
        cli_error("No finished templates found in benchmark")

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
            cli_error(str(e), e)

    console.print(f"[green]✓ Selected {len(selected_indices)} question(s)[/green]\n")

    # Step 2: Replicate count
    console.print("[bold]Step 2: Replicate Count[/bold]")
    replicate_count = _prompt_int_min("Number of replicates per verification", min_val=1, default="1")
    console.print(f"[green]✓ Replicate count: {replicate_count}[/green]\n")

    # Step 3: Feature flags
    console.print("[bold]Step 3: Feature Configuration[/bold]")

    # Evaluation mode - determines if/how rubrics are used
    console.print("\n[dim]Evaluation modes:[/dim]")
    console.print("[dim]  • template_only: Verify structured output only (fastest)[/dim]")
    console.print("[dim]  • template_and_rubric: Verify structure + evaluate quality criteria[/dim]")
    console.print("[dim]  • rubric_only: Evaluate quality criteria only (no structure required)[/dim]\n")

    evaluation_mode = Prompt.ask(
        "Evaluation mode",
        choices=["template_only", "template_and_rubric", "rubric_only"],
        default="template_only",
    )

    # Derive rubric_enabled from evaluation_mode
    rubric_enabled = evaluation_mode in ["template_and_rubric", "rubric_only"]

    abstention_enabled = Confirm.ask("Enable abstention detection?", default=False)
    sufficiency_enabled = Confirm.ask("Enable trace sufficiency detection?", default=False)

    # Embedding check
    embedding_check_enabled = Confirm.ask("Enable embedding check?", default=False)
    embedding_check_model = "all-MiniLM-L6-v2"
    embedding_check_threshold = 0.85

    if embedding_check_enabled:
        embedding_check_model = Prompt.ask("Embedding model", default="all-MiniLM-L6-v2")
        embedding_check_threshold = _prompt_float_range("Embedding similarity threshold (0.0-1.0)", 0.0, 1.0, "0.85")

    deep_judgment_enabled = Confirm.ask("Enable deep judgment?", default=False)

    console.print("[green]✓ Features configured[/green]\n")

    # Advanced mode: Additional configuration
    rubric_trait_names = None
    deep_judgment_max_excerpts = 3
    deep_judgment_fuzzy_threshold = 0.80
    deep_judgment_retry_attempts = 2
    deep_judgment_search_enabled = False
    deep_judgment_search_tool = "tavily"
    few_shot_config = None
    async_enabled = True
    async_max_workers = 2

    if mode == "advanced":
        console.print("[bold]Advanced Configuration[/bold]")

        # Rubric trait names filter (if rubric enabled)
        if rubric_enabled and Confirm.ask("Filter specific rubric traits?", default=False):
            traits_str = Prompt.ask("Rubric trait names (comma-separated)")
            rubric_trait_names = [t.strip() for t in traits_str.split(",")]

        if deep_judgment_enabled:
            dj = _configure_deep_judgment()
            deep_judgment_max_excerpts = dj["max_excerpts"]
            deep_judgment_fuzzy_threshold = dj["fuzzy_threshold"]
            deep_judgment_retry_attempts = dj["retry_attempts"]
            deep_judgment_search_enabled = dj["search_enabled"]
            deep_judgment_search_tool = dj["search_tool"]

        djr = _configure_deep_judgment_rubric(rubric_enabled)
        deep_judgment_rubric_mode = djr["mode"]
        deep_judgment_rubric_global_excerpts = djr["global_excerpts"]
        deep_judgment_rubric_max_excerpts_default = djr["max_excerpts_default"]
        deep_judgment_rubric_fuzzy_match_threshold_default = djr["fuzzy_match_threshold_default"]
        deep_judgment_rubric_excerpt_retry_attempts_default = djr["excerpt_retry_attempts_default"]
        deep_judgment_rubric_search_enabled = djr["search_enabled"]
        deep_judgment_rubric_search_tool = djr["search_tool"]
        deep_judgment_rubric_config = djr["config"]

        few_shot_config = _configure_few_shot()
        async_enabled, async_max_workers = _configure_async()

        console.print("[green]✓ Advanced configuration complete[/green]\n")

    # Step 4: Collect answering models
    console.print("[bold]Step 4: Answering Models[/bold]")
    answering_models = []

    while True:
        model = _prompt_for_model("answering", mode=mode)
        answering_models.append(model)
        console.print(f"[green]✓ Added answering model: {model.model_name}[/green]\n")

        if not Confirm.ask("Add another answering model?", default=False):
            break

    # Step 5: Collect parsing models
    console.print("[bold]Step 5: Parsing Models[/bold]")
    parsing_models = []

    while True:
        model = _prompt_for_model("parsing", mode=mode)
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
        rubric_trait_names=rubric_trait_names,
        evaluation_mode=evaluation_mode,
        abstention_enabled=abstention_enabled,
        sufficiency_enabled=sufficiency_enabled,
        embedding_check_enabled=embedding_check_enabled,
        embedding_check_model=embedding_check_model,
        embedding_check_threshold=embedding_check_threshold,
        deep_judgment_enabled=deep_judgment_enabled,
        deep_judgment_max_excerpts_per_attribute=deep_judgment_max_excerpts,
        deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_threshold,
        deep_judgment_excerpt_retry_attempts=deep_judgment_retry_attempts,
        deep_judgment_search_enabled=deep_judgment_search_enabled,
        deep_judgment_search_tool=deep_judgment_search_tool,
        deep_judgment_rubric_mode=deep_judgment_rubric_mode,
        deep_judgment_rubric_global_excerpts=deep_judgment_rubric_global_excerpts,
        deep_judgment_rubric_max_excerpts_default=deep_judgment_rubric_max_excerpts_default,
        deep_judgment_rubric_fuzzy_match_threshold_default=deep_judgment_rubric_fuzzy_match_threshold_default,
        deep_judgment_rubric_excerpt_retry_attempts_default=deep_judgment_rubric_excerpt_retry_attempts_default,
        deep_judgment_rubric_search_enabled=deep_judgment_rubric_search_enabled,
        deep_judgment_rubric_search_tool=deep_judgment_rubric_search_tool,
        deep_judgment_rubric_config=deep_judgment_rubric_config,
        few_shot_config=few_shot_config,
        async_enabled=async_enabled,
        async_max_workers=async_max_workers,
    )

    # Display summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Answering models: {len(answering_models)}")
    console.print(f"  Parsing models: {len(parsing_models)}")
    console.print(f"  Replicates: {replicate_count}")
    console.print(f"  Evaluation mode: {evaluation_mode}")
    console.print(f"  Abstention: {'enabled' if abstention_enabled else 'disabled'}")
    console.print(f"  Sufficiency: {'enabled' if sufficiency_enabled else 'disabled'}")
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

    # Step 7: Ask about progress bar
    console.print("\n[bold]Step 7: Progress Display[/bold]")
    show_progress_bar = Confirm.ask("Show progress bar during verification?", default=True)
    if show_progress_bar:
        console.print("[green]✓ Progress bar will be displayed during verification[/green]")
    else:
        console.print("[dim]Progress bar will be hidden[/dim]")

    # Step 8: Ask about progressive save
    console.print("\n[bold]Step 8: Progressive Save[/bold]")
    console.print("[dim]Progressive save enables crash recovery by saving results incrementally.[/dim]")
    console.print("[dim]If interrupted, you can resume with: karenina verify --resume <state_file>[/dim]")
    progressive_save_enabled = Confirm.ask("Enable progressive save?", default=False)
    if progressive_save_enabled:
        console.print("[green]✓ Progressive save enabled - results will be saved incrementally[/green]")
    else:
        console.print("[dim]Progressive save disabled[/dim]")

    console.print("\n[green]✓ Configuration complete![/green]\n")

    return config, selected_indices, show_progress_bar, progressive_save_enabled


def _configure_deep_judgment() -> dict[str, Any]:
    """Prompt for deep judgment settings. Returns dict with max_excerpts, fuzzy_threshold, retry_attempts, search_enabled, search_tool."""
    console.print("\n[cyan]Deep Judgment Settings:[/cyan]")
    max_excerpts = _prompt_int_min("Max excerpts per attribute", min_val=1, default="3")
    fuzzy_threshold = _prompt_float_range("Fuzzy match threshold (0.0-1.0)", 0.0, 1.0, "0.80")
    retry_attempts = _prompt_int_min("Excerpt retry attempts", min_val=0, default="2")

    search_enabled = Confirm.ask("Enable search validation?", default=False)
    search_tool = "tavily"
    if search_enabled:
        search_tool = Prompt.ask("Search tool", default="tavily")

    return {
        "max_excerpts": max_excerpts,
        "fuzzy_threshold": fuzzy_threshold,
        "retry_attempts": retry_attempts,
        "search_enabled": search_enabled,
        "search_tool": search_tool,
    }


def _configure_deep_judgment_rubric(rubric_enabled: bool) -> dict[str, Any]:
    """Prompt for deep judgment rubric settings. Returns dict with mode, global_excerpts, thresholds, search, config."""
    result: dict[str, Any] = {
        "mode": "disabled",
        "global_excerpts": True,
        "max_excerpts_default": 7,
        "fuzzy_match_threshold_default": 0.80,
        "excerpt_retry_attempts_default": 2,
        "search_enabled": False,
        "search_tool": "tavily",
        "config": None,
    }

    if not rubric_enabled:
        return result

    console.print("\n[cyan]Deep Judgment Rubric Settings:[/cyan]")
    console.print("[dim]Deep judgment modes:[/dim]")
    console.print("[dim]  • disabled: No deep judgment for rubrics (default)[/dim]")
    console.print("[dim]  • enable_all: Apply to all LLM traits with global settings[/dim]")
    console.print("[dim]  • use_checkpoint: Use settings saved in checkpoint file[/dim]")
    console.print("[dim]  • custom: Load per-trait config from JSON file[/dim]\n")

    result["mode"] = Prompt.ask(
        "Deep judgment rubric mode",
        choices=["disabled", "enable_all", "use_checkpoint", "custom"],
        default="disabled",
    )

    if result["mode"] == "enable_all":
        result["global_excerpts"] = Confirm.ask("Enable excerpts for all rubric traits?", default=True)
        result["max_excerpts_default"] = _prompt_int_min("Max excerpts per rubric trait", min_val=1, default="7")
        result["fuzzy_match_threshold_default"] = _prompt_float_range(
            "Fuzzy match threshold (0.0-1.0)", 0.0, 1.0, "0.80"
        )
        result["excerpt_retry_attempts_default"] = _prompt_int_min("Excerpt retry attempts", min_val=0, default="2")

        result["search_enabled"] = Confirm.ask("Enable search validation?", default=False)
        if result["search_enabled"]:
            result["search_tool"] = Prompt.ask("Search tool", default="tavily")

    elif result["mode"] == "custom":
        console.print("[yellow]Custom mode requires a JSON config file[/yellow]")
        config_path_str = Prompt.ask("Path to custom rubric config JSON")
        import json
        from pathlib import Path

        try:
            config_path = Path(config_path_str)
            with open(config_path) as f:
                result["config"] = json.load(f)
            console.print(f"[green]✓ Loaded custom config from {config_path}[/green]")
        except Exception as e:
            cli_error(f"loading config: {e}", e)

    elif result["mode"] == "use_checkpoint":
        console.print("[dim]Using deep judgment settings from checkpoint file[/dim]")

    return result


def _configure_few_shot() -> Any:
    """Prompt for few-shot configuration. Returns FewShotConfig or None."""
    console.print("\n[cyan]Few-Shot Configuration:[/cyan]")
    if not Confirm.ask("Enable few-shot prompting?", default=False):
        return None

    from karenina.schemas.workflow.models import FewShotConfig

    few_shot_mode_str = Prompt.ask("Few-shot mode", choices=["all", "k-shot", "custom", "none"], default="all")
    few_shot_mode = cast(Literal["all", "k-shot", "custom", "none"], few_shot_mode_str)
    few_shot_k = _prompt_int_min("Few-shot k (number of examples)", min_val=1, default="3")

    return FewShotConfig(enabled=True, global_mode=few_shot_mode, global_k=few_shot_k)


def _configure_async() -> tuple[bool, int]:
    """Prompt for async execution settings. Returns (async_enabled, async_max_workers)."""
    console.print("\n[cyan]Async Execution Settings:[/cyan]")
    async_enabled = Confirm.ask("Enable async execution?", default=True)
    async_max_workers = 2
    if async_enabled:
        async_max_workers = _prompt_int_min("Max parallel workers", min_val=1, default="2")
    return async_enabled, async_max_workers


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


def _prompt_for_model(model_type: str, mode: str = "basic") -> ModelConfig:
    """
    Prompt user for model configuration details.

    Args:
        model_type: Type of model ("answering" or "parsing")
        mode: Configuration mode ("basic" or "advanced")

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
    temperature = _prompt_float_range("Temperature (0.0-2.0)", 0.0, 2.0, "0.1")

    # Max retries
    max_retries = _prompt_int_min("Max retries", min_val=0, default="2")

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

    # MCP tools configuration (advanced mode only, answering model only)
    mcp_urls_dict = None
    mcp_tool_filter = None

    if mode == "advanced" and interface != INTERFACE_MANUAL and model_type == "answering":
        console.print("\n[cyan]MCP Tools Configuration (optional):[/cyan]")
        console.print("[dim]Note: MCP tools are only used by the answering model for generating responses[/dim]")

        if Confirm.ask("Configure MCP tools?", default=False):
            temp_mcp_urls: dict[str, str] = {}

            # Loop to add MCP servers one at a time
            while True:
                console.print("\n[cyan]Add MCP Server:[/cyan]")
                server_name = Prompt.ask("MCP server name")
                server_url = Prompt.ask("MCP server URL")

                # Add to dictionary
                temp_mcp_urls[server_name] = server_url

                # Validate this server immediately
                console.print(f"\n[cyan]Validating MCP server '{server_name}'...[/cyan]")
                try:
                    from karenina.utils.mcp import sync_fetch_tool_descriptions

                    tool_descriptions = sync_fetch_tool_descriptions({server_name: server_url})

                    if tool_descriptions:
                        console.print(f"[green]✓ Successfully connected to '{server_name}'[/green]")
                        console.print(f"[dim]Found {len(tool_descriptions)} available tool(s):[/dim]")
                        tool_items = list(tool_descriptions.items())
                        for name, desc in tool_items[:10]:  # Show first 10 tools
                            desc_preview = (desc[:60] + "...") if len(desc) > 60 else desc
                            console.print(f"  [dim]• {name}: {desc_preview}[/dim]")
                        if len(tool_items) > 10:
                            console.print(f"  [dim]... and {len(tool_items) - 10} more[/dim]")
                    else:
                        console.print(f"[yellow]Warning: No tools found from server '{server_name}'[/yellow]")

                except Exception as e:
                    console.print(f"[red]Failed to validate MCP server '{server_name}': {e}[/red]")
                    if not Confirm.ask("Continue with this MCP server anyway?", default=False):
                        console.print(f"[yellow]Removing server '{server_name}' from configuration.[/yellow]")
                        del temp_mcp_urls[server_name]

                # Ask if user wants to add another server
                if not Confirm.ask("\nAdd another MCP server?", default=False):
                    break

            # If servers were successfully added, assign to mcp_urls_dict
            if temp_mcp_urls:
                mcp_urls_dict = temp_mcp_urls
            else:
                console.print("[yellow]No MCP servers configured.[/yellow]")

            # MCP tool filter
            if mcp_urls_dict and Confirm.ask("\nFilter specific MCP tools?", default=False):  # type: ignore[arg-type]
                tools_str = Prompt.ask("MCP tool names (comma-separated)")
                mcp_tool_filter = [t.strip() for t in tools_str.split(",")]

                # Validate tool filter if we have tools
                try:
                    from karenina.utils.mcp import sync_fetch_tool_descriptions

                    all_tool_descriptions = sync_fetch_tool_descriptions(mcp_urls_dict, tool_filter=mcp_tool_filter)
                    available_tool_names = list(all_tool_descriptions.keys())

                    if not all_tool_descriptions:
                        console.print("[yellow]Warning: No tools match the specified filter[/yellow]")
                        console.print(f"[dim]Requested: {', '.join(mcp_tool_filter)}[/dim]")
                        if not Confirm.ask("Continue with this tool filter anyway?", default=False):
                            mcp_tool_filter = None
                    else:
                        console.print(
                            f"[green]✓ Tool filter validated ({len(all_tool_descriptions)} tool(s) selected)[/green]"
                        )
                        console.print(f"[dim]Selected tools: {', '.join(available_tool_names)}[/dim]")

                except Exception as e:
                    console.print(f"[yellow]Warning: Could not validate tool filter: {e}[/yellow]")

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

    if mcp_urls_dict:
        config_dict["mcp_urls_dict"] = mcp_urls_dict

    if mcp_tool_filter:
        config_dict["mcp_tool_filter"] = mcp_tool_filter

    # Skip validation for manual interface as it requires manual_traces
    if interface == INTERFACE_MANUAL:
        console.print("[yellow]Manual interface selected - validation skipped[/yellow]")

    return ModelConfig(**config_dict)
