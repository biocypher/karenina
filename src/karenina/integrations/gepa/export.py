"""Export utilities for GEPA-optimized prompts.

Provides functions to export optimized prompts in various formats:
- Karenina verification presets (JSON config files)
- Raw prompts with metadata (JSON)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from karenina.integrations.gepa.config import OptimizationTarget

if TYPE_CHECKING:
    from karenina.schemas.workflow.verification.config import VerificationConfig


def export_to_preset(
    optimized_prompts: dict[str, str],
    base_config: "VerificationConfig",
    output_path: Path | str,
    targets: list[OptimizationTarget] | None = None,
) -> Path:
    """Export optimized prompts as a karenina verification preset.

    Creates a JSON file compatible with `karenina verify --preset`.
    The preset contains the full VerificationConfig with optimized
    prompts injected into the appropriate locations.

    Args:
        optimized_prompts: Dict mapping component names to optimized text
            e.g., {"answering_system_prompt": "You are...", ...}
        base_config: Base VerificationConfig to use as template
        output_path: Path for the output JSON preset file
        targets: Optional list of OptimizationTargets to specify which
                 components were optimized. If None, infers from
                 optimized_prompts keys.

    Returns:
        Path to the created preset file

    Example:
        >>> export_to_preset(
        ...     {"answering_system_prompt": "You are an expert..."},
        ...     base_config=verification_config,
        ...     output_path="optimized_preset.json",
        ... )
    """
    output_path = Path(output_path)
    config = base_config.model_copy(deep=True)

    # Infer targets if not provided
    if targets is None:
        targets = []
        if "answering_system_prompt" in optimized_prompts:
            targets.append(OptimizationTarget.ANSWERING_SYSTEM_PROMPT)
        if "parsing_instructions" in optimized_prompts:
            targets.append(OptimizationTarget.PARSING_INSTRUCTIONS)
        if any(k.startswith("mcp_tool_") for k in optimized_prompts):
            targets.append(OptimizationTarget.MCP_TOOL_DESCRIPTIONS)

    # Inject optimized prompts based on targets
    if OptimizationTarget.ANSWERING_SYSTEM_PROMPT in targets:
        prompt = optimized_prompts.get("answering_system_prompt")
        if prompt:
            for model in config.answering_models:
                model.system_prompt = prompt

    if OptimizationTarget.PARSING_INSTRUCTIONS in targets:
        instructions = optimized_prompts.get("parsing_instructions")
        if instructions and hasattr(config, "parsing_instructions_override"):
            # Store as parsing_instructions_override in config
            # This will be used by the template evaluator
            config.parsing_instructions_override = instructions

    if OptimizationTarget.MCP_TOOL_DESCRIPTIONS in targets:
        # Collect MCP tool description overrides
        tool_overrides: dict[str, str] = {}
        for key, value in optimized_prompts.items():
            if key.startswith("mcp_tool_"):
                tool_name = key[9:]  # Remove "mcp_tool_" prefix
                tool_overrides[tool_name] = value

        if tool_overrides:
            for model in config.answering_models:
                if model.mcp_urls_dict:
                    model.mcp_tool_description_overrides = tool_overrides

    # Serialize config to JSON
    config_dict = config.model_dump(mode="json")

    # Add metadata about optimization
    config_dict["_gepa_optimization"] = {
        "exported_at": datetime.now().isoformat(),
        "targets": [t.value for t in targets],
        "optimized_components": list(optimized_prompts.keys()),
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return output_path


def export_prompts_json(
    optimized_prompts: dict[str, str],
    metadata: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Export just the optimized prompts with metadata.

    Creates a lightweight JSON file containing only the optimized
    prompts and associated metadata, without the full config.

    Output format:
    {
        "optimized_prompts": {...},
        "metadata": {
            "benchmark": "...",
            "timestamp": "...",
            "improvement": 0.15,
            ...
        }
    }

    Args:
        optimized_prompts: Dict mapping component names to optimized text
        metadata: Dict with optimization metadata (benchmark name, scores, etc.)
        output_path: Path for the output JSON file

    Returns:
        Path to the created JSON file

    Example:
        >>> export_prompts_json(
        ...     {"answering_system_prompt": "You are..."},
        ...     metadata={
        ...         "benchmark": "my_benchmark",
        ...         "improvement": 0.15,
        ...         "train_score": 0.85,
        ...         "val_score": 0.82,
        ...     },
        ...     output_path="optimized_prompts.json",
        ... )
    """
    output_path = Path(output_path)

    # Ensure timestamp is present
    if "timestamp" not in metadata:
        metadata["timestamp"] = datetime.now().isoformat()

    output = {
        "optimized_prompts": optimized_prompts,
        "metadata": metadata,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def load_prompts_json(path: Path | str) -> tuple[dict[str, str], dict[str, Any]]:
    """Load optimized prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file

    Returns:
        Tuple of (optimized_prompts, metadata)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if "optimized_prompts" not in data:
        raise ValueError(f"Invalid prompts file: missing 'optimized_prompts' key in {path}")

    return data["optimized_prompts"], data.get("metadata", {})


def export_comparison_report(
    runs: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Export a comparison report of multiple optimization runs.

    Creates a JSON report comparing multiple runs, useful for
    analyzing which optimization strategies work best.

    Args:
        runs: List of run data dicts (from OptimizationTracker.compare_runs)
        output_path: Path for the output JSON report

    Returns:
        Path to the created report file
    """
    output_path = Path(output_path)

    report = {
        "generated_at": datetime.now().isoformat(),
        "num_runs": len(runs),
        "runs": runs,
        "summary": _compute_run_summary(runs) if runs else {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return output_path


def _compute_run_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for a list of runs."""
    if not runs:
        return {}

    val_scores = [r.get("val_score", 0) for r in runs]
    improvements = [r.get("improvement", 0) for r in runs]

    return {
        "best_val_score": max(val_scores) if val_scores else 0,
        "worst_val_score": min(val_scores) if val_scores else 0,
        "avg_val_score": sum(val_scores) / len(val_scores) if val_scores else 0,
        "best_improvement": max(improvements) if improvements else 0,
        "worst_improvement": min(improvements) if improvements else 0,
        "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
        "unique_targets": list({target for r in runs for target in r.get("targets", [])}),
    }
