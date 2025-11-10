"""
CLI utility functions.

This module contains helper functions for the Karenina CLI:
- Preset discovery and resolution
- Question index parsing
- Output path validation
- Question filtering
- Export job creation
- Manual trace loading and validation
"""

from pathlib import Path
from typing import Any

from karenina.schemas import FinishedTemplate, VerificationConfig, VerificationJob, VerificationResult


def _get_presets_directory(presets_dir: Path | None = None) -> Path:
    """
    Get presets directory, using provided path, env var, or default.

    Args:
        presets_dir: Optional directory override

    Returns:
        Path to presets directory
    """
    import os

    if presets_dir is not None:
        return presets_dir

    # Check environment variable
    env_dir = os.getenv("KARENINA_PRESETS_DIR")
    if env_dir:
        return Path(env_dir)

    # Default to benchmark_presets/ in current directory
    return Path("benchmark_presets")


def list_presets(presets_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    List all available presets.

    Args:
        presets_dir: Directory containing presets. If None, uses default location.

    Returns:
        List of preset info dictionaries with name and filepath, sorted by name.
    """
    import os

    preset_dir = _get_presets_directory(presets_dir)

    if not preset_dir.exists():
        return []

    presets = []
    for preset_file in preset_dir.glob("*.json"):
        try:
            # Use filename (without .json extension) as the preset name
            name = preset_file.stem

            # Get file modification time for display
            mtime = os.path.getmtime(preset_file)
            from datetime import datetime

            modified = datetime.fromtimestamp(mtime).isoformat()

            preset_info = {
                "name": name,
                "filepath": str(preset_file),
                "modified": modified,
            }
            presets.append(preset_info)
        except Exception:
            # Skip invalid files
            continue

    # Sort by name
    presets.sort(key=lambda p: p["name"])
    return presets


def get_preset_path(name_or_path: str, presets_dir: Path | None = None) -> Path:
    """
    Resolve preset name to filepath.

    Args:
        name_or_path: Preset name (e.g., "gpt-oss-tools") or direct file path
        presets_dir: Directory to search for presets

    Returns:
        Path to preset file

    Raises:
        FileNotFoundError: If preset not found

    Examples:
        >>> get_preset_path("gpt-oss-tools")  # Finds presets/gpt-oss-tools.json
        >>> get_preset_path("/path/to/config.json")  # Uses direct path
    """
    # First, check if it's already a valid path
    path = Path(name_or_path)
    if path.exists() and path.is_file():
        return path.resolve()

    # Try to find by name in presets directory
    preset_dir = _get_presets_directory(presets_dir)

    if not preset_dir.exists():
        raise FileNotFoundError(f"Presets directory not found: {preset_dir}")

    # Try exact match (e.g., "gpt-oss-tools.json" or "gpt-oss-tools")
    candidate = preset_dir / name_or_path if name_or_path.endswith(".json") else preset_dir / f"{name_or_path}.json"

    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Preset '{name_or_path}' not found in {preset_dir}. Use 'karenina preset list' to see available presets."
    )


def parse_question_indices(indices_str: str, total_questions: int) -> list[int]:
    """
    Parse question indices string (e.g., '0,1,2,5-10') into list of integers.

    Args:
        indices_str: String with indices and ranges
        total_questions: Total number of questions for validation

    Returns:
        Sorted list of unique indices

    Raises:
        ValueError: If indices are invalid or out of range
    """
    indices: set[int] = set()
    parts = indices_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Handle range (e.g., "5-10")
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())

                if start < 0 or end < 0:
                    raise ValueError(f"Negative indices not allowed: {part}")

                if start > end:
                    raise ValueError(f"Invalid range (start > end): {part}")

                if start >= total_questions or end >= total_questions:
                    raise ValueError(f"Index out of range: {part} (total questions: {total_questions})")

                # Add all indices in range (inclusive)
                indices.update(range(start, end + 1))

            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid range format: {part}") from None
                raise

        # Handle single index
        else:
            try:
                index = int(part)
                if index < 0:
                    raise ValueError(f"Negative index not allowed: {index}")
                if index >= total_questions:
                    raise ValueError(f"Index out of range: {index} (total questions: {total_questions})")
                indices.add(index)
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid index: {part}") from None
                raise

    # Return sorted list
    return sorted(indices)


def validate_output_path(output_path: Path) -> str:
    """
    Validate output path and return format.

    Args:
        output_path: Path to output file

    Returns:
        Format string ('json' or 'csv')

    Raises:
        ValueError: If extension is not .json or .csv
    """
    extension = output_path.suffix.lower()

    if extension not in [".json", ".csv"]:
        raise ValueError(f"Invalid output format: {extension}. Output file must have .json or .csv extension.")

    # Ensure parent directory exists
    parent = output_path.parent
    if not parent.exists():
        raise ValueError(f"Parent directory does not exist: {parent}")

    # Return format string
    return extension[1:]  # Remove the dot (.json -> json)


def filter_templates_by_indices(templates: list[FinishedTemplate], indices: list[int]) -> list[FinishedTemplate]:
    """
    Filter templates by index positions.

    Args:
        templates: List of all templates
        indices: List of indices to keep

    Returns:
        Filtered list of templates
    """
    indices_set = set(indices)
    return [template for i, template in enumerate(templates) if i in indices_set]


def filter_templates_by_ids(templates: list[FinishedTemplate], ids: list[str]) -> list[FinishedTemplate]:
    """
    Filter templates by question IDs.

    Args:
        templates: List of all templates
        ids: List of question IDs to keep

    Returns:
        Filtered list of templates
    """
    ids_set = set(ids)
    return [template for template in templates if template.question_id in ids_set]


def create_export_job(
    results: dict[str, VerificationResult],
    config: VerificationConfig,
    run_name: str,
    start_time: float,
    end_time: float,
) -> VerificationJob:
    """
    Create VerificationJob object for export functions.

    Args:
        results: Verification results dictionary
        config: Verification configuration
        run_name: Name of the verification run
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        VerificationJob object suitable for export functions
    """
    import uuid

    # Calculate counts
    total = len(results)
    successful = sum(1 for r in results.values() if r.completed_without_errors)
    failed = total - successful

    # Create minimal job object
    job = VerificationJob(
        job_id=str(uuid.uuid4()),
        run_name=run_name or "cli-verification",
        status="completed",
        config=config,
        total_questions=total,
        successful_count=successful,
        failed_count=failed,
        start_time=start_time,  # Already a float timestamp
        end_time=end_time,  # Already a float timestamp
    )

    return job


def get_traces_path(name_or_path: str | Path) -> Path:
    """
    Resolve trace file path.

    Supports:
    - Absolute paths
    - Relative paths (from current directory)
    - Paths in traces/ directory (if exists)

    Args:
        name_or_path: Trace file path as string or Path

    Returns:
        Resolved absolute Path

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> get_traces_path("my_traces.json")  # Finds ./my_traces.json or traces/my_traces.json
        >>> get_traces_path("/path/to/traces.json")  # Uses direct path
    """
    trace_path = Path(name_or_path)

    # Try absolute/relative first
    if trace_path.exists():
        return trace_path.resolve()

    # Try in traces/ directory
    traces_dir = Path.cwd() / "traces"
    if traces_dir.exists():
        candidate = traces_dir / trace_path.name
        if candidate.exists():
            return candidate.resolve()

    # Not found
    raise FileNotFoundError(
        f"Trace file not found: {name_or_path}\n"
        f"Tried: {trace_path.absolute()}" + (f", {traces_dir / trace_path.name}" if traces_dir.exists() else "")
    )


def load_manual_traces_from_file(trace_file: Path, benchmark: Any) -> Any:
    """
    Load manual traces from JSON file and create ManualTraces object.

    Args:
        trace_file: Path to JSON file with traces (question_hash -> trace_string mapping)
        benchmark: Benchmark for question mapping

    Returns:
        ManualTraces object populated with traces

    Raises:
        FileNotFoundError: If trace file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        ManualTraceError: If trace validation fails

    Example JSON format:
        {
            "936dbc8755f623c951d96ea2b03e13bc": "Answer for question 1",
            "8f2e2b1e4d5c6a7b8c9d0e1f2a3b4c5d": "Answer for question 2"
        }
    """
    import json

    from karenina.infrastructure.llm.manual_traces import ManualTraces, load_manual_traces

    if not trace_file.exists():
        raise FileNotFoundError(f"Manual traces file not found: {trace_file}")

    # Load JSON
    with open(trace_file) as f:
        traces_data = json.load(f)

    # Validate it's a dictionary
    if not isinstance(traces_data, dict):
        raise ValueError(f"Invalid trace file format: expected JSON object (dict), got {type(traces_data).__name__}")

    # Load into global manager (validation happens here)
    load_manual_traces(traces_data)

    # Create ManualTraces object linked to benchmark
    manual_traces = ManualTraces(benchmark)

    return manual_traces
