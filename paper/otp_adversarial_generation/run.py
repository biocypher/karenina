"""Generate adversarial alternatives with standalone Claude Code sessions.

The default command calls ``claude`` once per non-binary item. Alternative
generation is independent of Karenina; the downstream scenario experiment
consumes the curator-approved samples. Pass ``--reuse-stored-samples`` for
fully offline validation and analysis of the approved archive.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    ADVERSARIAL_OUTPUT_DIR,
    OTP_ADVERSARIAL_BENCHMARK_CSV,
    OTP_ADVERSARIAL_CSV,
)
from paper.otp_adversarial_generation.analysis import write_analysis
from paper.otp_adversarial_generation.claude_code import (
    extract_sample_text,
    invoke_claude_code,
    trace_metadata,
    trace_to_markdown,
    validate_open_targets_mcp,
)
from paper.otp_adversarial_generation.config import (
    CLAUDE_TIMEOUT_SECONDS,
    GENERATOR_MODEL,
    INTER_SAMPLE_DELAY_SECONDS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)
from paper.otp_adversarial_generation.generation import (
    AdversarialPair,
    BenchmarkItem,
    binary_pair,
    load_approved_archive,
    load_benchmark_items,
    parse_sample_text,
)
from paper.otp_adversarial_generation.templates import GENERATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

def generation_prompt(item: BenchmarkItem, output_path: Path) -> str:
    """Fill the Claude Code generation prompt for one item."""
    return GENERATION_PROMPT_TEMPLATE.format(
        item_id=item.item_id,
        area=item.area,
        question_type=item.question_type,
        question=item.question,
        ground_truth=item.ground_truth,
        output_path=output_path,
    )


def _sample_paths(output_dir: Path, item_id: str) -> tuple[Path, Path, Path]:
    sample_dir = output_dir / "samples" / f"sample_{item_id}"
    return sample_dir / "adversarial.txt", sample_dir / "trace_raw.jsonl", sample_dir / "trace.md"


def _generate_non_binary(
    item: BenchmarkItem,
    output_dir: Path,
    *,
    model: str | None,
    timeout_seconds: int,
    max_retries: int,
    retry_delay: int,
) -> AdversarialPair:
    """Run one independent Claude Code session with configured retries."""
    sample_path, raw_trace_path, readable_trace_path = _sample_paths(output_dir, item.item_id)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = generation_prompt(item, sample_path.resolve())
    last_error = "Claude Code did not produce the required sample"
    for attempt in range(max_retries + 1):
        logger.info(
            "Running standalone Claude Code for item %s, attempt %d/%d",
            item.item_id,
            attempt + 1,
            max_retries + 1,
        )
        result = invoke_claude_code(
            prompt,
            model=model,
            workdir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        raw_trace_path.write_text(result.stdout, encoding="utf-8")
        readable_trace_path.write_text(trace_to_markdown(result.stdout), encoding="utf-8")
        mcp_valid = False
        if result.stdout:
            try:
                validate_open_targets_mcp(result.stdout)
                mcp_valid = True
            except RuntimeError as error:
                last_error = str(error)
        if not sample_path.exists():
            extracted = extract_sample_text(result.stdout)
            if extracted:
                sample_path.write_text(extracted, encoding="utf-8")
        if mcp_valid and sample_path.exists():
            try:
                model_name, session_id, _servers = trace_metadata(result.stdout)
                return parse_sample_text(
                    sample_path.read_text(encoding="utf-8"),
                    item,
                    model_name=model_name,
                    trace_id=session_id,
                )
            except ValueError as error:
                last_error = f"Generated sample was invalid: {error}"
        elif result.timed_out:
            last_error = "Claude Code timed out"
        elif not result.success:
            last_error = f"Claude Code exited {result.returncode}: {result.stderr[:500]}"
        if attempt < max_retries:
            time.sleep(retry_delay)
    raise RuntimeError(f"Item {item.item_id} failed after {max_retries + 1} attempts: {last_error}")


def _fresh_pairs(
    output_dir: Path,
    *,
    limit: int | None,
    non_binary_only: bool,
    model: str | None,
    timeout_seconds: int,
    max_retries: int,
    retry_delay: int,
    inter_sample_delay: int,
) -> list[AdversarialPair]:
    """Generate every selected pair using binary flips or Claude Code."""
    items = load_benchmark_items(input_path(OTP_ADVERSARIAL_BENCHMARK_CSV))
    if non_binary_only:
        items = [item for item in items if not item.is_binary]
    if limit is not None:
        items = items[:limit]
    pairs: list[AdversarialPair] = []
    for index, item in enumerate(items):
        pair = binary_pair(item) if item.is_binary else _generate_non_binary(
            item,
            output_dir,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        pairs.append(pair)
        if not item.is_binary and index < len(items) - 1 and inter_sample_delay:
            time.sleep(inter_sample_delay)
    return pairs


def run(
    output_dir: Path,
    *,
    reuse_stored_samples: bool,
    limit: int | None = None,
    model: str | None = GENERATOR_MODEL,
    timeout_seconds: int = CLAUDE_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
    inter_sample_delay: int = INTER_SAMPLE_DELAY_SECONDS,
    non_binary_only: bool = False,
) -> None:
    """Run fresh standalone generation or explicit offline archive validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if reuse_stored_samples:
        archive_path = input_path(OTP_ADVERSARIAL_CSV)
        pairs = load_approved_archive(archive_path)
        if limit is not None:
            pairs = pairs[:limit]
        mode = "reuse_stored_samples"
        inputs = {"approved_samples": str(archive_path)}
    else:
        pairs = _fresh_pairs(
            output_dir,
            limit=limit,
            non_binary_only=non_binary_only,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay=retry_delay,
            inter_sample_delay=inter_sample_delay,
        )
        mode = "fresh_standalone_claude_code"
        inputs = {
            "source_benchmark": str(
                input_path(OTP_ADVERSARIAL_BENCHMARK_CSV)
            )
        }
    write_analysis(pairs, output_dir / "analysis")
    non_binary_calls = mode == "fresh_standalone_claude_code" and any(
        pair.generation_route != "binary_flip" for pair in pairs
    )
    manifest = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "execution_interface": "standalone_claude_code_cli" if non_binary_calls else None,
        "karenina_generation": False,
        "model_calls": non_binary_calls,
        "mcp_calls": non_binary_calls,
        "configured_model": model if not reuse_stored_samples else None,
        "pair_count": len(pairs),
        "review_statuses": sorted({pair.review_status for pair in pairs}),
        "inputs": inputs,
        "analysis_directory": str(output_dir / "analysis"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse command-line options and run adversarial generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-stored-samples", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ADVERSARIAL_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=GENERATOR_MODEL)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=CLAUDE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--retry-delay", type=int, default=RETRY_DELAY_SECONDS
    )
    parser.add_argument(
        "--inter-sample-delay",
        type=int,
        default=INTER_SAMPLE_DELAY_SECONDS,
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_dir,
        reuse_stored_samples=args.reuse_stored_samples,
        limit=args.limit,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        inter_sample_delay=args.inter_sample_delay,
    )


if __name__ == "__main__":
    main()
