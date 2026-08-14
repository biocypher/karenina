"""Minimal example of standalone Claude Code alternative generation.

Adversarial generation is an independent preprocessing step. This example
calls the `claude` executable and expects a connected Claude Code MCP server
named `open-targets`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paper.common.bootstrap import bootstrap, input_path
from paper.config import ADVERSARIAL_OUTPUT_DIR, OTP_ADVERSARIAL_BENCHMARK_CSV
from paper.otp_adversarial_generation.claude_code import (
    extract_sample_text,
    invoke_claude_code,
    trace_metadata,
    validate_open_targets_mcp,
)
from paper.otp_adversarial_generation.config import CLAUDE_TIMEOUT_SECONDS, GENERATOR_MODEL
from paper.otp_adversarial_generation.generation import load_benchmark_items, parse_sample_text
from paper.otp_adversarial_generation.run import generation_prompt


def main() -> None:
    """Generate one draft in a standalone Claude Code session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_csv", type=Path, nargs="?")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--model", default=GENERATOR_MODEL)
    args = parser.parse_args()
    bootstrap()
    benchmark_csv = args.benchmark_csv or input_path(OTP_ADVERSARIAL_BENCHMARK_CSV)
    output_dir = args.output_dir or ADVERSARIAL_OUTPUT_DIR / "simplified"

    # ## 1. Select one non-binary source question
    # Binary questions are deterministic flips and do not need Claude Code.
    item = next(item for item in load_benchmark_items(benchmark_csv) if not item.is_binary)

    # ## 2. Prepare the per-item output path
    # The prompt tells Claude Code to write the structured sample.
    sample_dir = output_dir / f"sample_{item.item_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "adversarial.txt"
    prompt = generation_prompt(item, sample_path.resolve())

    # ## 3. Start a standalone Claude Code session
    # This calls the LLM. Claude Code discovers Open Targets through its own MCP
    # configuration; no Karenina model or evaluation interface is involved.
    result = invoke_claude_code(
        prompt,
        model=args.model,
        workdir=output_dir,
        timeout_seconds=CLAUDE_TIMEOUT_SECONDS,
    )
    if not result.success:
        raise RuntimeError(f"Claude Code failed: {result.stderr}")
    validate_open_targets_mcp(result.stdout)

    # ## 4. Recover the structured sample
    # Usually Claude writes the file. The trace contains the same Write payload,
    # so it is a safe recovery source if the file did not land.
    if not sample_path.exists():
        extracted = extract_sample_text(result.stdout)
        if extracted is None:
            raise RuntimeError("Claude Code returned no structured adversarial sample")
        sample_path.write_text(extracted, encoding="utf-8")

    # ## 5. Validate and label the draft
    # Human domain review is still required before scenario use.
    model_name, session_id, _servers = trace_metadata(result.stdout)
    pair = parse_sample_text(
        sample_path.read_text(encoding="utf-8"),
        item,
        model_name=model_name,
        trace_id=session_id,
    )
    print(f"Hard alternative: {pair.hard_adversarial}")
    print(f"Easy alternative: {pair.easy_adversarial}")
    print(f"Review status: {pair.review_status}")


if __name__ == "__main__":
    main()
