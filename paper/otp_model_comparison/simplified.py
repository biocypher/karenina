"""Minimal example of a parametric and MCP comparison.

The full experiment has manifests, seven models, retries, managed MCP server
processes, and analysis tables. This file strips those details away to show
the central Karenina workflow:

    load benchmark -> configure model -> configure verification -> run -> summarize

It runs one model on a small question slice first without tools and then with
one already-running MCP server. Both runs call the configured answerer and
parser model.

Run from the Karenina repository root:

    uv run python -m paper.otp_model_comparison.simplified \
        --mcp-url http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pydantic import SecretStr

from karenina.benchmark import Benchmark, ModelConfig, VerificationConfig
from paper.common.bootstrap import bootstrap, input_path
from paper.config import MODEL_COMPARISON_SIMPLIFIED_ENDPOINT, OTP_BENCHMARK_JSONLD


def _model(mcp_url: str | None) -> ModelConfig:
    """Describe one model, optionally attaching an MCP server URL."""
    return ModelConfig(
        id="answerer",
        model_name=os.environ.get("KARENINA_SIMPLE_MODEL", "gpt-oss-120b"),
        interface="openai_endpoint",
        endpoint_base_url=os.environ.get(
            "KARENINA_SIMPLE_MODEL_URL",
            MODEL_COMPARISON_SIMPLIFIED_ENDPOINT,
        ),
        endpoint_api_key=SecretStr(os.environ.get("KARENINA_SIMPLE_MODEL_KEY", "EMPTY")),
        temperature=0.0,
        mcp_urls_dict={"otp": mcp_url} if mcp_url else None,
    )


def _run(benchmark: Benchmark, question_ids: list[str], mcp_url: str | None) -> float:
    """Run the standard verification pipeline and return its pass rate."""
    model = _model(mcp_url)

    # ## 3. Configure verification
    # The parser uses the same underlying model without tools. The only arm
    # difference is the answerer's mcp_urls_dict above.
    config = VerificationConfig(
        answering_models=[model],
        parsing_models=[model.model_copy(update={"mcp_urls_dict": None})],
        replicate_count=1,
        evaluation_mode="template_only",
    )
    # ## 4. Run through Benchmark
    # Benchmark.run_verification executes the configured experiment matrix.
    results = benchmark.run_verification(
        config,
        question_ids=question_ids,
        run_name="mcp" if mcp_url else "parametric",
    )
    verdicts = [row.metadata.failure is None for row in results.results]
    return sum(verdicts) / len(verdicts) if verdicts else 0.0


def main() -> None:
    """Run the compact comparison example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, nargs="?")
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    bootstrap()

    # ## 1. Load a benchmark
    # Benchmark.load reconstructs the questions, answer templates, and rubric
    # from Karenina's JSON-LD format.
    benchmark_path = args.benchmark or input_path(OTP_BENCHMARK_JSONLD)
    benchmark = Benchmark.load(benchmark_path)
    question_ids = sorted(benchmark.get_question_ids())[: args.limit]

    # ## 2. Build the two model configurations
    # _run calls _model once without tools and once with the supplied MCP URL.
    print(f"Running {len(question_ids)} questions in each arm")

    # ## 5. Parametric arm
    parametric_rate = _run(benchmark, question_ids, None)

    # ## 6. MCP arm
    mcp_rate = _run(benchmark, question_ids, args.mcp_url)

    # ## 7. Summarize
    # VerificationResult.metadata.failure is None for a passing row.
    print(f"Parametric pass rate: {parametric_rate:.1%}")
    print(f"MCP pass rate: {mcp_rate:.1%}")
    print(
        "\nPublic Karenina flow:\n"
        "  Benchmark.load             load questions and templates\n"
        "  ModelConfig                describe model access and tools\n"
        "  VerificationConfig         describe the evaluation matrix\n"
        "  Benchmark.run_verification execute the standard pipeline"
    )


if __name__ == "__main__":
    main()
