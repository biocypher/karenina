"""Minimal example of one sycophancy scenario.

The first answer is replayed because that is part of the experimental method.
The challenge, parsing, and guardrail steps call configured models. Supply an
already-running MCP URL only for the MCP regime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    OTP_ADVERSARIAL_CSV,
    OTP_BENCHMARK_JSONLD,
    OTP_MCP_RESULTS,
    OTP_PARAMETRIC_RESULTS,
)
from paper.otp_sycophancy_scenarios.config import (
    answerer_model,
    build_config,
    guardrail_model,
    reference_parser_model,
)
from paper.otp_sycophancy_scenarios.replay import build_ask_replay
from paper.otp_sycophancy_scenarios.scenarios import build_scenario_benchmark


def main() -> None:
    """Run one scenario with Karenina."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_jsonld", type=Path, nargs="?")
    parser.add_argument("adversarial_csv", type=Path, nargs="?")
    parser.add_argument("qa_results", type=Path, nargs="?")
    parser.add_argument("--answerer", choices=("qwen3.5-122b-a10b", "claude-haiku-4-5"), required=True)
    parser.add_argument("--regime", choices=("parametric", "mcp"), default="parametric")
    parser.add_argument("--mcp-url")
    args = parser.parse_args()
    bootstrap()
    if args.regime == "mcp" and not args.mcp_url:
        parser.error("--mcp-url is required for the MCP regime")
    benchmark_path = args.benchmark_jsonld or input_path(OTP_BENCHMARK_JSONLD)
    adversarial_path = args.adversarial_csv or input_path(OTP_ADVERSARIAL_CSV)
    qa_path = args.qa_results or input_path(
        OTP_MCP_RESULTS if args.regime == "mcp" else OTP_PARAMETRIC_RESULTS
    )

    # ## 1. Describe the live model roles
    # The scenario answerer handles the challenge. Claude Opus parses the
    # challenge response. The same answerer model performs the guardrail check.
    answerer = answerer_model(args.answerer, mcp_url=args.mcp_url)
    parser_model = reference_parser_model()
    guardrail = guardrail_model(args.answerer)

    # ## 2. Build a four-node Scenario through Benchmark
    # The graph routes correct first answers to a challenge and incorrect first
    # answers to a neutral correction prompt.
    benchmark = build_scenario_benchmark(
        benchmark_path,
        adversarial_path,
        difficulty="easy",
        framing="casual",
        parser_model=parser_model,
        guardrail_model=guardrail,
    )
    selected = benchmark.get_scenarios()[0]
    for scenario in list(benchmark.get_scenarios())[1:]:
        benchmark.remove_scenario(scenario.name)

    # ## 3. Load stored QA results through ResultsIOManager
    # build_ask_replay uses ResultsIOManager, capture_from_result_set, and
    # ScenarioReplayBuilder. A missing replay raises instead of calling a model.
    replay = build_ask_replay(
        benchmark,
        qa_path,
        answerer=args.answerer,
        regime=args.regime,
    )

    # ## 4. Configure verification
    # Replay applies only to ask. The continuation, parser, and guardrail remain live.
    config = build_config(answerer, replay.store, workers=1)

    # ## 5. Run the standard Benchmark interface
    # This step calls models for all non-replayed nodes reached by the graph.
    result_set = replay.benchmark.run_verification(
        config,
        run_name="sycophancy_simplified",
    )

    # ## 6. Inspect scenario DataFrames
    # No raw JSON parsing is needed for scenario, turn, or outcome views.
    view = result_set.get_scenario_results()
    print(f"Scenario: {selected.name}")
    print(view.to_dataframe().to_string(index=False))
    print(view.to_turn_dataframe()[["node_id", "verify_result"]].to_string(index=False))


if __name__ == "__main__":
    main()
