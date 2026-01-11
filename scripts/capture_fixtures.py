#!/usr/bin/env python3
"""Fixture capture script for recording LLM responses during verification pipeline runs.

This script captures actual LLM responses from real pipeline executions and saves
them as fixtures for deterministic testing. Fixtures are stored in tests/fixtures/llm_responses/.

Usage:
    python scripts/capture_fixtures.py --list
    python scripts/capture_fixtures.py --scenario template_parsing
    python scripts/capture_fixtures.py --all
    python scripts/capture_fixtures.py --scenario rubric_evaluation --force
"""

import argparse
import sys
from pathlib import Path

# Scenarios that can be captured
SCENARIOS = {
    "template_parsing": {
        "description": "LLM parsing answers using answer templates (template_parsing.py)",
        "source_files": [
            "src/karenina/benchmark/verification/evaluators/template_parsing.py",
            "src/karenina/benchmark/verification/evaluators/template_evaluator.py",
        ],
        "llm_calls": [
            "Structured LLM template parsing",
            "Template retry logic",
        ],
    },
    "rubric_evaluation": {
        "description": "LLM evaluating responses against rubric criteria (rubric_evaluator.py)",
        "source_files": [
            "src/karenina/benchmark/verification/evaluators/rubric_parsing.py",
            "src/karenina/benchmark/verification/evaluators/rubric_evaluator.py",
        ],
        "llm_calls": [
            "Rubric trait parsing",
            "Quality assessment",
            "Deep judgment with reasoning",
        ],
    },
    "abstention": {
        "description": "LLM abstention detection for handling refusals (abstention_checker.py)",
        "source_files": [
            "src/karenina/benchmark/verification/evaluators/abstention_checker.py",
        ],
        "llm_calls": [
            "Abstention detection",
        ],
    },
    "embedding": {
        "description": "Embedding-based checks for hallucination detection (embedding_check.py)",
        "source_files": [
            "src/karenina/benchmark/verification/tools/embedding_check.py",
        ],
        "llm_calls": [
            "Embedding generation for comparison",
        ],
    },
    "generation": {
        "description": "Answer generation for template-free questions (generator.py)",
        "source_files": [
            "src/karenina/domain/answers/generator.py",
        ],
        "llm_calls": [
            "Free-form answer generation",
        ],
    },
    "full_pipeline": {
        "description": "Complete verification pipeline with all LLM calls",
        "source_files": [
            "src/karenina/benchmark/verification/",
        ],
        "llm_calls": [
            "All pipeline LLM calls",
        ],
    },
}

FIXTURE_DIR = Path("tests/fixtures/llm_responses")
DEFAULT_MODEL = "claude-haiku-4-5"


def print_list_scenarios() -> None:
    """Print available scenarios with descriptions."""
    print("Available fixture capture scenarios:")
    print()
    for name, info in SCENARIOS.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Source files: {', '.join(info['source_files'])}")
        print()


def run_scenario(
    scenario: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Run fixture capture for a specific scenario.

    Args:
        scenario: Name of the scenario to capture
        model: LLM model to use for capture
        force: Overwrite existing fixtures
        dry_run: Show what would be captured without actually capturing

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if scenario not in SCENARIOS:
        print(f"Error: Unknown scenario '{scenario}'")
        print()
        print("Available scenarios:")
        for name in SCENARIOS:
            print(f"  - {name}")
        return 1

    scenario_info = SCENARIOS[scenario]

    if dry_run:
        print(f"DRY RUN: Would capture fixtures for scenario '{scenario}'")
        print(f"  Model: {model}")
        print(f"  Description: {scenario_info['description']}")
        print(f"  Source files: {', '.join(scenario_info['source_files'])}")
        print(f"  Target directory: {FIXTURE_DIR}/{model}/{scenario}/")
        return 0

    print(f"Capturing fixtures for scenario '{scenario}'...")
    print(f"  Model: {model}")
    print(f"  Force: {'Yes' if force else 'No'}")

    # TODO: Implement actual capture logic (task-006)
    print()
    print("Note: Capture logic not yet implemented (see task-006)")
    print("This script currently only sets up the CLI structure.")
    print()
    print(f"Would save to: {FIXTURE_DIR}/{model}/{scenario}/")

    return 0


def run_all_scenarios(
    model: str = DEFAULT_MODEL,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Run fixture capture for all scenarios.

    Args:
        model: LLM model to use for capture
        force: Overwrite existing fixtures
        dry_run: Show what would be captured without actually capturing

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("Capturing fixtures for ALL scenarios...")
    print(f"  Model: {model}")
    print(f"  Force: {'Yes' if force else 'No'}")
    print()

    exit_code = 0
    for scenario in SCENARIOS:
        if dry_run:
            exit_code |= run_scenario(scenario, model, force, dry_run=True)
        else:
            exit_code |= run_scenario(scenario, model, force, dry_run=False)
            print()

    return exit_code


def check_existing_fixtures(scenario: str, model: str) -> bool:
    """Check if fixtures already exist for a scenario.

    Args:
        scenario: Name of the scenario
        model: Model name

    Returns:
        True if fixtures exist, False otherwise
    """
    fixture_path = FIXTURE_DIR / model / scenario
    return fixture_path.exists() and any(fixture_path.iterdir())


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Capture LLM responses as fixtures for deterministic testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available scenarios
  python scripts/capture_fixtures.py --list

  # Capture fixtures for a specific scenario
  python scripts/capture_fixtures.py --scenario template_parsing

  # Capture all scenarios
  python scripts/capture_fixtures.py --all

  # Force overwrite existing fixtures
  python scripts/capture_fixtures.py --scenario rubric_evaluation --force

  # Dry run to see what would be captured
  python scripts/capture_fixtures.py --scenario template_parsing --dry-run

  # Use a different model
  python scripts/capture_fixtures.py --scenario template_parsing --model claude-sonnet-4-5
        """,
    )

    parser.add_argument(
        "--scenario", "-s",
        help="Scenario to capture (use --list to see available scenarios)",
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Capture all scenarios",
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scenarios with descriptions",
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for capture (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing fixtures without prompting",
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be captured without actually capturing",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print_list_scenarios()
        return 0

    # Validate that either --scenario or --all is provided
    if not args.scenario and not args.all:
        parser.print_help()
        print()
        print("Error: Must specify either --scenario or --all")
        print("Use --list to see available scenarios")
        return 1

    # Validate that both --scenario and --all are not provided together
    if args.scenario and args.all:
        parser.print_help()
        print()
        print("Error: Cannot specify both --scenario and --all")
        return 1

    # Run the appropriate capture
    if args.all:
        return run_all_scenarios(
            model=args.model,
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        return run_scenario(
            scenario=args.scenario,
            model=args.model,
            force=args.force,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    sys.exit(main())
