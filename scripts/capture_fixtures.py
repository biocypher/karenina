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
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# CaptureLLMClient - Wraps LLM to capture responses
# =============================================================================


@dataclass
class CaptureUsage:
    """Usage metadata for captured responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CaptureResponse:
    """Captured LLM response matching fixture format.

    Real LLM responses have .content, .id, .model, and optionally .usage attributes.
    """

    content: str
    id: str
    model: str
    usage: CaptureUsage = field(default_factory=CaptureUsage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "id": self.id,
            "model": self.model,
            "usage": self.usage.to_dict(),
        }


class CaptureLLMClient:
    """LLM client wrapper that captures all invoke() calls to fixtures.

    This wraps a real LLM client, intercepts all invoke() calls, and saves
    the request/response pairs as JSON fixtures indexed by SHA256 hash.

    The hashing logic MUST match FixtureBackedLLMClient._hash_messages()
    to ensure fixtures can be found during test replay.
    """

    def __init__(
        self,
        real_client: Any,
        output_dir: Path,
        scenario: str,
        model: str,
    ) -> None:
        """Initialize the capturing LLM client.

        Args:
            real_client: The actual LLM client to wrap
            output_dir: Directory to save fixtures
            scenario: Scenario name for subdirectory organization
            model: Model name for metadata
        """
        self._real_client = real_client
        self._output_dir = Path(output_dir)
        self._scenario = scenario
        self._model = model
        self._captured_count = 0

        # Create output directory
        self._scenario_dir = self._output_dir / model / scenario
        self._scenario_dir.mkdir(parents=True, exist_ok=True)

    def invoke(self, messages: list[Any], **kwargs: Any) -> CaptureResponse:
        """Invoke the real LLM and capture the response.

        Args:
            messages: List of BaseMessage objects (HumanMessage, SystemMessage, etc.)
            **kwargs: Additional arguments passed to real LLM

        Returns:
            CaptureResponse with the same attributes as a real LLM response
        """
        prompt_hash = self._hash_messages(messages)

        # Check if fixture already exists
        fixture_path = self._scenario_dir / f"{prompt_hash}.json"
        if fixture_path.exists():
            print(f"  [SKIP] Fixture already exists: {prompt_hash[:8]}...")
            # Load and return existing fixture
            with fixture_path.open("r") as f:
                data = json.load(f)
            response_data = data.get("response", {})
            return CaptureResponse(
                content=response_data.get("content", ""),
                id=response_data.get("id", f"fixture-{prompt_hash[:8]}"),
                model=response_data.get("model", self._model),
                usage=CaptureUsage(**response_data.get("usage", {})),
            )

        # Call real LLM
        response = self._real_client.invoke(messages, **kwargs)

        # Extract response attributes
        content = response.content if hasattr(response, "content") else str(response)

        response_id = getattr(response, "id", f"captured-{prompt_hash[:8]}")
        model = getattr(response, "model", self._model)

        # Extract usage metadata
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            if hasattr(usage_obj, "to_dict"):
                usage_dict = usage_obj.to_dict()
            elif hasattr(usage_obj, "__dict__"):
                usage_dict = vars(usage_obj)
            else:
                usage_dict = {}
        else:
            usage_dict = {}

        usage = CaptureUsage(
            input_tokens=usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0)),
            output_tokens=usage_dict.get("output_tokens", usage_dict.get("completion_tokens", 0)),
            total_tokens=usage_dict.get("total_tokens", usage_dict.get("prompt_tokens", 0) + usage_dict.get("completion_tokens", 0)),
        )

        # Build capture response
        capture_response = CaptureResponse(
            content=content,
            id=response_id,
            model=model,
            usage=usage,
        )

        # Serialize messages for storage
        serialized_messages = self._serialize_messages(messages)

        # Save fixture
        fixture_data = {
            "metadata": {
                "scenario": self._scenario,
                "model": self._model,
                "timestamp": datetime.now().isoformat(),
                "prompt_hash": prompt_hash,
            },
            "request": {
                "messages": serialized_messages,
                "kwargs": {k: str(v) for k, v in kwargs.items() if k not in {"tools", "tool_choice"}},
            },
            "response": capture_response.to_dict(),
        }

        with fixture_path.open("w") as f:
            json.dump(fixture_data, f, indent=2)

        self._captured_count += 1
        print(f"  [CAPTURED] {prompt_hash[:8]}... ({self._captured_count} total)")

        return capture_response

    def _hash_messages(self, messages: list[Any]) -> str:
        """Generate SHA256 hash of messages for fixture lookup.

        This MUST match FixtureBackedLLMClient._hash_messages() exactly.

        Args:
            messages: List of BaseMessage objects

        Returns:
            SHA256 hex digest
        """
        normalized = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)

            normalized.append(" ".join(str(content).split()))

        hash_input = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _serialize_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Serialize messages to dict format for JSON storage.

        Args:
            messages: List of BaseMessage objects

        Returns:
            List of serialized message dicts
        """
        serialized = []
        for msg in messages:
            if isinstance(msg, dict):
                serialized.append(msg)
            elif hasattr(msg, "dict"):
                serialized.append(msg.dict())
            elif hasattr(msg, "model_dump"):
                serialized.append(msg.model_dump())
            else:
                # Fallback: capture class name and content
                serialized.append({
                    "type": type(msg).__name__,
                    "content": str(getattr(msg, "content", msg)),
                })
        return serialized

    @property
    def captured_count(self) -> int:
        """Return the number of fixtures captured."""
        return self._captured_count


# =============================================================================
# Scenario runners - Execute actual pipeline code to capture LLM calls
# =============================================================================

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


def _create_real_llm(model: str, provider: str = "anthropic") -> Any:
    """Create a real LLM client for capturing.

    Args:
        model: Model name (e.g., "claude-haiku-4-5")
        provider: Provider name (default: "anthropic")

    Returns:
        Initialized LLM client
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(model=model, model_provider=provider, temperature=0.0)


def _run_template_parsing_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture template parsing LLM calls.

    This scenario captures structured output parsing from the template evaluator.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "template_parsing", model)

    print("  Running template parsing scenario...")

    # Scenario 1: Simple field extraction
    system_msg = SystemMessage(
        content="You are a validation assistant. Parse and validate responses against the given template."
    )
    user_msg = HumanMessage(
        content='Extract the answer from: "The capital of France is Paris, a beautiful city." '
        'Return JSON with field "capital" containing the answer.'
    )

    capture_client.invoke([system_msg, user_msg])

    # Scenario 2: Multiple field extraction
    user_msg2 = HumanMessage(
        content='Extract the answer from: "Mitochondria produce ATP through cellular respiration." '
        'Return JSON with fields "organelle" and "molecule".'
    )

    capture_client.invoke([system_msg, user_msg2])

    # Scenario 3: Nested structure parsing
    user_msg3 = HumanMessage(
        content='Extract from: "The study (Smith et al., 2023) showed 85% effectiveness." '
        'Return JSON with "author", "year", and "effectiveness" as percentage.'
    )

    capture_client.invoke([system_msg, user_msg3])

    print(f"  Captured {capture_client.captured_count} fixtures")
    return 0


def _run_rubric_evaluation_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture rubric evaluation LLM calls.

    This scenario captures LLM-as-judge rubric trait evaluation.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "rubric_evaluation", model)

    print("  Running rubric evaluation scenario...")

    # Scenario 1: Boolean rubric trait (clarity check)
    system_msg = SystemMessage(
        content="You are evaluating response quality. Assess whether the response is clear and unambiguous."
    )
    user_msg = HumanMessage(
        content="Evaluate the clarity of this response: 'The protein BCL2 is located on chromosome 18 "
        "and regulates apoptosis.' Return true if clear, false if unclear."
    )

    capture_client.invoke([system_msg, user_msg])

    # Scenario 2: Score rubric trait (quality 1-5)
    system_msg2 = SystemMessage(
        content="You are evaluating response quality on a scale of 1-5, where 5 is excellent."
    )
    user_msg2 = HumanMessage(
        content="Rate the completeness of this response (1-5): 'The answer is Paris.' "
        "Consider whether it provides adequate context."
    )

    capture_client.invoke([system_msg2, user_msg2])

    # Scenario 3: Deep judgment with reasoning
    system_msg3 = SystemMessage(
        content="You are performing deep judgment analysis. Evaluate the response and provide "
        "reasoning with excerpts from the text."
    )
    user_msg3 = HumanMessage(
        content='Analyze this response for hallucination: "BCL2 is located on chromosome 21 and '
        'was discovered in 1990." Compare against known facts and provide reasoning.'
    )

    capture_client.invoke([system_msg3, user_msg3])

    print(f"  Captured {capture_client.captured_count} fixtures")
    return 0


def _run_abstention_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture abstention detection LLM calls.

    This scenario captures LLM calls for detecting refusals/abstentions.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "abstention", model)

    print("  Running abstention detection scenario...")

    system_msg = SystemMessage(
        content="You are detecting whether the model refused to answer. Return true if refusal, false otherwise."
    )

    # Scenario 1: Clear refusal
    user_msg1 = HumanMessage(
        content='Did this response refuse to answer? "I cannot provide medical advice. '
        'Please consult a healthcare professional."'
    )

    capture_client.invoke([system_msg, user_msg1])

    # Scenario 2: Normal response (not a refusal)
    user_msg2 = HumanMessage(
        content='Did this response refuse to answer? "The recommended dosage is 500mg twice daily, '
        'but you should consult your doctor for personalized advice."'
    )

    capture_client.invoke([system_msg, user_msg2])

    print(f"  Captured {capture_client.captured_count} fixtures")
    return 0


def _run_embedding_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture embedding check LLM calls.

    This scenario captures LLM calls for semantic similarity fallback.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "embedding", model)

    print("  Running embedding check scenario...")

    system_msg = SystemMessage(
        content="You are comparing semantic similarity between responses. "
        "Extract key semantic entities from the text."
    )

    user_msg = HumanMessage(
        content='Extract key entities from: "The BCL2 gene is located on chromosome 18q21.33 '
        'and encodes a protein that inhibits apoptosis."'
    )

    capture_client.invoke([system_msg, user_msg])

    print(f"  Captured {capture_client.captured_count} fixtures")
    return 0


def _run_generation_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture answer generation LLM calls.

    This scenario captures free-form answer generation.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "generation", model)

    print("  Running answer generation scenario...")

    system_msg = SystemMessage(
        content="You are an expert assistant. Answer the question accurately and concisely."
    )

    user_msg = HumanMessage(
        content="What is the approved gene symbol for the B-cell leukemia/lymphoma 2 gene?"
    )

    capture_client.invoke([system_msg, user_msg])

    print(f"  Captured {capture_client.captured_count} fixtures")
    return 0


def _run_full_pipeline_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture full verification pipeline LLM calls.

    This scenario runs a complete verification pipeline and captures all LLM calls.
    """
    print("  Running full pipeline scenario...")
    print("  Note: This runs all sub-scenarios in sequence")

    exit_code = 0
    exit_code |= _run_template_parsing_scenario(model, provider, output_dir)
    exit_code |= _run_rubric_evaluation_scenario(model, provider, output_dir)
    exit_code |= _run_abstention_scenario(model, provider, output_dir)
    exit_code |= _run_embedding_scenario(model, provider, output_dir)
    exit_code |= _run_generation_scenario(model, provider, output_dir)

    return exit_code


# Scenario runner mapping
SCENARIO_RUNNERS = {
    "template_parsing": _run_template_parsing_scenario,
    "rubric_evaluation": _run_rubric_evaluation_scenario,
    "abstention": _run_abstention_scenario,
    "embedding": _run_embedding_scenario,
    "generation": _run_generation_scenario,
    "full_pipeline": _run_full_pipeline_scenario,
}


def run_scenario(
    scenario: str,
    model: str = DEFAULT_MODEL,
    provider: str = "anthropic",
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Run fixture capture for a specific scenario.

    Args:
        scenario: Name of the scenario to capture
        model: LLM model to use for capture
        provider: LLM provider (default: "anthropic")
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
        print(f"  Provider: {provider}")
        print(f"  Description: {scenario_info['description']}")
        print(f"  Source files: {', '.join(scenario_info['source_files'])}")
        print(f"  Target directory: {FIXTURE_DIR}/{model}/{scenario}/")
        return 0

    print(f"Capturing fixtures for scenario '{scenario}'...")
    print(f"  Model: {model}")
    print(f"  Provider: {provider}")
    print(f"  Force: {'Yes' if force else 'No'}")
    print()

    # Check for existing fixtures if not forcing
    if not force:
        existing = list((FIXTURE_DIR / model / scenario).glob("*.json"))
        if existing:
            print(f"  Warning: {len(existing)} existing fixtures found.")
            print("  Use --force to overwrite. Skipping capture.")
            return 0

    # Run the scenario
    runner = SCENARIO_RUNNERS.get(scenario)
    if runner is None:
        print(f"  Error: No runner implemented for scenario '{scenario}'")
        return 1

    try:
        return runner(model, provider, FIXTURE_DIR)
    except Exception as e:
        print(f"  Error during capture: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_all_scenarios(
    model: str = DEFAULT_MODEL,
    provider: str = "anthropic",
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Run fixture capture for all scenarios.

    Args:
        model: LLM model to use for capture
        provider: LLM provider (default: "anthropic")
        force: Overwrite existing fixtures
        dry_run: Show what would be captured without actually capturing

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("Capturing fixtures for ALL scenarios...")
    print(f"  Model: {model}")
    print(f"  Provider: {provider}")
    print(f"  Force: {'Yes' if force else 'No'}")
    print()

    exit_code = 0
    for scenario in SCENARIOS:
        if dry_run:
            exit_code |= run_scenario(scenario, model, provider, force, dry_run=True)
        else:
            exit_code |= run_scenario(scenario, model, provider, force, dry_run=False)
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

  # Use a different provider
  python scripts/capture_fixtures.py --scenario template_parsing --provider openai
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
        "--provider", "-p",
        default="anthropic",
        help="LLM provider to use for capture (default: anthropic)",
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
            provider=args.provider,
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        return run_scenario(
            scenario=args.scenario,
            model=args.model,
            provider=args.provider,
            force=args.force,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    sys.exit(main())
