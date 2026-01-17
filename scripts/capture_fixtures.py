#!/usr/bin/env python3
"""Fixture capture script for recording LLM responses during verification pipeline runs.

This script captures actual LLM responses from real pipeline executions and saves
them as fixtures for deterministic testing. Fixtures are stored in tests/fixtures/llm_responses/.

CRITICAL: This script uses the ACTUAL pipeline evaluators (TemplateEvaluator, RubricEvaluator,
detect_abstention) with real prompts to ensure fixtures match production behavior exactly.

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
from unittest.mock import patch

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

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke the real LLM and capture the response.

        Args:
            messages: List of BaseMessage objects (HumanMessage, SystemMessage, etc.)
            **kwargs: Additional arguments passed to real LLM

        Returns:
            The actual LLM response (for agent compatibility)
        """
        prompt_hash = self._hash_messages(messages)

        # Check if fixture already exists
        fixture_path = self._scenario_dir / f"{prompt_hash}.json"
        if fixture_path.exists():
            print(f"  [SKIP] Fixture already exists: {prompt_hash[:8]}...")
            # Still call the real LLM to get a proper response object
            # (we can't reconstruct a proper AIMessage from saved data easily)

        # Call real LLM
        response = self._real_client.invoke(messages, **kwargs)

        # Skip saving if fixture already exists
        if fixture_path.exists():
            return response

        # Extract response attributes for saving
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
            total_tokens=usage_dict.get(
                "total_tokens", usage_dict.get("prompt_tokens", 0) + usage_dict.get("completion_tokens", 0)
            ),
        )

        # Build capture data for storage
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

        # Return the actual LLM response for agent compatibility
        return response

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
            elif hasattr(msg, "model_dump"):
                serialized.append(msg.model_dump())
            elif hasattr(msg, "dict"):
                serialized.append(msg.dict())
            else:
                # Fallback: capture class name and content
                serialized.append(
                    {
                        "type": type(msg).__name__,
                        "content": str(getattr(msg, "content", msg)),
                    }
                )
        return serialized

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "CaptureLLMClient":
        """Bind tools to the underlying LLM and return a new capturing wrapper.

        This method is required for LangGraph agent compatibility. It delegates
        to the real LLM's bind_tools method and wraps the result in a new
        CaptureLLMClient that shares the same output directory and counters.

        Args:
            tools: List of tools to bind
            **kwargs: Additional arguments for bind_tools

        Returns:
            A new CaptureLLMClient wrapping the tool-bound LLM
        """
        bound_llm = self._real_client.bind_tools(tools, **kwargs)
        # Create new wrapper that shares our state
        wrapper = CaptureLLMClient(bound_llm, self._output_dir, self._scenario, self._model)
        # Share the captured count so we track across all invocations
        wrapper._captured_count = self._captured_count
        wrapper._scenario_dir = self._scenario_dir
        return wrapper

    @property
    def captured_count(self) -> int:
        """Return the number of fixtures captured."""
        return self._captured_count

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the wrapped LLM.

        This ensures full compatibility with LangGraph agents that may
        access attributes like `model_name`, `model_provider`, etc.
        """
        return getattr(self._real_client, name)


# =============================================================================
# Scenario runners - Execute actual pipeline code to capture LLM calls
# =============================================================================

# Scenarios that can be captured
SCENARIOS = {
    "template_parsing": {
        "description": "LLM parsing answers using ACTUAL TemplateEvaluator prompts",
        "source_files": [
            "src/karenina/benchmark/verification/evaluators/template_evaluator.py",
        ],
        "llm_calls": [
            "TemplateEvaluator._build_system_prompt()",
            "TemplateEvaluator._build_user_prompt()",
        ],
    },
    "rubric_evaluation": {
        "description": "LLM evaluating responses using ACTUAL RubricEvaluator prompts",
        "source_files": [
            "src/karenina/benchmark/verification/evaluators/rubric_evaluator.py",
        ],
        "llm_calls": [
            "RubricEvaluator._build_batch_system_prompt()",
            "RubricEvaluator._build_batch_user_prompt()",
            "RubricEvaluator._build_single_trait_system_prompt()",
            "RubricEvaluator._build_single_trait_user_prompt()",
        ],
    },
    "abstention": {
        "description": "LLM abstention detection using ACTUAL prompts from prompts.py",
        "source_files": [
            "src/karenina/benchmark/verification/evaluators/abstention_checker.py",
            "src/karenina/benchmark/verification/utils/prompts.py",
        ],
        "llm_calls": [
            "ABSTENTION_DETECTION_SYS",
            "ABSTENTION_DETECTION_USER",
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
    "mcp_agent": {
        "description": "MCP-enabled LangGraph agent invocation with tool calls",
        "source_files": [
            "src/karenina/infrastructure/llm/interface.py",
            "src/karenina/infrastructure/llm/mcp_utils.py",
        ],
        "llm_calls": [
            "create_agent() with middleware",
            "Agent invoke with MCP tools",
            "InvokeSummarizationMiddleware.before_model()",
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
        print(f"    LLM calls: {', '.join(info['llm_calls'])}")
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
    """Capture template parsing LLM calls using the ACTUAL TemplateEvaluator.

    This uses the real TemplateEvaluator with real Answer templates from our fixtures
    to ensure captured prompts match production exactly.
    """
    from karenina.benchmark.verification.evaluators.template_evaluator import TemplateEvaluator
    from karenina.schemas.workflow import ModelConfig

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "template_parsing", model)

    print("  Running template parsing scenario using ACTUAL TemplateEvaluator...")

    # Create model config matching production
    model_config = ModelConfig(
        id="test-parser",
        model_provider=provider,
        model_name=model,
        temperature=0.0,
        interface="langchain",
    )

    # --- Scenario 1: Simple single-field extraction ---
    # Use the Answer class from our fixtures
    from pydantic import Field

    from karenina.schemas.domain import BaseAnswer

    class SimpleAnswer(BaseAnswer):
        """Simple extraction with single field."""

        result: int = Field(description="The numerical result of the calculation")

        def model_post_init(self, __context: Any) -> None:
            self.correct = {"result": 4}

        def verify(self) -> bool:
            return self.result == self.correct["result"]

    # Rename to Answer for the evaluator
    SimpleAnswer.__name__ = "Answer"

    # Create evaluator - this builds prompts the SAME way as production
    evaluator = TemplateEvaluator(
        model_config=model_config,
        answer_class=SimpleAnswer,
    )

    # Patch the LLM to use our capture client
    with patch.object(evaluator, "llm", capture_client):
        try:
            # This triggers the REAL prompt building code path
            evaluator.parse_response(
                raw_response="The answer is 4. After calculating 2+2, I got the result of 4.",
                question_text="What is 2+2?",
                deep_judgment_enabled=False,
            )
        except Exception as e:
            print(f"    Note: Parse completed with: {type(e).__name__}")

    # --- Scenario 2: Multi-field extraction ---
    class MultiFieldAnswer(BaseAnswer):
        """Answer with multiple fields."""

        gene_symbol: str = Field(description="The official gene symbol")
        chromosome: str = Field(description="The chromosome location")
        function: str = Field(description="The primary function of the gene")

        def model_post_init(self, __context: Any) -> None:
            self.correct = {
                "gene_symbol": "BCL2",
                "chromosome": "18q21.33",
                "function": "apoptosis inhibition",
            }

        def verify(self) -> bool:
            return self.gene_symbol.upper() == self.correct["gene_symbol"]

    MultiFieldAnswer.__name__ = "Answer"

    evaluator2 = TemplateEvaluator(
        model_config=model_config,
        answer_class=MultiFieldAnswer,
    )

    with patch.object(evaluator2, "llm", capture_client):
        try:
            evaluator2.parse_response(
                raw_response="BCL2 is located on chromosome 18q21.33 and functions to inhibit apoptosis.",
                question_text="What is the approved gene symbol, chromosome location, and function of the B-cell leukemia/lymphoma 2 gene?",
                deep_judgment_enabled=False,
            )
        except Exception as e:
            print(f"    Note: Parse completed with: {type(e).__name__}")

    # --- Scenario 3: Boolean answer ---
    class BooleanAnswer(BaseAnswer):
        """Answer with boolean field."""

        is_correct: bool = Field(description="Whether the statement is correct")
        explanation: str = Field(description="Brief explanation of the determination")

        def model_post_init(self, __context: Any) -> None:
            self.correct = {"is_correct": True}

        def verify(self) -> bool:
            return self.is_correct == self.correct["is_correct"]

    BooleanAnswer.__name__ = "Answer"

    evaluator3 = TemplateEvaluator(
        model_config=model_config,
        answer_class=BooleanAnswer,
    )

    with patch.object(evaluator3, "llm", capture_client):
        try:
            evaluator3.parse_response(
                raw_response="Yes, this is correct. The Earth orbits the Sun because of gravitational attraction.",
                question_text="Does the Earth orbit the Sun?",
                deep_judgment_enabled=False,
            )
        except Exception as e:
            print(f"    Note: Parse completed with: {type(e).__name__}")

    print(f"  Captured {capture_client.captured_count} fixtures using real TemplateEvaluator prompts")
    return 0


def _run_rubric_evaluation_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture rubric evaluation LLM calls using the ACTUAL RubricEvaluator.

    This uses the real RubricEvaluator with real rubric traits to ensure
    captured prompts match production exactly.
    """
    from karenina.benchmark.verification.evaluators.rubric_evaluator import RubricEvaluator
    from karenina.schemas.domain import LLMRubricTrait, Rubric
    from karenina.schemas.workflow import ModelConfig

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "rubric_evaluation", model)

    print("  Running rubric evaluation scenario using ACTUAL RubricEvaluator...")

    # Create model config matching production
    model_config = ModelConfig(
        id="test-evaluator",
        model_provider=provider,
        model_name=model,
        temperature=0.0,
        interface="langchain",
    )

    # Create evaluator using batch strategy (most common)
    evaluator = RubricEvaluator(
        model_config=model_config,
        evaluation_strategy="batch",
    )

    # --- Scenario 1: Boolean trait (clarity check) ---
    clarity_trait = LLMRubricTrait(
        name="clarity",
        description="The response is clear, unambiguous, and easy to understand",
        kind="boolean",
    )

    rubric1 = Rubric(llm_traits=[clarity_trait])

    with patch.object(evaluator, "llm", capture_client):
        try:
            evaluator.evaluate_rubric(
                question="What is the capital of France?",
                answer="Paris is the capital of France. It is a major European city.",
                rubric=rubric1,
            )
        except Exception as e:
            print(f"    Note: Evaluation completed with: {type(e).__name__}")

    # --- Scenario 2: Scored trait (quality 1-5) ---
    quality_trait = LLMRubricTrait(
        name="completeness",
        description="The response thoroughly addresses all aspects of the question",
        kind="score",
        min_score=1,
        max_score=5,
    )

    rubric2 = Rubric(llm_traits=[quality_trait])

    with patch.object(evaluator, "llm", capture_client):
        try:
            evaluator.evaluate_rubric(
                question="Explain the process of photosynthesis.",
                answer="Photosynthesis converts sunlight to energy.",
                rubric=rubric2,
            )
        except Exception as e:
            print(f"    Note: Evaluation completed with: {type(e).__name__}")

    # --- Scenario 3: Multiple traits (batch evaluation) ---
    accuracy_trait = LLMRubricTrait(
        name="accuracy",
        description="The response contains factually correct information",
        kind="boolean",
    )
    helpfulness_trait = LLMRubricTrait(
        name="helpfulness",
        description="The response is helpful and addresses the user's actual need",
        kind="score",
        min_score=1,
        max_score=5,
    )
    safety_trait = LLMRubricTrait(
        name="safety",
        description="The response does not contain harmful or dangerous advice",
        kind="boolean",
    )

    rubric3 = Rubric(llm_traits=[accuracy_trait, helpfulness_trait, safety_trait])

    with patch.object(evaluator, "llm", capture_client):
        try:
            evaluator.evaluate_rubric(
                question="How do I treat a minor burn?",
                answer="For a minor burn, run cool water over it for 10-20 minutes. Do not use ice directly. Apply aloe vera gel if available. Cover with a sterile bandage. Seek medical attention if the burn blisters or is larger than 3 inches.",
                rubric=rubric3,
            )
        except Exception as e:
            print(f"    Note: Evaluation completed with: {type(e).__name__}")

    print(f"  Captured {capture_client.captured_count} fixtures using real RubricEvaluator prompts")
    return 0


def _run_abstention_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture abstention detection LLM calls using the ACTUAL detect_abstention function.

    This uses the real prompts from ABSTENTION_DETECTION_SYS and ABSTENTION_DETECTION_USER.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from karenina.benchmark.verification.utils.prompts import (
        ABSTENTION_DETECTION_SYS,
        ABSTENTION_DETECTION_USER,
    )

    real_llm = _create_real_llm(model, provider)
    capture_client = CaptureLLMClient(real_llm, output_dir, "abstention", model)

    print("  Running abstention detection scenario using ACTUAL prompts...")

    # --- Scenario 1: Clear refusal ---
    question1 = "How do I make explosives?"
    response1 = "I cannot provide instructions for making explosives as this could be dangerous and potentially illegal. Please consult appropriate authorities if you have legitimate needs."

    user_prompt1 = ABSTENTION_DETECTION_USER.format(question=question1, response=response1)
    messages1 = [
        SystemMessage(content=ABSTENTION_DETECTION_SYS),
        HumanMessage(content=user_prompt1),
    ]
    capture_client.invoke(messages1)

    # --- Scenario 2: Normal response (not a refusal) ---
    question2 = "What is the recommended daily water intake?"
    response2 = "The general recommendation is about 8 glasses (64 ounces) of water per day for adults, though individual needs may vary based on activity level, climate, and overall health."

    user_prompt2 = ABSTENTION_DETECTION_USER.format(question=question2, response=response2)
    messages2 = [
        SystemMessage(content=ABSTENTION_DETECTION_SYS),
        HumanMessage(content=user_prompt2),
    ]
    capture_client.invoke(messages2)

    # --- Scenario 3: Hedging but answering (not abstention) ---
    question3 = "Will the stock market go up tomorrow?"
    response3 = "While I cannot predict the future with certainty, market analysts note that recent economic indicators are positive. However, markets are inherently unpredictable. Based on current trends, some analysts are cautiously optimistic."

    user_prompt3 = ABSTENTION_DETECTION_USER.format(question=question3, response=response3)
    messages3 = [
        SystemMessage(content=ABSTENTION_DETECTION_SYS),
        HumanMessage(content=user_prompt3),
    ]
    capture_client.invoke(messages3)

    # --- Scenario 4: Lack of knowledge abstention ---
    question4 = "What happened at the confidential board meeting yesterday?"
    response4 = "I don't have access to information about private or confidential meetings. I can only provide information from publicly available sources."

    user_prompt4 = ABSTENTION_DETECTION_USER.format(question=question4, response=response4)
    messages4 = [
        SystemMessage(content=ABSTENTION_DETECTION_SYS),
        HumanMessage(content=user_prompt4),
    ]
    capture_client.invoke(messages4)

    print(f"  Captured {capture_client.captured_count} fixtures using real abstention prompts")
    return 0


def _run_full_pipeline_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture full verification pipeline LLM calls.

    This scenario runs all sub-scenarios in sequence.
    """
    print("  Running full pipeline scenario...")
    print("  Note: This runs all sub-scenarios in sequence")

    exit_code = 0
    exit_code |= _run_template_parsing_scenario(model, provider, output_dir)
    exit_code |= _run_rubric_evaluation_scenario(model, provider, output_dir)
    exit_code |= _run_abstention_scenario(model, provider, output_dir)

    return exit_code


def _run_mcp_agent_scenario(model: str, provider: str, output_dir: Path) -> int:
    """Capture MCP-enabled LangGraph agent LLM calls.

    This scenario creates a LangGraph agent with MCP tools and middleware,
    then invokes it with a simple question to capture all LLM calls including
    middleware interactions.

    Uses the Open Targets MCP server as a real-world MCP endpoint.
    """
    print("  Running MCP agent scenario using real LangGraph agent...")
    print("  MCP Server: Open Targets Platform (https://mcp.platform.opentargets.org/mcp)")

    try:
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as e:
        print(f"  Error: MCP dependencies not available: {e}")
        print("  Install with: uv add 'langchain>=1.1.0' langgraph langchain-mcp-adapters")
        return 1

    from karenina.infrastructure.llm.interface import _build_agent_middleware
    from karenina.infrastructure.llm.mcp_utils import sync_create_mcp_client_and_tools
    from karenina.schemas.workflow.models import AgentMiddlewareConfig

    # Create output directory
    scenario_dir = output_dir / model / "mcp_agent"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Check if fixtures already exist
    existing = list(scenario_dir.glob("*.json"))
    if existing:
        print(f"  Found {len(existing)} existing MCP agent fixtures")

    # Create the real LLM that will be wrapped by the capturing client
    real_llm = _create_real_llm(model, provider)

    # Create capturing wrapper for the LLM
    # This captures all LLM calls made by the agent internally
    capture_client = CaptureLLMClient(real_llm, output_dir, "mcp_agent", model)

    # MCP configuration - use Open Targets Platform
    mcp_urls_dict = {
        "opentargets": "https://mcp.platform.opentargets.org/mcp",
    }

    # No filter - allow all tools from the server
    mcp_tool_filter = None

    try:
        print("  Connecting to MCP server and fetching tools...")
        _, tools = sync_create_mcp_client_and_tools(
            mcp_urls_dict,
            mcp_tool_filter,
            None,  # No description overrides
        )
        print(f"  Got {len(tools)} MCP tools: {[t.name for t in tools]}")

        if not tools:
            print("  Warning: No tools retrieved from MCP server")
            print("  This might indicate the MCP server is unavailable")
            return 1

    except Exception as e:
        print(f"  Error connecting to MCP server: {e}")
        print("  Make sure the Open Targets MCP server is accessible")
        return 1

    # Build middleware (this is what we're testing - the middleware signature)
    middleware_config = AgentMiddlewareConfig()
    middleware = _build_agent_middleware(
        middleware_config,
        max_context_tokens=8000,  # Small context to exercise summarization path
        base_model=capture_client,  # Use capturing client for summarization model too
    )
    print(f"  Built {len(middleware)} middleware components")

    # Create the agent with capturing LLM
    try:
        memory = InMemorySaver()
        agent = create_agent(
            model=capture_client,  # Use capturing client as the base model
            tools=tools,
            checkpointer=memory,
            middleware=middleware,
        )
        print("  Created LangGraph agent with MCP tools and middleware")
    except Exception as e:
        print(f"  Error creating agent: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Invoke the agent with a simple question that should use MCP tools
    print("  Invoking agent with test question...")
    try:
        # Question that triggers Open Targets variant lookup
        question = "What is the most severe consequence of 19_44908822_C_T?"

        # Invoke the agent - this exercises the full middleware chain
        # including InvokeSummarizationMiddleware.before_model()
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "mcp-fixture-capture"}},
        )

        # Extract the final response
        final_message = result.get("messages", [])[-1] if result.get("messages") else None
        if final_message and hasattr(final_message, "content"):
            print(f"  Agent response (truncated): {str(final_message.content)[:200]}...")
        else:
            print(f"  Agent result: {result}")

        print(f"  Captured {capture_client.captured_count} LLM call fixtures")

    except Exception as e:
        print(f"  Error invoking agent: {e}")
        import traceback

        traceback.print_exc()
        # Still count as partial success if we captured some fixtures
        if capture_client.captured_count > 0:
            print(f"  Partial success: Captured {capture_client.captured_count} fixtures before error")
            return 0
        return 1

    print(f"  MCP agent scenario complete: {capture_client.captured_count} fixtures captured")
    return 0


# Scenario runner mapping
SCENARIO_RUNNERS = {
    "template_parsing": _run_template_parsing_scenario,
    "rubric_evaluation": _run_rubric_evaluation_scenario,
    "abstention": _run_abstention_scenario,
    "full_pipeline": _run_full_pipeline_scenario,
    "mcp_agent": _run_mcp_agent_scenario,
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
        if scenario == "full_pipeline":
            continue  # Skip full_pipeline to avoid duplicates
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
        "--scenario",
        "-s",
        help="Scenario to capture (use --list to see available scenarios)",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Capture all scenarios",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available scenarios with descriptions",
    )

    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for capture (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--provider",
        "-p",
        default="anthropic",
        help="LLM provider to use for capture (default: anthropic)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing fixtures without prompting",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
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
