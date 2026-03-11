"""Capture workflow-taskeval LLM calls as fixture files.

Patches BaseChatModel.ainvoke to record every LLM call, then runs the
TaskEval workflow examples. Each call is saved as a numbered JSON file in
docs/data/workflow-taskeval/.

Usage (from karenina/ directory):
    uv run python docs/data/workflow-taskeval/capture_fixtures.py

Requires ANTHROPIC_API_KEY in the environment.
"""

import hashlib
import json
from pathlib import Path

# Recording infrastructure

DATA_DIR = Path(__file__).parent
_call_counter = 0
_calls: list[dict] = []


def _hash_messages(messages) -> str:
    """Compute a stable hash of message content for fixture matching."""
    normalized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        msg_type = msg.type if hasattr(msg, "type") else "unknown"
        if isinstance(content, str):
            normalized.append(f"{msg_type}:{content}")
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(str(block.get("text", block.get("input", ""))))
                else:
                    text_parts.append(str(block))
            normalized.append(f"{msg_type}:{'|'.join(text_parts)}")
    return hashlib.sha256("|".join(normalized).encode()).hexdigest()[:16]


def _serialize_message(msg) -> dict:
    """Serialize a LangChain BaseMessage to a JSON-safe dict."""
    data: dict = {
        "type": msg.type,
        "content": msg.content,
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        data["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "name") and msg.name:
        data["name"] = msg.name
    return data


def _serialize_response(msg) -> dict:
    """Serialize an AIMessage response to a JSON-safe dict."""
    data: dict = {
        "content": msg.content,
        "id": getattr(msg, "id", None),
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        um = msg.usage_metadata
        data["usage_metadata"] = {
            "input_tokens": getattr(um, "input_tokens", 0),
            "output_tokens": getattr(um, "output_tokens", 0),
            "total_tokens": getattr(um, "total_tokens", 0),
        }
    if hasattr(msg, "response_metadata") and msg.response_metadata:
        rm = msg.response_metadata
        data["response_metadata"] = {
            "model_name": rm.get("model_name") or rm.get("model"),
            "stop_reason": rm.get("stop_reason") or rm.get("finish_reason"),
        }
    return data


async def _recording_ainvoke(self, input, config=None, **kwargs):
    """Wrapper around BaseChatModel.ainvoke that records calls."""
    global _call_counter

    response = await _original_ainvoke(self, input, config=config, **kwargs)

    _call_counter += 1
    seq = _call_counter

    messages = input if isinstance(input, list) else [input]
    prompt_hash = _hash_messages(messages)
    record = {
        "sequence": seq,
        "prompt_hash": prompt_hash,
        "model": getattr(self, "model_name", None) or getattr(self, "model", "unknown"),
        "request": {
            "messages": [_serialize_message(m) for m in messages],
        },
        "response": _serialize_response(response),
    }
    _calls.append(record)

    filepath = DATA_DIR / f"{seq:03d}_{prompt_hash}.json"
    filepath.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print(f"  [recorded call {seq} ({prompt_hash}): {getattr(self, 'model_name', '?')}]")

    return response


# Workflow code (mirrors the notebook cells that call evaluate())


def run_workflow_examples():
    """Run the TaskEval workflow evaluation examples with real API calls."""
    from pydantic import Field

    from karenina.benchmark.task_eval import TaskEval
    from karenina.schemas.config.models import ModelConfig
    from karenina.schemas.entities import BaseAnswer
    from karenina.schemas.entities.rubric import LLMRubricTrait, RegexTrait, Rubric
    from karenina.schemas.verification.config import VerificationConfig

    # === Main flow (Steps 1-5) ===

    # Step 1: Create TaskEval
    task = TaskEval(
        task_id="agent-drug-target",
        metadata={"model": "claude-haiku-4-5", "run_date": "2025-02-15"},
    )
    print(f"Step 1: Created TaskEval: {task.task_id}")

    # Step 2: Log the output to evaluate
    task.log(
        "The approved drug target of venetoclax is BCL2 (B-cell lymphoma 2). "
        "Venetoclax is a selective BCL2 inhibitor that works by displacing pro-apoptotic "
        "proteins, triggering programmed cell death in cancer cells [1]."
    )
    print(f"Step 2: Logged {len(task.global_logs)} event(s)")

    # Step 3: Define answer template
    class Answer(BaseAnswer):
        identifies_bcl2: bool = Field(
            description=(
                "True if the response identifies BCL2 (including Bcl-2, BCL-2, or "
                "B-cell lymphoma 2) as the direct pharmacological target of venetoclax. "
                "False if BCL2 is mentioned only as a pathway member or a different "
                "protein is identified as the primary target."
            )
        )

        def ground_truth(self):
            self.correct = {"identifies_bcl2": True}

        def verify(self) -> bool:
            return self.identifies_bcl2 == self.correct["identifies_bcl2"]

    task.add_template(Answer)

    # Step 3: Add rubric
    rubric = Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="conciseness",
                kind="boolean",
                description=(
                    "True if the response answers directly without unnecessary "
                    "elaboration. False if the response is verbose or includes "
                    "tangential information."
                ),
            ),
        ],
        regex_traits=[
            RegexTrait(
                name="has_citations",
                pattern=r"\[\d+\]",
                description="Response includes numbered citations in bracket notation.",
                case_sensitive=False,
            ),
        ],
    )
    task.add_rubric(rubric)
    print("Step 3: Attached template and rubric")

    # Step 4: Configure and evaluate
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="claude-haiku-4-5",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                interface="langchain",
                temperature=0.0,
            )
        ],
        parsing_only=True,
    )
    result = task.evaluate(config)
    print(f"Step 4: Evaluation complete: {result.summary()}")

    # Step 5: Inspect results
    print(f"\nStep 5: Display:\n{result.display()}")

    # === Multi-step evaluation ===

    multi_task = TaskEval(task_id="multi-step-agent")

    multi_task.log(
        "Found 3 relevant papers on venetoclax mechanism of action.",
        step_id="retrieval",
    )
    multi_task.log(
        "BCL2 is the primary target of venetoclax. It selectively inhibits BCL2, triggering apoptosis in CLL cells.",
        step_id="synthesis",
    )

    multi_task.add_rubric(
        Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="retrieval_quality",
                    kind="boolean",
                    description="True if relevant sources were found for the query.",
                )
            ]
        ),
        step_id="retrieval",
    )
    multi_task.add_rubric(
        Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="synthesis_accuracy",
                    kind="boolean",
                    description="True if the synthesis accurately reflects the retrieved information.",
                )
            ]
        ),
        step_id="synthesis",
    )

    print("\nMulti-step results:")
    for step_id in ["retrieval", "synthesis"]:
        step_result = multi_task.evaluate(config, step_id=step_id)
        step_eval = step_result.per_step[step_id]
        stats = step_eval.get_summary_stats()
        print(f"  Step '{step_id}': {stats['rubric_traits_passed']}/{stats['rubric_traits_total']} traits passed")


if __name__ == "__main__":
    from langchain_core.language_models.chat_models import BaseChatModel

    _original_ainvoke = BaseChatModel.ainvoke

    # Clear existing fixtures
    for f in DATA_DIR.glob("*.json"):
        f.unlink()

    # Patch and run
    BaseChatModel.ainvoke = _recording_ainvoke
    try:
        run_workflow_examples()
    finally:
        BaseChatModel.ainvoke = _original_ainvoke

    print(f"\n=== Captured {_call_counter} LLM calls to {DATA_DIR}/ ===")
