"""Capture quickstart LLM calls as fixture files.

Patches BaseChatModel.ainvoke to record every LLM call, then runs the
quickstart flow. Each call is saved as a numbered JSON file in
docs/data/quickstart/.

Usage (from karenina/ directory):
    uv run python docs/data/quickstart/capture_fixtures.py

Requires ANTHROPIC_API_KEY in the environment.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

# ── Recording infrastructure ──────────────────────────────────────────

DATA_DIR = Path(__file__).parent
_call_counter = 0
_calls: list[dict] = []


def _hash_messages(messages) -> str:
    """Compute a stable hash of message content for fixture matching."""
    normalized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        msg_type = msg.type if hasattr(msg, "type") else "unknown"
        # Only hash role + text content (ignore tool IDs and other metadata)
        if isinstance(content, str):
            normalized.append(f"{msg_type}:{content}")
        elif isinstance(content, list):
            # For content blocks, extract text parts
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
        # Only keep serializable parts
        rm = msg.response_metadata
        data["response_metadata"] = {
            "model_name": rm.get("model_name") or rm.get("model"),
            "stop_reason": rm.get("stop_reason") or rm.get("finish_reason"),
        }
    return data


async def _recording_ainvoke(self, input, config=None, **kwargs):
    """Wrapper around BaseChatModel.ainvoke that records calls."""
    global _call_counter

    # Call the real model
    response = await _original_ainvoke(self, input, config=config, **kwargs)

    _call_counter += 1
    seq = _call_counter

    # Serialize
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

    # Save with hash in filename for lookup
    filepath = DATA_DIR / f"{seq:03d}_{prompt_hash}.json"
    filepath.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print(f"  [recorded call {seq} ({prompt_hash}): {getattr(self, 'model_name', '?')}]")

    return response


# ── Quickstart code (same as the notebook, minus mocks) ──────────────

def run_quickstart():
    """Run the quickstart flow end-to-end with real API calls."""
    from karenina import Benchmark
    from karenina.schemas.entities import BaseAnswer
    from karenina.schemas import (
        LLMRubricTrait,
        ModelConfig,
        RegexTrait,
        VerificationConfig,
    )
    from pydantic import Field

    # Step 1
    benchmark = Benchmark.create(
        name="Genomics Knowledge Benchmark",
        description="Testing LLM knowledge of genomics and molecular biology",
        version="1.0.0",
        creator="Your Name",
    )
    print(f"Step 1: Created benchmark: {benchmark.name}")

    # Step 2
    questions = [
        {"question": "How many chromosomes are in a human somatic cell?", "answer": "46"},
        {"question": "What is the approved drug target of Venetoclax?", "answer": "BCL2"},
        {"question": "How many protein subunits does hemoglobin A have?", "answer": "4"},
    ]
    question_ids = []
    for q in questions:
        qid = benchmark.add_question(
            question=q["question"],
            raw_answer=q["answer"],
            author={"name": "Bio Curator", "email": "curator@example.com"},
        )
        question_ids.append(qid)
    print(f"Step 2: Added {len(question_ids)} questions")

    # Step 3a: Generate templates
    print("Step 3a: Generating templates...")
    benchmark.generate_all_templates(
        model="claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0.0,
    )
    print(f"Step 3a: Generated templates for {benchmark.question_count} questions")

    # Step 3b: Review
    generated_code = benchmark.get_template(question_ids[0])
    print(f"Step 3b: Generated template preview:\n{generated_code[:200]}...")

    # Step 3c: Class-based override
    class Answer(BaseAnswer):
        is_bcl2: bool = Field(
            description="Whether the response identifies BCL2 as the putative target of the drug"
        )

        def model_post_init(self, __context):
            self.correct = {"is_bcl2": True}

        def verify(self) -> bool:
            return self.is_bcl2 == self.correct["is_bcl2"]

    benchmark.update_template(question_ids[1], Answer)
    print("Step 3c: Updated template with class-based definition")

    # Step 4
    benchmark.add_global_rubric_trait(
        LLMRubricTrait(
            name="Conciseness",
            description="Rate how concise the answer is on a scale of 1-5, where 1 is very verbose and 5 is extremely concise.",
            kind="score",
        )
    )
    venetoclax_qid = question_ids[1]
    benchmark.add_question_rubric_trait(
        venetoclax_qid,
        RegexTrait(
            name="Contains BCL2",
            description="The response must mention BCL2",
            pattern=r"\bBCL2\b",
            case_sensitive=True,
        ),
    )
    print("Step 4: Added rubric traits")

    # Step 5
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="claude-haiku-4-5",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                interface="langchain",
                temperature=0.7,
                system_prompt="You are a knowledgeable assistant. Answer accurately and concisely.",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="claude-haiku-4-5",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                interface="langchain",
                temperature=0.0,
            )
        ],
        evaluation_mode="template_and_rubric",
        rubric_enabled=True,
    )
    results = benchmark.run_verification(config)
    print(f"Step 5: Verification complete — {len(results.results)} results")

    # Step 6
    template_results = results.get_template_results()
    df_templates = template_results.to_dataframe()
    print("\nStep 6a: Template results")
    print(df_templates[["question_id", "field_name", "gt_value", "llm_value", "field_match"]].to_string(index=False))

    pass_rates = template_results.aggregate_pass_rate(by="question_id")
    print("\nStep 6b: Pass rates")
    for qid, rate in pass_rates.items():
        print(f"  {qid[:40]}...  {rate:.0%}")

    rubric_results = results.get_rubrics_results()
    df_rubrics = rubric_results.to_dataframe()
    print("\nStep 6c: Rubric results")
    print(df_rubrics[["question_id", "trait_name", "trait_score", "trait_type"]].to_string(index=False))

    # Step 7
    tmpdir = tempfile.mkdtemp()
    checkpoint_path = Path(tmpdir) / "genomics_benchmark.jsonld"
    benchmark.save(checkpoint_path)
    print(f"\nStep 7: Saved and loaded OK")
    loaded = Benchmark.load(checkpoint_path)
    print(f"  Loaded '{loaded.name}' with {loaded.question_count} questions")
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    from langchain_core.language_models.chat_models import BaseChatModel

    _original_ainvoke = BaseChatModel.ainvoke

    # Clear existing fixtures
    for f in DATA_DIR.glob("*.json"):
        f.unlink()

    # Patch and run
    BaseChatModel.ainvoke = _recording_ainvoke
    try:
        run_quickstart()
    finally:
        BaseChatModel.ainvoke = _original_ainvoke

    print(f"\n=== Captured {_call_counter} LLM calls to {DATA_DIR}/ ===")
