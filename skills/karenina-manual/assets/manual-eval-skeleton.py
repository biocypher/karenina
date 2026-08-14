"""Minimal manual evaluation example.

Demonstrates using the ManualAdapter to replay pre-recorded responses
through karenina's full verification pipeline.
"""

from karenina.benchmark import Benchmark
from karenina.adapters.manual import ManualTraces
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

# --- Create benchmark with questions ---

benchmark = Benchmark.create(
    name="Drug Knowledge QA",
    description="Basic pharmaceutical knowledge evaluation",
    version="1.0.0",
)

benchmark.add_question(
    question="What is the mechanism of action of aspirin?",
    raw_answer="Irreversible inhibition of cyclooxygenase (COX) enzymes",
    answer_template="""
class AspirinMOA(BaseAnswer):
    mechanism: str = VerifiedField(
        description="The mechanism of action of aspirin",
        ground_truth="Irreversible inhibition of cyclooxygenase (COX) enzymes",
        verify_with=ContainsAll(substrings=["irreversible", "cyclooxygenase", "COX"]),
    )
""",
)

benchmark.add_question(
    question="What is the primary target of venetoclax?",
    raw_answer="BCL-2",
    answer_template="""
class VenetoclaxTarget(BaseAnswer):
    target: str = VerifiedField(
        description="The primary molecular target of venetoclax",
        ground_truth="BCL-2",
        verify_with=ExactMatch(case_sensitive=False),
    )
""",
)

# --- Register pre-recorded traces ---

manual_traces = ManualTraces(benchmark)
manual_traces.register_traces(
    {
        "What is the mechanism of action of aspirin?": (
            "Aspirin works through the irreversible inhibition of "
            "cyclooxygenase (COX) enzymes, specifically COX-1 and COX-2. "
            "This blocks the production of prostaglandins and thromboxanes."
        ),
        "What is the primary target of venetoclax?": (
            "Venetoclax is a selective BCL-2 inhibitor. It binds directly "
            "to the BCL-2 protein, displacing pro-apoptotic proteins and "
            "triggering programmed cell death in cancer cells."
        ),
    },
    map_to_id=True,
)

# --- Configure verification ---

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            interface="manual",
            manual_traces=manual_traces,
        )
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            temperature=0.0,
        )
    ],
)

# --- Run and display results ---

results = benchmark.run_verification(config)

print(f"Evaluated {len(results)} questions using manual interface\n")
for r in results:
    status = "PASS" if r.template.verify_result else "FAIL"
    print(f"[{status}] {r.metadata.question_text[:60]}")
    print(f"  Model: {r.metadata.answering.display_string}")
    print(f"  Parsed: {r.template.parsed_llm_response}")
    print()
