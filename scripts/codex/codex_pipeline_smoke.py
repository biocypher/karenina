"""Pipeline integration smoke for the codex_sdk adapter.

Three-question end-to-end `run_verification` with interface="codex_sdk" as
the answering model against the live vLLM endpoint, executed in parallel
(async_enabled=True, async_max_workers=3), with a plain OpenAI-compatible
langchain model (interface="openai_endpoint") as the parsing judge on the
same endpoint.

Confirms: the pipeline completes under parallelism, every
VerificationResult carries its own codex trace with canonical delimiters,
parsing consumed each response, results are correct per question (no
cross-contamination), and no stage crashed.

Run from the worktree:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/codex/codex_pipeline_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

from karenina.benchmark import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

BASE_URL = os.environ.get("CODEX_GATE_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CODEX_GATE_MODEL", "qwen3.5-122b-a10b")

QUESTIONS = [
    ("What is the capital of France? Answer with just the city name.", "Paris"),
    ("What is the capital of Japan? Answer with just the city name.", "Tokyo"),
    ("What is the capital of Italy? Answer with just the city name.", "Rome"),
]

ANSWER_TEMPLATE = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    city: str = Field(description="The city named in the response")

    def model_post_init(self, __context):
        self.correct = {{"city": "{expected}"}}

    def verify(self) -> bool:
        return self.city.strip().lower() == self.correct["city"].lower()
"""


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" ({detail})" if detail else ""))
    return condition


def main() -> int:
    print(f"endpoint: {BASE_URL}  model: {MODEL}")

    benchmark = Benchmark.create(
        name="codex-pipeline-smoke",
        description="3-question parallel codex_sdk pipeline integration smoke",
        version="1.0.0",
    )
    for question, expected in QUESTIONS:
        benchmark.add_question(
            question,
            expected,
            answer_template=ANSWER_TEMPLATE.format(expected=expected),
            finished=True,
        )

    answering = ModelConfig(
        id="codex-answerer",
        model_name=MODEL,
        model_provider="openai",
        interface="codex_sdk",
        endpoint_base_url=BASE_URL,
    )
    # Parsing judge: plain langchain ChatOpenAI against the same endpoint,
    # mirroring tests/live/_async_live_helpers.openai_model.
    parsing = ModelConfig(
        id="qwen-parser",
        model_name=MODEL,
        model_provider="openai",
        interface="openai_endpoint",
        endpoint_base_url=BASE_URL,
        endpoint_api_key="EMPTY",
        temperature=0.0,
        extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )

    config = VerificationConfig(
        answering_models=[answering],
        parsing_models=[parsing],
        replicate_count=1,
        evaluation_mode="template_only",
        async_enabled=True,
        async_max_workers=3,
    )

    t0 = time.time()
    result_set = benchmark.run_verification(config=config, run_name="codex-pipeline-smoke")
    print(f"run_verification wall time: {time.time() - t0:.1f}s (3 questions, 3 workers)")

    ok = _check("exactly three results", len(result_set.results) == 3, str(len(result_set.results)))
    if not result_set.results:
        print("\nRESULT: FAIL")
        return 1

    expected_by_question = dict(QUESTIONS)
    for result in result_set.results:
        metadata = result.metadata
        template = result.template
        expected = expected_by_question.get(metadata.question_text, "?")
        parsed = (template.parsed_llm_response or {}) if template else {}
        raw = (template.raw_llm_response if template else None) or ""
        print(f"\n--- {metadata.question_text[:45]}... (expected {expected}) ---")
        print(f"  failure:       {metadata.failure}")
        print(f"  verify_result: {template.verify_result if template else None}")
        print(f"  parsed_llm:    {parsed}")
        print(f"  raw trace:     {len(raw)} chars")

        ok &= _check("no failure recorded", metadata.failure is None, str(metadata.failure))
        ok &= _check("template result present", template is not None)
        if template is not None:
            ok &= _check("raw trace has canonical AI Message delimiter", "--- AI Message ---" in raw)
            ok &= _check("parsing produced a parsed response", template.parsed_llm_response is not None)
            ok &= _check(
                "parsed city matches its own question",
                str(parsed.get("city", "")).strip().lower() == expected.lower(),
                str(parsed.get("city")),
            )
            ok &= _check("verify_result is True", template.verify_result is True, str(template.verify_result))

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
