"""Simplified karenina walkthrough of the response-characterization analyses.

The full analyses in this package (failure_tree, no_tool_call,
missing_final_message, evidence_grounding) all share one karenina pattern:

    load archived results -> attach a rubric -> evaluate -> summarize

This reduced example shows the three rubric forms used by the analyses:

  1. Regex rubric  -> deterministic response-shape flags   (failure_tree, no_tool_call)
  2. Literal rubric -> classify blank-final traces with an LLM (missing_final_message)
  3. Boolean rubric -> judge evidence grounding with an LLM    (evidence_grounding)

Step 1 is offline. Steps 2-3 call the GPT-OSS judge endpoint configured
through KARENINA_PAPER_GPT_OSS_URL / KARENINA_PAPER_GPT_OSS_KEY.

Run from the karenina repository root:

    uv run python -m paper.otp_response_characterization.analyses.simplified [N]

The optional N limits how many traces each LLM step judges (default 5).
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from karenina.benchmark import (
    ResultsIOManager,
    RowContext,
    evaluate_rubric_on_results,
)
from karenina.schemas.entities.rubric import (
    LLMRubricTrait,
    RegexRubricTrait,
    Rubric,
)
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import bootstrap, input_path
from paper.common.qa_results import load_benchmark_text
from paper.config import OTP_BENCHMARK_JSONLD, OTP_MCP_RESULTS
from paper.otp_response_characterization.config import gpt_oss_judge

# ---------------------------------------------------------------------------
# Rubric prompts (compact versions of the prompts in missing_final_message.py
# and evidence_grounding.py).

BLANK_FINAL_CLASSES = {
    "answer_present_no_final_message": ("The correct answer, or an answer-equivalent tool result, is present."),
    "wrong_result_no_final_message": (
        "The trace reaches a substantive result that conflicts with the reference answer."
    ),
    "no_answer_gave_up": ("No usable answer is present: failed tool calls, schema exploration, or irrelevant work."),
}

BLANK_FINAL_PROMPT = """\
Classify why this agent trace ended with an empty final assistant message.
Use exactly one class:
- answer_present_no_final_message: the correct answer, or an answer-equivalent
  tool result, is present in the trace.
- wrong_result_no_final_message: the trace reaches a substantive result, but
  it conflicts with or misses the reference answer.
- no_answer_gave_up: no usable answer is present. The trace mainly shows
  failed tool calls, schema exploration, or irrelevant work.
Judge the trace against the reference answer. Prefer no_answer_gave_up when
there is no clear answer-bearing result.
"""

GROUNDING_PROMPT = (
    "You are reviewing an agent trace produced while answering a benchmark "
    "question with tool access to the Open Targets Platform. "
    "Answer True if the trace's final answer is grounded in evidence the "
    "agent actually retrieved: cited tool results, entity IDs, or data that "
    "appear in the trace. "
    "Answer False if the answer is asserted without retrieved support, or if "
    "the trace contains no usable answer."
)


def load_results(path: Path) -> Iterator[VerificationResult]:
    """Reopen archived QA results with ``ResultsIOManager``.

    ``ResultsIOManager.iter_from_json`` streams one JSON export of stored
    verification results. ``raw=False`` validates every row into a
    ``VerificationResult``.
    """
    yield from ResultsIOManager.iter_from_json(path, raw=False)


def context_for(
    result: VerificationResult,
    benchmark: dict[str, tuple[str, str]],
) -> RowContext:
    """Return the question and reference answer for one stored row.

    LLM rubrics see the judged text plus this per-row context.
    """
    question, reference = benchmark[result.metadata.question_id]
    return RowContext(question=question, ground_truth=reference)


def main() -> None:
    bootstrap()
    args = parse_args()
    judge = gpt_oss_judge()
    results_path = input_path(OTP_MCP_RESULTS)
    benchmark = load_benchmark_text(input_path(OTP_BENCHMARK_JSONLD))

    print(f"results: {results_path}")
    print(f"judge:   {judge.model_name} via {judge.interface}")
    print(f"llm limit: {args.limit} traces per live step\n")

    # -- 1. load archived results -------------------------------------------
    # Every analysis starts here: stream stored results and read metadata.
    head = list(itertools.islice(load_results(results_path), 5))
    print(f"[1] archived results - first {len(head)} rows:")
    for result in head:
        meta = result.metadata
        print(f"    {meta.question_id[:28]:<28} {meta.answering.model_name:<22} rep {meta.replicate}")

    # -- 2. regex rubric: response shapes (offline) --------------------------
    # RegexRubricTrait scores every trace deterministically against a pattern.
    # No LLM is called anywhere in this step.
    shape_rubric = Rubric(
        regex_traits=[
            RegexRubricTrait(
                name="EmptyTrace",
                summary="trace contains no usable output",
                description="Matches when the answerer produced no output at all.",
                pattern=r"\A\s*\Z",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=False,
            ),
            RegexRubricTrait(
                name="EmptyTrailingAI",
                summary="trace ends in an empty AI message",
                description="Matches when the trace ends on an AI message header with no content after it.",
                pattern=r"--- AI Message ---\s*\Z",
                case_sensitive=True,
                invert_result=False,
                higher_is_better=False,
            ),
            RegexRubricTrait(
                name="NoToolCall",
                summary="trace contains no tool call",
                description="Matches when the trace contains no tool-call request and no tool-result block.",
                pattern=r"(?m)^Tool Calls:\s*$|^--- Tool Message \(call_id:",
                case_sensitive=True,
                invert_result=True,  # pattern absent -> trait True
                higher_is_better=False,
            ),
        ]
    )

    shape_judgments = list(
        evaluate_rubric_on_results(
            load_results(results_path),
            shape_rubric,
            judge,  # required by the facade, never called for regex traits
            collapse_parser_siblings=True,  # judge each unique trace once
            max_workers=8,
        )
    )
    for judgment in shape_judgments:
        if judgment.error is not None:
            raise RuntimeError(f"shape scoring failed: {judgment.error}")
    print(f"\n[2] response shapes - {len(shape_judgments)} unique traces scored offline")
    for name in ("EmptyTrace", "EmptyTrailingAI", "NoToolCall"):
        matches = sum(1 for j in shape_judgments if j.scores.get(name) is True)
        print(f"    {name:<18} {matches}/{len(shape_judgments)}")

    # -- 3. literal rubric: classify blank-final traces (LLM) ----------------
    # Take the traces the regex flagged as ending in a blank AI message and
    # ask the judge for one label per trace (kind="literal").
    selected = [judgment for judgment in shape_judgments if judgment.scores.get("EmptyTrailingAI") is True][
        : args.limit
    ]
    print(f"\n[3] blank-final traces - {len(selected)} sent to the judge")

    if selected:
        final_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="MissingFinalMessageOutcome",
                    summary="classifies a blank final assistant response",
                    description=BLANK_FINAL_PROMPT,
                    kind="literal",  # judge picks exactly one label
                    classes=BLANK_FINAL_CLASSES,
                    higher_is_better=None,
                )
            ]
        )

        # Rejoin selected rows from a fresh load by result ID.
        selected_ids = {judgment.representative_result_id for judgment in selected}
        rows = [result for result in load_results(results_path) if result.metadata.result_id in selected_ids]

        judged = list(
            evaluate_rubric_on_results(
                rows,
                final_rubric,
                judge,
                row_context=lambda result: context_for(result, benchmark),
                max_workers=1,  # live LLM call; keep it serial
            )
        )
        labels: Counter[str] = Counter()
        for judgment in judged:
            if judgment.error is not None:
                raise RuntimeError(f"blank-final judgment failed: {judgment.error}")
            label = judgment.labels.get("MissingFinalMessageOutcome")
            if label not in BLANK_FINAL_CLASSES:
                raise RuntimeError(f"blank-final judge returned no valid class: {label!r}")
            labels[label] += 1
        print("    classifications:")
        for label, count in labels.most_common():
            print(f"    {label:<34} {count}")
    else:
        print("    (no blank-final traces found in the archive)")

    # -- 4. boolean rubric: evidence grounding (LLM) -------------------------
    # Mirror evidence_grounding.py: one True/False judgment per trace. Only
    # rows whose template verification passed are judged.
    grounding_rubric = Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="EvidenceGroundedAnswer",
                summary="answer grounded in retrieved evidence",
                description=GROUNDING_PROMPT,
                kind="boolean",
                higher_is_better=True,  # True is the good outcome
            )
        ]
    )

    def verified(result: VerificationResult) -> bool:
        template = result.template
        return template is not None and template.verify_result is True

    eligible = itertools.islice(
        (result for result in load_results(results_path) if verified(result)),
        args.limit,
    )
    judged = list(
        evaluate_rubric_on_results(
            eligible,
            grounding_rubric,
            judge,
            row_context=lambda result: context_for(result, benchmark),
            max_workers=1,
        )
    )
    grounded = 0
    for judgment in judged:
        if judgment.error is not None:
            raise RuntimeError(f"grounding judgment failed: {judgment.error}")
        verdict = judgment.scores.get("EvidenceGroundedAnswer")
        if not isinstance(verdict, bool):
            raise RuntimeError(f"grounding judge returned no verdict: {verdict!r}")
        grounded += int(verdict)
    print(f"\n[4] evidence grounding - {grounded}/{len(judged)} verified traces grounded")

    print(
        "\nEach step above is one karenina function:\n"
        "  ResultsIOManager.iter_from_json   load stored results\n"
        "  Rubric(regex_traits=/llm_traits=) define the criteria\n"
        "  evaluate_rubric_on_results        judge rows with those criteria"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=5,
        help="traces each LLM step judges (default: 5)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be at least 1")
    return args


if __name__ == "__main__":
    main()
