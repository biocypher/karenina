"""Configuration and model builders for response characterization.

Shared paths and endpoints live in :mod:`paper.config`.
"""

from __future__ import annotations

import os

from pydantic import SecretStr

from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import RegexRubricTrait
from paper.config import RESPONSE_GPT_OSS_ENDPOINT

FAILURE_SHAPE_TRAITS = [
    RegexRubricTrait(
        name="NoAIFinalMessage",
        summary="trace ends without AI message",
        description=(
            "Matches when the trace ends with a tool result and no subsequent "
            "AI message. Indicates the agent halted mid-tool-call."
        ),
        pattern=r"(?s)--- Tool Message \(call_id:[^)]*\) ---(?:(?!--- AI Message ---).)*\Z",
        case_sensitive=True,
        invert_result=False,
        higher_is_better=False,
    ),
    RegexRubricTrait(
        name="EmptyTrace",
        summary="trace contains no usable output",
        description=(
            "Matches when the answerer produced no output at all: an empty "
            "raw response or one consisting solely of whitespace."
        ),
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
        name="BracketedTraceNote",
        summary="trace ends in a bracketed pipeline annotation",
        description=(
            "Matches when the response ends in a bracketed runtime sentinel, "
            "indicating that the agent was cut off."
        ),
        pattern=r"\[[^\[\]\n]+\]\s*\Z",
        case_sensitive=True,
        invert_result=False,
        higher_is_better=False,
    ),
]

NO_TOOL_CALL_TRAIT = RegexRubricTrait(
    name="NoToolCall",
    summary="trace contains no tool call",
    description="Matches when the trace contains no tool-call request and no tool-result block.",
    pattern=r"(?m)^Tool Calls:\s*$|^--- Tool Message \(call_id:",
    case_sensitive=True,
    invert_result=True,
    higher_is_better=False,
)

EMPTY_TRAILING_CLASSES = {
    "answer_present_no_final_message": (
        "The trace contains the correct answer, or an answer-equivalent tool "
        "result matching the reference answer, but the model stops before "
        "writing a final assistant interpretation."
    ),
    "wrong_result_no_final_message": (
        "The trace reaches a substantive result, but that result is not the "
        "reference answer, and the model stops before writing a final assistant "
        "interpretation."
    ),
    "no_answer_gave_up": (
        "The trace does not contain a usable answer to the benchmark question. "
        "It mainly shows failed tool calls, schema exploration, irrelevant "
        "work, or an incomplete search path."
    ),
}

MAX_STAGE1_INPUT_TOKENS = 120_000
POST_HOC_WORKERS = 32

ANSWERERS = [
    "qwen3.5-a3b",
    "qwen3.6-a3b",
    "qwen3.5-122b-a10b",
    "gpt-oss-120b",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]


def gpt_oss_judge() -> ModelConfig:
    """Return the temperature-zero GPT-OSS judge used by fresh runs."""
    base_url = os.environ.get(
        "KARENINA_PAPER_GPT_OSS_URL",
        RESPONSE_GPT_OSS_ENDPOINT,
    )
    return ModelConfig(
        id="gpt-oss-judge",
        model_provider="openai",
        model_name="gpt-oss-120b",
        interface="openai_endpoint",
        endpoint_base_url=base_url,
        endpoint_api_key=SecretStr(os.environ.get("KARENINA_PAPER_GPT_OSS_KEY", "EMPTY")),
        temperature=0.0,
        extra_kwargs={"max_retries": 0},
    )


def evidence_grounding_judge() -> ModelConfig:
    """Return the GPT-OSS judge with a 16,384-token output cap."""
    judge = gpt_oss_judge()
    return judge.model_copy(update={"max_tokens": 16_384})
