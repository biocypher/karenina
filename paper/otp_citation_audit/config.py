"""Configuration for the citation integrity audit."""

from __future__ import annotations

import os

from pydantic import SecretStr

from karenina.benchmark import ModelConfig
from paper.config import CITATION_GPT_OSS_ENDPOINT

CANONICAL_PARSER = "claude-opus-4-6"
CLAUDE_MODELS = {
    "claude-haiku-4-5-20251001": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "claude-opus-4-6": "opus",
}
TARGET_PER_CONDITION = 12
MAX_CITATIONS_PER_TRACE = 10


def screener_model() -> ModelConfig:
    """Return the temperature-zero GPT-OSS citation screening model."""
    return ModelConfig(
        id="citation-screen-gpt-oss",
        model_name="gpt-oss-120b",
        interface="openai_endpoint",
        endpoint_base_url=os.environ.get("KARENINA_PAPER_GPT_OSS_URL", CITATION_GPT_OSS_ENDPOINT),
        endpoint_api_key=SecretStr(os.environ.get("KARENINA_PAPER_GPT_OSS_KEY", "EMPTY")),
        temperature=0.0,
        extra_kwargs={"max_retries": 0},
    )


def citation_judge(model_name: str = "claude-opus-4-6") -> ModelConfig:
    """Return the web-enabled Claude Agent SDK citation judge."""
    return ModelConfig(
        id="citation-investigator",
        interface="claude_agent_sdk",
        model_provider="anthropic",
        model_name=model_name,
        extra_kwargs={"parse_via_agent_sdk": True},
    )
