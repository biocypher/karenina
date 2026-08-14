"""Configuration for the Open Targets model comparison."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from pydantic import SecretStr

from karenina.benchmark import ModelConfig, VerificationConfig
from karenina.schemas.config.models import (
    AgentLimitConfig,
    AgentMiddlewareConfig,
    ModelRetryConfig,
    SummarizationConfig,
    ToolRetryConfig,
)
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, TimeoutEscalationConfig
from paper.config import MODEL_COMPARISON_ENDPOINTS

ANSWERER_NAMES = (
    "gpt-oss-120b",
    "qwen3.5-a3b",
    "qwen3.6-a3b",
    "qwen3.5-122b-a10b",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)
REFERENCE_PARSER = "claude-opus-4-6"

_ANTHROPIC_NAMES = frozenset(name for name in ANSWERER_NAMES if name.startswith("claude-"))

_SYSTEM_PROMPT = (
    "You are an expert biomedical assistant with deep knowledge of drug "
    "targets, diseases, genetics, and clinical data.\n\n"
    "Think through each question carefully before answering. Consider "
    "relevant evidence, mechanisms, and data sources.\n\n"
    "Guidelines:\n"
    "- Be precise with identifiers (gene symbols, MONDO IDs, MeSH IDs, "
    "UniProt accessions).\n"
    "- Match your reasoning depth to the question. Simple lookups need "
    "minimal reasoning; complex questions deserve more analysis."
)
_MCP_SYSTEM_PROMPT = _SYSTEM_PROMPT + (
    "\n\nYou have access to the Open Targets Platform via MCP tools. Use them "
    "to verify identifiers, fetch evidence, and ground your answers. Tools: "
    "get_open_targets_graphql_schema, get_type_dependencies, search_entities, "
    "query_open_targets_graphql, batch_query_open_targets_graphql."
)


def _middleware() -> AgentMiddlewareConfig:
    return AgentMiddlewareConfig(
        limits=AgentLimitConfig(model_call_limit=30, tool_call_limit=60),
        tool_retry=ToolRetryConfig(max_retries=1),
        model_retry=ModelRetryConfig(max_retries=10, max_delay=30.0, on_failure="raise"),
        summarization=SummarizationConfig(enabled=False),
    )


def _model(name: str, *, role: str, mcp_url: str | None, temperature: float) -> ModelConfig:
    system_prompt = _MCP_SYSTEM_PROMPT if mcp_url else _SYSTEM_PROMPT
    if name in _ANTHROPIC_NAMES:
        model = ModelConfig(
            id=name,
            model_name=name,
            interface="langchain",
            model_provider="anthropic",
            temperature=temperature,
            system_prompt=system_prompt if role == "answering" else None,
            extra_kwargs={"max_retries": 0},
        )
    else:
        extra_kwargs: dict[str, object] = {"max_retries": 0}
        if name.startswith("qwen"):
            extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        model = ModelConfig(
            id=name,
            model_name=name,
            interface="openai_endpoint",
            endpoint_base_url=os.environ.get(
                f"KARENINA_PAPER_{name.upper().replace('.', '_').replace('-', '_')}_URL",
                MODEL_COMPARISON_ENDPOINTS[name],
            ),
            endpoint_api_key=SecretStr(os.environ.get("KARENINA_PAPER_VLLM_KEY", "EMPTY")),
            temperature=temperature,
            system_prompt=system_prompt if role == "answering" else None,
            extra_kwargs=extra_kwargs,
        )
    if role == "answering" and mcp_url is not None:
        model = model.model_copy(
            update={
                "mcp_urls_dict": {"otp": mcp_url},
                "agent_middleware": _middleware(),
                "agent_timeout": 900,
                "mcp_http_timeout": 240.0,
                "mcp_sse_read_timeout": 600.0,
            }
        )
    return model


def build_config(
    mcp_urls: Mapping[str, str] | None,
    *,
    model_names: Sequence[str] = ANSWERER_NAMES,
    temperature: float = 0.0,
    replicates: int = 3,
    workers: int = 16,
) -> VerificationConfig:
    """Build the seven-answerer by seven-parser comparison configuration."""
    unknown = set(model_names) - set(ANSWERER_NAMES)
    if unknown:
        raise ValueError(f"Unknown comparison models: {sorted(unknown)}")
    answerers = [
        _model(name, role="answering", mcp_url=mcp_urls.get(name) if mcp_urls else None, temperature=temperature)
        for name in model_names
    ]
    parsers = [_model(name, role="parsing", mcp_url=None, temperature=temperature) for name in model_names]
    retry = RetryPolicy(
        timeout=CategoryRetryConfig(max_attempts=4, backoff_min=5.0, backoff_max=30.0),
        connection=CategoryRetryConfig(max_attempts=5, backoff_min=1.0, backoff_max=10.0),
        timeout_escalation=TimeoutEscalationConfig(strategy="linear", max_timeout=900.0),
    )
    return VerificationConfig(
        answering_models=answerers,
        parsing_models=parsers,
        evaluation_mode="template_only",
        replicate_count=replicates,
        request_timeout=180.0,
        async_enabled=True,
        async_max_workers=workers,
        retry_policy=retry,
        abstention_enabled=True,
        sufficiency_enabled=False,
        answerer_concurrency_limits=dict.fromkeys(_ANTHROPIC_NAMES, 4),
    )
