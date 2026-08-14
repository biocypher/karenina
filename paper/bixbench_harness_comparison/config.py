"""Configuration for the BixBench agent-harness comparison."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import SecretStr

from karenina.benchmark import ModelConfig, VerificationConfig
from karenina.schemas.config import (
    AgentLimitConfig,
    AgentMiddlewareConfig,
    AgentRuntimeConfig,
    ModelRetryConfig,
    ToolRetryConfig,
)
from karenina.utils import select_openai_endpoint
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy
from paper.config import (
    BIXBENCH_QWEN_ENDPOINTS,
    ZAI_ANTHROPIC_ENDPOINT,
    ZAI_OPENAI_ENDPOINT,
)

ModelLabel = Literal["Qwen 3.5 122B A10B", "GLM-5.1", "Claude Opus 4.6"]
HarnessLabel = Literal["Claude Code", "DeepAgents"]
RuntimeLabel = Literal["docker", "singularity", "apptainer", "host"]

MODELS: tuple[ModelLabel, ...] = (
    "Qwen 3.5 122B A10B",
    "GLM-5.1",
    "Claude Opus 4.6",
)
HARNESSES: tuple[HarnessLabel, ...] = ("Claude Code", "DeepAgents")
MODEL_CLI_LABELS = {
    "qwen": "Qwen 3.5 122B A10B",
    "glm": "GLM-5.1",
    "opus": "Claude Opus 4.6",
}
HARNESS_CLI_LABELS = {
    "claude-code": "Claude Code",
    "deepagents": "DeepAgents",
}

QWEN_MODEL = "qwen3.5-122b-a10b"
GLM_MODEL = "glm-5.1"
OPUS_MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = (
    "You are a biostatistician. Work in the provided workspace directory. "
    "A standard analysis environment is available, with Python and R and the "
    "usual scientific and bioinformatics packages already installed. Install "
    "anything else the task needs. Load the workspace data, perform the "
    "requested analyses with code, answer every requested item, and save your "
    "scripts, interpretation, and result files in the workspace for "
    "independent inspection."
)

@dataclass(frozen=True)
class ExperimentCondition:
    """One model and harness combination in the comparison matrix."""

    model: ModelLabel
    harness: HarnessLabel

    @property
    def slug(self) -> str:
        """Return a stable filesystem label for experiment outputs."""

        model = {
            "Qwen 3.5 122B A10B": "qwen-3-5-122b-a10b",
            "GLM-5.1": "glm-5-1",
            "Claude Opus 4.6": "claude-opus-4-6",
        }[self.model]
        harness = {"Claude Code": "claude-code", "DeepAgents": "deepagents"}[
            self.harness
        ]
        return f"{model}_{harness}"


def selected_conditions(
    model: str = "all",
    harness: str = "both",
) -> list[ExperimentCondition]:
    """Resolve CLI selections into final display labels."""

    models = (
        MODELS
        if model == "all"
        else (cast(ModelLabel, MODEL_CLI_LABELS[model]),)
    )
    harnesses: tuple[HarnessLabel, ...] = (
        HARNESSES
        if harness == "both"
        else (cast(HarnessLabel, HARNESS_CLI_LABELS[harness]),)
    )
    return [
        ExperimentCondition(model=model_label, harness=harness_label)
        for model_label in models
        for harness_label in harnesses
    ]


def _zai_key() -> SecretStr:
    value = os.environ.get("ZAI_API_KEY") or os.environ.get("KARENINA_BIX_API_KEY")
    if not value:
        raise RuntimeError(
            "GLM-5.1 requires ZAI_API_KEY or KARENINA_BIX_API_KEY in the environment"
        )
    return SecretStr(value)


def _container_image(harness: HarnessLabel, runtime: RuntimeLabel) -> str | None:
    if runtime == "host":
        return None
    env_name = (
        "KARENINA_BIX_CLAUDE_CONTAINER_IMAGE"
        if harness == "Claude Code"
        else "KARENINA_BIX_DEEPAGENTS_CONTAINER_IMAGE"
    )
    fallback = (
        "karenina-bixbench-claude:latest"
        if harness == "Claude Code"
        else "karenina-bixbench:latest"
    )
    return os.environ.get(env_name, fallback)


def _runtime(
    harness: HarnessLabel,
    runtime: RuntimeLabel,
    *,
    access_mode: Literal["read_write", "read_only"],
) -> AgentRuntimeConfig:
    if runtime == "host":
        backend: Literal["native", "local_shell"] = (
            "native" if harness == "Claude Code" else "local_shell"
        )
        return AgentRuntimeConfig(backend=backend, access_mode=access_mode)
    return AgentRuntimeConfig(
        backend="container",
        access_mode=access_mode,
        container_runtime=runtime,
        container_image=_container_image(harness, runtime),
    )


def _middleware() -> AgentMiddlewareConfig:
    return AgentMiddlewareConfig(
        limits=AgentLimitConfig(model_call_limit=10_000, tool_call_limit=10_000),
        model_retry=ModelRetryConfig(max_retries=0),
        tool_retry=ToolRetryConfig(max_retries=0),
    )


def _qwen_endpoint() -> str:
    explicit = os.environ.get("KARENINA_BIX_QWEN_ENDPOINT")
    return explicit or BIXBENCH_QWEN_ENDPOINTS[0]


def _answerer(
    condition: ExperimentCondition,
    runtime: RuntimeLabel,
    timeout: int,
) -> ModelConfig:
    runtime_config = _runtime(condition.harness, runtime, access_mode="read_write")
    interface = (
        "claude_agent_sdk" if condition.harness == "Claude Code" else "langchain_deep_agents"
    )
    common: dict[str, Any] = {
        "id": condition.slug,
        "interface": interface,
        "temperature": 0.0,
        "system_prompt": SYSTEM_PROMPT,
        "agent_timeout": timeout,
        "agent_middleware": _middleware(),
        "agent_runtime": runtime_config,
    }
    if condition.model == "GLM-5.1":
        key = _zai_key()
        if interface == "claude_agent_sdk":
            return ModelConfig(
                **common,
                model_name=GLM_MODEL,
                anthropic_base_url=ZAI_ANTHROPIC_ENDPOINT,
                anthropic_api_key=key,
            )
        return ModelConfig(
            **common,
            model_name=GLM_MODEL,
            model_provider="openai",
            endpoint_base_url=ZAI_OPENAI_ENDPOINT,
            endpoint_api_key=key,
            extra_kwargs={"endpoint_base_url_mode": "raw"},
        )
    if condition.model == "Qwen 3.5 122B A10B":
        key = SecretStr(os.environ.get("KARENINA_BIX_QWEN_API_KEY", "EMPTY"))
        if interface == "claude_agent_sdk":
            return ModelConfig(
                **common,
                model_name=QWEN_MODEL,
                anthropic_base_url=_qwen_endpoint(),
                anthropic_api_key=key,
            )
        return ModelConfig(
            **common,
            model_name=QWEN_MODEL,
            model_provider="openai",
            endpoint_base_url=_qwen_endpoint(),
            endpoint_api_key=key,
        )
    if interface == "claude_agent_sdk":
        return ModelConfig(
            **common,
            model_name=OPUS_MODEL,
            extra_kwargs={"effort": "high"},
        )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("Claude Opus with DeepAgents requires ANTHROPIC_API_KEY")
    return ModelConfig(
        **common,
        model_name=OPUS_MODEL,
        model_provider="anthropic",
        anthropic_base_url="https://api.anthropic.com",
        anthropic_api_key=SecretStr(anthropic_key),
        extra_kwargs={"endpoint_base_url_mode": "raw", "effort": "high"},
    )


def _parser(runtime: RuntimeLabel, timeout: int) -> ModelConfig:
    """Build the final GLM-5.1 DeepAgents parser and judge."""

    return ModelConfig(
        id="glm-5-1-parser",
        model_name=GLM_MODEL,
        model_provider="openai",
        interface="langchain_deep_agents",
        temperature=0.0,
        endpoint_base_url=ZAI_OPENAI_ENDPOINT,
        endpoint_api_key=_zai_key(),
        agent_runtime=_runtime(
            "DeepAgents",
            runtime,
            access_mode="read_only",
        ).model_copy(update={"read_max_bytes": 20_000}),
        agent_timeout=timeout,
        extra_kwargs={"endpoint_base_url_mode": "raw"},
    )


def build_config(
    condition: ExperimentCondition,
    *,
    runtime: RuntimeLabel,
    replicates: int,
    workers: int,
    timeout: int,
    workspace_output_dir: Path,
) -> VerificationConfig:
    """Build one condition using the final experiment settings."""

    no_retry = CategoryRetryConfig(max_attempts=0)
    retry = RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=3, backoff_min=1.0, backoff_max=10.0),
        timeout=no_retry,
        rate_limit=no_retry,
        server_error=no_retry,
    )
    return VerificationConfig(
        answering_models=[_answerer(condition, runtime, timeout)],
        parsing_models=[_parser(runtime, timeout)],
        evaluation_mode="template_only",
        replicate_count=replicates,
        agentic_parsing=True,
        agentic_parsing_trigger="dynamic",
        agentic_judge_context="trace_and_workspace",
        agentic_parsing_materialize_trace=True,
        agentic_parsing_persist_trace=True,
        agentic_parsing_max_turns=60,
        agentic_parsing_timeout=float(timeout),
        request_timeout=float(timeout),
        workspace_copy=True,
        workspace_cleanup=True,
        workspace_output_mode="full",
        workspace_output_dir=workspace_output_dir,
        async_enabled=True,
        async_max_workers=workers,
        allow_partial_trace_scoring=True,
        retry_policy=retry,
    )


def failure_burden_model(timeout: int = 600) -> ModelConfig:
    """Build the GLM-5.1 subscription judge used for trace burdens."""

    return ModelConfig(
        id="glm-5-1-failure-burden",
        model_name=GLM_MODEL,
        model_provider="openai",
        interface="langchain_deep_agents",
        temperature=0.0,
        max_tokens=8192,
        endpoint_base_url=ZAI_OPENAI_ENDPOINT,
        endpoint_api_key=_zai_key(),
        extra_kwargs={"endpoint_base_url_mode": "raw"},
        agent_runtime=AgentRuntimeConfig(backend="filesystem", access_mode="read_only"),
        agent_timeout=timeout,
        request_timeout=float(timeout),
    )


def probe_condition_endpoints(conditions: list[ExperimentCondition]) -> None:
    """Fail early when a required OpenAI-compatible route is unavailable."""

    zai_key = _zai_key()
    select_openai_endpoint(
        [ZAI_OPENAI_ENDPOINT],
        GLM_MODEL,
        api_key=zai_key,
    )
    if any(condition.model == "Qwen 3.5 122B A10B" for condition in conditions):
        explicit = os.environ.get("KARENINA_BIX_QWEN_ENDPOINT")
        candidates = [explicit] if explicit else list(BIXBENCH_QWEN_ENDPOINTS)
        selected = select_openai_endpoint(
            candidates,
            QWEN_MODEL,
            api_key=os.environ.get("KARENINA_BIX_QWEN_API_KEY", "EMPTY"),
        )
        os.environ["KARENINA_BIX_QWEN_ENDPOINT"] = selected
