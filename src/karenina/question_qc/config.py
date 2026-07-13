"""Configuration for question quality control runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from karenina.schemas.config import ModelConfig


class RoleModelConfig(BaseModel):
    """Model + tool policy for one QC role (proposer / validator / reviewer)."""

    model: ModelConfig
    system_prompt_override: str | None = None
    allowed_tool_substrings: list[str] = Field(default_factory=list)


class QcRuntimeConfig(BaseModel):
    """Runtime knobs for a QC run.

    Time budgets use **active** agent time by default: tool-call windows can be
    excluded from the stage budget when ``exclude_tool_time`` is True and the
    agent reports tool activity (or when a wall-clock tool buffer is applied).
    """

    max_attempts: int = 3
    invalid_output_retries: int = 1
    # Staged active-time budgets (seconds)
    investigation_seconds: float = 180.0
    wrap_up_seconds: float = 90.0
    conclusion_seconds: float = 30.0
    # When True, tool execution does not consume the active stage budget
    # (event-driven pause when the agent emits tool activity; otherwise a
    # wall-clock buffer is added so tools are less likely to exhaust the stage).
    exclude_tool_time: bool = True
    tool_time_buffer_seconds: float = 600.0
    # Optional absolute wall-clock cap per stage (None = budget + buffer only)
    role_timeout_seconds: float | None = None
    async_max_workers: int | None = None
    # Substrings matching tool names/titles whose time is excluded
    exclude_tool_name_substrings: list[str] = Field(
        default_factory=lambda: [
            "query",
            "search",
            "execute",
            "fetch",
            "cypher",
            "tool",
        ]
    )


class QcConfig(BaseModel):
    """Configuration for a question QC run (analogue of VerificationConfig).

    Question selection and run control (``question_ids``, ``run_name``,
    ``async_enabled``, ``progress_callback``) are method arguments on
    ``Benchmark.run_qc``, not fields on this config.
    """

    proposer: RoleModelConfig
    validator: RoleModelConfig
    reviewer: RoleModelConfig
    runtime: QcRuntimeConfig = Field(default_factory=QcRuntimeConfig)
    prompt_profile: str | None = None
    evidence_context_path: Path | None = None
