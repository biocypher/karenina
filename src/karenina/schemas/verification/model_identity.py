"""Model identity for verification results.

ModelIdentity is the single paradigm for model identification in verification
results. It replaces all string-based model identifiers and per-adapter
format_model_string functions.

Identity dimensions:
- Answering models: (interface, model_name, tools) — tools = MCP server names
- Parsing models: (interface, model_name) — parsing models never have tools
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from karenina.schemas.config.models import ModelConfig


class ModelIdentity(BaseModel):
    """Composite model identity for verification results.

    Captures all dimensions that make a model configuration unique:
    interface, model name, and (for answering models) MCP tool names.
    """

    model_config = ConfigDict(extra="forbid")

    interface: str = Field(..., json_schema_extra={"max_length": 50})
    model_name: str = Field(..., json_schema_extra={"max_length": 255})
    tools: list[str] = Field(default_factory=list)
    config_id: str | None = Field(
        default=None,
        description="ModelConfig.id when it differs from model_name",
    )

    @classmethod
    def from_model_config(cls, config: ModelConfig, *, role: str = "answering") -> ModelIdentity:
        """Create a ModelIdentity from a ModelConfig.

        Args:
            config: The model configuration to extract identity from.
            role: Either "answering" or "parsing". Parsing models always
                  produce tools=[] regardless of config's mcp_urls_dict.

        Returns:
            A ModelIdentity capturing the relevant identity dimensions.
        """
        tools: list[str] = []
        if role == "answering" and config.mcp_urls_dict:
            tools = sorted(config.mcp_urls_dict.keys())
        effective_name = config.model_name or "unknown"
        config_id = config.id if config.id and config.id != effective_name else None
        return cls(
            interface=config.interface,
            model_name=effective_name,
            tools=tools,
            config_id=config_id,
        )

    @property
    def display_string(self) -> str:
        """Human-readable model identity string.

        Format examples:
        - "interface:model_name"
        - "interface:model_name (config_id)"
        - "interface:model_name +[tool1, tool2]"
        - "interface:model_name (config_id) +[tool1, tool2]"
        """
        base = f"{self.interface}:{self.model_name}"
        if self.config_id:
            base = f"{base} ({self.config_id})"
        if self.tools:
            return f"{base} +[{', '.join(self.tools)}]"
        return base

    @property
    def canonical_key(self) -> str:
        """Deterministic identity key for hashing and comparison.

        Format without config_id: "interface:model_name:tools" (3 segments).
        Format with config_id: "interface:model_name:config_id:tools" (4 segments).
        Tools are sorted and pipe-delimited.
        """
        tools_part = "|".join(sorted(self.tools)) if self.tools else ""
        if self.config_id:
            return f"{self.interface}:{self.model_name}:{self.config_id}:{tools_part}"
        return f"{self.interface}:{self.model_name}:{tools_part}"
