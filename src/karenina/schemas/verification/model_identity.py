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

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from karenina.schemas.config.models import ModelConfig


class ModelIdentity(BaseModel):
    """Composite model identity for verification results.

    Captures all dimensions that make a model configuration unique:
    interface, model name, and (for answering models) MCP tool names.
    """

    interface: str = Field(..., json_schema_extra={"max_length": 50})
    model_name: str = Field(..., json_schema_extra={"max_length": 255})
    tools: list[str] = Field(default_factory=list)

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
        return cls(
            interface=config.interface,
            model_name=config.model_name or "unknown",
            tools=tools,
        )

    @property
    def display_string(self) -> str:
        """Human-readable model identity string.

        Format: "interface:model_name" or "interface:model_name +[tool1, tool2]"
        """
        base = f"{self.interface}:{self.model_name}"
        if self.tools:
            return f"{base} +[{', '.join(self.tools)}]"
        return base

    @property
    def canonical_key(self) -> str:
        """Deterministic identity key for hashing and comparison.

        Format: "interface:model_name:tool1|tool2" (tools sorted, pipe-delimited).
        """
        tools_part = "|".join(sorted(self.tools)) if self.tools else ""
        return f"{self.interface}:{self.model_name}:{tools_part}"
