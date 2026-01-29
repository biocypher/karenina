"""Registration module for LangChain adapters.

This module registers the LangChain interface and related routing interfaces
(openrouter, openai_endpoint) with the AdapterRegistry.

Also registers adapter instructions for PARSING, RUBRIC, and DEEP JUDGMENT tasks
that append format instructions to system and user prompts (since LangChain does
not have native structured output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec
from karenina.ports.adapter_instruction import AdapterInstructionRegistry

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.workflow.models import ModelConfig


def _check_langchain_availability() -> AdapterAvailability:
    """Check if LangChain is available."""
    try:
        import langchain_core  # noqa: F401

        return AdapterAvailability(
            available=True,
            reason="LangChain is installed and available",
        )
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason="LangChain packages not installed. Install with: pip install langchain-core langchain-anthropic",
            fallback_interface=None,  # No fallback if LangChain unavailable
        )


def _create_langchain_agent(config: ModelConfig) -> AgentPort:
    """Factory function to create LangChain agent adapter."""
    from karenina.adapters.langchain.agent import LangChainAgentAdapter

    return LangChainAgentAdapter(config)


def _create_langchain_llm(config: ModelConfig) -> LLMPort:
    """Factory function to create LangChain LLM adapter."""
    from karenina.adapters.langchain.llm import LangChainLLMAdapter

    return LangChainLLMAdapter(config)


def _create_langchain_parser(config: ModelConfig) -> ParserPort:
    """Factory function to create LangChain parser adapter."""
    from karenina.adapters.langchain.parser import LangChainParserAdapter

    return LangChainParserAdapter(config)


def _format_langchain_model_string(config: ModelConfig) -> str:
    """Format model string for standard LangChain interface."""
    if config.model_provider:
        return f"{config.model_provider}/{config.model_name}"
    return config.model_name or "unknown"


def _format_openrouter_model_string(config: ModelConfig) -> str:
    """Format model string for OpenRouter interface.

    OpenRouter uses just the model name, not provider/model format.
    """
    return config.model_name or "unknown"


def _format_openai_endpoint_model_string(config: ModelConfig) -> str:
    """Format model string for OpenAI endpoint interface."""
    return f"endpoint/{config.model_name}" if config.model_name else "endpoint/unknown"


# Register the langchain adapter (primary interface)
_langchain_spec = AdapterSpec(
    interface="langchain",
    description="LangChain adapter for standard LLM providers",
    agent_factory=_create_langchain_agent,
    llm_factory=_create_langchain_llm,
    parser_factory=_create_langchain_parser,
    availability_checker=_check_langchain_availability,
    fallback_interface=None,
    model_string_formatter=_format_langchain_model_string,
    routes_to=None,
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_langchain_spec)


# Register openrouter adapter (routes through langchain)
_openrouter_spec = AdapterSpec(
    interface="openrouter",
    description="OpenRouter interface (routes through LangChain)",
    agent_factory=_create_langchain_agent,
    llm_factory=_create_langchain_llm,
    parser_factory=_create_langchain_parser,
    availability_checker=_check_langchain_availability,
    fallback_interface=None,
    model_string_formatter=_format_openrouter_model_string,
    routes_to="langchain",  # Indicates this routes to langchain
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_openrouter_spec)


# Register openai_endpoint adapter (routes through langchain)
_openai_endpoint_spec = AdapterSpec(
    interface="openai_endpoint",
    description="OpenAI-compatible endpoint interface (routes through LangChain)",
    agent_factory=_create_langchain_agent,
    llm_factory=_create_langchain_llm,
    parser_factory=_create_langchain_parser,
    availability_checker=_check_langchain_availability,
    fallback_interface=None,
    model_string_formatter=_format_openai_endpoint_model_string,
    routes_to="langchain",  # Indicates this routes to langchain
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_openai_endpoint_spec)


# =============================================================================
# Adapter instructions for PARSING
# =============================================================================


@dataclass
class _LangChainFormatInstruction:
    """Append format instructions for LangChain parsing.

    LangChain does not have native structured output, so both system and
    user prompts need explicit format instructions with the JSON schema.
    """

    json_schema: dict[str, Any] | None = None
    format_instructions: str = ""

    @property
    def system_addition(self) -> str:
        if not self.format_instructions:
            return ""
        return (
            "# Output Format\n\n"
            "Return only the completed JSON object - no surrounding text, no markdown fences:\n\n"
            f"<format_instructions>\n{self.format_instructions}\n</format_instructions>"
        )

    @property
    def user_addition(self) -> str:
        if self.json_schema is None:
            return ""
        schema_json = json.dumps(self.json_schema, indent=2)
        return (
            f"**JSON SCHEMA (your response MUST conform to this):**\n"
            f"```json\n{schema_json}\n```\n\n"
            f"**PARSING NOTES:**\n"
            f"- Extract values for each field based on its description in the schema\n"
            f"- If information for a field is not present, use null (if field allows null) or your best inference\n"
            f"- Return ONLY the JSON object - no surrounding text\n\n"
            f"**YOUR JSON RESPONSE:**"
        )


def _langchain_format_instruction_factory(**kwargs: object) -> _LangChainFormatInstruction:
    """Factory producing LangChain format instructions.

    Expects ``json_schema`` and optionally ``format_instructions`` in the
    instruction context dict.
    """
    return _LangChainFormatInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
        format_instructions=kwargs.get("format_instructions", "") or "",  # type: ignore[arg-type]
    )


AdapterInstructionRegistry.register("langchain", "parsing", _langchain_format_instruction_factory)
AdapterInstructionRegistry.register("openrouter", "parsing", _langchain_format_instruction_factory)
AdapterInstructionRegistry.register("openai_endpoint", "parsing", _langchain_format_instruction_factory)


# =============================================================================
# Adapter instructions for RUBRIC tasks
# =============================================================================


@dataclass
class _LangChainRubricFormatInstruction:
    """Append format instructions for LangChain rubric evaluation.

    LangChain does not have native structured output, so rubric prompts need
    explicit JSON schema and format sections appended by the adapter.
    """

    json_schema: dict[str, Any] | None = None
    example_json: str = ""
    output_format_hint: str = ""

    @property
    def system_addition(self) -> str:
        return (
            "**RESPONSE FORMAT:**\n"
            "You will receive a JSON Schema specifying the exact output structure. "
            "Your response MUST conform to this schema.\n"
            "Return ONLY a JSON object - no explanations, no markdown, no surrounding text.\n\n"
            "**WHAT NOT TO DO:**\n"
            "- Do NOT wrap JSON in markdown code blocks (no ```)\n"
            "- Do NOT add explanatory text before or after the JSON"
        )

    @property
    def user_addition(self) -> str:
        parts: list[str] = []
        if self.json_schema is not None:
            schema_json = json.dumps(self.json_schema, indent=2)
            parts.append(f"**JSON SCHEMA (your response MUST conform to this):**\n```json\n{schema_json}\n```")
        if self.example_json:
            parts.append(
                f"**REQUIRED OUTPUT FORMAT:**\n"
                f"Return a JSON object matching the schema above.\n\n"
                f"Example:\n{self.example_json}"
            )
        elif self.output_format_hint:
            parts.append(f"**REQUIRED OUTPUT FORMAT:**\n{self.output_format_hint}")
        parts.append("**YOUR JSON RESPONSE:**")
        return "\n\n".join(parts)


def _langchain_rubric_format_factory(**kwargs: object) -> _LangChainRubricFormatInstruction:
    """Factory producing LangChain rubric format instructions."""
    return _LangChainRubricFormatInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
        example_json=kwargs.get("example_json", "") or "",  # type: ignore[arg-type]
        output_format_hint=kwargs.get("output_format_hint", "") or "",  # type: ignore[arg-type]
    )


# Register rubric tasks for all LangChain-family interfaces
_RUBRIC_TASKS = [
    "rubric_llm_trait_batch",
    "rubric_llm_trait_single",
    "rubric_literal_trait_batch",
    "rubric_literal_trait_single",
    "rubric_metric_trait",
]
_LANGCHAIN_INTERFACES = ["langchain", "openrouter", "openai_endpoint"]

for _task in _RUBRIC_TASKS:
    for _iface in _LANGCHAIN_INTERFACES:
        AdapterInstructionRegistry.register(_iface, _task, _langchain_rubric_format_factory)


# =============================================================================
# Adapter instructions for DEEP JUDGMENT tasks
# =============================================================================


@dataclass
class _LangChainDJFormatInstruction:
    """Append format instructions for LangChain deep judgment tasks.

    Deep judgment excerpt extraction, hallucination assessment, and score
    extraction all produce structured JSON output that needs explicit
    format instructions for LangChain adapters.
    """

    json_schema: dict[str, Any] | None = None
    parsing_notes: str = ""

    @property
    def system_addition(self) -> str:
        return (
            "You will receive a JSON Schema specifying the exact output structure. "
            "Your response MUST conform to this schema.\n"
            "Return ONLY a JSON object - no explanations, no markdown, no surrounding text.\n\n"
            "**WHAT NOT TO DO:**\n"
            "- Do NOT wrap JSON in markdown code blocks (no ```)\n"
            "- Do NOT add explanatory text before or after the JSON"
        )

    @property
    def user_addition(self) -> str:
        parts: list[str] = []
        if self.json_schema is not None:
            schema_json = json.dumps(self.json_schema, indent=2)
            parts.append(f"**JSON SCHEMA (your response MUST conform to this):**\n```json\n{schema_json}\n```")
        if self.parsing_notes:
            parts.append(f"**PARSING NOTES:**\n{self.parsing_notes}")
        parts.append("**YOUR JSON RESPONSE:**")
        return "\n\n".join(parts)


def _langchain_dj_format_factory(**kwargs: object) -> _LangChainDJFormatInstruction:
    """Factory producing LangChain deep judgment format instructions."""
    return _LangChainDJFormatInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
        parsing_notes=kwargs.get("parsing_notes", "") or "",  # type: ignore[arg-type]
    )


# Register DJ tasks (excluding reasoning tasks which produce free-form text)
_DJ_STRUCTURED_TASKS = [
    "dj_rubric_excerpt_extraction",
    "dj_rubric_hallucination",
    "dj_rubric_score_extraction",
    "dj_template_excerpt_extraction",
    "dj_template_hallucination",
]

for _task in _DJ_STRUCTURED_TASKS:
    for _iface in _LANGCHAIN_INTERFACES:
        AdapterInstructionRegistry.register(_iface, _task, _langchain_dj_format_factory)
