"""Self-registration for the google_genai adapter.

Registers the ``google_genai`` interface (Gemini through the google-genai
SDK, on Vertex AI or the Gemini API) with the AdapterRegistry. The adapter
has no fallback: a silent fallback to LangChain would drop the grounding
audit trail and the search budget (design D11).

No adapter-specific prompt instructions are registered: the default
parsing, rubric, abstention and sufficiency instructions apply.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)


def _check_availability() -> AdapterAvailability:
    """Check that the google-genai SDK is installed.

    Returns:
        AdapterAvailability for the google_genai interface.
    """
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason="google-genai not installed. Install with: pip install 'karenina[google-genai]'",
            fallback_interface=None,
        )
    return AdapterAvailability(available=True, reason="google-genai is installed")


def _create_llm(config: ModelConfig) -> LLMPort:
    """Create a GoogleGenAILLMAdapter for the given config."""
    from karenina.adapters.google_genai.llm import GoogleGenAILLMAdapter

    return GoogleGenAILLMAdapter(config)


def _create_agent(config: ModelConfig) -> AgentPort:
    """Create a GoogleGenAIAgentAdapter for the given config."""
    from karenina.adapters.google_genai.agent import GoogleGenAIAgentAdapter

    return GoogleGenAIAgentAdapter(config)


def _create_parser(config: ModelConfig) -> ParserPort:
    """Create a GoogleGenAIParserAdapter for the given config."""
    from karenina.adapters.google_genai.parser import GoogleGenAIParserAdapter

    return GoogleGenAIParserAdapter(config)


_google_genai_spec = AdapterSpec(
    interface="google_genai",
    description="Google GenAI SDK adapter (Gemini on Vertex AI or the Gemini API) with a karenina-owned tool loop",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface=None,
    supports_mcp=True,
    supports_tools=True,
    requires_provider=False,
    agent_tier="tool_loop",
)

AdapterRegistry.register(_google_genai_spec)

logger.debug("Registered google_genai adapter with AdapterRegistry")
