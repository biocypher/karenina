"""Configuration models for LLM and workflow settings.

This module contains models for configuring LLM behavior:
- ModelConfig: LLM provider and model selection
- FewShotConfig: Few-shot prompting configuration
"""

from .models import (
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENAI_ENDPOINT,
    INTERFACE_OPENROUTER,
    INTERFACES_NO_PROVIDER_REQUIRED,
    AgentLimitConfig,
    AgentMiddlewareConfig,
    FewShotConfig,
    ModelConfig,
    ModelRetryConfig,
    PromptCachingConfig,
    QuestionFewShotConfig,
    SummarizationConfig,
    ToolRetryConfig,
)

__all__ = [
    # Interface constants
    "INTERFACE_OPENROUTER",
    "INTERFACE_MANUAL",
    "INTERFACE_LANGCHAIN",
    "INTERFACE_OPENAI_ENDPOINT",
    "INTERFACES_NO_PROVIDER_REQUIRED",
    # Model configuration
    "ModelConfig",
    "FewShotConfig",
    "QuestionFewShotConfig",
    # Agent middleware configuration
    "AgentMiddlewareConfig",
    "AgentLimitConfig",
    "ModelRetryConfig",
    "ToolRetryConfig",
    "SummarizationConfig",
    "PromptCachingConfig",
]
