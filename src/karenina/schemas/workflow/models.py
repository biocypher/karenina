"""Model configuration and few-shot configuration models.

DEPRECATED: Import from `karenina.schemas.config` instead.
"""

# Re-export from new location for backward compatibility
from ..config.models import (
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
