"""Data models for benchmark verification system."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..utils.async_utils import AsyncConfig

# Interface constants
INTERFACE_OPENROUTER = "openrouter"
INTERFACE_MANUAL = "manual"
INTERFACE_LANGCHAIN = "langchain"
INTERFACES_NO_PROVIDER_REQUIRED = [INTERFACE_OPENROUTER, INTERFACE_MANUAL]


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    id: str
    model_provider: str
    model_name: str
    temperature: float = 0.1
    interface: Literal["langchain", "openrouter", "manual"] = "langchain"
    system_prompt: str


class VerificationConfig(BaseModel):
    """Configuration for verification run with multiple models."""

    answering_models: list[ModelConfig]
    parsing_models: list[ModelConfig]
    replicate_count: int = 1  # Number of times to run each test combination

    # Rubric evaluation settings
    rubric_enabled: bool = False
    rubric_trait_names: list[str] | None = None  # Optional filter for specific traits

    # Legacy fields for backward compatibility (deprecated)
    answering_model_provider: str | None = None
    answering_model_name: str | None = None
    answering_temperature: float | None = None
    answering_interface: str | None = None
    answering_system_prompt: str | None = None

    parsing_model_provider: str | None = None
    parsing_model_name: str | None = None
    parsing_temperature: float | None = None
    parsing_interface: str | None = None
    parsing_system_prompt: str | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize with backward compatibility for single model configs."""
        # If legacy single model fields are provided, convert to arrays
        if "answering_models" not in data and any(k.startswith("answering_") for k in data):
            answering_model = ModelConfig(
                id="answering-legacy",
                model_provider=data.get("answering_model_provider", ""),
                model_name=data.get("answering_model_name", ""),
                temperature=data.get("answering_temperature", 0.1),
                interface=data.get("answering_interface", INTERFACE_LANGCHAIN),
                system_prompt=data.get(
                    "answering_system_prompt",
                    "You are an expert assistant. Answer the question accurately and concisely.",
                ),
            )
            data["answering_models"] = [answering_model]

        if "parsing_models" not in data and any(k.startswith("parsing_") for k in data):
            parsing_model = ModelConfig(
                id="parsing-legacy",
                model_provider=data.get("parsing_model_provider", ""),
                model_name=data.get("parsing_model_name", ""),
                temperature=data.get("parsing_temperature", 0.1),
                interface=data.get("parsing_interface", INTERFACE_LANGCHAIN),
                system_prompt=data.get(
                    "parsing_system_prompt",
                    "You are a validation assistant. Parse and validate responses against the given Pydantic template.",
                ),
            )
            data["parsing_models"] = [parsing_model]

        super().__init__(**data)

        # Validate configuration after initialization
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration, especially for rubric-enabled scenarios.

        Validates that:
        - At least one answering and parsing model is configured
        - Required fields are present for each model
        - Model provider is provided for interfaces that require it
        - Rubric-specific requirements are met when enabled

        Raises:
            ValueError: If any validation rule fails
        """
        # Check that we have at least one answering and parsing model
        if not self.answering_models:
            raise ValueError("At least one answering model must be configured")

        if not self.parsing_models:
            raise ValueError("At least one parsing model must be configured")

        # Validate model configurations
        for model in self.answering_models + self.parsing_models:
            # Model provider is optional for OpenRouter and manual interfaces
            if model.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model.model_provider:
                raise ValueError(
                    f"Model provider is required for model {model.id} "
                    f"(interface: {model.interface}). Only {INTERFACES_NO_PROVIDER_REQUIRED} "
                    f"interfaces allow empty providers."
                )
            if not model.model_name:
                raise ValueError(f"Model name is required for model {model.id}")
            if not model.system_prompt:
                raise ValueError(f"System prompt is required for model {model.id}")

        # Additional validation for rubric-enabled scenarios
        if self.rubric_enabled:
            # Ensure parsing models are configured since they're needed for rubric evaluation
            if not self.parsing_models:
                raise ValueError("Parsing models are required when rubric evaluation is enabled")

            # Check that replicate count is valid
            if self.replicate_count < 1:
                raise ValueError("Replicate count must be at least 1")


class VerificationResult(BaseModel):
    """Result of verifying a single question."""

    question_id: str
    success: bool
    error: str | None = None

    # Raw data
    question_text: str
    raw_llm_response: str
    parsed_gt_response: dict[str, Any] | None = None  # Ground truth from 'correct' field
    parsed_llm_response: dict[str, Any] | None = None  # LLM extracted fields (excluding 'id' and 'correct')

    # Verification outcomes
    verify_result: Any | None = None
    verify_granular_result: Any | None = None
    verify_rubric: dict[str, int | bool] | None = None  # Rubric trait scores
    evaluation_rubric: dict[str, Any] | None = None  # The merged rubric used for evaluation

    # Question metadata
    keywords: list[str] | None = None  # Keywords associated with the question

    # Metadata
    answering_model: str
    parsing_model: str
    execution_time: float
    timestamp: str
    answering_system_prompt: str | None = None
    parsing_system_prompt: str | None = None

    # Run identification
    run_name: str | None = None
    job_id: str | None = None

    # Replicate tracking
    answering_replicate: int | None = None  # Replicate number for answering model (1, 2, 3, ...)
    parsing_replicate: int | None = None  # Replicate number for parsing model (1, 2, 3, ...)


class VerificationJob(BaseModel):
    """Represents a verification job."""

    job_id: str
    run_name: str  # User-defined or auto-generated run name
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    config: VerificationConfig
    async_config: AsyncConfig | None = None

    # Progress tracking
    total_questions: int
    processed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    percentage: float = 0.0
    current_question: str = ""
    estimated_time_remaining: float | None = None

    # Timing
    start_time: float | None = None
    end_time: float | None = None

    # Results
    results: dict[str, VerificationResult] = Field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "run_name": self.run_name,
            "status": self.status,
            "total_questions": self.total_questions,
            "processed_count": self.processed_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "percentage": self.percentage,
            "current_question": self.current_question,
            "estimated_time_remaining": self.estimated_time_remaining,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class FinishedTemplate(BaseModel):
    """Metadata for a finished answer template."""

    question_id: str
    question_text: str
    question_preview: str  # Truncated version for UI
    template_code: str
    last_modified: str
    finished: bool = True
    question_rubric: dict[str, Any] | None = None  # Question-specific rubric as dict
    keywords: list[str] | None = None  # Keywords associated with the question


class VerificationRequest(BaseModel):
    """Request to start verification."""

    config: VerificationConfig
    question_ids: list[str] | None = None  # If None, verify all finished templates
    run_name: str | None = None  # Optional user-defined run name


class VerificationStatusResponse(BaseModel):
    """Response for verification status."""

    job_id: str
    run_name: str
    status: str
    percentage: float
    current_question: str
    processed_count: int
    total_count: int
    successful_count: int
    failed_count: int
    estimated_time_remaining: float | None = None
    error: str | None = None
    results: dict[str, VerificationResult] | None = None


class VerificationStartResponse(BaseModel):
    """Response when starting verification."""

    job_id: str
    run_name: str
    status: str
    message: str
