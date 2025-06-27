"""Data models for benchmark verification system."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ModelConfiguration(BaseModel):
    """Configuration for a single model."""

    id: str
    model_provider: str
    model_name: str
    temperature: float = 0.1
    interface: Literal["langchain", "openrouter"] = "langchain"
    system_prompt: str


class VerificationConfig(BaseModel):
    """Configuration for verification run with multiple models."""

    answering_models: List[ModelConfiguration]
    parsing_models: List[ModelConfiguration]
    replicate_count: int = 1  # Number of times to run each test combination

    # Legacy fields for backward compatibility (deprecated)
    answering_model_provider: Optional[str] = None
    answering_model_name: Optional[str] = None
    answering_temperature: Optional[float] = None
    answering_interface: Optional[str] = None
    answering_system_prompt: Optional[str] = None

    parsing_model_provider: Optional[str] = None
    parsing_model_name: Optional[str] = None
    parsing_temperature: Optional[float] = None
    parsing_interface: Optional[str] = None
    parsing_system_prompt: Optional[str] = None

    def __init__(self, **data):
        """Initialize with backward compatibility for single model configs."""
        # If legacy single model fields are provided, convert to arrays
        if "answering_models" not in data and any(k.startswith("answering_") for k in data):
            answering_model = ModelConfiguration(
                id="answering-legacy",
                model_provider=data.get("answering_model_provider", ""),
                model_name=data.get("answering_model_name", ""),
                temperature=data.get("answering_temperature", 0.1),
                interface=data.get("answering_interface", "langchain"),
                system_prompt=data.get(
                    "answering_system_prompt",
                    "You are an expert assistant. Answer the question accurately and concisely.",
                ),
            )
            data["answering_models"] = [answering_model]

        if "parsing_models" not in data and any(k.startswith("parsing_") for k in data):
            parsing_model = ModelConfiguration(
                id="parsing-legacy",
                model_provider=data.get("parsing_model_provider", ""),
                model_name=data.get("parsing_model_name", ""),
                temperature=data.get("parsing_temperature", 0.1),
                interface=data.get("parsing_interface", "langchain"),
                system_prompt=data.get(
                    "parsing_system_prompt",
                    "You are a validation assistant. Parse and validate responses against the given Pydantic template.",
                ),
            )
            data["parsing_models"] = [parsing_model]

        super().__init__(**data)


class VerificationResult(BaseModel):
    """Result of verifying a single question."""

    question_id: str
    success: bool
    error: Optional[str] = None

    # Raw data
    question_text: str
    raw_llm_response: str
    parsed_response: Optional[Dict[str, Any]] = None

    # Verification outcomes
    verify_result: Optional[Any] = None
    verify_granular_result: Optional[Any] = None

    # Metadata
    answering_model: str
    parsing_model: str
    execution_time: float
    timestamp: str
    answering_system_prompt: Optional[str] = None
    parsing_system_prompt: Optional[str] = None

    # Run identification
    run_name: Optional[str] = None
    job_id: Optional[str] = None

    # Replicate tracking
    answering_replicate: Optional[int] = None  # Replicate number for answering model (1, 2, 3, ...)
    parsing_replicate: Optional[int] = None  # Replicate number for parsing model (1, 2, 3, ...)


class VerificationJob(BaseModel):
    """Represents a verification job."""

    job_id: str
    run_name: str  # User-defined or auto-generated run name
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    config: VerificationConfig

    # Progress tracking
    total_questions: int
    processed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    percentage: float = 0.0
    current_question: str = ""
    estimated_time_remaining: Optional[float] = None

    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Results
    results: Dict[str, VerificationResult] = Field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self):
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


class VerificationRequest(BaseModel):
    """Request to start verification."""

    config: VerificationConfig
    question_ids: Optional[List[str]] = None  # If None, verify all finished templates
    run_name: Optional[str] = None  # Optional user-defined run name


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
    estimated_time_remaining: Optional[float] = None
    error: Optional[str] = None
    results: Optional[Dict[str, VerificationResult]] = None


class VerificationStartResponse(BaseModel):
    """Response when starting verification."""

    job_id: str
    run_name: str
    status: str
    message: str
