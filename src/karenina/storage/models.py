"""SQLAlchemy ORM models for Karenina database storage.

This module defines the database schema including tables for benchmarks,
questions, verification runs, and results.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class BenchmarkModel(Base):
    """Database model for benchmarks.

    Stores benchmark metadata and tracks the checkpoint file that serves
    as the source of truth.
    """

    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    creator: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False
    )

    # Relationships
    benchmark_questions: Mapped[list["BenchmarkQuestionModel"]] = relationship(
        "BenchmarkQuestionModel", back_populates="benchmark", cascade="all, delete-orphan"
    )
    verification_runs: Mapped[list["VerificationRunModel"]] = relationship(
        "VerificationRunModel", back_populates="benchmark", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<Benchmark(name='{self.name}', version='{self.version}')>"


class QuestionModel(Base):
    """Database model for questions.

    Questions are shared across benchmarks (normalized). Each unique question
    appears only once in this table.
    """

    __tablename__ = "questions"

    # Use MD5 hash of question text as primary key
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    raw_answer: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    few_shot_examples: Mapped[list[dict[str, str]] | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False
    )

    # Relationships
    benchmark_questions: Mapped[list["BenchmarkQuestionModel"]] = relationship(
        "BenchmarkQuestionModel", back_populates="question"
    )
    verification_results: Mapped[list["VerificationResultModel"]] = relationship(
        "VerificationResultModel", back_populates="question"
    )

    # Index for full-text search on question_text
    __table_args__ = (Index("idx_question_text", "question_text"),)

    def __repr__(self) -> str:
        """String representation."""
        preview = self.question_text[:50] + "..." if len(self.question_text) > 50 else self.question_text
        return f"<Question(id='{self.id}', text='{preview}')>"


class BenchmarkQuestionModel(Base):
    """Database model for benchmark-question associations.

    Junction table linking benchmarks to questions with benchmark-specific metadata
    like answer templates, completion status, and question-specific rubrics.
    """

    __tablename__ = "benchmark_questions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)

    # Benchmark-specific question data
    answer_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_answer_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    finished: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    keywords: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    question_rubric: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False
    )

    # Relationships
    benchmark: Mapped["BenchmarkModel"] = relationship("BenchmarkModel", back_populates="benchmark_questions")
    question: Mapped["QuestionModel"] = relationship("QuestionModel", back_populates="benchmark_questions")

    # Constraints
    __table_args__ = (
        UniqueConstraint("benchmark_id", "question_id", name="uq_benchmark_question"),
        Index("idx_benchmark_finished", "benchmark_id", "finished"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<BenchmarkQuestion(benchmark_id={self.benchmark_id}, question_id='{self.question_id}', finished={self.finished})>"


class VerificationRunModel(Base):
    """Database model for verification runs.

    Tracks verification job metadata including configuration, status, and timing.
    """

    __tablename__ = "verification_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # pending, running, completed, failed, cancelled
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Progress and timing
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    successful_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    start_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Summary statistics (JSON for flexibility)
    summary_stats: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False
    )

    # Relationships
    benchmark: Mapped["BenchmarkModel"] = relationship("BenchmarkModel", back_populates="verification_runs")
    results: Mapped[list["VerificationResultModel"]] = relationship(
        "VerificationResultModel", back_populates="run", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_run_benchmark_status", "benchmark_id", "status"),
        Index("idx_run_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<VerificationRun(id='{self.id}', run_name='{self.run_name}', status='{self.status}')>"


class VerificationResultModel(Base):
    """Database model for verification results.

    Stores comprehensive verification result data for each question-model combination.
    """

    __tablename__ = "verification_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.id", ondelete="CASCADE"), nullable=False)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)

    # Basic result information
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Raw data
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_llm_response: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_gt_response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    parsed_llm_response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Verification outcomes
    verify_result: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    verify_granular_result: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    verify_rubric: Mapped[dict[str, int | bool] | None] = mapped_column(JSON, nullable=True)
    evaluation_rubric: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Question metadata
    keywords: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Model information
    answering_model: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    parsing_model: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    answering_system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsing_system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Execution metadata
    execution_time: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[str] = mapped_column(String(50), nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Replicate tracking
    answering_replicate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parsing_replicate: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Embedding check metadata
    embedding_check_performed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    embedding_similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_override_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    embedding_model_used: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Regex validation metadata
    regex_validations_performed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    regex_validation_results: Mapped[dict[str, bool] | None] = mapped_column(JSON, nullable=True)
    regex_validation_details: Mapped[dict[str, dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    regex_overall_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    regex_extraction_results: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Recursion limit metadata
    recursion_limit_reached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # MCP server metadata
    answering_mcp_servers: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    parsing_mcp_servers: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Relationships
    run: Mapped["VerificationRunModel"] = relationship("VerificationRunModel", back_populates="results")
    question: Mapped["QuestionModel"] = relationship("QuestionModel", back_populates="verification_results")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_result_run_question", "run_id", "question_id"),
        Index("idx_result_success", "success"),
        Index("idx_result_models", "answering_model", "parsing_model"),
        Index("idx_result_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<VerificationResult(id={self.id}, run_id='{self.run_id}', question_id='{self.question_id}', success={self.success})>"
