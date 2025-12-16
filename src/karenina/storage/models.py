"""SQLAlchemy ORM models for Karenina database storage.

This module defines the database schema including tables for benchmarks,
questions, verification runs, and results.

Note: VerificationResultModel is auto-generated from Pydantic schemas
and is defined in generated_models.py to keep it in sync with domain models.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    # VerificationResultModel is dynamically generated, so we import it for type checking
    from .generated_models import VerificationResultModel  # type: ignore[misc]


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
    question_text: Mapped[str] = mapped_column(Text, nullable=False)  # Removed unique=True for composite key
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
    template_id: Mapped[str] = mapped_column(String(32), nullable=False)  # MD5 of template or "no_template"

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
        UniqueConstraint("benchmark_id", "question_id", "template_id", name="uq_benchmark_question_template"),
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


class ImportMetadataModel(Base):
    """Database model for tracking verification result imports.

    Records audit information about imports for traceability.
    """

    __tablename__ = "import_metadata"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.id", ondelete="CASCADE"), nullable=False)
    import_source: Mapped[str] = mapped_column(String(50), nullable=False)  # 'json_file', 'api', etc.
    source_format_version: Mapped[str] = mapped_column(String(20), nullable=False)  # '2.0', '1.0', 'legacy'
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_job_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_export_timestamp: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_karenina_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    results_count: Mapped[int] = mapped_column(Integer, nullable=False)
    shared_rubric_definition: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Relationships
    run: Mapped["VerificationRunModel"] = relationship("VerificationRunModel")

    def __repr__(self) -> str:
        """String representation."""
        return f"<ImportMetadata(id={self.id}, run_id='{self.run_id}', source='{self.import_source}')>"


# Note: VerificationResultModel is auto-generated from Pydantic schemas
# Import it from generated_models.py when needed
