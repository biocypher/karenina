"""Checkpoint schemas for JSON-LD format compatibility with frontend.

This module defines Pydantic models that exactly match the TypeScript
types in the frontend (karenina-gui), enabling seamless data exchange
between Python library and GUI.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Schema.org Person type
class SchemaOrgPerson(BaseModel):
    """Schema.org Person representation."""

    type: Literal["Person"] = Field(alias="@type", default="Person")
    name: str
    url: str | None = None
    email: str | None = None


# Schema.org CreativeWork type for sources
class SchemaOrgCreativeWork(BaseModel):
    """Schema.org CreativeWork for document sources."""

    type: Literal["CreativeWork"] = Field(alias="@type", default="CreativeWork")
    name: str
    url: str | None = None
    author: str | None = None
    datePublished: str | None = None


# Schema.org PropertyValue for additional metadata
class SchemaOrgPropertyValue(BaseModel):
    """Schema.org PropertyValue for additional properties."""

    type: Literal["PropertyValue"] = Field(alias="@type", default="PropertyValue")
    name: str
    value: Any


# Schema.org Rating for rubric traits
class SchemaOrgRating(BaseModel):
    """Schema.org Rating for rubric trait definitions."""

    type: Literal["Rating"] = Field(alias="@type", default="Rating")
    id: str | None = Field(alias="@id", default=None)
    name: str
    description: str | None = None
    bestRating: float
    worstRating: float
    ratingValue: float | None = None  # Only present in evaluation results
    ratingExplanation: str | None = None  # Only present in evaluation results
    additionalType: Literal[
        "karenina:GlobalRubricTrait",
        "karenina:QuestionSpecificRubricTrait",
        "karenina:GlobalRegexTrait",
        "karenina:QuestionSpecificRegexTrait",
        "karenina:GlobalCallableTrait",
        "karenina:QuestionSpecificCallableTrait",
        "karenina:GlobalMetricRubricTrait",
        "karenina:QuestionSpecificMetricRubricTrait",
        "karenina:GlobalLLMRubricTrait",
        "karenina:QuestionSpecificLLMRubricTrait",
    ]

    @model_validator(mode="before")
    @classmethod
    def _normalize_additional_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Add karenina: prefix if missing (for old-format checkpoints)."""
        if isinstance(data, dict):
            at = data.get("additionalType", "")
            if at and not at.startswith("karenina:"):
                data["additionalType"] = f"karenina:{at}"
        return data

    additionalProperty: list[SchemaOrgPropertyValue] | None = None  # For metric trait instructions


# Schema.org SoftwareSourceCode for Pydantic templates
class SchemaOrgSoftwareSourceCode(BaseModel):
    """Schema.org SoftwareSourceCode for answer templates."""

    type: Literal["SoftwareSourceCode"] = Field(alias="@type", default="SoftwareSourceCode")
    id: str | None = Field(alias="@id", default=None)
    name: str
    text: str  # The actual Python code as string
    programmingLanguage: Literal["Python"] = "Python"
    codeRepository: str | None = None


# Schema.org Answer
class SchemaOrgAnswer(BaseModel):
    """Schema.org Answer for expected answer content."""

    type: Literal["Answer"] = Field(alias="@type", default="Answer")
    id: str | None = Field(alias="@id", default=None)
    text: str  # The raw answer text


# Schema.org Question
class SchemaOrgQuestion(BaseModel):
    """Schema.org Question with answer and template."""

    type: Literal["Question"] = Field(alias="@type", default="Question")
    id: str | None = Field(alias="@id", default=None)
    text: str  # The question text
    acceptedAnswer: SchemaOrgAnswer
    hasPart: SchemaOrgSoftwareSourceCode  # The Pydantic template
    rating: list[SchemaOrgRating] | None = None  # Question-specific rubric traits
    additionalProperty: list[SchemaOrgPropertyValue] | None = None
    keywords: list[str] | None = None


# Schema.org DataFeedItem
class SchemaOrgDataFeedItem(BaseModel):
    """Schema.org DataFeedItem for timestamped questions."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["DataFeedItem"] = Field(alias="@type", default="DataFeedItem")
    id: str | None = Field(alias="@id", default=None)
    dateCreated: str  # ISO timestamp
    dateModified: str  # ISO timestamp
    keywords: list[str] | None = None  # Deprecated: kept for loading old checkpoints
    item: SchemaOrgQuestion


# Dataset metadata
class DatasetMetadata(BaseModel):
    """Metadata for the dataset."""

    name: str | None = None
    description: str | None = None
    version: str | None = None
    dateCreated: str | None = None
    dateModified: str | None = None
    creator: SchemaOrgPerson | None = None
    url: str | None = None
    identifier: str | None = None


# Schema.org DataFeed (root container)
class SchemaOrgDataFeed(BaseModel):
    """Schema.org DataFeed as root container."""

    type: Literal["DataFeed"] = Field(alias="@type", default="DataFeed")
    id: str | None = Field(alias="@id", default=None)
    name: str
    description: str | None = None
    version: str | None = None
    creator: str | SchemaOrgPerson | None = None  # Can be string or SchemaOrgPerson
    dateCreated: str  # ISO timestamp
    dateModified: str  # ISO timestamp
    rating: list[SchemaOrgRating] | None = None  # Global rubric traits
    dataFeedElement: list[SchemaOrgDataFeedItem]  # The questions
    additionalProperty: list[SchemaOrgPropertyValue] | None = None


# Main JSON-LD checkpoint format
class JsonLdContext(BaseModel):
    """JSON-LD context definition."""

    context: dict[str, Any] = Field(alias="@context")


class JsonLdCheckpoint(BaseModel):
    """Complete JSON-LD checkpoint format matching frontend exactly."""

    context: dict[str, Any] = Field(alias="@context")
    type: Literal["DataFeed"] = Field(alias="@type", default="DataFeed")
    id: str | None = Field(alias="@id", default=None)
    name: str
    description: str | None = None
    version: str | None = None
    creator: str | SchemaOrgPerson | None = None  # Can be string or SchemaOrgPerson
    dateCreated: str
    dateModified: str
    rating: list[SchemaOrgRating] | None = None  # Global rubric
    dataFeedElement: list[SchemaOrgDataFeedItem]  # Questions
    additionalProperty: list[SchemaOrgPropertyValue] | None = None

    model_config = ConfigDict(populate_by_name=True)


# Standard JSON-LD context for schema.org
SCHEMA_ORG_CONTEXT = {
    "@version": 1.1,
    "@vocab": "https://schema.org/",
    "karenina": "urn:karenina:vocab:",
    "dataFeedElement": {"@id": "dataFeedElement", "@container": "@set"},
    "item": {"@id": "item", "@type": "@id"},
    "acceptedAnswer": {"@id": "acceptedAnswer", "@type": "@id"},
    "rating": {"@id": "contentRating", "@container": "@set"},
    "additionalProperty": {"@id": "additionalProperty", "@container": "@set"},
    "keywords": {"@id": "keywords", "@container": "@set"},
}
