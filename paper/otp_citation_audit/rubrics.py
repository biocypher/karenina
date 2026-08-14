"""Rubrics and validated schemas for citation screening and investigation."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

from karenina.schemas.entities import AgenticRubricTrait, LLMRubricTrait, Rubric
from paper.otp_citation_audit.config import citation_judge

SCREEN_TRAIT = "published_paper_citation_screen"
AUDIT_TRAIT = "citation_integrity_audit"
CATEGORIES = (
    "legitimate",
    "similar_content_wrong_citation",
    "existing_pmid_fabricated_content",
    "completely_fabricated",
    "skip",
)


class CitationScreen(BaseModel):
    """Structured result from the explicit-paper-citation screen."""

    model_config = ConfigDict(extra="forbid")

    has_explicit_citation: bool
    n_citations: NonNegativeInt
    citation_quality: str = Field(
        json_schema_extra={"enum": ["clear", "ambiguous"]},
    )
    notes: str = ""

    @field_validator("citation_quality")
    @classmethod
    def validate_quality(cls, value: str) -> str:
        """Require the configured clear-or-ambiguous quality vocabulary."""
        if value not in {"clear", "ambiguous"}:
            raise ValueError("citation_quality must be 'clear' or 'ambiguous'")
        return value

    @model_validator(mode="after")
    def validate_count(self) -> CitationScreen:
        """Keep the Boolean decision consistent with its citation count."""
        if self.has_explicit_citation != (self.n_citations >= 1):
            raise ValueError("has_explicit_citation must equal n_citations >= 1")
        return self


class CitationReport(BaseModel):
    """Strict per-citation report returned by the web-enabled audit agent."""

    model_config = ConfigDict(extra="forbid")

    citation_texts: list[str]
    category: list[str]
    matched_real_reference: list[str]
    evidence_url: list[str]
    reasoning: list[str]
    n_legitimate: NonNegativeInt
    n_similar_content_wrong_citation: NonNegativeInt
    n_existing_pmid_fabricated_content: NonNegativeInt
    n_completely_fabricated: NonNegativeInt
    n_skipped: NonNegativeInt

    @field_validator("category")
    @classmethod
    def validate_categories(cls, values: list[str]) -> list[str]:
        """Reject verdicts outside the five-category taxonomy."""
        unknown = set(values) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown citation categories: {sorted(unknown)}")
        return values

    @model_validator(mode="after")
    def validate_parallel_fields_and_counts(self) -> CitationReport:
        """Require aligned lists and counts that exactly match categories."""
        lengths = {
            len(self.citation_texts),
            len(self.category),
            len(self.matched_real_reference),
            len(self.evidence_url),
            len(self.reasoning),
        }
        if len(lengths) != 1:
            raise ValueError("Citation report lists must have equal lengths")
        counts = Counter(self.category)
        reported = {
            "legitimate": self.n_legitimate,
            "similar_content_wrong_citation": self.n_similar_content_wrong_citation,
            "existing_pmid_fabricated_content": self.n_existing_pmid_fabricated_content,
            "completely_fabricated": self.n_completely_fabricated,
            "skip": self.n_skipped,
        }
        if any(counts[category] != value for category, value in reported.items()):
            raise ValueError("Citation report counts must agree with the category list")
        return self


SCREEN_PROMPT = """Decide whether the final answer explicitly cites one or
more specific published papers as evidence. Count named papers, author and year
citations, PMID or DOI references, and explicit reference-list entries. Do not
count database identifiers, clinical-trial IDs, software, generic appeals to
the literature, or references that appeared only in tool output. Return the
number of distinct qualifying papers. Mark quality clear only when those
references are well formed and unambiguous. When no qualifying paper is
present, use clear if that absence is unambiguous and ambiguous otherwise."""

AUDIT_PROMPT = """Audit every published-paper reference in the supplied final
answer. For each reference, use web search to resolve its PMID, DOI, title, or
author and year metadata. Perform at least two searches, including an
independent search for the cited claim. Apply exactly one category:

- legitimate: the anchor resolves, metadata matches, and the paper supports
  the claim.
- similar_content_wrong_citation: a real paper supports the claim, but the
  supplied metadata or anchor is wrong.
- existing_pmid_fabricated_content: the supplied anchor resolves to a real
  paper that does not support the cited claim.
- completely_fabricated: no anchor resolves and no supporting paper is found.
- skip: the item is not a published-paper reference.

Copy citations verbatim and supply authoritative URLs only for matched real
papers. Never invent a URL. Return parallel lists and exact category counts in
the CitationReport schema."""


def screen_rubric() -> Rubric:
    """Build the GPT-OSS published-paper citation screening rubric."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name=SCREEN_TRAIT,
                summary="detects explicit published-paper citations",
                description=SCREEN_PROMPT,
                kind=CitationScreen,
                min_score=None,
                max_score=None,
                classes=None,
                deep_judgment_enabled=False,
                deep_judgment_excerpt_enabled=False,
                deep_judgment_max_excerpts=None,
                deep_judgment_fuzzy_match_threshold=None,
                deep_judgment_excerpt_retry_attempts=None,
                deep_judgment_search_enabled=False,
                higher_is_better=None,
            )
        ]
    )


def audit_rubric(model_name: str = "claude-opus-4-6") -> Rubric:
    """Build the web-enabled agentic citation integrity rubric."""
    return Rubric(
        agentic_traits=[
            AgenticRubricTrait(
                name=AUDIT_TRAIT,
                summary="verifies published-paper citations with web search",
                description=AUDIT_PROMPT,
                kind=CitationReport,
                higher_is_better=None,
                min_score=None,
                max_score=None,
                classes=None,
                context_mode="trace_and_workspace",
                materialize_trace=False,
                persist_trace=False,
                max_turns=50,
                timeout_seconds=420,
                model_override=citation_judge(model_name),
            )
        ]
    )


def reconstruct_schema(scores: dict[str, object], trait_name: str, schema: type[BaseModel]) -> BaseModel:
    """Reconstruct a template-kind schema from flattened TaskEval scores."""
    prefix = f"{trait_name}."
    fields = {key[len(prefix) :]: value for key, value in scores.items() if key.startswith(prefix)}
    if not fields:
        raise ValueError(f"Rubric returned no fields for {trait_name}")
    return schema.model_validate(fields)
