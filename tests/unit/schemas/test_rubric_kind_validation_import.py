"""Behavior tests for ``_validate_template_fields`` wired into rubric trait kinds.

The validator lives in ``karenina.schemas.entities._rubric_kind_validation`` and
is invoked from ``LLMRubricTrait.validate_kind`` / ``AgenticRubricTrait.validate_kind``
whenever a trait is constructed with ``kind=<SomeBaseModel>``. The realistic
regression signal is *not* "is the function importable" but "does a malformed
template kind surface a clear error at trait construction, and does a clean
template round-trip through serialization". These tests pin that contract so a
future refactor cannot silently disconnect the validator from the trait.
"""

import pytest
from pydantic import BaseModel, Field, ValidationError

from karenina.schemas.entities._rubric_kind_validation import _validate_template_fields
from karenina.schemas.entities.rubric import AgenticRubricTrait, LLMRubricTrait


class _CleanTemplate(BaseModel):
    """A template kind that satisfies the primitive-only constraint."""

    count: int = Field(description="A count")
    items: list[str] = Field(description="Items")
    optional_score: float | None = None


# -----------------------------------------------------------------------------
# Direct validator: realistic shape of inputs that the trait path can produce
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateTemplateFieldsShapes:
    """The validator is the gatekeeper for what may be used as a template kind."""

    def test_clean_template_passes(self) -> None:
        _validate_template_fields(_CleanTemplate)  # no exception

    def test_rejects_template_with_nested_basemodel_field(self) -> None:
        class Inner(BaseModel):
            x: int

        class Bad(BaseModel):
            nested: Inner

        with pytest.raises(ValueError, match="not allowed"):
            _validate_template_fields(Bad)

    def test_rejects_template_with_dict_field(self) -> None:
        class Bad(BaseModel):
            payload: dict[str, int]

        with pytest.raises(ValueError, match="not allowed"):
            _validate_template_fields(Bad)

    def test_rejects_list_of_basemodel(self) -> None:
        class Inner(BaseModel):
            x: int

        class Bad(BaseModel):
            items: list[Inner]

        with pytest.raises(ValueError, match="list must contain primitive"):
            _validate_template_fields(Bad)

    def test_rejects_optional_without_default(self) -> None:
        # Optional-without-default breaks the JSON-Schema round-trip the trait
        # serializer relies on, so this must be rejected at validation time.
        class Bad(BaseModel):
            score: float | None  # no default

        with pytest.raises(ValueError, match="must have an explicit default"):
            _validate_template_fields(Bad)


# -----------------------------------------------------------------------------
# Wiring: the validator actually runs from trait construction paths
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatorWiredIntoTraitKinds:
    """Constructing a trait with a bad template kind must raise during validation."""

    def test_llm_trait_clean_template_accepted_and_round_trips(self) -> None:
        trait = LLMRubricTrait(
            name="structure",
            description="Inspect structured findings.",
            kind=_CleanTemplate,
            higher_is_better=None,
        )
        assert trait.is_template_kind is True
        assert trait.kind is _CleanTemplate

        rebuilt = LLMRubricTrait.model_validate(trait.model_dump())
        assert rebuilt.is_template_kind is True
        # The reconstructed kind must carry the same primitive-only fields
        assert set(rebuilt.kind.model_fields) == {"count", "items", "optional_score"}

    def test_llm_trait_rejects_nested_basemodel_kind(self) -> None:
        class Inner(BaseModel):
            x: int

        class BadTemplate(BaseModel):
            nested: Inner

        with pytest.raises(ValidationError) as exc:
            LLMRubricTrait(
                name="bad",
                description="Should not validate.",
                kind=BadTemplate,
                higher_is_better=None,
            )
        assert "not allowed" in str(exc.value)

    def test_agentic_trait_rejects_dict_field_kind(self) -> None:
        class BadTemplate(BaseModel):
            payload: dict[str, int]

        with pytest.raises(ValidationError) as exc:
            AgenticRubricTrait(
                name="bad",
                description="Should not validate.",
                kind=BadTemplate,
                higher_is_better=None,
            )
        assert "not allowed" in str(exc.value)
