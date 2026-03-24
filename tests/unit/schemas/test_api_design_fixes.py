"""Tests for type improvements: repr, config, group_by_model, group_by_replicate."""

import pytest
from pydantic import ValidationError

from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.verification.config import (
    DeepJudgmentRubricCustomConfig,
    DeepJudgmentTraitConfig,
    VerificationConfig,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def _make_model_identity(model_name: str = "gpt-4") -> ModelIdentity:
    """Create a ModelIdentity for testing."""
    return ModelIdentity(interface="langchain", model_name=model_name)


def _make_result(
    question_id="q1",
    answering_model="gpt-4",
    parsing_model="gpt-4",
    replicate=1,
    mcp_servers=None,
):
    """Create a real VerificationResult for testing."""
    metadata = VerificationResultMetadata(
        question_id=question_id,
        template_id="tmpl_test",
        completed_without_errors=True,
        question_text="Test question?",
        answering=_make_model_identity(model_name=answering_model),
        parsing=_make_model_identity(model_name=parsing_model),
        execution_time=1.0,
        timestamp="2026-01-01T00:00:00",
        result_id="test_result_id",
        replicate=replicate,
    )
    template = VerificationResultTemplate(
        raw_llm_response="test response",
        template_verification_performed=True,
        verify_result=True,
        answering_mcp_servers=mcp_servers,
    )
    return VerificationResult(metadata=metadata, template=template)


def _make_minimal_config(**overrides):
    """Create a minimal VerificationConfig with required fields."""
    from karenina.schemas.config import ModelConfig

    defaults = {
        "answering_models": [],
        "parsing_models": [
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        "parsing_only": True,
    }
    defaults.update(overrides)
    return VerificationConfig(**defaults)


# =============================================================================
# Sub-task A: VerificationResultSet.__repr__
# =============================================================================


@pytest.mark.unit
class TestVerificationResultSetRepr:
    """Tests for the simplified __repr__ method."""

    def test_empty_repr(self):
        """Empty result set should show 'empty'."""
        rs = VerificationResultSet(results=[])
        assert repr(rs) == "VerificationResultSet(empty)"

    def test_single_result_repr(self):
        """Single result should use singular 'result'."""
        rs = VerificationResultSet(results=[_make_result()])
        result = repr(rs)
        assert result == "VerificationResultSet(1 result)"

    def test_multiple_results_repr(self):
        """Multiple results should use plural 'results'."""
        rs = VerificationResultSet(results=[_make_result(), _make_result()])
        result = repr(rs)
        assert result == "VerificationResultSet(2 results)"

    def test_repr_is_concise(self):
        """Repr should not contain expensive summary data."""
        rs = VerificationResultSet(results=[_make_result() for _ in range(10)])
        result = repr(rs)
        # Should be a short string, not a multi-line summary
        assert "\n" not in result
        assert "10 results" in result


# =============================================================================
# Sub-task B: group_by_replicate preserves None
# =============================================================================


@pytest.mark.unit
class TestGroupByReplicate:
    """Tests for group_by_replicate preserving None keys."""

    def test_none_replicate_preserved_as_key(self):
        """Results with None replicate should use None as key, not 0."""
        r1 = _make_result(replicate=None)
        r2 = _make_result(replicate=1)
        rs = VerificationResultSet(results=[r1, r2])

        grouped = rs.group_by_replicate()
        assert None in grouped
        assert 1 in grouped
        assert 0 not in grouped

    def test_all_none_replicates(self):
        """All None replicates should group under single None key."""
        results = [_make_result(replicate=None) for _ in range(3)]
        rs = VerificationResultSet(results=results)

        grouped = rs.group_by_replicate()
        assert list(grouped.keys()) == [None]
        assert len(grouped[None]) == 3

    def test_integer_replicates_unchanged(self):
        """Integer replicates should still work as before."""
        r1 = _make_result(replicate=1)
        r2 = _make_result(replicate=2)
        r3 = _make_result(replicate=1)
        rs = VerificationResultSet(results=[r1, r2, r3])

        grouped = rs.group_by_replicate()
        assert set(grouped.keys()) == {1, 2}
        assert len(grouped[1]) == 2
        assert len(grouped[2]) == 1


# =============================================================================
# Sub-task C: View-level group_by_model with 'by' parameter
# =============================================================================


@pytest.mark.unit
class TestViewGroupByModel:
    """Tests for the 'by' parameter on view-level group_by_model methods."""

    def test_template_results_group_by_answering(self):
        """TemplateResults.group_by_model(by='answering') groups by answering model."""
        from karenina.schemas.results.template import TemplateResults

        r1 = _make_result(answering_model="gpt-4", parsing_model="gpt-3.5")
        r2 = _make_result(answering_model="claude-3", parsing_model="gpt-3.5")
        tr = TemplateResults(results=[r1, r2])

        grouped = tr.group_by_model(by="answering")
        assert "langchain:gpt-4" in grouped
        assert "langchain:claude-3" in grouped

    def test_template_results_group_by_parsing(self):
        """TemplateResults.group_by_model(by='parsing') groups by parsing model."""
        from karenina.schemas.results.template import TemplateResults

        r1 = _make_result(answering_model="gpt-4", parsing_model="gpt-3.5")
        r2 = _make_result(answering_model="gpt-4", parsing_model="claude-3")
        tr = TemplateResults(results=[r1, r2])

        grouped = tr.group_by_model(by="parsing")
        assert "langchain:gpt-3.5" in grouped
        assert "langchain:claude-3" in grouped

    def test_template_results_group_by_both(self):
        """TemplateResults.group_by_model(by='both') groups by answering / parsing."""
        from karenina.schemas.results.template import TemplateResults

        r1 = _make_result(answering_model="gpt-4", parsing_model="gpt-3.5")
        r2 = _make_result(answering_model="gpt-4", parsing_model="claude-3")
        tr = TemplateResults(results=[r1, r2])

        grouped = tr.group_by_model(by="both")
        assert "langchain:gpt-4 / langchain:gpt-3.5" in grouped
        assert "langchain:gpt-4 / langchain:claude-3" in grouped

    def test_template_results_group_by_answering_with_mcp(self):
        """MCP servers should be appended to the answering model key."""
        from karenina.schemas.results.template import TemplateResults

        r1 = _make_result(answering_model="gpt-4", mcp_servers=["server1", "server2"])
        r2 = _make_result(answering_model="gpt-4", mcp_servers=None)
        tr = TemplateResults(results=[r1, r2])

        grouped = tr.group_by_model(by="answering")
        assert "langchain:gpt-4 + MCP[server1,server2]" in grouped
        assert "langchain:gpt-4" in grouped

    def test_rubric_results_group_by_parsing(self):
        """RubricResults.group_by_model supports the 'by' parameter."""
        from karenina.schemas.results.rubric import RubricResults

        r1 = _make_result(answering_model="gpt-4", parsing_model="gpt-3.5")
        r2 = _make_result(answering_model="gpt-4", parsing_model="claude-3")
        rr = RubricResults(results=[r1, r2])

        grouped = rr.group_by_model(by="parsing")
        assert "langchain:gpt-3.5" in grouped
        assert "langchain:claude-3" in grouped

    def test_judgment_results_group_by_both(self):
        """JudgmentResults.group_by_model supports the 'by' parameter."""
        from karenina.schemas.results.judgment import JudgmentResults

        r1 = _make_result(answering_model="gpt-4", parsing_model="gpt-3.5")
        r2 = _make_result(answering_model="claude-3", parsing_model="gpt-3.5")
        jr = JudgmentResults(results=[r1, r2])

        grouped = jr.group_by_model(by="both")
        assert "langchain:gpt-4 / langchain:gpt-3.5" in grouped
        assert "langchain:claude-3 / langchain:gpt-3.5" in grouped

    def test_default_is_answering(self):
        """Default 'by' parameter should be 'answering'."""
        from karenina.schemas.results.template import TemplateResults

        r1 = _make_result(answering_model="gpt-4", parsing_model="gpt-3.5")
        tr = TemplateResults(results=[r1])

        # Call without explicit 'by' argument
        grouped = tr.group_by_model()
        assert "langchain:gpt-4" in grouped

    def test_invalid_by_raises(self):
        """Invalid 'by' value should raise ValueError."""
        from karenina.schemas.results.template import TemplateResults

        r1 = _make_result()
        tr = TemplateResults(results=[r1])

        with pytest.raises(ValueError, match="Invalid grouping mode"):
            tr.group_by_model(by="invalid")


# =============================================================================
# Sub-task D: DeepJudgmentRubricCustomConfig
# =============================================================================


@pytest.mark.unit
class TestDeepJudgmentRubricCustomConfig:
    """Tests for the new typed config model."""

    def test_valid_config_creation(self):
        """Can create config with global traits and question-specific traits."""
        config = DeepJudgmentRubricCustomConfig(
            **{
                "global": {
                    "Accuracy": DeepJudgmentTraitConfig(enabled=True, excerpt_enabled=True),
                },
                "question_specific": {
                    "q-123": {
                        "Completeness": DeepJudgmentTraitConfig(enabled=False),
                    }
                },
            }
        )
        assert "Accuracy" in config.global_traits
        assert config.global_traits["Accuracy"].enabled is True
        assert "q-123" in config.question_specific
        assert config.question_specific["q-123"]["Completeness"].enabled is False

    def test_empty_config_defaults(self):
        """Empty config should have empty dicts."""
        config = DeepJudgmentRubricCustomConfig()
        assert config.global_traits == {}
        assert config.question_specific == {}

    def test_extra_fields_forbidden(self):
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError):
            DeepJudgmentRubricCustomConfig(unknown_field="value")

    def test_custom_mode_requires_config(self):
        """Custom mode without config should raise validation error."""
        with pytest.raises(ValidationError, match="deep_judgment_rubric_config is required"):
            _make_minimal_config(
                deep_judgment_rubric_mode="custom",
                deep_judgment_rubric_config=None,
            )

    def test_custom_mode_with_config_succeeds(self):
        """Custom mode with config should succeed."""
        config = _make_minimal_config(
            deep_judgment_rubric_mode="custom",
            deep_judgment_rubric_config=DeepJudgmentRubricCustomConfig(
                **{
                    "global": {
                        "Accuracy": DeepJudgmentTraitConfig(enabled=True),
                    }
                }
            ),
        )
        assert config.deep_judgment_rubric_mode == "custom"
        assert config.deep_judgment_rubric_config is not None

    def test_non_custom_mode_without_config_ok(self):
        """Non-custom modes should not require config."""
        config = _make_minimal_config(
            deep_judgment_rubric_mode="enable_all",
            deep_judgment_rubric_config=None,
        )
        assert config.deep_judgment_rubric_mode == "enable_all"
        assert config.deep_judgment_rubric_config is None

    def test_dict_accepted_and_coerced(self):
        """Raw dicts should be coerced to the typed model by Pydantic."""
        config = _make_minimal_config(
            deep_judgment_rubric_mode="custom",
            deep_judgment_rubric_config={
                "global": {
                    "Accuracy": {"enabled": True, "excerpt_enabled": True},
                },
                "question_specific": {},
            },
        )
        assert isinstance(config.deep_judgment_rubric_config, DeepJudgmentRubricCustomConfig)
        assert config.deep_judgment_rubric_config.global_traits["Accuracy"].enabled is True

    def test_from_overrides_accepts_dict(self):
        """from_overrides should accept a raw dict for deep_judgment_rubric_config."""
        raw_dict = {
            "global": {
                "Accuracy": {"enabled": True},
            },
            "question_specific": {},
        }
        config = VerificationConfig.from_overrides(
            answering_model="gpt-4",
            answering_provider="openai",
            answering_id="answering",
            answering_interface="langchain",
            parsing_model="gpt-4",
            parsing_provider="openai",
            parsing_id="parsing",
            parsing_interface="langchain",
            deep_judgment_rubric_mode="custom",
            deep_judgment_rubric_config=raw_dict,
        )
        assert isinstance(config.deep_judgment_rubric_config, DeepJudgmentRubricCustomConfig)
