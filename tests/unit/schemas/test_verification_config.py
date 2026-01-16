"""Unit tests for VerificationConfig and related models.

Tests cover:
- VerificationConfig field validation and defaults
- Backward compatibility (legacy single-model fields)
- Environment variable handling (EMBEDDING_CHECK, KARENINA_ASYNC_*)
- _validate_config() method with all error conditions
- __repr__() output formatting
- get_few_shot_config() and is_few_shot_enabled()
- Preset utility class methods (sanitize_model_config, sanitize_preset_name, validate_preset_metadata, create_preset_structure)
- save_preset() and from_preset() with temp directories
- DeepJudgmentTraitConfig validation
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from karenina.schemas.workflow.models import FewShotConfig, ModelConfig
from karenina.schemas.workflow.verification.config import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    DeepJudgmentTraitConfig,
    VerificationConfig,
)

# =============================================================================
# VerificationConfig Field Defaults Tests
# =============================================================================


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {},
    clear=True,
)
@patch("karenina.schemas.workflow.verification.config.os.getenv", return_value=None)
def test_verification_config_default_values(_mock_getenv) -> None:
    """Test VerificationConfig default field values.

    Patches os.getenv to ensure environment variables don't affect defaults.
    """
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    assert len(config.answering_models) == 1
    assert config.replicate_count == 1
    assert config.parsing_only is False
    assert config.rubric_enabled is False
    assert config.rubric_trait_names is None
    assert config.rubric_evaluation_strategy == "batch"
    assert config.evaluation_mode == "template_only"
    assert config.use_full_trace_for_template is False
    assert config.use_full_trace_for_rubric is True
    assert config.abstention_enabled is False
    assert config.embedding_check_enabled is False
    assert config.embedding_check_model == "all-MiniLM-L6-v2"
    assert config.embedding_check_threshold == 0.85
    assert config.async_enabled is True
    assert config.async_max_workers == 2
    assert config.deep_judgment_enabled is False
    assert config.deep_judgment_max_excerpts_per_attribute == 3
    assert config.deep_judgment_fuzzy_match_threshold == 0.80
    assert config.deep_judgment_excerpt_retry_attempts == 2
    assert config.deep_judgment_search_enabled is False
    assert config.deep_judgment_search_tool == "tavily"


@pytest.mark.unit
def test_deep_judgment_trait_config_defaults() -> None:
    """Test DeepJudgmentTraitConfig default values."""
    config = DeepJudgmentTraitConfig()

    assert config.enabled is True
    assert config.excerpt_enabled is True
    assert config.max_excerpts is None
    assert config.fuzzy_match_threshold is None
    assert config.excerpt_retry_attempts is None
    assert config.search_enabled is False


# =============================================================================
# VerificationConfig Validation Tests
# =============================================================================


@pytest.mark.unit
def test_validation_requires_parsing_model() -> None:
    """Test that validation requires at least one parsing model."""
    with pytest.raises(ValueError, match="At least one parsing model"):
        VerificationConfig(parsing_models=[])


@pytest.mark.unit
def test_validation_requires_answering_model_when_not_parsing_only() -> None:
    """Test that validation requires answering models unless parsing_only=True."""
    with pytest.raises(ValueError, match="At least one answering model"):
        VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
        )


@pytest.mark.unit
def test_validation_allows_no_answering_model_when_parsing_only() -> None:
    """Test that parsing_only mode allows empty answering_models."""
    # Should not raise
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )
    assert config.parsing_only is True


@pytest.mark.unit
def test_validation_requires_model_provider_for_non_excluded_interfaces() -> None:
    """Test that validation requires provider for most interfaces."""
    from karenina.schemas.workflow.models import ModelConfig

    model = ModelConfig(
        id="test",
        model_name="gpt-4",
        model_provider=None,  # Missing provider
        interface="langchain",  # Requires provider
        system_prompt="test",
        temperature=0.1,
    )

    with pytest.raises(ValueError, match="Model provider is required"):
        VerificationConfig(
            parsing_models=[model],
            answering_models=[],
            parsing_only=True,
        )


@pytest.mark.unit
def test_validation_allows_empty_provider_for_openrouter() -> None:
    """Test that openrouter interface allows empty provider."""
    from karenina.schemas.workflow.models import ModelConfig

    model = ModelConfig(
        id="test",
        model_name="gpt-4",
        model_provider=None,  # Empty provider
        interface="openrouter",  # Interface that doesn't require provider
        system_prompt="test",
        temperature=0.1,
    )

    # Should not raise
    config = VerificationConfig(
        parsing_models=[model],
        answering_models=[],
        parsing_only=True,
    )
    assert config.parsing_models[0].model_provider is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "evaluation_mode,rubric_enabled,error_match",
    [
        ("rubric_only", False, "evaluation_mode='rubric_only' requires rubric_enabled=True"),
        ("template_and_rubric", False, "evaluation_mode='template_and_rubric' requires rubric_enabled=True"),
        ("template_only", True, "evaluation_mode='template_only' is incompatible with rubric_enabled=True"),
    ],
    ids=[
        "rubric_only_requires_rubric",
        "template_and_rubric_requires_rubric",
        "template_only_incompatible_with_rubric",
    ],
)
def test_validation_evaluation_mode_rubric_consistency(
    evaluation_mode: str, rubric_enabled: bool, error_match: str
) -> None:
    """Test that evaluation_mode and rubric_enabled settings are consistent."""
    with pytest.raises(ValueError, match=error_match):
        VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
            evaluation_mode=evaluation_mode,
            rubric_enabled=rubric_enabled,
            parsing_only=True,
        )


@pytest.mark.unit
def test_validation_invalid_search_tool_name() -> None:
    """Test that validation rejects unknown search tool names."""
    with pytest.raises(ValueError, match="Unknown search tool"):
        VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
            deep_judgment_search_enabled=True,
            deep_judgment_search_tool="unknown_tool",
            parsing_only=True,
        )


@pytest.mark.unit
def test_validation_search_tool_must_be_string_or_callable() -> None:
    """Test that validation rejects invalid search tool types."""
    with pytest.raises(ValueError, match="Search tool must be either a supported tool name"):
        VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
            deep_judgment_search_enabled=True,
            deep_judgment_search_tool=123,  # Invalid type
            parsing_only=True,
        )


# =============================================================================
# VerificationConfig Backward Compatibility Tests
# =============================================================================


@pytest.mark.unit
def test_backward_compat_legacy_answering_fields() -> None:
    """Test backward compatibility with legacy single answering model fields."""
    config = VerificationConfig(
        answering_model_provider="openai",
        answering_model_name="gpt-4",
        answering_temperature=0.5,
        answering_interface="langchain",
        answering_system_prompt="Custom prompt",
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    assert len(config.answering_models) == 1
    assert config.answering_models[0].id == "answering-legacy"
    assert config.answering_models[0].model_provider == "openai"
    assert config.answering_models[0].model_name == "gpt-4"
    assert config.answering_models[0].temperature == 0.5
    assert config.answering_models[0].interface == "langchain"
    assert config.answering_models[0].system_prompt == "Custom prompt"


@pytest.mark.unit
def test_backward_compat_legacy_parsing_fields() -> None:
    """Test backward compatibility with legacy single parsing model fields."""
    config = VerificationConfig(
        parsing_model_provider="anthropic",
        parsing_model_name="claude-haiku-4-5",
        parsing_temperature=0.3,
        parsing_interface="langchain",
        parsing_system_prompt="Parse this",
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    assert len(config.parsing_models) == 1
    assert config.parsing_models[0].id == "parsing-legacy"
    assert config.parsing_models[0].model_provider == "anthropic"
    assert config.parsing_models[0].model_name == "claude-haiku-4-5"
    assert config.parsing_models[0].temperature == 0.3


@pytest.mark.unit
def test_backward_compat_legacy_few_shot_fields() -> None:
    """Test backward compatibility with legacy few-shot fields."""

    config = VerificationConfig(
        few_shot_enabled=True,
        few_shot_mode="k-shot",
        few_shot_k=5,
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    assert config.few_shot_config is not None
    assert config.few_shot_config.enabled is True
    assert config.few_shot_config.global_mode == "k-shot"
    assert config.few_shot_config.global_k == 5


@pytest.mark.unit
def test_backward_compat_default_system_prompts() -> None:
    """Test that default system prompts are applied when not provided."""
    from karenina.schemas.workflow.models import ModelConfig

    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="p1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="",  # Empty, should be replaced
                temperature=0.1,
            )
        ],
        answering_models=[
            ModelConfig(
                id="a1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt=None,  # None, should be replaced
                temperature=0.1,
            )
        ],
    )

    # Default prompts should be applied
    assert config.parsing_models[0].system_prompt == DEFAULT_PARSING_SYSTEM_PROMPT
    assert config.answering_models[0].system_prompt == DEFAULT_ANSWERING_SYSTEM_PROMPT


# =============================================================================
# VerificationConfig Environment Variable Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "env_var,env_value,config_attr,expected_value",
    [
        ("EMBEDDING_CHECK", "true", "embedding_check_enabled", True),
        ("EMBEDDING_CHECK_MODEL", "custom-model", "embedding_check_model", "custom-model"),
        ("EMBEDDING_CHECK_THRESHOLD", "0.95", "embedding_check_threshold", 0.95),
        ("KARENINA_ASYNC_ENABLED", "false", "async_enabled", False),
        ("KARENINA_ASYNC_MAX_WORKERS", "4", "async_max_workers", 4),
    ],
    ids=[
        "embedding_check_enabled",
        "embedding_check_model",
        "embedding_check_threshold",
        "async_enabled",
        "async_max_workers",
    ],
)
def test_env_var_sets_config_value(env_var: str, env_value: str, config_attr: str, expected_value: object) -> None:
    """Test that environment variables correctly set config values."""
    with patch.dict("os.environ", {env_var: env_value}):
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
            parsing_only=True,
        )
        assert getattr(config, config_attr) == expected_value


@pytest.mark.unit
def test_env_var_invalid_embedding_threshold_ignored() -> None:
    """Test that invalid EMBEDDING_CHECK_THRESHOLD is ignored (uses default)."""
    with patch.dict("os.environ", {"EMBEDDING_CHECK_THRESHOLD": "not-a-number"}):
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
            parsing_only=True,
        )
        # Should use default value
        assert config.embedding_check_threshold == 0.85


@pytest.mark.unit
def test_explicit_value_overrides_env_var() -> None:
    """Test that explicit values override environment variables."""
    with patch.dict("os.environ", {"EMBEDDING_CHECK": "true"}):
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.1,
                )
            ],
            answering_models=[],
            parsing_only=True,
            embedding_check_enabled=False,  # Explicit value
        )
        # Explicit value should override env var
        assert config.embedding_check_enabled is False


# =============================================================================
# VerificationConfig Repr Tests
# =============================================================================


@pytest.mark.unit
def test_repr_shows_models() -> None:
    """Test that __repr__ shows model information."""
    from karenina.schemas.workflow.models import ModelConfig

    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="a1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.5,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="p1",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    repr_str = repr(config)
    assert "Answering (1):" in repr_str
    assert "gpt-4" in repr_str
    assert "Parsing (1):" in repr_str
    assert "claude-haiku-4-5" in repr_str


@pytest.mark.unit
def test_repr_shows_execution_settings() -> None:
    """Test that __repr__ shows execution settings."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        replicate_count=3,
        async_enabled=False,
    )

    repr_str = repr(config)
    assert "Replicates: 3" in repr_str
    assert "Async: False" in repr_str
    assert "Parsing Only: True" in repr_str


@pytest.mark.unit
def test_repr_shows_features() -> None:
    """Test that __repr__ shows enabled features."""
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        evaluation_mode="template_and_rubric",
        rubric_enabled=True,
        abstention_enabled=True,
        deep_judgment_enabled=True,
    )

    repr_str = repr(config)
    assert "Rubric: enabled" in repr_str
    assert "Abstention: enabled" in repr_str
    assert "Deep Judgment (Template):" in repr_str


# =============================================================================
# VerificationConfig Few-Shot Tests
# =============================================================================


@pytest.mark.unit
def test_get_few_shot_config_returns_new_config() -> None:
    """Test get_few_shot_config returns new FewShotConfig when present."""

    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        few_shot_config=FewShotConfig(enabled=True, global_mode="all", global_k=3),
    )

    result = config.get_few_shot_config()
    assert result is not None
    assert result.enabled is True
    assert result.global_mode == "all"


@pytest.mark.unit
def test_get_few_shot_config_returns_none_when_disabled() -> None:
    """Test get_few_shot_config returns None when few-shot is disabled."""
    # When no few_shot fields are set at all, get_few_shot_config returns None
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    result = config.get_few_shot_config()
    assert result is None


@pytest.mark.unit
def test_is_few_shot_enabled_true() -> None:
    """Test is_few_shot_enabled returns True when enabled."""

    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        few_shot_config=FewShotConfig(enabled=True, global_mode="all", global_k=3),
    )

    assert config.is_few_shot_enabled() is True


@pytest.mark.unit
def test_is_few_shot_enabled_false() -> None:
    """Test is_few_shot_enabled returns False when disabled."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    assert config.is_few_shot_enabled() is False


# =============================================================================
# VerificationConfig Preset Utility Tests
# =============================================================================


@pytest.mark.unit
def test_sanitize_model_config_basic() -> None:
    """Test sanitize_model_config with basic configuration."""
    model = {
        "id": "test-model",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.5,
        "interface": "langchain",
        "system_prompt": "You are helpful.",
    }

    result = VerificationConfig.sanitize_model_config(model)

    assert result == {
        "id": "test-model",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.5,
        "interface": "langchain",
        "system_prompt": "You are helpful.",
    }


@pytest.mark.unit
def test_sanitize_model_config_includes_max_retries() -> None:
    """Test sanitize_model_config includes max_retries when present."""
    model = {
        "id": "test-model",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.5,
        "interface": "langchain",
        "system_prompt": "test",
        "max_retries": 3,
    }

    result = VerificationConfig.sanitize_model_config(model)

    assert result["max_retries"] == 3


@pytest.mark.unit
def test_sanitize_model_config_openai_endpoint_includes_endpoint_fields() -> None:
    """Test sanitize_model_config includes endpoint fields for openai_endpoint."""
    model = {
        "id": "test-model",
        "model_provider": None,
        "model_name": "gpt-4",
        "temperature": 0.5,
        "interface": "openai_endpoint",
        "system_prompt": "test",
        "endpoint_base_url": "http://localhost:8000",
        "endpoint_api_key": "sk-test",
    }

    result = VerificationConfig.sanitize_model_config(model)

    assert result["endpoint_base_url"] == "http://localhost:8000"
    assert result["endpoint_api_key"] == "sk-test"


@pytest.mark.unit
def test_sanitize_model_config_excludes_endpoint_fields_for_non_openai_endpoint() -> None:
    """Test sanitize_model_config excludes endpoint fields for other interfaces."""
    model = {
        "id": "test-model",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.5,
        "interface": "langchain",
        "system_prompt": "test",
        "endpoint_base_url": "http://localhost:8000",  # Should be excluded
        "endpoint_api_key": "sk-test",  # Should be excluded
    }

    result = VerificationConfig.sanitize_model_config(model)

    assert "endpoint_base_url" not in result
    assert "endpoint_api_key" not in result


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_name,expected_output",
    [
        ("My Test Preset", "my-test-preset.json"),
        ("Test@#$% Config!", "test-config.json"),
        ("Test---Config", "test-config.json"),
        ("   ", "preset.json"),
    ],
    ids=["basic", "special_chars", "consecutive_hyphens", "empty_defaults"],
)
def test_sanitize_preset_name(input_name: str, expected_output: str) -> None:
    """Test sanitize_preset_name with various inputs."""
    result = VerificationConfig.sanitize_preset_name(input_name)
    assert result == expected_output


@pytest.mark.unit
def test_sanitize_preset_name_limits_length() -> None:
    """Test sanitize_preset_name limits length to 96 chars."""
    long_name = "a" * 100
    result = VerificationConfig.sanitize_preset_name(long_name)

    assert len(result) == len(".json") + 96  # 96 chars + .json
    assert result.endswith(".json")


@pytest.mark.unit
def test_validate_preset_metadata_success() -> None:
    """Test validate_preset_metadata with valid input."""
    # Should not raise
    VerificationConfig.validate_preset_metadata("Test Preset", "A test description")


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,description,error_match",
    [
        ("", None, "Preset name cannot be empty"),
        ("   ", None, "Preset name cannot be empty"),
        ("a" * 101, None, "Preset name cannot exceed 100 characters"),
        ("Test", "a" * 501, "Description cannot exceed 500 characters"),
    ],
    ids=["empty_name", "whitespace_name", "name_too_long", "description_too_long"],
)
def test_validate_preset_metadata_errors(name: str, description: str | None, error_match: str) -> None:
    """Test validate_preset_metadata rejects invalid inputs."""
    with pytest.raises(ValueError, match=error_match):
        VerificationConfig.validate_preset_metadata(name, description)


@pytest.mark.unit
def test_create_preset_structure() -> None:
    """Test create_preset_structure creates standardized structure."""
    config_dict = {"parsing_models": [...]}

    result = VerificationConfig.create_preset_structure(
        preset_id="abc-123",
        name="Test",
        description="Test preset",
        config_dict=config_dict,
        created_at="2025-01-11T12:00:00Z",
        updated_at="2025-01-11T12:00:00Z",
    )

    assert result["id"] == "abc-123"
    assert result["name"] == "Test"
    assert result["description"] == "Test preset"
    assert result["config"] is config_dict
    assert result["created_at"] == "2025-01-11T12:00:00Z"
    assert result["updated_at"] == "2025-01-11T12:00:00Z"


# =============================================================================
# VerificationConfig Save/Load Preset Tests
# =============================================================================


@pytest.mark.unit
def test_save_preset_creates_file(tmp_path: Path) -> None:
    """Test save_preset creates a JSON file."""
    from karenina.schemas.workflow.models import ModelConfig

    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="a1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.5,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="p1",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    metadata = config.save_preset(
        name="Test Preset",
        description="A test preset",
        presets_dir=tmp_path,
    )

    assert metadata["name"] == "Test Preset"
    assert metadata["description"] == "A test preset"
    assert "filepath" in metadata
    assert "id" in metadata
    assert "created_at" in metadata

    # Check file exists
    filepath = Path(metadata["filepath"])
    assert filepath.exists()

    # Check file content
    with open(filepath) as f:
        preset = json.load(f)

    assert preset["name"] == "Test Preset"
    assert preset["description"] == "A test preset"
    assert "config" in preset
    assert "id" in preset


@pytest.mark.unit
def test_save_preset_generates_safe_filename(tmp_path: Path) -> None:
    """Test save_preset generates safe filename."""
    from karenina.schemas.workflow.models import ModelConfig

    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    metadata = config.save_preset(
        name="My Test@#$ Preset!",
        presets_dir=tmp_path,
    )

    # Filename should be sanitized
    filepath = Path(metadata["filepath"])
    assert filepath.name == "my-test-preset.json"


@pytest.mark.unit
def test_save_perset_existing_file_raises(tmp_path: Path) -> None:
    """Test save_preset raises error if file already exists."""
    from karenina.schemas.workflow.models import ModelConfig

    # Create existing file
    (tmp_path / "test-preset.json").write_text("{}")

    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    with pytest.raises(ValueError, match="already exists"):
        config.save_preset(name="test preset", presets_dir=tmp_path)


@pytest.mark.unit
def test_from_preset_loads_config(tmp_path: Path) -> None:
    """Test from_preset loads VerificationConfig from file."""
    from karenina.schemas.workflow.models import ModelConfig

    # Create a preset file
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="a1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.5,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="p1",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    metadata = config.save_preset("Test Preset", presets_dir=tmp_path)

    # Load it back
    loaded_config = VerificationConfig.from_preset(Path(metadata["filepath"]))

    assert len(loaded_config.answering_models) == 1
    assert loaded_config.answering_models[0].model_name == "gpt-4"
    assert len(loaded_config.parsing_models) == 1
    assert loaded_config.parsing_models[0].model_name == "claude-haiku-4-5"


@pytest.mark.unit
def test_from_preset_nonexistent_file_raises(tmp_path: Path) -> None:
    """Test from_preset raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="Preset file not found"):
        VerificationConfig.from_preset(tmp_path / "nonexistent.json")


@pytest.mark.unit
def test_from_preset_invalid_json_raises(tmp_path: Path) -> None:
    """Test from_preset raises error for corrupted JSON file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{ invalid json }")

    with pytest.raises(json.JSONDecodeError):
        VerificationConfig.from_preset(invalid_file)


@pytest.mark.unit
def test_from_preset_missing_config_raises(tmp_path: Path) -> None:
    """Test from_preset raises error when preset has no config."""
    invalid_file = tmp_path / "no-config.json"
    invalid_file.write_text(json.dumps({"name": "Test"}))

    with pytest.raises(ValueError, match="no configuration data"):
        VerificationConfig.from_preset(invalid_file)


# =============================================================================
# VerificationConfig Deep Judgment Rubric Mode Tests
# =============================================================================


@pytest.mark.unit
def test_deep_judgment_rubric_mode_default() -> None:
    """Test default deep_judgment_rubric_mode is 'disabled'."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    assert config.deep_judgment_rubric_mode == "disabled"


@pytest.mark.unit
def test_deep_judgment_rubric_enable_all_mode() -> None:
    """Test deep_judgment_rubric_mode='enable_all' setting."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        deep_judgment_rubric_mode="enable_all",
        deep_judgment_rubric_global_excerpts=False,
    )

    assert config.deep_judgment_rubric_mode == "enable_all"
    assert config.deep_judgment_rubric_global_excerpts is False


@pytest.mark.unit
def test_deep_judgment_rubric_custom_mode() -> None:
    """Test deep_judgment_rubric_mode='custom' with config."""
    custom_config = {
        "global": {
            "Clarity": {"enabled": True, "excerpt_enabled": False},
        },
        "question_specific": {
            "q-123": {
                "Completeness": {"enabled": False},
            }
        },
    }

    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        deep_judgment_rubric_mode="custom",
        deep_judgment_rubric_config=custom_config,
    )

    assert config.deep_judgment_rubric_mode == "custom"
    assert config.deep_judgment_rubric_config == custom_config


# =============================================================================
# VerificationConfig Sufficiency Detection Tests
# =============================================================================


@pytest.mark.unit
def test_sufficiency_enabled_default_false() -> None:
    """Test that sufficiency_enabled defaults to False."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
    )

    assert config.sufficiency_enabled is False


@pytest.mark.unit
def test_sufficiency_enabled_can_be_set_true() -> None:
    """Test that sufficiency_enabled can be set to True."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        sufficiency_enabled=True,
    )

    assert config.sufficiency_enabled is True


@pytest.mark.unit
def test_sufficiency_and_abstention_can_both_be_enabled() -> None:
    """Test that sufficiency_enabled and abstention_enabled can both be True."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        abstention_enabled=True,
        sufficiency_enabled=True,
    )

    assert config.abstention_enabled is True
    assert config.sufficiency_enabled is True


@pytest.mark.unit
def test_repr_shows_sufficiency_when_enabled() -> None:
    """Test that __repr__ shows sufficiency when enabled."""
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[],
        parsing_only=True,
        sufficiency_enabled=True,
    )

    repr_str = repr(config)
    assert "Sufficiency: enabled" in repr_str
