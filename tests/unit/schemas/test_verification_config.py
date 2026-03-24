"""Unit tests for VerificationConfig and related models.

Tests cover:
- VerificationConfig field validation and defaults
- Default system prompt application
- Environment variable handling (EMBEDDING_CHECK, KARENINA_ASYNC_*)
- _validate_config() method with all error conditions
- __repr__() output formatting
- get_few_shot_config() and is_few_shot_enabled()
- Preset utility class methods (sanitize_model_config, sanitize_preset_name, validate_preset_metadata, create_preset_structure)
- save_preset() and from_preset() with temp directories
- DeepJudgmentTraitConfig validation
- Validation constraints: extra="forbid", Field ge/le, bool rejection
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from karenina.schemas.config import FewShotConfig, ModelConfig
from karenina.schemas.config.models import ModelRetryConfig, ToolRetryConfig
from karenina.schemas.verification import (
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
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
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
    assert config.deep_judgment_mode == "disabled"
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
    from karenina.schemas.config import ModelConfig

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
    from karenina.schemas.config import ModelConfig

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
    with pytest.raises(ValueError, match="Input should be"):
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
            deep_judgment_search_tool=123,  # Invalid type — rejected by Pydantic (str | Callable)
            parsing_only=True,
        )


# =============================================================================
# VerificationConfig Default System Prompt Tests
# =============================================================================


@pytest.mark.unit
def test_default_system_prompts() -> None:
    """Test that default system prompts are applied when not provided."""
    from karenina.schemas.config import ModelConfig

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
    from karenina.schemas.config import ModelConfig

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
        abstention_enabled=True,
        deep_judgment_mode="full",
    )

    repr_str = repr(config)
    assert "Rubric: enabled" in repr_str
    assert "Abstention: enabled" in repr_str
    assert "Deep Judgment (Template): mode=full" in repr_str


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
    from karenina.schemas.config import ModelConfig

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
    from karenina.schemas.config import ModelConfig

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
    from karenina.schemas.config import ModelConfig

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
    from karenina.schemas.config import ModelConfig

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


# =============================================================================
# VerificationConfig Agentic Rubric Config Fields Tests
# =============================================================================


@pytest.mark.unit
class TestAgenticRubricConfigFields:
    """Tests for agentic rubric strategy and parallel fields on VerificationConfig."""

    def test_defaults(self):
        """New fields have correct defaults."""
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test",
                    model_name="test",
                    model_provider="test",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )
        assert config.agentic_rubric_strategy == "individual"
        assert config.agentic_rubric_parallel is False

    def test_shared_strategy(self):
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test",
                    model_name="test",
                    model_provider="test",
                    interface="langchain",
                )
            ],
            parsing_only=True,
            agentic_rubric_strategy="shared",
        )
        assert config.agentic_rubric_strategy == "shared"

    def test_parallel_enabled(self):
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test",
                    model_name="test",
                    model_provider="test",
                    interface="langchain",
                )
            ],
            parsing_only=True,
            agentic_rubric_parallel=True,
        )
        assert config.agentic_rubric_parallel is True

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError):
            VerificationConfig(
                parsing_models=[
                    ModelConfig(
                        id="test",
                        model_name="test",
                        model_provider="test",
                        interface="langchain",
                    )
                ],
                parsing_only=True,
                agentic_rubric_strategy="invalid",
            )


# =============================================================================
# Issue 013: VerificationConfig extra="forbid"
# =============================================================================


@pytest.mark.unit
class TestVerificationConfigExtraForbid:
    """Tests for VerificationConfig rejecting unknown extra fields."""

    def _make_parsing_model(self) -> ModelConfig:
        """Create a minimal valid parsing model."""
        return ModelConfig(
            id="parsing",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
            system_prompt="test",
            temperature=0.1,
        )

    def test_rejects_unknown_field(self) -> None:
        """VerificationConfig rejects unknown extra fields."""
        with pytest.raises(ValidationError, match="extra_forbidden"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                totally_unknown_field="should_fail",
            )

    def test_accepts_known_fields(self) -> None:
        """VerificationConfig accepts all known fields without error."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            replicate_count=3,
            abstention_enabled=True,
        )
        assert config.replicate_count == 3
        assert config.abstention_enabled is True

    def test_round_trip_dump_and_recreate(self) -> None:
        """VerificationConfig can be dumped and recreated (round-trip).

        This ensures computed fields like rubric_enabled do not break
        reconstruction from a model_dump() output.
        """
        original = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.5,
                )
            ],
            parsing_models=[self._make_parsing_model()],
            evaluation_mode="template_and_rubric",
        )
        dumped = original.model_dump()
        # rubric_enabled appears in the dump as a computed field
        assert "rubric_enabled" in dumped

        # Recreating from dump should work (rubric_enabled is popped in __init__)
        recreated = VerificationConfig(**dumped)
        assert recreated.evaluation_mode == "template_and_rubric"
        assert recreated.rubric_enabled is True

    def test_from_overrides_round_trip(self) -> None:
        """from_overrides can rebuild from a base config without errors.

        This tests that model_dump() output passed back to the constructor
        does not include fields that would be rejected by extra='forbid'.
        """
        base = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering",
                    model_name="gpt-4",
                    model_provider="openai",
                    interface="langchain",
                    system_prompt="test",
                    temperature=0.5,
                )
            ],
            parsing_models=[self._make_parsing_model()],
        )
        # from_overrides dumps the base and re-creates; must not raise
        rebuilt = VerificationConfig.from_overrides(base, replicate_count=5)
        assert rebuilt.replicate_count == 5


# =============================================================================
# Issue 014: replicate_count rejects 0 and negative in all modes
# =============================================================================


@pytest.mark.unit
class TestReplicateCountValidation:
    """Tests for replicate_count rejecting zero and negative values."""

    def _make_parsing_model(self) -> ModelConfig:
        return ModelConfig(
            id="parsing",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
            system_prompt="test",
            temperature=0.1,
        )

    def test_rejects_zero_in_template_only_mode(self) -> None:
        """replicate_count=0 is rejected even in template_only mode."""
        with pytest.raises(ValidationError, match="replicate_count"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                replicate_count=0,
            )

    def test_rejects_negative_in_template_only_mode(self) -> None:
        """replicate_count=-1 is rejected even in template_only mode."""
        with pytest.raises(ValidationError, match="replicate_count"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                replicate_count=-1,
            )

    def test_accepts_one(self) -> None:
        """replicate_count=1 is accepted."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            replicate_count=1,
        )
        assert config.replicate_count == 1

    def test_accepts_positive(self) -> None:
        """replicate_count=5 is accepted."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            replicate_count=5,
        )
        assert config.replicate_count == 5


# =============================================================================
# Issue 134: Numeric fields missing range constraints
# =============================================================================


@pytest.mark.unit
class TestNumericFieldRangeConstraints:
    """Tests for numeric field range constraints on VerificationConfig."""

    def _make_parsing_model(self) -> ModelConfig:
        return ModelConfig(
            id="parsing",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
            system_prompt="test",
            temperature=0.1,
        )

    # async_max_workers: ge=1
    def test_async_max_workers_rejects_zero(self) -> None:
        """async_max_workers=0 is rejected (must be >= 1)."""
        with pytest.raises(ValidationError, match="async_max_workers"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                async_max_workers=0,
            )

    def test_async_max_workers_accepts_one(self) -> None:
        """async_max_workers=1 is accepted."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            async_max_workers=1,
        )
        assert config.async_max_workers == 1

    # embedding_check_threshold: ge=0.0, le=1.0
    def test_embedding_threshold_rejects_negative(self) -> None:
        """embedding_check_threshold=-0.1 is rejected."""
        with pytest.raises(ValidationError, match="embedding_check_threshold"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                embedding_check_threshold=-0.1,
            )

    def test_embedding_threshold_rejects_above_one(self) -> None:
        """embedding_check_threshold=1.1 is rejected."""
        with pytest.raises(ValidationError, match="embedding_check_threshold"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                embedding_check_threshold=1.1,
            )

    def test_embedding_threshold_accepts_boundaries(self) -> None:
        """embedding_check_threshold=0.0 and 1.0 are both accepted."""
        config_low = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            embedding_check_threshold=0.0,
        )
        assert config_low.embedding_check_threshold == 0.0

        config_high = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            embedding_check_threshold=1.0,
        )
        assert config_high.embedding_check_threshold == 1.0

    # agentic_parsing_max_turns: ge=1
    def test_agentic_parsing_max_turns_rejects_zero(self) -> None:
        """agentic_parsing_max_turns=0 is rejected."""
        with pytest.raises(ValidationError, match="agentic_parsing_max_turns"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                agentic_parsing_max_turns=0,
            )

    def test_agentic_parsing_max_turns_accepts_one(self) -> None:
        """agentic_parsing_max_turns=1 is accepted."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            agentic_parsing_max_turns=1,
        )
        assert config.agentic_parsing_max_turns == 1

    # agentic_parsing_timeout: ge=0.0
    def test_agentic_parsing_timeout_rejects_negative(self) -> None:
        """agentic_parsing_timeout=-1.0 is rejected."""
        with pytest.raises(ValidationError, match="agentic_parsing_timeout"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                agentic_parsing_timeout=-1.0,
            )

    def test_agentic_parsing_timeout_accepts_zero(self) -> None:
        """agentic_parsing_timeout=0.0 is accepted."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            agentic_parsing_timeout=0.0,
        )
        assert config.agentic_parsing_timeout == 0.0

    # scenario_turn_limit: ge=1
    def test_scenario_turn_limit_rejects_zero(self) -> None:
        """scenario_turn_limit=0 is rejected."""
        with pytest.raises(ValidationError, match="scenario_turn_limit"):
            VerificationConfig(
                parsing_models=[self._make_parsing_model()],
                parsing_only=True,
                scenario_turn_limit=0,
            )

    def test_scenario_turn_limit_accepts_one(self) -> None:
        """scenario_turn_limit=1 is accepted."""
        config = VerificationConfig(
            parsing_models=[self._make_parsing_model()],
            parsing_only=True,
            scenario_turn_limit=1,
        )
        assert config.scenario_turn_limit == 1


# =============================================================================
# Issue 030: ModelConfig rejects manual_traces=True (bool)
# =============================================================================


@pytest.mark.unit
class TestModelConfigManualTracesBoolRejection:
    """Tests for ModelConfig rejecting bool values for manual_traces."""

    def test_rejects_true_bool(self) -> None:
        """ModelConfig rejects manual_traces=True with interface='manual'."""
        with pytest.raises(ValueError, match="ManualTraces instance"):
            ModelConfig(
                interface="manual",
                manual_traces=True,
            )

    def test_rejects_false_bool(self) -> None:
        """ModelConfig rejects manual_traces=False with interface='manual'.

        False is not None, so it passes the None check but should be caught
        as a bool.
        """
        with pytest.raises(ValueError, match="ManualTraces instance"):
            ModelConfig(
                interface="manual",
                manual_traces=False,
            )

    def test_accepts_none_for_non_manual(self) -> None:
        """ModelConfig accepts manual_traces=None for non-manual interfaces."""
        config = ModelConfig(
            id="test",
            model_name="gpt-4",
            model_provider="openai",
            interface="langchain",
        )
        assert config.manual_traces is None


# =============================================================================
# Issue 039: ModelRetryConfig and ToolRetryConfig reject negative max_retries
# =============================================================================


@pytest.mark.unit
class TestRetryConfigMaxRetriesConstraint:
    """Tests for max_retries rejecting negative values."""

    def test_model_retry_rejects_negative(self) -> None:
        """ModelRetryConfig rejects max_retries=-1."""
        with pytest.raises(ValidationError, match="max_retries"):
            ModelRetryConfig(max_retries=-1)

    def test_model_retry_accepts_zero(self) -> None:
        """ModelRetryConfig accepts max_retries=0 (no retries)."""
        config = ModelRetryConfig(max_retries=0)
        assert config.max_retries == 0

    def test_model_retry_accepts_positive(self) -> None:
        """ModelRetryConfig accepts max_retries=5."""
        config = ModelRetryConfig(max_retries=5)
        assert config.max_retries == 5

    def test_tool_retry_rejects_negative(self) -> None:
        """ToolRetryConfig rejects max_retries=-1."""
        with pytest.raises(ValidationError, match="max_retries"):
            ToolRetryConfig(max_retries=-1)

    def test_tool_retry_accepts_zero(self) -> None:
        """ToolRetryConfig accepts max_retries=0 (no retries)."""
        config = ToolRetryConfig(max_retries=0)
        assert config.max_retries == 0

    def test_tool_retry_accepts_positive(self) -> None:
        """ToolRetryConfig accepts max_retries=3."""
        config = ToolRetryConfig(max_retries=3)
        assert config.max_retries == 3
