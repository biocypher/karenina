"""Tests for preset utility class methods in VerificationConfig."""

import pytest

from karenina.schemas.workflow.verification import VerificationConfig


class TestSanitizeModelConfig:
    """Tests for VerificationConfig.sanitize_model_config() class method."""

    def test_sanitize_basic_model(self):
        """Test sanitizing a basic model configuration."""
        model = {
            "id": "test-id",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "interface": "langchain",
            "system_prompt": "You are a helpful assistant",
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert result == model  # No extra fields to remove
        assert "id" in result
        assert "model_provider" in result
        assert "model_name" in result
        assert "temperature" in result
        assert "interface" in result
        assert "system_prompt" in result

    def test_sanitize_model_with_max_retries(self):
        """Test that max_retries is preserved if present."""
        model = {
            "id": "test-id",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "interface": "langchain",
            "system_prompt": "Test",
            "max_retries": 5,
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert result["max_retries"] == 5

    def test_sanitize_removes_endpoint_fields_for_langchain(self):
        """Test that endpoint fields are removed for langchain interface."""
        model = {
            "id": "test-id",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "interface": "langchain",
            "system_prompt": "Test",
            "endpoint_base_url": "http://example.com",
            "endpoint_api_key": "secret-key",
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert "endpoint_base_url" not in result
        assert "endpoint_api_key" not in result

    def test_sanitize_preserves_endpoint_fields_for_openai_endpoint(self):
        """Test that endpoint fields are preserved for openai_endpoint interface."""
        model = {
            "id": "test-id",
            "model_provider": "custom",
            "model_name": "custom-model",
            "temperature": 0.0,
            "interface": "openai_endpoint",
            "system_prompt": "Test",
            "endpoint_base_url": "http://example.com",
            "endpoint_api_key": "secret-key",
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert result["endpoint_base_url"] == "http://example.com"
        assert result["endpoint_api_key"] == "secret-key"

    def test_sanitize_removes_empty_endpoint_fields(self):
        """Test that empty endpoint fields are removed even for openai_endpoint."""
        model = {
            "id": "test-id",
            "model_provider": "custom",
            "model_name": "custom-model",
            "temperature": 0.0,
            "interface": "openai_endpoint",
            "system_prompt": "Test",
            "endpoint_base_url": "",
            "endpoint_api_key": None,
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert "endpoint_base_url" not in result
        assert "endpoint_api_key" not in result

    def test_sanitize_preserves_mcp_fields_with_values(self):
        """Test that MCP fields are preserved when they have values."""
        model = {
            "id": "test-id",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "interface": "langchain",
            "system_prompt": "Test",
            "mcp_urls_dict": {"tool1": "http://tool1.com"},
            "mcp_tool_filter": ["tool1", "tool2"],
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert result["mcp_urls_dict"] == {"tool1": "http://tool1.com"}
        assert result["mcp_tool_filter"] == ["tool1", "tool2"]

    def test_sanitize_removes_empty_mcp_fields(self):
        """Test that empty MCP fields are removed."""
        model = {
            "id": "test-id",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "interface": "langchain",
            "system_prompt": "Test",
            "mcp_urls_dict": {},
            "mcp_tool_filter": [],
        }

        result = VerificationConfig.sanitize_model_config(model)

        assert "mcp_urls_dict" not in result
        assert "mcp_tool_filter" not in result


class TestSanitizePresetName:
    """Tests for VerificationConfig.sanitize_preset_name() class method."""

    def test_basic_name(self):
        """Test basic name sanitization."""
        result = VerificationConfig.sanitize_preset_name("Quick Test")
        assert result == "quick-test.json"

    def test_lowercase_conversion(self):
        """Test that names are converted to lowercase."""
        result = VerificationConfig.sanitize_preset_name("UPPERCASE TEST")
        assert result == "uppercase-test.json"

    def test_space_to_hyphen(self):
        """Test that spaces are converted to hyphens."""
        result = VerificationConfig.sanitize_preset_name("Multi Word Name")
        assert result == "multi-word-name.json"

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        result = VerificationConfig.sanitize_preset_name("Test!@#$%^&*()Name")
        assert result == "testname.json"

    def test_consecutive_hyphens_removed(self):
        """Test that consecutive hyphens are collapsed."""
        result = VerificationConfig.sanitize_preset_name("test---name")
        assert result == "test-name.json"

    def test_leading_trailing_hyphens_removed(self):
        """Test that leading/trailing hyphens are removed."""
        result = VerificationConfig.sanitize_preset_name("---test-name---")
        assert result == "test-name.json"

    def test_empty_name_fallback(self):
        """Test that empty names fallback to 'preset'."""
        result = VerificationConfig.sanitize_preset_name("!@#$")
        assert result == "preset.json"

    def test_length_limit(self):
        """Test that names are limited to 96 characters."""
        long_name = "a" * 150
        result = VerificationConfig.sanitize_preset_name(long_name)
        # Should be "aaa...aaa.json" with 96 'a's
        assert len(result) == 101  # 96 + ".json" (5 chars)
        assert result.endswith(".json")
        assert result.count("a") == 96

    def test_complex_name(self):
        """Test a complex real-world name."""
        result = VerificationConfig.sanitize_preset_name("My Test Config (v2.1)")
        assert result == "my-test-config-v21.json"


class TestValidatePresetMetadata:
    """Tests for VerificationConfig.validate_preset_metadata() class method."""

    def test_valid_name_and_description(self):
        """Test that valid name and description pass validation."""
        # Should not raise
        VerificationConfig.validate_preset_metadata("Test Preset", "A test description")

    def test_valid_name_without_description(self):
        """Test that valid name without description passes."""
        # Should not raise
        VerificationConfig.validate_preset_metadata("Test Preset")

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Preset name cannot be empty"):
            VerificationConfig.validate_preset_metadata("")

    def test_whitespace_only_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Preset name cannot be empty"):
            VerificationConfig.validate_preset_metadata("   ")

    def test_none_name_raises_error(self):
        """Test that None name raises ValueError."""
        with pytest.raises(ValueError, match="Preset name cannot be empty"):
            VerificationConfig.validate_preset_metadata(None)

    def test_name_too_long_raises_error(self):
        """Test that name exceeding 100 characters raises ValueError."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Preset name cannot exceed 100 characters"):
            VerificationConfig.validate_preset_metadata(long_name)

    def test_name_exactly_100_chars_is_valid(self):
        """Test that name with exactly 100 characters is valid."""
        name = "a" * 100
        # Should not raise
        VerificationConfig.validate_preset_metadata(name)

    def test_description_too_long_raises_error(self):
        """Test that description exceeding 500 characters raises ValueError."""
        long_desc = "a" * 501
        with pytest.raises(ValueError, match="Description cannot exceed 500 characters"):
            VerificationConfig.validate_preset_metadata("Test", long_desc)

    def test_description_exactly_500_chars_is_valid(self):
        """Test that description with exactly 500 characters is valid."""
        desc = "a" * 500
        # Should not raise
        VerificationConfig.validate_preset_metadata("Test", desc)

    def test_empty_string_description_is_valid(self):
        """Test that empty string description is valid."""
        # Should not raise
        VerificationConfig.validate_preset_metadata("Test", "")


class TestCreatePresetStructure:
    """Tests for VerificationConfig.create_preset_structure() class method."""

    def test_basic_structure(self):
        """Test creating a basic preset structure."""
        result = VerificationConfig.create_preset_structure(
            preset_id="test-uuid",
            name="Test Preset",
            description="Test description",
            config_dict={"test": "config"},
            created_at="2025-11-03T12:00:00Z",
            updated_at="2025-11-03T12:00:00Z",
        )

        assert result["id"] == "test-uuid"
        assert result["name"] == "Test Preset"
        assert result["description"] == "Test description"
        assert result["config"] == {"test": "config"}
        assert result["created_at"] == "2025-11-03T12:00:00Z"
        assert result["updated_at"] == "2025-11-03T12:00:00Z"

    def test_structure_without_description(self):
        """Test creating preset structure without description."""
        result = VerificationConfig.create_preset_structure(
            preset_id="test-uuid",
            name="Test Preset",
            description=None,
            config_dict={"test": "config"},
            created_at="2025-11-03T12:00:00Z",
            updated_at="2025-11-03T12:00:00Z",
        )

        assert result["description"] is None

    def test_structure_has_all_required_fields(self):
        """Test that structure contains all required fields."""
        result = VerificationConfig.create_preset_structure(
            preset_id="test-uuid",
            name="Test",
            description=None,
            config_dict={},
            created_at="now",
            updated_at="now",
        )

        assert "id" in result
        assert "name" in result
        assert "description" in result
        assert "config" in result
        assert "created_at" in result
        assert "updated_at" in result
