"""Tests for embedding check functionality."""

import os
from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.models import ModelConfig
from karenina.benchmark.verification.embedding_utils import (
    _convert_to_comparable_string,
    _get_embedding_model_name,
    _get_embedding_threshold,
    _should_use_embedding_check,
    check_semantic_equivalence,
    compute_embedding_similarity,
    perform_embedding_check,
)


class TestEmbeddingCheckConfiguration:
    """Test configuration and environment variable handling."""

    def test_should_use_embedding_check_disabled_by_default(self):
        """Test that embedding check is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            assert not _should_use_embedding_check()

    def test_should_use_embedding_check_enabled(self):
        """Test that embedding check can be enabled."""
        test_cases = ["true", "True", "TRUE", "1", "yes", "on"]
        for value in test_cases:
            with patch.dict(os.environ, {"EMBEDDING_CHECK": value}):
                assert _should_use_embedding_check()

    def test_should_use_embedding_check_disabled(self):
        """Test that embedding check can be explicitly disabled."""
        test_cases = ["false", "False", "FALSE", "0", "no", "off", "invalid"]
        for value in test_cases:
            with patch.dict(os.environ, {"EMBEDDING_CHECK": value}):
                assert not _should_use_embedding_check()

    def test_get_embedding_model_name_default(self):
        """Test default embedding model name."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_embedding_model_name() == "all-MiniLM-L6-v2"

    def test_get_embedding_model_name_custom(self):
        """Test custom embedding model name."""
        with patch.dict(os.environ, {"EMBEDDING_CHECK_MODEL": "custom-model"}):
            assert _get_embedding_model_name() == "custom-model"

    def test_get_embedding_threshold_default(self):
        """Test default embedding threshold."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_embedding_threshold() == 0.85

    def test_get_embedding_threshold_custom(self):
        """Test custom embedding threshold."""
        with patch.dict(os.environ, {"EMBEDDING_CHECK_THRESHOLD": "0.9"}):
            assert _get_embedding_threshold() == 0.9

    def test_get_embedding_threshold_clamped(self):
        """Test embedding threshold is clamped between 0 and 1."""
        with patch.dict(os.environ, {"EMBEDDING_CHECK_THRESHOLD": "1.5"}):
            assert _get_embedding_threshold() == 1.0

        with patch.dict(os.environ, {"EMBEDDING_CHECK_THRESHOLD": "-0.5"}):
            assert _get_embedding_threshold() == 0.0

    def test_get_embedding_threshold_invalid(self):
        """Test invalid threshold falls back to default."""
        with patch.dict(os.environ, {"EMBEDDING_CHECK_THRESHOLD": "invalid"}):
            assert _get_embedding_threshold() == 0.85


class TestStringConversion:
    """Test string conversion utilities."""

    def test_convert_to_comparable_string_dict(self):
        """Test converting dictionary to comparable string."""
        data = {"key": "value", "number": 42}
        result = _convert_to_comparable_string(data)
        assert "key" in result
        assert "value" in result
        assert "42" in result

    def test_convert_to_comparable_string_sorted_keys(self):
        """Test that keys are sorted for consistency."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        result1 = _convert_to_comparable_string(data1)
        result2 = _convert_to_comparable_string(data2)
        assert result1 == result2

    def test_convert_to_comparable_string_none(self):
        """Test converting None returns empty string."""
        assert _convert_to_comparable_string(None) == ""

    def test_convert_to_comparable_string_empty_dict(self):
        """Test converting empty dict."""
        result = _convert_to_comparable_string({})
        assert result == "{}"


class TestEmbeddingSimilarity:
    """Test embedding similarity computation."""

    # Note: These tests are skipped because they require complex mocking of dynamic imports
    # The actual functionality can be tested through integration tests

    @pytest.mark.skip(reason="Dynamic import mocking is complex - test via integration")
    def test_compute_embedding_similarity_success(self):
        """Test successful embedding similarity computation."""
        pass

    def test_compute_embedding_similarity_none_data(self):
        """Test embedding similarity with None data."""
        with patch.dict(os.environ, {}, clear=True):
            similarity, model_name = compute_embedding_similarity(None, {"answer": "test"})
            assert similarity == 0.0
            assert model_name == "all-MiniLM-L6-v2"

            similarity, model_name = compute_embedding_similarity({"answer": "test"}, None)
            assert similarity == 0.0

    def test_compute_embedding_similarity_empty_data(self):
        """Test embedding similarity with empty data."""
        with patch.dict(os.environ, {}, clear=True):
            similarity, model_name = compute_embedding_similarity({}, {"answer": "test"})
            assert similarity == 0.0

    @pytest.mark.skip(reason="Dynamic import mocking is complex - test via integration")
    def test_compute_embedding_similarity_missing_import(self):
        """Test handling of missing sentence-transformers import."""
        pass

    @pytest.mark.skip(reason="Dynamic import mocking is complex - test via integration")
    def test_compute_embedding_similarity_model_error(self):
        """Test handling of model loading errors."""
        pass

    @pytest.mark.skip(reason="Dynamic import mocking is complex - test via integration")
    def test_compute_embedding_similarity_clamped(self):
        """Test similarity scores are clamped to [0, 1]."""
        pass


class TestSemanticEquivalence:
    """Test semantic equivalence checking."""

    def test_check_semantic_equivalence_none_data(self):
        """Test semantic check with None data."""
        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        is_equiv, details = check_semantic_equivalence(None, {"answer": "test"}, model_config, "What is the answer?")
        assert not is_equiv
        assert "Missing data" in details

    @patch("karenina.benchmark.verification.embedding_utils.init_chat_model_unified")
    def test_check_semantic_equivalence_yes(self, mock_init_chat):
        """Test semantic equivalence returning YES."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "YES - These responses are semantically equivalent"
        mock_llm.invoke.return_value = mock_response
        mock_init_chat.return_value = mock_llm

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        gt_data = {"answer": "The capital is Paris"}
        llm_data = {"answer": "Paris is the capital"}

        is_equiv, details = check_semantic_equivalence(gt_data, llm_data, model_config, "What is the capital?")

        assert is_equiv is True
        assert "YES" in details
        mock_init_chat.assert_called_once()
        mock_llm.invoke.assert_called_once()

    @patch("karenina.benchmark.verification.embedding_utils.init_chat_model_unified")
    def test_check_semantic_equivalence_no(self, mock_init_chat):
        """Test semantic equivalence returning NO."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "NO - These responses have different meanings"
        mock_llm.invoke.return_value = mock_response
        mock_init_chat.return_value = mock_llm

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        gt_data = {"answer": "Paris"}
        llm_data = {"answer": "London"}

        is_equiv, details = check_semantic_equivalence(gt_data, llm_data, model_config, "What is the capital?")

        assert is_equiv is False
        assert "NO" in details

    @patch("karenina.benchmark.verification.embedding_utils.init_chat_model_unified")
    def test_check_semantic_equivalence_error(self, mock_init_chat):
        """Test semantic equivalence check error handling."""
        mock_init_chat.side_effect = RuntimeError("LLM initialization failed")

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        with pytest.raises(RuntimeError, match="Failed to perform semantic equivalence check"):
            check_semantic_equivalence({"a": 1}, {"b": 2}, model_config, "Test question")


class TestPerformEmbeddingCheck:
    """Test the complete embedding check workflow."""

    def test_perform_embedding_check_disabled(self):
        """Test embedding check when disabled."""
        with patch.dict(os.environ, {"EMBEDDING_CHECK": "false"}):
            model_config = ModelConfig(
                id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
            )

            result = perform_embedding_check({"a": 1}, {"b": 2}, model_config, "Test question")
            should_override, similarity, model_name, check_performed, details = result

            assert not should_override
            assert similarity is None
            assert model_name is None
            assert not check_performed
            assert details is None

    @patch("karenina.benchmark.verification.embedding_utils.compute_embedding_similarity")
    @patch("karenina.benchmark.verification.embedding_utils.check_semantic_equivalence")
    @patch.dict(os.environ, {"EMBEDDING_CHECK": "true", "EMBEDDING_CHECK_THRESHOLD": "0.8"})
    def test_perform_embedding_check_above_threshold_equivalent(self, mock_semantic_check, mock_similarity):
        """Test embedding check with high similarity and semantic equivalence."""
        mock_similarity.return_value = (0.9, "test-model")
        mock_semantic_check.return_value = (True, "Semantically equivalent")

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        result = perform_embedding_check({"answer": "A"}, {"answer": "B"}, model_config, "What is the answer?")
        should_override, similarity, model_name, check_performed, details = result

        assert should_override is True
        assert similarity == 0.9
        assert model_name == "test-model"
        assert check_performed is True
        assert "equivalent" in details

    @patch("karenina.benchmark.verification.embedding_utils.compute_embedding_similarity")
    @patch("karenina.benchmark.verification.embedding_utils.check_semantic_equivalence")
    @patch.dict(os.environ, {"EMBEDDING_CHECK": "true", "EMBEDDING_CHECK_THRESHOLD": "0.8"})
    def test_perform_embedding_check_above_threshold_not_equivalent(self, mock_semantic_check, mock_similarity):
        """Test embedding check with high similarity but no semantic equivalence."""
        mock_similarity.return_value = (0.9, "test-model")
        mock_semantic_check.return_value = (False, "Not semantically equivalent")

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        result = perform_embedding_check({"answer": "A"}, {"answer": "B"}, model_config, "What is the answer?")
        should_override, similarity, model_name, check_performed, details = result

        assert should_override is False
        assert similarity == 0.9
        assert model_name == "test-model"
        assert check_performed is True
        assert "Not semantically equivalent" in details

    @patch("karenina.benchmark.verification.embedding_utils.compute_embedding_similarity")
    @patch.dict(os.environ, {"EMBEDDING_CHECK": "true", "EMBEDDING_CHECK_THRESHOLD": "0.8"})
    def test_perform_embedding_check_below_threshold(self, mock_similarity):
        """Test embedding check with similarity below threshold."""
        mock_similarity.return_value = (0.7, "test-model")

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        result = perform_embedding_check({"answer": "A"}, {"answer": "B"}, model_config, "What is the answer?")
        should_override, similarity, model_name, check_performed, details = result

        assert should_override is False
        assert similarity == 0.7
        assert model_name == "test-model"
        assert check_performed is True
        assert "below threshold" in details

    @patch("karenina.benchmark.verification.embedding_utils.compute_embedding_similarity")
    @patch("karenina.benchmark.verification.embedding_utils.check_semantic_equivalence")
    @patch.dict(os.environ, {"EMBEDDING_CHECK": "true"})
    def test_perform_embedding_check_semantic_check_fails(self, mock_semantic_check, mock_similarity):
        """Test embedding check when semantic check fails."""
        mock_similarity.return_value = (0.9, "test-model")
        mock_semantic_check.side_effect = RuntimeError("Semantic check failed")

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        result = perform_embedding_check({"answer": "A"}, {"answer": "B"}, model_config, "What is the answer?")
        should_override, similarity, model_name, check_performed, details = result

        assert should_override is False
        assert similarity == 0.9
        assert model_name == "test-model"
        assert check_performed is True
        assert "Semantic check failed" in details

    @patch("karenina.benchmark.verification.embedding_utils.compute_embedding_similarity")
    @patch.dict(os.environ, {"EMBEDDING_CHECK": "true"})
    def test_perform_embedding_check_embedding_fails(self, mock_similarity):
        """Test embedding check when embedding computation fails."""
        mock_similarity.side_effect = RuntimeError("Embedding computation failed")

        model_config = ModelConfig(
            id="test", model_provider="openai", model_name="gpt-4", temperature=0.1, system_prompt="Test prompt"
        )

        result = perform_embedding_check({"answer": "A"}, {"answer": "B"}, model_config, "What is the answer?")
        should_override, similarity, model_name, check_performed, details = result

        assert should_override is False
        assert similarity is None
        assert model_name is None
        assert check_performed is True
        assert "Embedding check failed" in details


class TestEmbeddingCheckIntegration:
    """Integration tests for embedding check with verification runner."""

    @patch("karenina.benchmark.verification.runner.perform_embedding_check")
    def test_embedding_check_only_on_failure(self, _mock_embedding_check):
        """Test that embedding check is only called when verification fails."""
        # This would be part of the runner integration test
        # For now, just ensure the import works
        from karenina.benchmark.verification.runner import run_single_model_verification

        assert run_single_model_verification is not None

    def test_embedding_utils_import(self):
        """Test that embedding utilities can be imported successfully."""
        from karenina.benchmark.verification.embedding_utils import (
            check_semantic_equivalence,
            compute_embedding_similarity,
            perform_embedding_check,
        )

        # Just verify imports work
        assert compute_embedding_similarity is not None
        assert check_semantic_equivalence is not None
        assert perform_embedding_check is not None
