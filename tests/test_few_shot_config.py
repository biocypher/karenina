"""Tests for the new FewShotConfig system."""

import pytest

from karenina.benchmark.models import FewShotConfig, ModelConfig, QuestionFewShotConfig, VerificationConfig


class TestFewShotConfig:
    """Test the FewShotConfig class functionality."""

    def test_from_index_selections(self) -> None:
        """Test creating FewShotConfig from index selections."""
        selections = {
            "question_1": [0, 2, 4],
            "question_2": [1, 3],
        }

        config = FewShotConfig.from_index_selections(selections)

        assert config.global_mode == "custom"
        assert config.enabled is True
        assert len(config.question_configs) == 2

        q1_config = config.question_configs["question_1"]
        assert q1_config.mode == "custom"
        assert q1_config.selected_examples == [0, 2, 4]

        q2_config = config.question_configs["question_2"]
        assert q2_config.mode == "custom"
        assert q2_config.selected_examples == [1, 3]

    def test_from_hash_selections(self) -> None:
        """Test creating FewShotConfig from hash selections."""
        selections = {
            "question_1": ["abc123def456789", "ghi789jkl012345"],
            "question_2": ["mno345pqr678901"],
        }

        config = FewShotConfig.from_hash_selections(selections)

        assert config.global_mode == "custom"
        assert config.enabled is True
        assert len(config.question_configs) == 2

        q1_config = config.question_configs["question_1"]
        assert q1_config.mode == "custom"
        assert q1_config.selected_examples == ["abc123def456789", "ghi789jkl012345"]

    def test_k_shot_for_questions(self) -> None:
        """Test creating FewShotConfig with different k values per question."""
        k_mapping = {
            "question_1": 5,
            "question_2": 2,
        }

        config = FewShotConfig.k_shot_for_questions(k_mapping, global_k=3)

        assert config.global_mode == "k-shot"
        assert config.global_k == 3
        assert config.enabled is True

        q1_config = config.question_configs["question_1"]
        assert q1_config.mode == "k-shot"
        assert q1_config.k == 5

        q2_config = config.question_configs["question_2"]
        assert q2_config.mode == "k-shot"
        assert q2_config.k == 2

    def test_add_selections_by_index(self) -> None:
        """Test dynamically adding selections by index."""
        config = FewShotConfig()

        config.add_selections_by_index(
            {
                "question_1": [0, 1, 2],
                "question_2": [3, 4],
            }
        )

        assert len(config.question_configs) == 2
        assert config.question_configs["question_1"].selected_examples == [0, 1, 2]
        assert config.question_configs["question_2"].selected_examples == [3, 4]

    def test_add_selections_by_hash(self) -> None:
        """Test dynamically adding selections by hash."""
        config = FewShotConfig()

        config.add_selections_by_hash(
            {
                "question_1": ["hash1", "hash2"],
            }
        )

        assert len(config.question_configs) == 1
        assert config.question_configs["question_1"].selected_examples == ["hash1", "hash2"]

    def test_get_effective_config_inherit(self) -> None:
        """Test getting effective config with inheritance."""
        config = FewShotConfig(
            global_mode="k-shot",
            global_k=5,
        )

        # Question with no specific config should inherit
        effective = config.get_effective_config("question_1")
        assert effective.mode == "k-shot"
        assert effective.k == 5

    def test_get_effective_config_override(self) -> None:
        """Test getting effective config with overrides."""
        config = FewShotConfig(
            global_mode="k-shot",
            global_k=5,
            question_configs={
                "question_1": QuestionFewShotConfig(
                    mode="custom",
                    selected_examples=[0, 1],
                )
            },
        )

        effective = config.get_effective_config("question_1")
        assert effective.mode == "custom"
        assert effective.selected_examples == [0, 1]

    def test_resolve_examples_all_mode(self) -> None:
        """Test resolving examples in 'all' mode."""
        config = FewShotConfig(global_mode="all")
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
        ]

        resolved = config.resolve_examples_for_question("q1", available_examples)
        assert len(resolved) == 2
        assert resolved == available_examples

    def test_resolve_examples_k_shot_mode(self) -> None:
        """Test resolving examples in 'k-shot' mode."""
        config = FewShotConfig(global_mode="k-shot", global_k=1)
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

        resolved = config.resolve_examples_for_question("q1", available_examples)
        assert len(resolved) == 1
        assert resolved[0] in available_examples

    def test_resolve_examples_custom_mode_index(self) -> None:
        """Test resolving examples in 'custom' mode with indices."""
        config = FewShotConfig.from_index_selections(
            {
                "q1": [0, 2],
            }
        )
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

        resolved = config.resolve_examples_for_question("q1", available_examples)
        assert len(resolved) == 2
        assert resolved[0] == available_examples[0]
        assert resolved[1] == available_examples[2]

    def test_resolve_examples_custom_mode_hash(self) -> None:
        """Test resolving examples in 'custom' mode with hashes."""
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
        ]

        # Generate correct hashes for the questions
        FewShotConfig.generate_example_hash("What is 1+1?")
        hash2 = FewShotConfig.generate_example_hash("What is 2+2?")

        config = FewShotConfig.from_hash_selections(
            {
                "q1": [hash2],  # Select only the second example
            }
        )

        resolved = config.resolve_examples_for_question("q1", available_examples, "dummy")
        assert len(resolved) == 1
        assert resolved[0] == available_examples[1]

    def test_resolve_examples_with_exclusions(self) -> None:
        """Test resolving examples with exclusions."""
        config = FewShotConfig(
            global_mode="all",
            question_configs={
                "q1": QuestionFewShotConfig(
                    mode="all",
                    excluded_examples=[1],  # Exclude index 1
                )
            },
        )
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

        resolved = config.resolve_examples_for_question("q1", available_examples)
        assert len(resolved) == 2
        assert available_examples[1] not in resolved

    def test_resolve_examples_with_external_examples(self) -> None:
        """Test resolving examples with external examples."""
        config = FewShotConfig(
            global_mode="custom",
            global_external_examples=[{"question": "Global example", "answer": "Global answer"}],
            question_configs={
                "q1": QuestionFewShotConfig(
                    mode="custom",
                    selected_examples=[0],
                    external_examples=[{"question": "Question example", "answer": "Question answer"}],
                )
            },
        )
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
        ]

        resolved = config.resolve_examples_for_question("q1", available_examples)
        assert len(resolved) == 3  # 1 selected + 1 question external + 1 global external
        assert {"question": "What is 1+1?", "answer": "2"} in resolved
        assert {"question": "Question example", "answer": "Question answer"} in resolved
        assert {"question": "Global example", "answer": "Global answer"} in resolved

    def test_generate_example_hash(self) -> None:
        """Test MD5 hash generation for examples."""
        hash1 = FewShotConfig.generate_example_hash("What is 1+1?")
        hash2 = FewShotConfig.generate_example_hash("What is 1+1?")
        hash3 = FewShotConfig.generate_example_hash("What is 2+2?")

        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
        assert len(hash1) == 32  # MD5 hash length

    def test_validate_selections(self) -> None:
        """Test validation of selections against available examples."""
        config = FewShotConfig.from_index_selections(
            {
                "q1": [0, 5],  # Index 5 will be out of range
                "q2": [1, 2],
            }
        )

        question_examples = {
            "q1": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ],  # Only indices 0, 1 available
            "q2": [
                {"question": "Q3", "answer": "A3"},
                {"question": "Q4", "answer": "A4"},
                {"question": "Q5", "answer": "A5"},
            ],
        }

        errors = config.validate_selections(question_examples)
        assert len(errors) == 1
        assert "Index 5 out of range" in errors[0]
        assert "q1" in errors[0]

    def test_disabled_config(self) -> None:
        """Test that disabled config returns no examples."""
        config = FewShotConfig(enabled=False, global_mode="all")
        available_examples = [
            {"question": "What is 1+1?", "answer": "2"},
        ]

        resolved = config.resolve_examples_for_question("q1", available_examples)
        assert len(resolved) == 0


class TestVerificationConfigIntegration:
    """Test integration with VerificationConfig."""

    def test_verification_config_with_few_shot_config(self) -> None:
        """Test VerificationConfig with new FewShotConfig."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        few_shot_config = FewShotConfig.from_index_selections(
            {
                "question_1": [0, 1, 2],
            }
        )

        config = VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
            few_shot_config=few_shot_config,
        )

        assert config.few_shot_config is not None
        assert config.is_few_shot_enabled() is True
        assert config.get_few_shot_config() == few_shot_config

    def test_legacy_few_shot_compatibility(self) -> None:
        """Test backward compatibility with legacy few-shot fields."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        # Use legacy fields
        config = VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
            few_shot_enabled=True,
            few_shot_mode="k-shot",
            few_shot_k=5,
        )

        # Should convert to new FewShotConfig automatically
        few_shot_config = config.get_few_shot_config()
        assert few_shot_config is not None
        assert few_shot_config.enabled is True
        assert few_shot_config.global_mode == "k-shot"
        assert few_shot_config.global_k == 5

    def test_legacy_custom_mode_support(self) -> None:
        """Test that legacy 'custom' mode works correctly."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        config = VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
            few_shot_enabled=True,
            few_shot_mode="custom",  # Legacy mode
            few_shot_k=3,
        )

        few_shot_config = config.get_few_shot_config()
        assert few_shot_config is not None
        assert few_shot_config.global_mode == "custom"

    def test_few_shot_validation_new_config(self) -> None:
        """Test validation with new FewShotConfig."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        # Invalid k value should raise error
        few_shot_config = FewShotConfig(
            global_mode="k-shot",
            global_k=0,  # Invalid
        )

        with pytest.raises(ValueError, match="Global few-shot k value must be at least 1"):
            VerificationConfig(
                answering_models=[answering_model],
                parsing_models=[parsing_model],
                few_shot_config=few_shot_config,
            )

    def test_is_few_shot_enabled_methods(self) -> None:
        """Test the few-shot enabled check methods."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test answering prompt",
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="test",
            model_name="test-model",
            temperature=0.1,
            interface="langchain",
            system_prompt="Test parsing prompt",
        )

        # Test with new config enabled
        config1 = VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
            few_shot_config=FewShotConfig(enabled=True),
        )
        assert config1.is_few_shot_enabled() is True

        # Test with new config disabled
        config2 = VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
            few_shot_config=FewShotConfig(enabled=False),
        )
        assert config2.is_few_shot_enabled() is False

        # Test with no config
        config3 = VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
        )
        assert config3.is_few_shot_enabled() is False
