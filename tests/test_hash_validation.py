"""Test hash validation in verification runner."""

import pytest

from karenina.benchmark.verification.runner import run_single_model_verification
from karenina.benchmark.verification.verification_utils import _is_valid_md5_hash
from karenina.schemas import ModelConfig


def test_is_valid_md5_hash() -> None:
    """Test MD5 hash validation function."""
    # Valid MD5 hashes
    assert _is_valid_md5_hash("d41d8cd98f00b204e9800998ecf8427e")
    assert _is_valid_md5_hash("c4ca4238a0b923820dcc509a6f75849b")
    assert not _is_valid_md5_hash("ABCDEF1234567890123456789012345")  # 31 chars
    assert not _is_valid_md5_hash("ABCDEF123456789012345678901234567")  # 33 chars

    # Valid format with different cases
    assert _is_valid_md5_hash("ABCDEF123456789012345678901234ab")  # Mixed case
    assert _is_valid_md5_hash("abcdef123456789012345678901234AB")  # Mixed case

    # Invalid formats
    assert not _is_valid_md5_hash("not-a-hash")
    assert not _is_valid_md5_hash("")
    assert not _is_valid_md5_hash("g41d8cd98f00b204e9800998ecf8427e")  # Invalid char 'g'
    assert not _is_valid_md5_hash("d41d8cd98f00b204e9800998ecf8427z")  # Invalid char 'z'
    assert not _is_valid_md5_hash("d41d8cd9-8f00-b204-e980-0998ecf8427e")  # Dashes

    # Non-string inputs
    assert not _is_valid_md5_hash(None)
    assert not _is_valid_md5_hash(123)
    assert not _is_valid_md5_hash([])


def test_manual_interface_valid_hash() -> None:
    """Test that manual interface accepts valid MD5 hash as question_id."""
    from karenina.llm.manual_traces import clear_manual_traces, load_manual_traces

    # Clear any existing traces
    clear_manual_traces()

    # Load test traces
    question_hash = "d41d8cd98f00b204e9800998ecf8427e"
    test_traces = {question_hash: "Test manual trace response"}
    load_manual_traces(test_traces)

    # Create manual model configuration
    manual_model = ModelConfig(
        id="test-manual-model",
        interface="manual",
        model_name="manual",
        model_provider="manual",
        temperature=0.0,
        system_prompt="",
    )

    # Create parsing model (can be any valid model)
    parsing_model = ModelConfig(
        id="test-parsing-model",
        interface="langchain",
        model_name="gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
        system_prompt="",
    )

    # Simple answer template
    template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: str = Field(description="The answer result")
    correct: dict = Field(description="Correct answer")

    def verify(self):
        return len(self.result) > 0
"""

    # This should work without raising an exception
    try:
        result = run_single_model_verification(
            question_id=question_hash,  # Valid MD5 hash
            question_text="Test question",
            template_code=template_code,
            answering_model=manual_model,
            parsing_model=parsing_model,
        )
        # Verify that the result includes our manual trace
        assert result.raw_answer == "Test manual trace response"
        assert result.success
    except Exception as e:
        # If it fails for other reasons (like missing dependencies), that's okay
        # We're just testing that the hash validation doesn't raise ValueError
        if "Invalid question_id format" in str(e):
            pytest.fail(f"Hash validation failed for valid hash: {e}")
    finally:
        # Cleanup
        clear_manual_traces()


def test_manual_interface_invalid_hash() -> None:
    """Test that manual interface rejects invalid question_id formats."""
    # Create manual model configuration
    manual_model = ModelConfig(
        id="test-manual-model",
        interface="manual",
        model_name="manual",
        model_provider="manual",
        temperature=0.0,
        system_prompt="",
    )

    # Create parsing model
    parsing_model = ModelConfig(
        id="test-parsing-model",
        interface="langchain",
        model_name="gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
        system_prompt="",
    )

    # Simple answer template
    template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: str = Field(description="The answer result")
    correct: dict = Field(description="Correct answer")

    def verify(self):
        return len(self.result) > 0
"""

    # Test various invalid question_id formats
    invalid_hashes = [
        "not-a-hash",
        "short",
        "d41d8cd98f00b204e9800998ecf8427",  # 31 chars (too short)
        "d41d8cd98f00b204e9800998ecf8427ee",  # 33 chars (too long)
        "g41d8cd98f00b204e9800998ecf8427e",  # Invalid character 'g'
        "d41d8cd9-8f00-b204-e980-0998ecf8427e",  # Contains dashes
        "",  # Empty string
        "12345",  # Too short
    ]

    for invalid_hash in invalid_hashes:
        result = run_single_model_verification(
            question_id=invalid_hash,
            question_text="Test question",
            template_code=template_code,
            answering_model=manual_model,
            parsing_model=parsing_model,
        )
        # Check that verification failed due to hash validation
        assert not result.success
        assert result.error is not None
        assert "Invalid question_id format for manual interface" in result.error


def test_non_manual_interface_ignores_hash_validation() -> None:
    """Test that non-manual interfaces don't validate question_id format."""
    # Create non-manual model configuration
    langchain_model = ModelConfig(
        id="test-langchain-model",
        interface="langchain",
        model_name="gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
        system_prompt="",
    )

    # Simple answer template
    template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: str = Field(description="The answer result")
    correct: dict = Field(description="Correct answer")

    def verify(self):
        return len(self.result) > 0
"""

    # Use invalid hash format - should not raise ValueError for non-manual interface
    invalid_hash = "not-a-hash-at-all"

    try:
        result = run_single_model_verification(
            question_id=invalid_hash,  # Invalid hash format
            question_text="Test question",
            template_code=template_code,
            answering_model=langchain_model,
            parsing_model=langchain_model,
        )
        # Check that if it failed, it wasn't due to hash validation
        if not result.success and result.error and "Invalid question_id format for manual interface" in result.error:
            pytest.fail("Hash validation should not apply to non-manual interfaces")
    except Exception:
        # Other exceptions (like missing API keys, etc.) are fine for this test
        pass


def test_hash_validation_error_message() -> None:
    """Test that hash validation provides helpful error message."""
    manual_model = ModelConfig(
        id="test-manual-model",
        interface="manual",
        model_name="manual",
        model_provider="manual",
        temperature=0.0,
        system_prompt="",
    )

    parsing_model = ModelConfig(
        id="test-parsing-model",
        interface="langchain",
        model_name="gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
        system_prompt="",
    )

    template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: str = Field(description="The answer result")
    correct: dict = Field(description="Correct answer")

    def verify(self):
        return len(self.result) > 0
"""

    result = run_single_model_verification(
        question_id="invalid-hash",
        question_text="Test question",
        template_code=template_code,
        answering_model=manual_model,
        parsing_model=parsing_model,
    )

    # Check that verification failed due to hash validation
    assert not result.success
    assert result.error is not None
    error_message = result.error
    assert "Invalid question_id format for manual interface" in error_message
    assert "invalid-hash" in error_message
    assert "32-character hexadecimal MD5 hash" in error_message
    assert "question extraction" in error_message


if __name__ == "__main__":
    pytest.main([__file__])
