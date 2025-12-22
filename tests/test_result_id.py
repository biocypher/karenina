"""Tests for deterministic verification result ID computation."""

import json

import pytest

from karenina.schemas.workflow.verification.result_components import (
    VerificationResultMetadata,
)


class TestComputeResultId:
    """Tests for VerificationResultMetadata.compute_result_id()."""

    def test_deterministic_same_inputs_same_id(self):
        """Same inputs should always produce the same ID."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id2 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        assert id1 == id2

    def test_different_question_id_different_id(self):
        """Different question_id should produce different ID."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id2 = VerificationResultMetadata.compute_result_id(
            question_id="q456",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        assert id1 != id2

    def test_different_timestamp_different_id(self):
        """Different timestamp should produce different ID."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id2 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:31:00",
        )
        assert id1 != id2

    def test_different_answering_model_different_id(self):
        """Different answering_model should produce different ID."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id2 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="openai/gpt-4o",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        assert id1 != id2

    def test_different_parsing_model_different_id(self):
        """Different parsing_model should produce different ID."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id2 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="openai/gpt-4o-mini",
            timestamp="2025-12-15 10:30:00",
        )
        assert id1 != id2

    def test_null_replicates_handled(self):
        """None replicates should be handled correctly."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            replicate=None,
        )
        # Should produce a valid 16-char hex string
        assert len(id1) == 16
        assert all(c in "0123456789abcdef" for c in id1)

    def test_replicates_affect_id(self):
        """Different replicate values should produce different IDs."""
        id_no_rep = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id_rep1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            replicate=1,
        )
        id_rep2 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            replicate=2,
        )
        assert id_no_rep != id_rep1
        assert id_rep1 != id_rep2

    def test_mcp_servers_affect_id(self):
        """Different MCP servers should produce different IDs."""
        id_no_mcp = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        id_with_mcp = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            answering_mcp_servers=["brave_search"],
        )
        assert id_no_mcp != id_with_mcp

    def test_mcp_servers_order_independent(self):
        """MCP server order should not affect ID (servers are sorted)."""
        id1 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            answering_mcp_servers=["brave_search", "biocontext"],
        )
        id2 = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            answering_mcp_servers=["biocontext", "brave_search"],
        )
        assert id1 == id2

    def test_empty_mcp_servers_same_as_none(self):
        """Empty MCP server list should be same as None."""
        id_none = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            answering_mcp_servers=None,
        )
        id_empty = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
            answering_mcp_servers=[],
        )
        assert id_none == id_empty

    def test_id_is_16_char_hex(self):
        """Result ID should be exactly 16 hex characters."""
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )
        assert len(result_id) == 16
        assert all(c in "0123456789abcdef" for c in result_id)


class TestVerificationResultMetadataWithResultId:
    """Tests for VerificationResultMetadata with result_id field."""

    def test_metadata_requires_result_id(self):
        """VerificationResultMetadata should require result_id field."""
        # Compute result_id first
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )

        # Create metadata with result_id
        metadata = VerificationResultMetadata(
            question_id="q123",
            template_id="template_abc",
            completed_without_errors=True,
            question_text="What is the answer?",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-12-15 10:30:00",
            result_id=result_id,
        )

        assert metadata.result_id == result_id
        assert len(metadata.result_id) == 16

    def test_metadata_without_result_id_raises(self):
        """Creating metadata without result_id should raise validation error."""
        with pytest.raises(ValueError):  # Pydantic validation error
            VerificationResultMetadata(
                question_id="q123",
                template_id="template_abc",
                completed_without_errors=True,
                question_text="What is the answer?",
                answering_model="anthropic/claude-haiku-4-5",
                parsing_model="anthropic/claude-haiku-4-5",
                execution_time=1.5,
                timestamp="2025-12-15 10:30:00",
                # Missing result_id
            )

    def test_metadata_serialization_includes_result_id(self):
        """Serialized metadata should include result_id field."""
        result_id = VerificationResultMetadata.compute_result_id(
            question_id="q123",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            timestamp="2025-12-15 10:30:00",
        )

        metadata = VerificationResultMetadata(
            question_id="q123",
            template_id="template_abc",
            completed_without_errors=True,
            question_text="What is the answer?",
            answering_model="anthropic/claude-haiku-4-5",
            parsing_model="anthropic/claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-12-15 10:30:00",
            result_id=result_id,
        )

        serialized = metadata.model_dump()
        assert "result_id" in serialized
        assert serialized["result_id"] == result_id

        # Also check JSON serialization
        json_str = metadata.model_dump_json()
        parsed = json.loads(json_str)
        assert "result_id" in parsed
        assert parsed["result_id"] == result_id
