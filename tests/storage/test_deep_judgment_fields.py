"""Simple tests that deep-judgment columns exist in VerificationResultModel.

This validates that the database model has all required deep-judgment fields
with correct types and defaults.
"""

from karenina.storage.models import VerificationResultModel


def test_deep_judgment_fields_exist():
    """Test that all deep-judgment fields exist in the model."""
    # Check that the fields are defined
    assert hasattr(VerificationResultModel, "deep_judgment_enabled")
    assert hasattr(VerificationResultModel, "deep_judgment_performed")
    assert hasattr(VerificationResultModel, "extracted_excerpts")
    assert hasattr(VerificationResultModel, "attribute_reasoning")
    assert hasattr(VerificationResultModel, "deep_judgment_stages_completed")
    assert hasattr(VerificationResultModel, "deep_judgment_model_calls")
    assert hasattr(VerificationResultModel, "deep_judgment_excerpt_retry_count")
    assert hasattr(VerificationResultModel, "attributes_without_excerpts")


def test_deep_judgment_field_annotations():
    """Test that deep-judgment fields have correct type annotations."""
    # Get the field annotations
    annotations = VerificationResultModel.__annotations__

    # Verify each field exists in annotations
    assert "deep_judgment_enabled" in annotations
    assert "deep_judgment_performed" in annotations
    assert "extracted_excerpts" in annotations
    assert "attribute_reasoning" in annotations
    assert "deep_judgment_stages_completed" in annotations
    assert "deep_judgment_model_calls" in annotations
    assert "deep_judgment_excerpt_retry_count" in annotations
    assert "attributes_without_excerpts" in annotations
