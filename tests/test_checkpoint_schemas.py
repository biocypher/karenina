"""Test checkpoint schema validation."""

import pytest
from pydantic import ValidationError

from karenina.schemas.checkpoint import SCHEMA_ORG_CONTEXT, JsonLdCheckpoint, SchemaOrgPerson


def test_jsonld_checkpoint_with_string_creator() -> None:
    """Test JsonLdCheckpoint accepts string creator."""
    checkpoint = JsonLdCheckpoint(
        **{
            "@context": SCHEMA_ORG_CONTEXT,
            "@type": "DataFeed",
            "name": "Test Benchmark",
            "description": "Test description",
            "version": "1.0.0",
            "creator": "Karenina Benchmarking System",
            "dateCreated": "2025-10-10T10:00:00Z",
            "dateModified": "2025-10-10T10:00:00Z",
            "dataFeedElement": [],
        }
    )
    assert checkpoint.name == "Test Benchmark"
    assert checkpoint.creator == "Karenina Benchmarking System"
    assert isinstance(checkpoint.creator, str)


def test_jsonld_checkpoint_with_person_creator() -> None:
    """Test JsonLdCheckpoint accepts SchemaOrgPerson creator."""
    person_creator = SchemaOrgPerson(
        name="Karenina Benchmarking System",
        url="https://github.com/karenina",
        email="noreply@karenina.dev",
    )

    checkpoint = JsonLdCheckpoint(
        **{
            "@context": SCHEMA_ORG_CONTEXT,
            "@type": "DataFeed",
            "name": "Test Benchmark",
            "description": "Test description",
            "version": "1.0.0",
            "creator": person_creator.model_dump(by_alias=True),
            "dateCreated": "2025-10-10T10:00:00Z",
            "dateModified": "2025-10-10T10:00:00Z",
            "dataFeedElement": [],
        }
    )
    assert checkpoint.name == "Test Benchmark"
    assert isinstance(checkpoint.creator, SchemaOrgPerson)
    assert checkpoint.creator.name == "Karenina Benchmarking System"
    assert checkpoint.creator.url == "https://github.com/karenina"
    assert checkpoint.creator.email == "noreply@karenina.dev"


def test_jsonld_checkpoint_with_none_creator() -> None:
    """Test JsonLdCheckpoint accepts None creator."""
    checkpoint = JsonLdCheckpoint(
        **{
            "@context": SCHEMA_ORG_CONTEXT,
            "@type": "DataFeed",
            "name": "Test Benchmark",
            "description": "Test description",
            "version": "1.0.0",
            "creator": None,
            "dateCreated": "2025-10-10T10:00:00Z",
            "dateModified": "2025-10-10T10:00:00Z",
            "dataFeedElement": [],
        }
    )
    assert checkpoint.name == "Test Benchmark"
    assert checkpoint.creator is None


def test_jsonld_checkpoint_creator_serialization() -> None:
    """Test that creator serializes correctly in both formats."""
    # Test with string creator
    checkpoint1 = JsonLdCheckpoint(
        **{
            "@context": SCHEMA_ORG_CONTEXT,
            "@type": "DataFeed",
            "name": "Test Benchmark",
            "creator": "Test Creator",
            "dateCreated": "2025-10-10T10:00:00Z",
            "dateModified": "2025-10-10T10:00:00Z",
            "dataFeedElement": [],
        }
    )
    data1 = checkpoint1.model_dump(by_alias=True)
    assert data1["creator"] == "Test Creator"

    # Test with SchemaOrgPerson creator
    checkpoint2 = JsonLdCheckpoint(
        **{
            "@context": SCHEMA_ORG_CONTEXT,
            "@type": "DataFeed",
            "name": "Test Benchmark",
            "creator": {"@type": "Person", "name": "Test Person"},
            "dateCreated": "2025-10-10T10:00:00Z",
            "dateModified": "2025-10-10T10:00:00Z",
            "dataFeedElement": [],
        }
    )
    data2 = checkpoint2.model_dump(by_alias=True)
    assert isinstance(data2["creator"], dict)
    assert data2["creator"]["@type"] == "Person"
    assert data2["creator"]["name"] == "Test Person"


def test_jsonld_checkpoint_invalid_creator() -> None:
    """Test that invalid creator types are rejected."""
    with pytest.raises(ValidationError):
        JsonLdCheckpoint(
            **{
                "@context": SCHEMA_ORG_CONTEXT,
                "@type": "DataFeed",
                "name": "Test Benchmark",
                "creator": 123,  # Invalid: number
                "dateCreated": "2025-10-10T10:00:00Z",
                "dateModified": "2025-10-10T10:00:00Z",
                "dataFeedElement": [],
            }
        )

    with pytest.raises(ValidationError):
        JsonLdCheckpoint(
            **{
                "@context": SCHEMA_ORG_CONTEXT,
                "@type": "DataFeed",
                "name": "Test Benchmark",
                "creator": ["invalid", "list"],  # Invalid: list
                "dateCreated": "2025-10-10T10:00:00Z",
                "dateModified": "2025-10-10T10:00:00Z",
                "dataFeedElement": [],
            }
        )


def test_schema_org_person() -> None:
    """Test SchemaOrgPerson schema validation."""
    # Test valid person with all fields
    person = SchemaOrgPerson(
        name="Test Person",
        url="https://example.com",
        email="test@example.com",
    )
    assert person.type == "Person"
    assert person.name == "Test Person"
    assert person.url == "https://example.com"
    assert person.email == "test@example.com"

    # Test with only required field
    person = SchemaOrgPerson(name="Test Person")
    assert person.name == "Test Person"
    assert person.url is None
    assert person.email is None

    # Test validation error for missing name
    with pytest.raises(ValidationError):
        SchemaOrgPerson()
