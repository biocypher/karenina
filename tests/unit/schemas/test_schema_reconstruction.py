"""Tests for JSON Schema to Pydantic model reconstruction."""

import pytest
from pydantic import BaseModel, Field

from karenina.schemas.entities._schema_reconstruction import (
    _reconstruct_model_from_schema,
)


@pytest.mark.unit
class TestReconstructModelFromSchema:
    def test_roundtrip_primitives(self):
        class Original(BaseModel):
            count: int = Field(description="A count")
            name: str = Field(description="A name")
            score: float = Field(description="A score")
            found: bool = Field(description="A flag")

        schema = Original.model_json_schema()
        Rebuilt = _reconstruct_model_from_schema(schema)

        instance = Rebuilt(count=5, name="test", score=3.14, found=True)
        assert instance.count == 5
        assert instance.name == "test"
        assert instance.score == 3.14
        assert instance.found is True

    def test_roundtrip_list_fields(self):
        class Original(BaseModel):
            items: list[str] = Field(description="Items")
            counts: list[int] = Field(description="Counts")

        schema = Original.model_json_schema()
        Rebuilt = _reconstruct_model_from_schema(schema)

        instance = Rebuilt(items=["a", "b"], counts=[1, 2])
        assert instance.items == ["a", "b"]
        assert instance.counts == [1, 2]

    def test_roundtrip_optional_with_default(self):
        class Original(BaseModel):
            score: float | None = None

        schema = Original.model_json_schema()
        Rebuilt = _reconstruct_model_from_schema(schema)

        instance = Rebuilt()
        assert instance.score is None

        instance2 = Rebuilt(score=3.5)
        assert instance2.score == 3.5

    def test_preserves_title(self):
        class MyFindings(BaseModel):
            x: int

        schema = MyFindings.model_json_schema()
        Rebuilt = _reconstruct_model_from_schema(schema)
        assert Rebuilt.__name__ == "MyFindings"

    def test_model_dump_roundtrip(self):
        class Original(BaseModel):
            count: int = Field(description="N")
            items: list[str] = Field(description="L")
            flag: bool | None = None

        schema = Original.model_json_schema()
        Rebuilt = _reconstruct_model_from_schema(schema)

        data = {"count": 3, "items": ["a"], "flag": True}
        orig_instance = Original(**data)
        rebuilt_instance = Rebuilt(**data)
        assert orig_instance.model_dump() == rebuilt_instance.model_dump()
