"""Build JSON schemas for the judge LLM, filtering out verification metadata.

Two filtering operations:
1. Remove trace fields (judge should not parse these)
2. Strip __verification__ metadata (judge must not see ground truth)
"""

import logging
from typing import Any

from karenina.schemas.entities.answer import BaseAnswer

logger = logging.getLogger(__name__)


def build_parsing_schema(answer_class: type[BaseAnswer]) -> dict[str, Any]:
    """Build JSON schema for the judge, filtering out trace fields and metadata.

    Handles Pydantic v2 schema shapes including $defs and allOf references.

    Args:
        answer_class: The Answer class to generate a schema for.

    Returns:
        Filtered JSON schema dict safe to send to the judge LLM.
    """
    schema = answer_class.model_json_schema()
    verified = answer_class._get_verified_fields()

    if not verified:
        return schema

    from karenina.schemas.entities.primitives import TracePrimitive, _reconstruct_primitive

    trace_fields: set[str] = set()
    for name, meta in verified.items():
        primitive = _reconstruct_primitive(meta.verify_with)
        if isinstance(primitive, TracePrimitive):
            trace_fields.add(name)

    def _strip_properties(props: dict[str, Any]) -> None:
        """Remove trace fields and __verification__ from a properties dict."""
        for field_name in list(props.keys()):
            if field_name in trace_fields:
                del props[field_name]
            elif isinstance(props.get(field_name), dict):
                props[field_name].pop("__verification__", None)

    if "properties" in schema:
        _strip_properties(schema["properties"])

    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r not in trace_fields]

    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if "properties" in def_schema:
                _strip_properties(def_schema["properties"])

    return schema
