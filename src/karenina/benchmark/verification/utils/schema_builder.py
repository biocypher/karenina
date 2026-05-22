"""Build JSON schemas for the judge LLM, filtering out verification metadata.

Two filtering operations:
1. Remove trace fields (judge should not parse these)
2. Strip __verification__ metadata (judge must not see ground truth)

Also exposes ``build_extraction_relaxed_class`` which produces a sibling
Pydantic model with every VerifiedField coerced to ``Optional[T] = None``.
This relaxed class is used at agentic-extraction time so that a single
null field from the LLM does not reject the entire record. The caller
recombines null-valued fields into the strict template afterwards via
``BaseAnswer.model_construct`` plus a private ``_null_fields`` set.
"""

import logging
from typing import Any

from pydantic import create_model

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

    from karenina.schemas.primitives import TracePrimitive
    from karenina.schemas.primitives.registry import _reconstruct_primitive

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


def build_extraction_relaxed_class(answer_class: type[BaseAnswer]) -> type[BaseAnswer]:
    """Build a sibling Pydantic class with VerifiedFields relaxed to Optional.

    Each VerifiedField on ``answer_class`` is rewritten as ``Optional[T] = None``
    so that the agentic parser accepts ``null`` for any single field without
    failing the whole record. The relaxed class is used only at extraction
    time; downstream verification then reconstructs the strict template and
    tracks which fields came back null via a private ``_null_fields`` set.

    Non-VerifiedField fields (e.g. ``id``) and unverified plain fields are
    preserved as-is; only fields that carry ``__verification__`` metadata are
    loosened, because only those participate in scoring.

    Args:
        answer_class: The strict template class to derive from.

    Returns:
        A new Pydantic ``BaseAnswer`` subclass with relaxed field types.
    """
    relaxed_fields: dict[str, Any] = {}
    for name, field_info in answer_class.model_fields.items():
        extra = field_info.json_schema_extra
        is_verified = isinstance(extra, dict) and "__verification__" in extra

        if not is_verified:
            # Preserve non-verified fields verbatim
            relaxed_fields[name] = (field_info.annotation, field_info)
            continue

        # Coerce the annotation to Optional[T] with default None.
        annotation = field_info.annotation or Any
        relaxed_annotation = annotation | None
        relaxed_fields[name] = (relaxed_annotation, None)

    relaxed_cls = create_model(
        f"{answer_class.__name__}__Relaxed",
        __base__=BaseAnswer,
        **relaxed_fields,
    )
    return relaxed_cls


def rebuild_strict_answer_with_null_fields(
    answer_class: type[BaseAnswer],
    relaxed_instance: Any,
) -> BaseAnswer:
    """Rebuild a strict Answer while preserving null verified fields.

    Parser adapters receive the relaxed class from
    ``build_extraction_relaxed_class``. This helper converts the parsed relaxed
    instance back to the original strict answer class, omitting null field
    values and storing the names of verified null fields on ``_null_fields``.
    ``BaseAnswer._compute_field_results`` consumes that marker and reports
    those fields as ``None`` instead of conflating them with ``False``.
    """
    extracted_dict = relaxed_instance.model_dump()
    verified_names = set(answer_class._get_verified_fields().keys())
    null_fields = {name for name, value in extracted_dict.items() if value is None and name in verified_names}
    strict_payload = {name: value for name, value in extracted_dict.items() if value is not None}
    strict_instance = answer_class.model_construct(**strict_payload)
    strict_instance.__dict__["_null_fields"] = null_fields
    return strict_instance
