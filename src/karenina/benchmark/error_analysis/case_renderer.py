"""Render one VerificationResult (or one ScenarioExecutionResult) as a
Markdown case file with YAML frontmatter.

This module contains only the single-case rendering logic. Bucket
placement, filename sanitization, and directory orchestration are in
materializer.py (Task 8).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from karenina.replay.ports_message_hydration import hydrate_trace_messages
from karenina.scenario.handover import TaggedMessage, format_transcript
from karenina.scenario.trace_materialization import (
    DEFAULT_TRACE_TRUNCATION_THRESHOLD,
    format_turns_as_xml,
    group_entries_into_turns,
    parse_transcript_entries,
)
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import VerificationResultMetadata

logger = logging.getLogger(__name__)

TEMPLATE_INLINE_THRESHOLD_LINES = 100
TEMPLATE_EXCERPT_LINES = 20


def _qa_case_id(metadata: Any) -> str:
    """Build the case ID stamped into the frontmatter.

    Args:
        metadata: The VerificationResultMetadata for the case.

    Returns:
        A string like ``q_<question_id>`` or ``q_<question_id>__rep_<n>``
        when a replicate number is present.
    """
    base = f"q_{metadata.question_id}"
    if metadata.replicate is not None:
        return f"{base}__rep_{metadata.replicate}"
    return base


def _qa_frontmatter(result: VerificationResult) -> dict[str, Any]:
    """Compute the YAML frontmatter mapping for a QA case.

    Args:
        result: The VerificationResult being rendered.

    Returns:
        A plain dict ready for ``yaml.safe_dump``.
    """
    md = result.metadata
    failure = md.failure
    fm: dict[str, Any] = {
        "id": _qa_case_id(md),
        "outcome": "failure" if failure is not None else "pass",
        "question_id": md.question_id,
        "replicate": md.replicate,
        "model": md.answering.display_string,
        "parsing_model": md.parsing.display_string,
        "category": failure.category.value if failure else None,
        "group": failure.group.value if failure else None,
        "stage": failure.stage if failure else None,
        "rubric_scores": _collect_rubric_scores(result),
    }
    return fm


def _collect_rubric_scores(result: VerificationResult) -> dict[str, Any]:
    """Use the rubric's native get_all_trait_scores() aggregator.

    Args:
        result: The VerificationResult whose rubric scores to collect.

    Returns:
        An empty dict when result.rubric is None (rubric is Optional on
        VerificationResult) or when no trait scores are present.
    """
    if result.rubric is None:
        return {}
    # VerificationResultRubric.get_all_trait_scores() unions every
    # *_trait_scores dict into a single mapping, which is exactly what the
    # frontmatter wants.
    return dict(result.rubric.get_all_trait_scores())


def _render_template_section(template_source: str | None, template_link: str | None) -> str:
    """Render the Template section: link + inline source or excerpt.

    Args:
        template_source: Raw Python source of the answer template, or None.
        template_link: Relative path to the template file, or None.

    Returns:
        The rendered section as a string (empty when both inputs are None).
    """
    if template_source is None and template_link is None:
        return ""
    lines: list[str] = ["# Template", ""]
    if template_link:
        lines.append(f"Source: [{template_link}]({template_link})")
        lines.append("")
    if template_source is not None:
        source_lines = template_source.splitlines()
        if len(source_lines) <= TEMPLATE_INLINE_THRESHOLD_LINES:
            lines.append("```python")
            lines.extend(source_lines)
            lines.append("```")
        else:
            lines.append("```python")
            lines.extend(source_lines[:TEMPLATE_EXCERPT_LINES])
            lines.append("```")
            lines.append(f"_Excerpt: first {TEMPLATE_EXCERPT_LINES} lines of {len(source_lines)}._")
    lines.append("")
    return "\n".join(lines)


def _render_section(title: str, body: str) -> str:
    """Render a ``# Title`` section with the supplied body.

    Args:
        title: Heading text (without the leading ``#``).
        body: Body content; returns empty string when body is falsy.

    Returns:
        The full section string or an empty string.
    """
    if not body:
        return ""
    return f"# {title}\n\n{body}\n"


def _resolve_truncation_threshold(override: int | None) -> int:
    """Pick the effective truncation threshold for trace rendering.

    Args:
        override: Caller-supplied max_trace_chars; takes precedence over env.

    Returns:
        The chosen threshold, falling back to the env var and then to
        DEFAULT_TRACE_TRUNCATION_THRESHOLD when the env var is missing or
        invalid.
    """
    if override is not None:
        return override
    raw = os.environ.get("KARENINA_TRACE_TRUNCATION_THRESHOLD")
    if raw is None:
        return DEFAULT_TRACE_TRUNCATION_THRESHOLD
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid KARENINA_TRACE_TRUNCATION_THRESHOLD=%s, using default %d",
            raw,
            DEFAULT_TRACE_TRUNCATION_THRESHOLD,
        )
        return DEFAULT_TRACE_TRUNCATION_THRESHOLD


def _render_trace(
    trace_messages: list[dict[str, Any]],
    metadata: VerificationResultMetadata,
    artifacts_dir: Path | None,
    max_trace_chars: int | None,
) -> str:
    """Convert stored trace_messages into the XML turn body.

    Args:
        trace_messages: Stored raw message dicts from the template record.
        metadata: Case metadata; supplies the answering agent identity.
        artifacts_dir: Directory for offloaded long-message artifacts.
        max_trace_chars: Optional threshold override.

    Returns:
        An XML-formatted transcript string, or a stub when no messages
        are available.
    """
    if not trace_messages:
        return "_No trace captured for this case._"

    messages = hydrate_trace_messages(trace_messages)
    agent_id = metadata.answering.display_string
    tagged = [
        TaggedMessage(
            message=msg,
            agent_id="__user__" if msg.role.value == "user" else agent_id,
        )
        for msg in messages
    ]
    transcript = format_transcript(tagged)
    if not transcript:
        return "_No trace captured for this case._"
    entries = parse_transcript_entries(transcript)
    turns = group_entries_into_turns(entries)
    threshold = _resolve_truncation_threshold(max_trace_chars)
    xml = format_turns_as_xml(
        turns,
        artifacts_dir=artifacts_dir,
        truncation_threshold=threshold,
    )
    return xml


def _render_failure(result: VerificationResult) -> str:
    """Render the Failure section for cases that carry a Failure.

    Args:
        result: The VerificationResult being rendered.

    Returns:
        The rendered section, or empty string when the case is a pass.
    """
    failure = result.metadata.failure
    if failure is None:
        return ""
    lines = [
        f"- Category: `{failure.category.value}`",
        f"- Group: `{failure.group.value}`",
        f"- Stage: `{failure.stage}`",
        f"- Reason: {failure.reason}",
    ]
    if failure.details:
        lines.append(f"- Details: `{json.dumps(failure.details, sort_keys=True)}`")
    return "# Failure\n\n" + "\n".join(lines) + "\n"


def render_qa_case(
    result: VerificationResult,
    *,
    template_source: str | None,
    template_link: str | None = None,
    artifacts_dir: Path | None,
    max_trace_chars: int | None = None,
) -> str:
    """Render a QA-style VerificationResult as markdown with YAML frontmatter.

    Args:
        result: The VerificationResult to render.
        template_source: Raw Python source of the answer template, or None.
        template_link: Relative path to the template file, or None.
        artifacts_dir: Directory where artifacts (traces, attachments) may
            be written. None disables offloading.
        max_trace_chars: Optional override of the trace truncation
            threshold. When None, the env var
            ``KARENINA_TRACE_TRUNCATION_THRESHOLD`` is consulted, falling
            back to the module default.

    Returns:
        The full markdown body, ready to be written to a ``.md`` file.
    """
    fm = _qa_frontmatter(result)
    parts: list[str] = [
        "---",
        yaml.safe_dump(fm, sort_keys=True).rstrip(),
        "---",
        "",
    ]

    md = result.metadata
    tmpl = result.template

    question_body_lines = [md.question_text.strip()]
    if md.keywords:
        question_body_lines.append("")
        question_body_lines.append(f"Keywords: {', '.join(md.keywords)}")
    if md.raw_answer is not None:
        question_body_lines.append("")
        question_body_lines.append("Expected answer:")
        question_body_lines.append("")
        question_body_lines.append(md.raw_answer)
    parts.append(_render_section("Question", "\n".join(question_body_lines)))

    template_section = _render_template_section(template_source, template_link)
    if template_section:
        parts.append(template_section)

    if tmpl is not None and tmpl.raw_llm_response:
        parts.append(_render_section("LLM Response", tmpl.raw_llm_response))

    trace_messages = tmpl.trace_messages if tmpl is not None else []
    trace_body = _render_trace(
        trace_messages,
        md,
        artifacts_dir,
        max_trace_chars,
    )
    parts.append(_render_section("Trace", trace_body))

    if tmpl is not None and tmpl.parsed_llm_response is not None:
        parts.append(
            _render_section(
                "Parsed Answer",
                "```json\n" + json.dumps(tmpl.parsed_llm_response, sort_keys=True, indent=2) + "\n```",
            )
        )

    if tmpl is not None and tmpl.field_results:
        rows = "\n".join(f"- `{field}`: {'pass' if ok else 'fail'}" for field, ok in tmpl.field_results.items())
        parts.append(_render_section("Field Results", rows))

    failure_section = _render_failure(result)
    if failure_section:
        parts.append(failure_section)

    return "\n".join(p for p in parts if p)
