"""Render one VerificationResult (or one ScenarioExecutionResult) as a
Markdown case file with YAML frontmatter.

This module contains only the single-case rendering logic. Bucket
placement, filename sanitization, and directory orchestration are in
materializer.py (Task 8).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from karenina.schemas.verification.result import VerificationResult

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
    artifacts_dir: Path | None,  # noqa: ARG001  # threaded for Task 3 trace rendering
) -> str:
    """Render a QA-style VerificationResult as markdown with YAML frontmatter.

    The ``artifacts_dir`` parameter is threaded to the trace-rendering code
    (added in Task 3). Pass None to disable trace offloading.

    Args:
        result: The VerificationResult to render.
        template_source: Raw Python source of the answer template, or None.
        template_link: Relative path to the template file, or None.
        artifacts_dir: Directory where artifacts (traces, attachments) may
            be written. None disables offloading.

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

    # Trace section is added in Task 3. For now emit a stub so existing
    # body-layout assertions remain stable.
    parts.append(_render_section("Trace", "_No trace captured for this case._"))

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
