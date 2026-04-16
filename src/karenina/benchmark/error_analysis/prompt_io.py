"""Resolve and write PROMPT.md for an analysis directory.

Resolution order:
  1. User-supplied prompt_path, if provided.
  2. Packaged default_prompt.md loaded via importlib.resources.

Placeholders are substituted via string.Template before writing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PACKAGE = "karenina.benchmark.error_analysis.prompts"
_DEFAULT_PROMPT_FILENAME = "default_prompt.md"


@dataclass(frozen=True)
class PromptContext:
    """Values substituted into the analyst prompt before it is written.

    Attributes:
        benchmark_name: Human-readable benchmark identifier.
        answering_model: Model identifier used to produce responses under test.
        total: Total number of verification results.
        passed: Number of passing results.
        failed: Number of failing results.
        failure_categories: Ordered list of failure categories observed.
    """

    benchmark_name: str
    answering_model: str
    total: int
    passed: int
    failed: int
    failure_categories: list[str]

    def as_mapping(self) -> dict[str, str]:
        """Return the string mapping used by string.Template.safe_substitute."""
        return {
            "BENCHMARK_NAME": self.benchmark_name,
            "ANSWERING_MODEL": self.answering_model,
            "TOTAL": str(self.total),
            "PASSED": str(self.passed),
            "FAILED": str(self.failed),
            "FAILURE_CATEGORIES": ", ".join(self.failure_categories),
        }


def _load_default_prompt() -> str:
    return resources.files(_DEFAULT_PROMPT_PACKAGE).joinpath(_DEFAULT_PROMPT_FILENAME).read_text(encoding="utf-8")


def resolve_and_write_prompt(
    *,
    prompt_path: Path | None,
    out_dir: Path,
    context: PromptContext,
) -> Path:
    """Write PROMPT.md under out_dir and return its path.

    Substitutes placeholders via string.Template. If the source prompt
    contains no placeholders, the substitute call is still safe (Template
    leaves non-identifier $-sequences alone thanks to safe_substitute).

    Args:
        prompt_path: Optional user-supplied prompt file. If None, the
            packaged default is used.
        out_dir: Directory into which PROMPT.md is written.
        context: Values substituted into the prompt.

    Returns:
        The path to the written PROMPT.md.
    """
    source_text = prompt_path.read_text(encoding="utf-8") if prompt_path else _load_default_prompt()
    rendered = Template(source_text).safe_substitute(context.as_mapping())
    target = out_dir / "PROMPT.md"
    target.write_text(rendered, encoding="utf-8")
    logger.debug("Wrote PROMPT.md to %s", target)
    return target
