"""Text normalizers for verification primitives.

Normalizers transform strings before comparison. They are composable:
apply them in sequence via apply_normalizers().
"""

import logging
import re
import string
from typing import TypeAlias

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SynonymMap(BaseModel):
    """Maps known synonyms to canonical forms before comparison."""

    mapping: dict[str, str]


Normalizer: TypeAlias = str | SynonymMap


def apply_normalizer(normalizer: Normalizer, text: str) -> str:
    """Apply a single normalizer to text.

    Args:
        normalizer: A string normalizer name or SynonymMap instance.
        text: The text to normalize.

    Returns:
        Normalized text.

    Raises:
        ValueError: If normalizer name is not recognized.
    """
    if isinstance(normalizer, SynonymMap):
        return normalizer.mapping.get(text, text)

    if normalizer == "lowercase":
        return text.lower()
    if normalizer == "strip":
        return text.strip()
    if normalizer == "remove_punctuation":
        return text.translate(str.maketrans("", "", string.punctuation))
    if normalizer == "collapse_whitespace":
        return re.sub(r"\s+", " ", text).strip()

    raise ValueError(f"Unknown normalizer: {normalizer!r}")


def apply_normalizers(normalizers: list[Normalizer], text: str) -> str:
    """Apply a chain of normalizers in sequence.

    Args:
        normalizers: Ordered list of normalizers to apply.
        text: The text to normalize.

    Returns:
        Text after all normalizers have been applied.
    """
    for n in normalizers:
        text = apply_normalizer(n, text)
    return text
