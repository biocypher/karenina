"""Question extraction and processing functionality for Karenina.

DEPRECATED: This module is deprecated. Import from `services.questions` instead.

This module re-exports from `services.questions` for backward compatibility.
"""

import warnings

# Re-export from services.questions for backward compatibility
from karenina.services.questions import (
    extract_and_generate_questions,
    extract_questions_from_file,
    generate_questions_file,
    read_file_to_dataframe,
    read_questions_from_file,
)

__all__ = [
    "extract_and_generate_questions",
    "extract_questions_from_file",
    "generate_questions_file",
    "read_file_to_dataframe",
    "read_questions_from_file",
]


def __getattr__(name: str) -> object:
    """Emit deprecation warning when accessing this module."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from 'karenina.domain.questions' is deprecated. "
            f"Use 'karenina.services.questions' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
