"""Domain operations for Karenina.

DEPRECATED: This module is deprecated. Import from `services` instead.

This module re-exports from `services` for backward compatibility.
"""

import warnings

# Re-export from services for backward compatibility
from karenina.services import generate_answer_template, read_questions_from_file

__all__ = [
    "generate_answer_template",
    "read_questions_from_file",
]


def __getattr__(name: str) -> object:
    """Emit deprecation warning when accessing this module."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from 'karenina.domain' is deprecated. Use 'karenina.services' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
