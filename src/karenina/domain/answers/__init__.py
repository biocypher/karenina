"""Answer generation and processing functionality for Karenina.

DEPRECATED: This module is deprecated. Import from `services.answers` instead.

This module re-exports from `services.answers` for backward compatibility.
"""

import warnings

# Re-export from services.answers for backward compatibility
from karenina.services.answers import generate_answer_template

__all__ = ["generate_answer_template"]


def __getattr__(name: str) -> object:
    """Emit deprecation warning when accessing this module."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from 'karenina.domain.answers' is deprecated. Use 'karenina.services.answers' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
