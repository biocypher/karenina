"""Error handling utilities for verification operations.

This module re-exports error detection utilities from the central karenina.utils
package for backward compatibility with existing verification code.

Functions:
    is_retryable_error: Check if an exception is a transient/retryable error
"""

from karenina.utils.errors import is_retryable_error

__all__ = [
    "is_retryable_error",
]
