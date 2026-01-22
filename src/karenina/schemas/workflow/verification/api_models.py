"""API models for verification endpoints.

DEPRECATED: Import from `karenina.schemas.verification` instead.
"""

# Re-export from new location for backward compatibility
from ...verification.api_models import (
    FinishedTemplate,
    VerificationRequest,
    VerificationStartResponse,
    VerificationStatusResponse,
)

__all__ = [
    "FinishedTemplate",
    "VerificationRequest",
    "VerificationStatusResponse",
    "VerificationStartResponse",
]
