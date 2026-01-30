"""Core infrastructure for the verification pipeline.

This package contains base classes, protocols, and orchestration logic:
- ArtifactKeys: Type-safe constants for artifact and result field keys
- VerificationContext: Shared state across stages
- VerificationStage: Protocol defining stage interface
- BaseVerificationStage: Abstract base class for stages
- BaseCheckStage: Base class for check stages (abstention, sufficiency)
- BaseAutoFailStage: Base class for auto-fail stages
- StageRegistry: Manages stage instances and dependencies
- StageOrchestrator: Executes the pipeline
"""

from .autofail_stage_base import BaseAutoFailStage
from .base import (
    ArtifactKeys,
    BaseVerificationStage,
    StageList,
    StageRegistry,
    VerificationContext,
    VerificationStage,
)
from .check_stage_base import BaseCheckStage
from .orchestrator import StageOrchestrator

__all__ = [
    # Core types from base
    "ArtifactKeys",
    "VerificationContext",
    "VerificationStage",
    "BaseVerificationStage",
    "StageRegistry",
    "StageList",
    # Base classes for extending
    "BaseAutoFailStage",
    "BaseCheckStage",
    # Orchestration
    "StageOrchestrator",
]
