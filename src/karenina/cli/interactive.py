"""
Interactive configuration builder.

This module implements the interactive configuration builder with two modes:
- Basic mode: Essential parameters only
- Advanced mode: All available parameters
"""

from karenina.benchmark import Benchmark
from karenina.schemas import VerificationConfig


def build_config_interactively(benchmark: Benchmark, mode: str = "basic") -> VerificationConfig:
    """
    Build VerificationConfig interactively through prompts.

    Args:
        benchmark: Loaded benchmark for question display
        mode: "basic" or "advanced"

    Returns:
        VerificationConfig object

    To be implemented in Phase 6 (basic) and Phase 7 (advanced).
    """
    raise NotImplementedError("To be implemented in Phase 6-7")
