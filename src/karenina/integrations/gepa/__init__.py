"""GEPA (Generative Evolutionary Prompt Advancement) integration for Karenina.

This module provides integration with GEPA to enable automated optimization of
text components used in LLM benchmark verification:
- System prompts for answering models
- Parsing instructions for judge models
- MCP tool descriptions

Example usage:
    >>> from karenina import Benchmark
    >>> from karenina.integrations.gepa import KareninaAdapter, split_benchmark
    >>>
    >>> benchmark = Benchmark.load("benchmark.jsonld")
    >>> split = split_benchmark(benchmark, seed=42)
    >>> # ... configure and run optimization
"""

from karenina.integrations.gepa.config import OptimizationConfig, OptimizationTarget
from karenina.integrations.gepa.data_types import (
    BenchmarkSplit,
    KareninaDataInst,
    KareninaOutput,
    KareninaTrajectory,
)
from karenina.integrations.gepa.export import (
    export_comparison_report,
    export_prompts_json,
    export_to_preset,
    load_prompts_json,
)
from karenina.integrations.gepa.feedback import LLMFeedbackGenerator
from karenina.integrations.gepa.scoring import (
    compute_improvement,
    compute_multi_model_score,
    compute_single_score,
    compute_weighted_score,
    extract_failed_fields,
)
from karenina.integrations.gepa.splitting import (
    questions_to_data_insts,
    split_benchmark,
    split_by_attribute,
)
from karenina.integrations.gepa.tracking import OptimizationRun, OptimizationTracker

# Adapter requires gepa to be installed
try:
    from karenina.integrations.gepa.adapter import KareninaAdapter

    GEPA_AVAILABLE = True
except ImportError:
    KareninaAdapter = None  # type: ignore[assignment, misc]
    GEPA_AVAILABLE = False

__all__ = [
    # Config
    "OptimizationConfig",
    "OptimizationTarget",
    # Data types
    "KareninaDataInst",
    "KareninaTrajectory",
    "KareninaOutput",
    "BenchmarkSplit",
    # Feedback generation
    "LLMFeedbackGenerator",
    # Splitting
    "split_benchmark",
    "split_by_attribute",
    "questions_to_data_insts",
    # Scoring
    "compute_single_score",
    "compute_weighted_score",
    "compute_multi_model_score",
    "extract_failed_fields",
    "compute_improvement",
    # Tracking
    "OptimizationRun",
    "OptimizationTracker",
    # Export
    "export_to_preset",
    "export_prompts_json",
    "load_prompts_json",
    "export_comparison_report",
    # Adapter (requires gepa)
    "KareninaAdapter",
    "GEPA_AVAILABLE",
]
