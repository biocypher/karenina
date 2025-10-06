"""Karenina: A benchmarking system for Large Language Models.

This package provides core functionality for:
- Question extraction from various file formats
- Answer template generation
- LLM benchmarking and verification
- Response evaluation
- Database storage for benchmarks and results
"""

from .storage import DBConfig, load_benchmark, save_benchmark

__version__ = "0.1.0"

__all__ = [
    "DBConfig",
    "save_benchmark",
    "load_benchmark",
]
