"""TaskEval module for rubric-driven failure characterization.

This module provides the TaskEval utility that reuses the existing benchmark
verification pipeline for evaluating task performance and characterizing failure modes.
"""

from .models import LogEvent, StepEval, TaskEvalResult
from .task_eval import TaskEval

__all__ = ["TaskEval", "LogEvent", "StepEval", "TaskEvalResult"]
