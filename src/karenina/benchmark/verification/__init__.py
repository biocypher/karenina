"""Verification components for Karenina."""

from .batch_runner import (
    execute_task,
    generate_task_queue,
    run_verification_batch,
)
from .executor import (
    ExecutorConfig,
    VerificationExecutor,
    get_async_portal,
    set_async_portal,
)
from .extension import extend_template_run
from .sinks import (
    AgenticProgressiveFileSink,
    CompositeSink,
    DBSink,
    InMemorySink,
    ProgressiveFileSink,
    ResultSink,
    is_completed_result,
)
from .stages import (
    export_verification_results_csv,
    export_verification_results_json_stream,
)
from .utils.embedding_check import clear_embedding_model_cache, preload_embedding_model
from .utils.storage_helpers import auto_save_results

__all__ = [
    # Batch runner
    "run_verification_batch",
    "generate_task_queue",
    "execute_task",
    "auto_save_results",
    # Extension
    "extend_template_run",
    # Executor
    "VerificationExecutor",
    "ExecutorConfig",
    "get_async_portal",
    "set_async_portal",
    # Embedding check
    "clear_embedding_model_cache",
    "preload_embedding_model",
    # Results export
    "export_verification_results_csv",
    "export_verification_results_json_stream",
    # Sinks
    "ResultSink",
    "ProgressiveFileSink",
    "AgenticProgressiveFileSink",
    "CompositeSink",
    "DBSink",
    "InMemorySink",
    "is_completed_result",
]
