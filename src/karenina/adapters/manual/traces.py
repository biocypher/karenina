"""ManualTraces class for benchmark-level manual trace management.

This module provides the ManualTraces class, a high-level API for registering
traces programmatically with support for string traces, port Message lists,
and LangChain message lists.
"""

from typing import Any

from karenina.ports.messages import Message

from . import ManualTraceError
from .helpers import get_manual_trace_manager
from .message_utils import is_langchain_message_list, is_port_message_list, preprocess_message_list


class ManualTraces:
    """
    Manages manual traces for a specific benchmark.

    This class provides a high-level API for registering traces programmatically,
    with support for:
    - String traces (plain text answers)
    - Port Message lists (new architecture)
    - LangChain message lists (backward compatibility)

    For message lists, automatic preprocessing extracts agent metrics
    (tool calls, failures, iterations) and harmonizes to a string trace.
    """

    def __init__(self, benchmark: "Any") -> None:
        """
        Initialize ManualTraces with a benchmark.

        Args:
            benchmark: The Benchmark object containing questions

        Note:
            This enables question text to hash mapping via the benchmark's question cache.
        """
        self._benchmark = benchmark
        self._trace_manager = get_manual_trace_manager()
        # Lazy-built index for O(1) question text to hash lookup
        self._question_text_index: dict[str, str] | None = None

    def register_trace(
        self,
        question_identifier: str,
        trace: str | list[Message] | list[Any],
        map_to_id: bool = False,
    ) -> None:
        """
        Register a single trace by question ID or text.

        Args:
            question_identifier: Either a question hash (32-char MD5) or question text
            trace: One of:
                - String trace (plain text answer)
                - List of port Message objects
                - List of LangChain messages (AIMessage, ToolMessage, etc.)
            map_to_id: If True, treat question_identifier as text and convert to hash

        Raises:
            ValueError: If question not found in benchmark (when map_to_id=True)
            ManualTraceError: If question_hash format is invalid
            TypeError: If trace format is invalid
        """
        # Convert text to hash if needed
        question_hash = self._question_text_to_hash(question_identifier) if map_to_id else question_identifier

        # Validate hash format
        if not self._trace_manager._is_valid_md5_hash(question_hash):
            raise ManualTraceError(
                f"Invalid question hash: '{question_hash}'. "
                "Question hashes must be 32-character hexadecimal MD5 hashes."
            )

        # Preprocess trace (handle string, port Message, and LangChain formats)
        original_question = question_identifier if map_to_id else None
        harmonized_trace, agent_metrics = self._preprocess_trace(trace, original_question)

        # Store in trace manager
        self._trace_manager.set_trace(question_hash=question_hash, trace=harmonized_trace, agent_metrics=agent_metrics)

    def register_traces(
        self,
        traces_dict: dict[str, str | list[Message] | list[Any]],
        map_to_id: bool = False,
    ) -> None:
        """
        Batch register traces.

        Args:
            traces_dict: Dictionary with question identifiers as keys and traces as values
            map_to_id: If True, treat keys as question text and convert to hashes

        Raises:
            ValueError: If any question not found in benchmark (when map_to_id=True)
            ManualTraceError: If any question_hash format is invalid
            TypeError: If any trace format is invalid
        """
        for identifier, trace in traces_dict.items():
            self.register_trace(identifier, trace, map_to_id=map_to_id)

    def _build_question_text_index(self) -> dict[str, str]:
        """
        Build a reverse index from question text to MD5 hash.

        This index is built lazily on first use and cached for subsequent lookups,
        converting O(n) searches to O(1) lookups.

        Returns:
            Dictionary mapping question text to MD5 hash
        """
        import hashlib

        if self._question_text_index is not None:
            return self._question_text_index

        # Build the index
        self._question_text_index = {}
        for question_urn_id in self._benchmark._questions_cache:
            question_data = self._benchmark.get_question(question_urn_id)
            question_text = question_data["question"]
            question_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()
            self._question_text_index[question_text] = question_hash

        return self._question_text_index

    def _question_text_to_hash(self, question_text: str) -> str:
        """
        Convert question text to MD5 hash using the same algorithm as Question.id.

        Uses a lazily-built index for O(1) lookup after the first call.

        Args:
            question_text: The question text

        Returns:
            MD5 hash of the question text

        Raises:
            ValueError: If question not found in benchmark
        """
        # Build or retrieve the question text index
        index = self._build_question_text_index()

        # O(1) lookup
        if question_text in index:
            return index[question_text]

        # Question not found - compute hash for error message
        import hashlib

        computed_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()
        raise ValueError(
            f"Question not found in benchmark: '{question_text[:50]}...'\n"
            f"Computed hash: {computed_hash}\n"
            f"Indexed questions in benchmark: {len(index)}\n"
            "Note: Question text must match EXACTLY (case-sensitive, including whitespace)."
        )

    def _preprocess_trace(
        self,
        trace: str | list[Message] | list[Any],
        original_question: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Process trace and extract agent metrics if applicable.

        For string traces: Returns as-is with no metrics.
        For message lists (port or LangChain): Extracts metrics and harmonizes to string.

        Args:
            trace: String trace or list of messages (port or LangChain format)
            original_question: The original user question (if known). Used for
                              detection of summary messages from SummarizationMiddleware.

        Returns:
            Tuple of (harmonized_trace_string, agent_metrics_dict_or_None)

        Raises:
            TypeError: If trace format is invalid
        """
        if isinstance(trace, str):
            # Plain string trace - no metrics
            return trace, None

        elif isinstance(trace, list):
            if not trace:
                # Empty list - treat as empty string
                return "", None

            # Check if it's a port Message list or LangChain list
            if is_port_message_list(trace) or is_langchain_message_list(trace):
                try:
                    harmonized_trace, agent_metrics = preprocess_message_list(trace, original_question)
                    return harmonized_trace, agent_metrics
                except ImportError as e:
                    raise ManualTraceError(
                        f"Failed to import required preprocessing functions: {e}\n"
                        "Ensure langchain-core is installed for LangChain message support."
                    ) from e
                except Exception as e:
                    raise ManualTraceError(
                        f"Failed to preprocess message list: {e}\n"
                        "Ensure the message list contains valid message objects."
                    ) from e
            else:
                raise TypeError(
                    f"Invalid list content: expected port Message or LangChain messages, got {type(trace[0]).__name__}"
                )

        else:
            raise TypeError(f"Invalid trace format: expected str or list of messages, got {type(trace).__name__}")
