"""Manual LLM implementation that returns precomputed traces."""

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

from .manual_traces import get_manual_trace

if TYPE_CHECKING:
    from .interface import LLMError
else:
    # Avoid circular import at runtime
    class LLMError(Exception):
        """Base exception for LLM-related errors."""
        pass


class ManualTraceNotFoundError(LLMError):
    """Raised when a manual trace is not found for a question."""
    pass


class ManualLLM:
    """
    LLM implementation that returns precomputed manual traces.
    
    This class mimics the behavior of a real LLM but instead of making
    API calls, it retrieves precomputed answer traces based on question hashes.
    """

    def __init__(self, question_hash: str, **_kwargs):
        """
        Initialize the ManualLLM.
        
        Args:
            question_hash: MD5 hash of the question this LLM will answer
            **kwargs: Additional arguments for compatibility (ignored)
        """
        self.question_hash = question_hash

    def invoke(self, _messages: list[BaseMessage]) -> AIMessage:
        """
        Return precomputed trace for the question.
        
        Args:
            messages: List of messages (ignored for manual traces)
            
        Returns:
            AIMessage containing the precomputed trace
            
        Raises:
            ManualTraceNotFoundError: If no trace is found for the question hash
        """
        trace = get_manual_trace(self.question_hash)

        if trace is None:
            from .manual_traces import get_manual_trace_count
            trace_count = get_manual_trace_count()
            
            raise ManualTraceNotFoundError(
                f"No manual trace found for question hash: '{self.question_hash}'. "
                f"Currently loaded {trace_count} trace(s). "
                "To resolve this issue: "
                "1) Upload a JSON file containing manual traces using the GUI upload feature, or "
                "2) Use load_manual_traces() to load traces programmatically, or "
                "3) Verify that the question hash matches one of your uploaded traces."
            )

        return AIMessage(content=trace)

    def with_structured_output(self, _schema: Any) -> "ManualLLM":
        """
        Return self for compatibility with structured output interface.
        
        Args:
            schema: Output schema (ignored for compatibility)
            
        Returns:
            Self for method chaining
        """
        return self

    @property
    def content(self) -> str:
        """
        Get the trace content directly.
        
        Returns:
            The precomputed trace content
            
        Raises:
            ManualTraceNotFoundError: If no trace is found
        """
        trace = get_manual_trace(self.question_hash)

        if trace is None:
            from .manual_traces import get_manual_trace_count
            trace_count = get_manual_trace_count()
            
            raise ManualTraceNotFoundError(
                f"No manual trace found for question hash: '{self.question_hash}'. "
                f"Currently loaded {trace_count} trace(s). "
                "To resolve this issue: "
                "1) Upload a JSON file containing manual traces using the GUI upload feature, or "
                "2) Use load_manual_traces() to load traces programmatically, or "
                "3) Verify that the question hash matches one of your uploaded traces."
            )

        return trace


def create_manual_llm(question_hash: str, **_kwargs) -> ManualLLM:
    """
    Create a ManualLLM instance for a specific question hash.
    
    Args:
        question_hash: MD5 hash of the question
        **kwargs: Additional arguments for compatibility
        
    Returns:
        ManualLLM instance configured for the question
    """
    return ManualLLM(question_hash=question_hash, **_kwargs)
