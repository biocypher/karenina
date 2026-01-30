"""Parser Port interface for LLM-based structured output parsing.

This module defines the ParserPort Protocol for extracting structured data
from LLM response text. IMPORTANT: This is NOT just JSON parsing - it invokes
an LLM to interpret and extract structured data from natural language responses.

Use this for:
- Parsing answer templates from free-form LLM traces
- Extracting structured information from conversational responses
- Converting natural language answers into typed Pydantic models
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message

# TypeVar bound to BaseModel for generic schema support
T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class ParserPort(Protocol):
    """Protocol for LLM-based structured output parsing.

    IMPORTANT: This is NOT just JSON parsing. Implementations invoke an LLM
    (the "judge" model) to interpret natural language responses and extract
    structured data according to a Pydantic schema.

    The typical flow is:
    1. The caller assembles prompt messages (system + user) via PromptAssembler
    2. The ParserPort receives pre-assembled messages and invokes the LLM
    3. The judge extracts attributes defined in the schema
    4. The result is a validated Pydantic model instance

    Implementations must provide:
    - aparse_to_pydantic(): Async parsing (primary API)
    - parse_to_pydantic(): Sync wrapper for convenience

    Example:
        >>> from pydantic import BaseModel, Field
        >>> class Answer(BaseModel):
        ...     gene_name: str = Field(description="The gene mentioned in the response")
        ...     is_oncogene: bool = Field(description="Whether it's an oncogene")

        >>> parser = get_parser(model_config)
        >>> messages = [Message.system("Extract..."), Message.user("BCL2 is...")]
        >>> answer = await parser.aparse_to_pydantic(messages, Answer)
        >>> print(answer.gene_name)
        'BCL2'
    """

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this parser adapter supports.

        Returns:
            PortCapabilities with adapter-specific feature flags.
            Defaults to PortCapabilities() (system prompts supported,
            structured output not supported).
        """
        return PortCapabilities()

    async def aparse_to_pydantic(self, messages: list[Message], schema: type[T]) -> T:
        """Parse using pre-assembled prompt messages into a structured Pydantic model.

        The caller is responsible for assembling the prompt messages
        (via PromptAssembler). The parser is a pure executor — it invokes
        the LLM with the provided messages and parses the result.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.
                    Field descriptions guide the LLM on what to extract.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.

        Example:
            >>> class DrugTarget(BaseModel):
            ...     target: str = Field(description="Drug target protein")
            ...     mechanism: str = Field(description="Mechanism of action")
            >>> messages = assembler.assemble(system_text=..., user_text=...)
            >>> result = await parser.aparse_to_pydantic(messages, DrugTarget)
            >>> result.target
            'BCL2'
        """
        ...

    def parse_to_pydantic(self, messages: list[Message], schema: type[T]) -> T:
        """Parse using pre-assembled prompt messages (sync).

        This is a convenience wrapper around aparse_to_pydantic() for sync code.
        Uses asyncio.run() internally.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.
        """
        ...
