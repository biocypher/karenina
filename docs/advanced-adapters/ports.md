# Port Types

Karenina defines three port protocols that form the interface boundary between the verification pipeline and LLM backends. Each port serves a distinct purpose and is used by specific pipeline stages.

## Port Summary

| Port | Purpose | Pipeline Usage | Async Primary |
|------|---------|----------------|---------------|
| **LLMPort** | Simple LLM invocation | Stages 5-6 (abstention/sufficiency), Stage 11 (rubric), deep judgment | `ainvoke()` |
| **ParserPort** | Structured output parsing | Stage 7 (parse template) | `aparse_to_pydantic()` |
| **AgentPort** | Multi-turn agent execution | Stage 2 (generate answer) | `arun()` |

All ports use duck typing via Python's `Protocol` class — implementations don't inherit from the port, they just implement the required methods.

## LLMPort

The simplest port. Makes stateless LLM calls without agent loops or tool use. Used for evaluation tasks that need a single LLM response.

### Protocol Signature

```python
from contextlib import AbstractAsyncContextManager
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.llm import LLMPort, LLMResponse, StreamingLLMResponse

class LLMPort(Protocol):
    @property
    def capabilities(self) -> PortCapabilities: ...

    async def ainvoke(self, messages: list[Message]) -> LLMResponse: ...

    def invoke(self, messages: list[Message]) -> LLMResponse: ...

    def with_structured_output(
        self, schema: type[BaseModel], *, max_retries: int | None = None
    ) -> "LLMPort": ...

    def astream(
        self, messages: list[Message]
    ) -> AbstractAsyncContextManager[StreamingLLMResponse]: ...

    def stream_invoke(
        self, messages: list[Message], timeout: float | None = None
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...
```

### Methods

| Method | Description |
|--------|-------------|
| `ainvoke(messages)` | Async invocation — primary API. Takes a list of `Message` objects, returns `LLMResponse`. |
| `invoke(messages)` | Sync wrapper around `ainvoke()`. Uses `asyncio.run()` internally. |
| `with_structured_output(schema, *, max_retries=None)` | Returns a new `LLMPort` configured for structured output using the provided Pydantic schema. Not all adapters support `max_retries`; those that do not will log a warning. |
| `astream(messages)` | Open a streaming connection. Returns an async context manager that yields a `StreamingLLMResponse`; the response is async-iterable, producing text chunks while accumulating content. Adapters that do not support streaming raise `NotImplementedError`. Check `capabilities.supports_streaming` before calling. |
| `stream_invoke(messages, timeout=None)` | Sync wrapper that streams tokens with an optional wall-clock timeout. On timeout, returns an `LLMResponse` with `is_partial=True` and whatever content arrived. Adapters that do not support streaming raise `NotImplementedError`. |
| `capabilities` | Property returning `PortCapabilities` declaring adapter feature support. |
| `aclose()` | Release adapter resources. Required; must be safe to call multiple times. |

### LLMResponse

```python
@dataclass
class LLMResponse:
    content: str                  # The text content of the response
    usage: UsageMetadata          # Token usage and cost metadata
    raw: Any = None               # Provider-specific raw response object
    is_partial: bool = False      # True when streaming was truncated (e.g., timeout)
    usage_unavailable: bool = False  # True when usage metadata could not be captured
```

`is_partial` is set by `stream_invoke()` when a wall-clock timeout interrupts the stream; the partial content is preserved. `usage_unavailable` is set when the streaming timeout interrupted the final chunk that carries token counts: usage fields are zero but do not reflect actual consumption.

### StreamingLLMResponse

```python
class StreamingLLMResponse:
    accumulated_content: str    # All text chunks received so far
    usage: UsageMetadata        # Populated when the stream completes cleanly
    is_complete: bool           # True after the stream finished without interruption
```

`StreamingLLMResponse` is async-iterable: iterating yields each `str` chunk as it arrives and appends it to `accumulated_content`. Use it inside an `async with llm.astream(...) as sr:` block.

### Pipeline Usage

- **Stage 5 (Abstention Check)**: Detects whether the model refused to answer
- **Stage 6 (Sufficiency Check)**: Determines if the response has enough information for parsing
- **Stage 11 (Rubric Evaluation)**: Evaluates LLM rubric traits (boolean, score, literal)
- **Deep Judgment**: Excerpt extraction, reasoning, and search-enhanced verification

### Example

```python
from karenina.adapters.factory import get_llm
from karenina.ports.messages import Message

llm = get_llm(model_config)
response = await llm.ainvoke([
    Message.system("You are a helpful assistant."),
    Message.user("What is 2+2?")
])
print(response.content)  # "4"
print(response.usage.total_tokens)  # e.g., 15
```

## ParserPort

Invokes an LLM (the "judge" model) to extract structured data from natural language responses. This is **not** JSON parsing — it uses an LLM to interpret free-form text and fill in a Pydantic schema.

### Protocol Signature

```python
from karenina.ports.parser import ParserPort, ParsePortResult

class ParserPort(Protocol):
    @property
    def capabilities(self) -> PortCapabilities: ...

    async def aparse_to_pydantic(
        self, messages: list[Message], schema: type[T]
    ) -> ParsePortResult[T]: ...

    def parse_to_pydantic(
        self, messages: list[Message], schema: type[T]
    ) -> ParsePortResult[T]: ...

    async def aclose(self) -> None: ...
```

### Methods

| Method | Description |
|--------|-------------|
| `aparse_to_pydantic(messages, schema)` | Async parsing — primary API. Receives pre-assembled prompt messages and a Pydantic schema, returns `ParsePortResult[T]`. |
| `parse_to_pydantic(messages, schema)` | Sync wrapper around `aparse_to_pydantic()`. Uses `asyncio.run()` internally. |
| `capabilities` | Property returning `PortCapabilities` declaring adapter feature support. |
| `aclose()` | Release adapter resources. Required; must be safe to call multiple times. |

### ParsePortResult

```python
@dataclass
class ParsePortResult(Generic[T]):
    parsed: T                 # The validated Pydantic model instance
    usage: UsageMetadata      # Token usage from the parsing LLM call(s)
```

The `T` type parameter is bound to `BaseModel`, so `parsed` is always a validated Pydantic instance.

### Key Design: Pure Executor

The ParserPort is a **pure executor** — it receives pre-assembled messages and doesn't build prompts internally. Prompt construction happens in the `PromptAssembler`:

```
TemplatePromptBuilder → PromptAssembler → list[Message] → ParserPort
```

This separation means parser adapters don't need to know about karenina-specific prompt formats. Adapter-specific prompt tuning (e.g., Claude Tool stripping JSON schema since it has native structured output) is handled by `AdapterInstructionRegistry`.

### Pipeline Usage

- **Stage 7 (Parse Template)**: The judge LLM parses the candidate's response into the answer template's Pydantic schema

### Example

```python
from pydantic import BaseModel, Field
from karenina.adapters.factory import get_parser
from karenina.ports.messages import Message

class Answer(BaseModel):
    gene_name: str = Field(
        description=(
            "The gene name or symbol mentioned in the response. Use the standard "
            "HGNC gene symbol in uppercase (e.g., 'BCL2' not 'Bcl-2')."
        )
    )
    is_oncogene: bool = Field(
        description=(
            "True if the response identifies the gene as an oncogene or "
            "proto-oncogene. False if not mentioned or classified differently."
        )
    )

parser = get_parser(model_config)
messages = [
    Message.system("Extract the following from the response..."),
    Message.user("BCL2 is a proto-oncogene that...")
]
result = await parser.aparse_to_pydantic(messages, Answer)
print(result.parsed.gene_name)  # "BCL2"
print(result.usage.total_tokens)  # e.g., 200
```

## AgentPort

The most complex port. Executes multi-turn agent loops with tool use and MCP server connections. Used for answer generation where models may need to call tools or interact with external services.

### Protocol Signature

```python
from karenina.ports.agent import AgentPort, AgentConfig, AgentResult, Tool, MCPServerConfig

class AgentPort(Protocol):
    @property
    def capabilities(self) -> PortCapabilities: ...

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult: ...

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult: ...

    async def aclose(self) -> None: ...
```

### Methods

| Method | Description |
|--------|-------------|
| `capabilities` | Property returning `PortCapabilities` declaring adapter feature support. |
| `arun(messages, tools=None, mcp_servers=None, config=None)` | Async agent execution — primary API. Runs an agent loop with optional tools and MCP servers. |
| `run(messages, tools=None, mcp_servers=None, config=None)` | Sync wrapper around `arun()`. Creates a new event loop — do not call from within an existing async context. |
| `aclose()` | Release adapter resources (MCP sessions, SDK clients). Required; must be safe to call multiple times. |

### AgentConfig

Controls agent execution behavior.

```python
@dataclass
class AgentConfig:
    max_turns: int = 25            # Maximum conversation turns before stopping
    system_prompt: str | None = None  # Optional system prompt (overrides adapter default)
    timeout: float | None = None   # Optional timeout in seconds
    question_hash: str | None = None  # MD5 hash for manual trace lookup
    extra: dict[str, Any] = field(default_factory=dict)  # Adapter-specific options
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turns` | `int` | `25` | Maximum conversation turns. Maps to `recursion_limit` in VerificationConfig. |
| `system_prompt` | `str \| None` | `None` | Override the adapter's default system prompt. |
| `timeout` | `float \| None` | `None` | Timeout in seconds for the entire run. `None` means no timeout. |
| `question_hash` | `str \| None` | `None` | MD5 hash for manual interface trace lookup. Ignored by other adapters. |
| `extra` | `dict[str, Any]` | `{}` | Adapter-specific options (e.g., Claude SDK `permission_mode`, `max_budget_usd`). |

### AgentResult

Contains the execution result with dual trace formats.

```python
@dataclass
class AgentResult:
    final_response: str            # Final text response from the agent
    raw_trace: str                 # Legacy string format with delimiters
    trace_messages: list[Message]  # Structured message list
    usage: UsageMetadata           # Token and cost usage for the entire run
    turns: int                     # Number of conversation turns completed
    limit_reached: bool            # True if stopped by max_turns limit
    session_id: str | None = None  # Adapter-specific session ID
    actual_model: str | None = None  # Actual model used (may differ from requested)
```

| Field | Type | Description |
|-------|------|-------------|
| `final_response` | `str` | The last assistant message text. |
| `raw_trace` | `str` | Legacy string format with `--- AI Message ---` delimiters. Used for database storage and regex-based processing. |
| `trace_messages` | `list[Message]` | Structured message objects for type-safe access. Used by the frontend structured trace display. |
| `usage` | `UsageMetadata` | Aggregate token/cost usage for the entire agent run. |
| `turns` | `int` | Number of agent iterations completed. |
| `limit_reached` | `bool` | `True` if the agent hit `max_turns` rather than completing naturally. |
| `session_id` | `str \| None` | Session identifier for checkpointing (adapter-specific). |
| `actual_model` | `str \| None` | The model that actually generated the response (may differ due to routing or fallback). |

!!! tip "Dual trace formats"
    Both `raw_trace` and `trace_messages` represent the same conversation. `raw_trace` is the legacy format for backward compatibility; `trace_messages` is the structured format for new features. Both are produced by all adapters.

### Tool

Definition for standalone tools (not from MCP servers).

```python
@dataclass(frozen=True)
class Tool:
    name: str                      # Unique tool identifier
    description: str               # Human-readable description
    input_schema: dict[str, Any]   # JSON Schema for tool input parameters
```

### MCP Server Configuration

MCP servers support two transport types:

```python
# Stdio transport (local process)
class MCPStdioServerConfig(TypedDict, total=False):
    type: Literal["stdio"]         # Transport type
    command: str                   # Command to run (e.g., "npx", "python")
    args: list[str]                # Command arguments
    env: dict[str, str]            # Environment variables

# HTTP/SSE transport (remote)
class MCPHttpServerConfig(TypedDict, total=False):
    type: Literal["http", "sse"]   # "http" for streamable HTTP, "sse" for SSE
    url: str                       # Server URL endpoint
    headers: dict[str, str]        # HTTP headers (e.g., Authorization)

MCPServerConfig = MCPStdioServerConfig | MCPHttpServerConfig
```

### Pipeline Usage

- **Stage 2 (Generate Answer)**: The answering model generates a response, potentially using tools and MCP servers

### Error Types

`AgentPort` implementations raise `AgentExecutionError` (and its subclass `AgentTimeoutError`) for runtime failures, and `AgentResponseError` when the agent produced unparsable output. Both inherit from `PortError`. See the consolidated [Error Hierarchy](#error-hierarchy) below for the full table including attributes (`stderr`, `limit_reached`) and guidance for custom adapter authors.

### Example

```python
from karenina.adapters.factory import get_agent
from karenina.ports.agent import AgentConfig
from karenina.ports.messages import Message

agent = get_agent(model_config)
result = await agent.arun(
    messages=[Message.user("What genes are associated with breast cancer?")],
    mcp_servers={
        "pubmed": {"type": "http", "url": "https://pubmed.example.com/mcp"}
    },
    config=AgentConfig(max_turns=10, timeout=60.0)
)
print(result.final_response)
print(f"Completed in {result.turns} turns")
print(f"Limit reached: {result.limit_reached}")
```

## Error Hierarchy

All port-layer exceptions inherit from `PortError`, which itself inherits from `KareninaError`. Adapters should raise these (never bare `Exception` or provider-specific exceptions); pipeline stages catch them at the port boundary and convert them into structured `Failure` records.

```
PortError
├── AdapterUnavailableError
├── AgentExecutionError
│   └── AgentTimeoutError
├── AgentResponseError
└── ParseError
```

### Class Reference

| Class | Inherits from | Constructor signature | Extra attributes | When to raise |
|-------|---------------|----------------------|------------------|---------------|
| `PortError` | `KareninaError` | `(message)` | (none) | Base class. Do not raise directly; raise a subclass. |
| `AdapterUnavailableError` | `PortError` | `(message, reason=None, fallback_interface=None)` | `reason: str` (defaults to `message` if not provided), `fallback_interface: str \| None` | Required infrastructure is missing: SDK not installed, CLI not in `PATH`, API key absent, missing `model_config` field. The factories raise this with `fallback_interface` populated when a sensible fallback exists. |
| `AgentExecutionError` | `PortError` | `(message, stderr=None, limit_reached=False)` | `stderr: str \| None`, `limit_reached: bool` | Runtime failure during `arun()`: subprocess crashed, SDK threw, agent loop exited with an error. Set `stderr` when a captured stderr stream is available; set `limit_reached=True` if the failure is "model hit `max_turns`/`recursion_limit`" rather than a hard error. |
| `AgentTimeoutError` | `AgentExecutionError` | inherited | inherited | Wall-clock timeout or recursion-limit timeout during agent execution. Raise instead of `AgentExecutionError` when the cause is specifically a timeout, so retry logic can route it to `RetryExecutor`'s `TIMEOUT` category. |
| `AgentResponseError` | `PortError` | `(message)` | (none) | Agent completed but produced output that cannot be processed: invalid JSON, unexpected message shape, missing required fields. |
| `ParseError` | `PortError` | `(message, raw_response=None)` | `raw_response: str \| None` | The judge LLM ran but its output could not be parsed into the requested Pydantic schema. Populate `raw_response` with the original text so callers can inspect it for debugging. |

Source: `karenina/src/karenina/ports/errors.py`.

### Custom Adapter Guidelines

When writing your own adapter, route every provider exception through this hierarchy at the port boundary:

```python
from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
    ParseError,
)

# Construction-time / config issue
if not _has_my_sdk():
    raise AdapterUnavailableError(
        "my_provider SDK not installed",
        reason="my_provider_sdk import failed",
        fallback_interface="langchain",
    )

# Agent runtime crash
try:
    result = await self._client.run(...)
except MyProviderTimeout as e:
    raise AgentTimeoutError(str(e), stderr=getattr(e, "stderr", None)) from e
except MyProviderError as e:
    raise AgentExecutionError(str(e), stderr=getattr(e, "stderr", None)) from e

# Parser could not extract the schema
try:
    parsed = schema.model_validate(data)
except ValidationError as e:
    raise ParseError("Failed to parse response", raw_response=raw_text) from e
```

Always chain via `from e` so the original exception is preserved. The pipeline's `failure_classifier` reads these attributes (`reason`, `fallback_interface`, `stderr`, `limit_reached`, `raw_response`) when building `Failure` records, so populating them is what makes downstream debugging useful.

## Supporting Types

### Message

The unified message format used across all ports. See [Adapter Architecture](index.md) for the full message design.

```python
@dataclass
class Message:
    role: Role                     # system, user, assistant, tool
    content: list[Content]         # List of content blocks

    @property
    def text(self) -> str: ...     # Extract all text content as a string

    @property
    def tool_calls(self) -> list[ToolUseContent]: ...  # Extract tool use blocks

    @classmethod
    def system(cls, text: str) -> Message: ...

    @classmethod
    def user(cls, text: str) -> Message: ...

    @classmethod
    def assistant(cls, text: str = "", tool_calls=None) -> Message: ...

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str, is_error=False) -> Message: ...
```

Content blocks can be:

| Type | Purpose |
|------|---------|
| `TextContent` | Plain text |
| `ToolUseContent` | Tool invocation (id, name, input) |
| `ToolResultContent` | Tool execution result (tool_use_id, content, is_error) |
| `ThinkingContent` | Extended thinking (Claude's reasoning trace) |

### UsageMetadata

Token and cost tracking for LLM invocations.

```python
@dataclass
class UsageMetadata:
    input_tokens: int = 0          # Tokens in the prompt
    output_tokens: int = 0         # Tokens in the response
    total_tokens: int = 0          # input + output
    cost_usd: float | None = None  # Cost in USD (if available)
    cache_read_tokens: int | None = None    # Anthropic prompt cache reads
    cache_creation_tokens: int | None = None  # Anthropic prompt cache writes
    model: str | None = None       # Model that generated this usage
```

### PortCapabilities

Declares what prompt features an adapter supports. Used by `PromptAssembler` to decide message formatting.

```python
@dataclass(frozen=True)
class PortCapabilities:
    supports_system_prompt: bool = True       # Separate system messages supported
    supports_structured_output: bool = False  # JSON schema enforcement supported
    supports_streaming: bool = False          # astream/stream_invoke implemented
```

When `supports_system_prompt` is `False`, the `PromptAssembler` prepends system text to the user message instead of sending it as a separate system message. `supports_streaming` is set by adapters that implement `LLMPort.astream` and `LLMPort.stream_invoke`; the default `False` reflects that streaming is not part of the standard verification path. Streaming is currently used only by callers that need partial-output recovery on wall-clock timeouts (see [`stream_invoke` and `is_partial`](#llmresponse) above) and is internal/experimental for most evaluation workflows: do not rely on it as a stable user-facing API. Always check `capabilities.supports_streaming` before invoking the streaming methods, since adapters that lack support raise `NotImplementedError`.

## Port Relationship

The three ports form a complexity hierarchy:

```
LLMPort          ← Simplest: single call, no tools, no agent loop
  │
ParserPort       ← Middle: single call, structured output, schema-driven
  │
AgentPort        ← Most complex: multi-turn, tools, MCP, trace capture
```

Each higher-level port can do everything the lower-level ports do, but the pipeline uses the simplest appropriate port for each task:

- **Evaluation tasks** (abstention, sufficiency, rubric) use `LLMPort` — they need a single LLM judgment, not an agent loop
- **Parsing** uses `ParserPort` — it needs structured output (filling a Pydantic schema), but doesn't need tools
- **Answer generation** uses `AgentPort` — the answering model may need tools, MCP servers, and multi-turn reasoning

## Import Paths

```python
# Port protocols
from karenina.ports.llm import LLMPort, LLMResponse
from karenina.ports.parser import ParserPort, ParsePortResult
from karenina.ports.agent import AgentPort, AgentConfig, AgentResult, Tool

# Supporting types
from karenina.ports.messages import Message, Role, ContentType
from karenina.ports.messages import TextContent, ToolUseContent, ToolResultContent, ThinkingContent
from karenina.ports.usage import UsageMetadata
from karenina.ports.capabilities import PortCapabilities

# MCP configuration
from karenina.ports.agent import MCPServerConfig, MCPStdioServerConfig, MCPHttpServerConfig

# Factory functions
from karenina.adapters.factory import get_llm, get_parser, get_agent
```

## Testing Utilities

Two registry-level helpers exist for test isolation only; they are not public API and must not be used in production code.

| Helper | Module | Purpose |
|--------|--------|---------|
| `AdapterRegistry._reset()` | `karenina.adapters.registry` | Clear `_specs` and reset initialization flags so tests can register their own `AdapterSpec` without colliding with built-ins. |
| `AdapterInstructionRegistry.clear()` | `karenina.ports.adapter_instruction` | Remove all `(interface, task)` factory mappings from the global instruction registry. |

Use them inside pytest fixtures with both setup and teardown so the global state is always restored. See [Writing Custom Adapters: Testing Utilities](writing-adapters.md#testing-utilities) for an example fixture.

## Related

- [Adapter Architecture](index.md) — how ports fit into the hexagonal architecture
- [Available Adapters](available-adapters.md) — implementations for each port
- [Prompt Assembly](../advanced-pipeline/prompt-assembly.md) — how messages are built before being passed to ports
- [Verification Pipeline Stages](../advanced-pipeline/stages.md) — which stages use which ports
- [Writing Custom Adapters](writing-adapters.md) — implementing your own port adapter
