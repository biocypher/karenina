# Writing Custom Adapters

This guide walks through creating a custom adapter that integrates a new LLM backend into karenina's verification pipeline. By the end, you'll have a fully registered adapter with its own prompt instructions, available through the standard factory functions.

---

## What You Need to Implement

A complete adapter provides up to three port implementations and a registration module:

| Component | Required | Purpose |
|-----------|----------|---------|
| `AgentPort` adapter | Optional | Multi-turn agent execution with tools/MCP |
| `ParserPort` adapter | Optional | LLM-based structured output parsing |
| `LLMPort` adapter | Optional | Simple LLM invocation |
| `registration.py` | Required | Register adapter with the factory system |
| Prompt instructions | Recommended | Adapter-specific prompt tuning |

You can implement any subset of the three ports. If your adapter only supports simple invocation, implement just `LLMPort` and set the other factory functions to `None`.

---

## Step 1: Create the Adapter Directory

Create a new directory under `karenina/src/karenina/adapters/`:

```
karenina/src/karenina/adapters/my_provider/
├── __init__.py          # Package marker
├── registration.py      # Adapter registration (required)
├── llm.py               # LLMPort implementation
├── parser.py            # ParserPort implementation
├── agent.py             # AgentPort implementation
└── prompts/             # Adapter-specific prompt instructions
    ├── __init__.py
    ├── parsing.py
    ├── rubric.py
    └── deep_judgment.py
```

---

## Step 2: Implement Port Protocols

Adapter classes use **duck typing** — they implement the port method signatures without inheriting from the protocol class. Any object with the right methods satisfies the protocol.

### Implementing LLMPort

The simplest port. Implement `ainvoke`, `invoke`, `with_structured_output`, `aclose`, and a `capabilities` property:

```python
from __future__ import annotations

from pydantic import BaseModel

from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.llm import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig


class MyProviderLLMAdapter:
    """LLM adapter for MyProvider.

    Implements LLMPort protocol via duck typing (no explicit inheritance).
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config
        # Initialize your provider's client here
        self._client = self._initialize_client()

    def _initialize_client(self):
        """Create the provider-specific client."""
        # Import your provider's SDK here (lazy import)
        from my_provider_sdk import Client
        return Client(
            model=self._config.model_name,
            api_key=...,
        )

    @property
    def capabilities(self) -> PortCapabilities:
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=False,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Async invocation — the primary API."""
        # Convert karenina Messages to your provider's format
        provider_messages = self._convert_messages(messages)

        # Call your provider
        response = await self._client.chat(provider_messages)

        # Return standardized response
        return LLMResponse(
            content=response.text,
            usage=UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            raw=response,
        )

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Sync wrapper — use asyncio.run()."""
        import asyncio
        return asyncio.run(self.ainvoke(messages))

    def with_structured_output(
        self, schema: type[BaseModel], *, max_retries: int | None = None
    ) -> "MyProviderLLMAdapter":
        """Return a new adapter configured for structured output.

        If your provider doesn't support native structured output,
        return self unchanged — the pipeline will handle JSON parsing.
        """
        if max_retries is not None:
            logger.warning(
                "%s does not support max_retries (got %d), ignoring",
                type(self).__name__, max_retries,
            )
        return self

    async def aclose(self) -> None:
        """Release adapter resources.

        Required protocol method. Implement even if the adapter holds
        no resources (as a no-op). Must be safe to call multiple times.
        """

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert karenina Messages to provider format."""
        result = []
        for msg in messages:
            result.append({
                "role": msg.role.value,
                "content": msg.text,
            })
        return result
```

### Implementing ParserPort

The parser is a **pure executor** — it receives pre-assembled prompt messages from `PromptAssembler` and doesn't build prompts internally:

```python
from typing import TypeVar

from pydantic import BaseModel

from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.parser import ParsePortResult
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig

T = TypeVar("T", bound=BaseModel)


class MyProviderParserAdapter:
    """Parser adapter for MyProvider.

    Implements ParserPort protocol via duck typing.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config

    @property
    def capabilities(self) -> PortCapabilities:
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=False,
        )

    async def aparse_to_pydantic(
        self, messages: list[Message], schema: type[T]
    ) -> ParsePortResult[T]:
        """Parse LLM response into a Pydantic model.

        The messages are pre-assembled by PromptAssembler with task
        instructions, adapter instructions, and user instructions.
        """
        # Call your provider with the pre-assembled messages
        response = await self._call_llm(messages)

        # Parse the JSON response into the schema
        import json
        data = json.loads(response.text)
        parsed = schema.model_validate(data)

        return ParsePortResult(
            parsed=parsed,
            usage=UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def parse_to_pydantic(
        self, messages: list[Message], schema: type[T]
    ) -> ParsePortResult[T]:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.aparse_to_pydantic(messages, schema))

    async def aclose(self) -> None:
        """Release adapter resources.

        Required protocol method. Must be safe to call multiple times.
        """
```

### Implementing AgentPort

The most complex port — handles multi-turn execution with optional tools and MCP servers:

```python
from karenina.ports.agent import AgentConfig, AgentResult
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig


class MyProviderAgentAdapter:
    """Agent adapter for MyProvider.

    Implements AgentPort protocol via duck typing.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Required on all three ports (AgentPort, LLMPort, ParserPort).
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=False,
        )

    async def arun(
        self,
        messages: list[Message],
        tools: list | None = None,
        mcp_servers: dict | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an agent loop."""
        config = config or AgentConfig()

        # Run your agent with the given messages and tools
        # ... your implementation here ...

        return AgentResult(
            final_response="The agent's final response",
            raw_trace="--- AI Message ---\nThe response...",
            trace_messages=[Message.assistant("The response...")],
            usage=UsageMetadata(),
            turns=1,
            limit_reached=False,
        )

    def run(
        self,
        messages: list[Message],
        tools: list | None = None,
        mcp_servers: dict | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Sync wrapper."""
        import asyncio
        return asyncio.run(self.arun(messages, tools, mcp_servers, config))

    async def aclose(self) -> None:
        """Release adapter resources (MCP sessions, SDK clients, etc.).

        Required protocol method. Must be safe to call multiple times.
        """
```

---

## Step 3: Write the Registration Module

The registration module connects your adapter to the factory system. This is the only required file.

### Availability Checker

Check whether your provider's SDK is installed:

```python
# karenina/src/karenina/adapters/my_provider/registration.py
import logging

from karenina.adapters.registry import (
    AdapterAvailability,
    AdapterRegistry,
    AdapterSpec,
)

logger = logging.getLogger(__name__)


def _check_availability() -> AdapterAvailability:
    """Check if MyProvider SDK is installed."""
    try:
        import my_provider_sdk  # noqa: F401
        return AdapterAvailability(
            available=True,
            reason="my_provider_sdk is installed and available",
        )
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason=(
                "my_provider_sdk not installed. "
                "Install with: pip install my-provider-sdk"
            ),
            fallback_interface="langchain",  # Fall back to langchain
        )
```

### Factory Functions

Each factory takes a `ModelConfig` and returns a port implementation. Use **lazy imports** to avoid importing heavy SDKs at module level:

```python
from karenina.ports.agent import AgentPort
from karenina.ports.llm import LLMPort
from karenina.ports.parser import ParserPort
from karenina.schemas.config import ModelConfig


def _create_agent(config: ModelConfig) -> AgentPort:
    from karenina.adapters.my_provider.agent import MyProviderAgentAdapter
    return MyProviderAgentAdapter(config)


def _create_llm(config: ModelConfig) -> LLMPort:
    from karenina.adapters.my_provider.llm import MyProviderLLMAdapter
    return MyProviderLLMAdapter(config)


def _create_parser(config: ModelConfig) -> ParserPort:
    from karenina.adapters.my_provider.parser import MyProviderParserAdapter
    return MyProviderParserAdapter(config)
```

### Register the AdapterSpec

```python
_my_provider_spec = AdapterSpec(
    interface="my_provider",
    description="MyProvider adapter for custom LLM backend",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface="langchain",
    routes_to=None,
    supports_mcp=False,      # Set True if your adapter handles MCP
    supports_tools=True,
)

AdapterRegistry.register(_my_provider_spec)

logger.debug("Registered my_provider adapter with AdapterRegistry")
```

### Trigger Prompt Instruction Registration

At the end of `registration.py`, import your prompt modules so their instruction registrations execute:

```python
# Import prompt modules to trigger adapter instruction registration
import karenina.adapters.my_provider.prompts.parsing  # noqa: E402, F401
import karenina.adapters.my_provider.prompts.rubric  # noqa: E402, F401
import karenina.adapters.my_provider.prompts.deep_judgment  # noqa: E402, F401
```

### Enable Discovery

There are two ways to make the registry discover your adapter.

**Option A: Built-in adapter** (for adapters inside the karenina package). Add an import to `_load_builtins()` in `karenina/src/karenina/adapters/registry.py`:

```python
# In AdapterRegistry._load_builtins():
try:
    from karenina.adapters.my_provider import registration as _mp  # noqa: F401
except ImportError:
    logger.debug("MyProvider registration module not available")
```

**Option B: Entry point** (for external adapter packages). Add a `karenina.adapters` entry point to your package's `pyproject.toml`:

```toml
[project.entry-points."karenina.adapters"]
my_provider = "my_package.registration"
```

The entry point module must call `AdapterRegistry.register()` when imported. The registry discovers entry points automatically after loading built-in adapters. If an entry point's name conflicts with a built-in interface, the entry point is skipped with a warning.

---

## Step 4: Register Prompt Instructions

Adapter instructions customize how prompts are assembled for your adapter. The `PromptAssembler` applies them after task instructions and before user instructions.

### How Prompt Assembly Works

```
Final prompt = Task instructions + Adapter instructions + User instructions
                                    ^^^^^^^^^^^^^^^^^^^^
                                    Your additions go here
```

Adapter instructions **append** text — they never replace the base prompt. This means your additions refine the instructions for your provider's specific capabilities.

### Create an Instruction Class

An instruction class provides `system_addition` and `user_addition` properties:

```python
# karenina/src/karenina/adapters/my_provider/prompts/parsing.py
from dataclasses import dataclass
from typing import Any

from karenina.ports.adapter_instruction import AdapterInstructionRegistry


@dataclass
class _MyProviderParsingInstruction:
    """Parsing instructions for MyProvider.

    Adds format guidance appropriate for this provider's capabilities.
    """

    json_schema: dict[str, Any] | None = None

    @property
    def system_addition(self) -> str:
        """Text appended to the system prompt."""
        return (
            "Return your response as a valid JSON object. "
            "Do not include markdown fences or surrounding text."
        )

    @property
    def user_addition(self) -> str:
        """Text appended to the user prompt."""
        if self.json_schema is None:
            return ""
        import json
        schema_json = json.dumps(self.json_schema, indent=2)
        return f"Your response must conform to this JSON schema:\n```json\n{schema_json}\n```"
```

### Write the Factory Function

The factory receives keyword arguments from the `instruction_context` dict. Common keys include `json_schema`, `format_instructions`, and `model_capabilities`:

```python
def _my_provider_parsing_factory(**kwargs: object) -> _MyProviderParsingInstruction:
    """Factory producing MyProvider parsing instructions."""
    return _MyProviderParsingInstruction(
        json_schema=kwargs.get("json_schema"),
    )
```

### Register with AdapterInstructionRegistry

```python
AdapterInstructionRegistry.register(
    "my_provider", "parsing", _my_provider_parsing_factory
)
```

The first argument is the interface name, the second is the `PromptTask` value string. Common task values:

| Task | Pipeline Stage | When to Register |
|------|---------------|-----------------|
| `parsing` | Stage 7 (parse_template) | Always — controls how your adapter receives parsing instructions |
| `rubric_llm_trait_batch` | Stage 11 | If rubric evaluation needs adapter-specific tuning |
| `rubric_llm_trait_single` | Stage 11 | Same, for single-trait evaluation |
| `rubric_literal_trait_batch` | Stage 11 | For literal trait evaluation |
| `rubric_literal_trait_single` | Stage 11 | Same, for single-trait evaluation |
| `rubric_metric_trait` | Stage 11 | For metric trait evaluation |
| `dj_template_excerpt_extraction` | Deep judgment | For excerpt extraction |
| `dj_template_hallucination` | Deep judgment | For hallucination risk assessment via search |
| `dj_template_reasoning` | Deep judgment | For reasoning generation |
| `dj_template_reasoning_only` | Deep judgment | For reasoning-only mode (no excerpts) |
| `dj_rubric_excerpt_extraction` | Deep judgment | For rubric excerpt extraction |
| `dj_rubric_hallucination` | Deep judgment | For rubric hallucination risk assessment |
| `dj_rubric_reasoning` | Deep judgment | For rubric reasoning |
| `dj_rubric_score_extraction` | Deep judgment | For rubric score extraction |

For most adapters, registering `parsing` instructions is sufficient. Rubric and deep judgment instructions are optional refinements.

### When Your Provider Has Native Structured Output

If your provider supports structured output natively (like Anthropic's `messages.parse`), your parsing instructions can be minimal — strip the JSON schema since the provider handles it:

```python
@dataclass
class _MyProviderParsingInstruction:
    @property
    def system_addition(self) -> str:
        return "Extract only what's stated — don't infer."

    @property
    def user_addition(self) -> str:
        return ""  # No schema needed — provider handles it natively
```

---

## Step 5: Use Your Adapter

Once registered, your adapter is available through the standard factory functions:

```python
from karenina.adapters.factory import get_agent, get_llm, get_parser
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="my-model",
    model_name="my-model-v1",
    interface="my_provider",
)

llm = get_llm(config)           # Returns your MyProviderLLMAdapter
parser = get_parser(config)     # Returns your MyProviderParserAdapter
agent = get_agent(config)       # Returns your MyProviderAgentAdapter
```

It also works in `VerificationConfig`:

```python
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig(
    answering_models=[ModelConfig(
        id="my-model",
        model_name="my-model-v1",
        interface="my_provider",
    )],
    parsing_models=[ModelConfig(
        id="my-parser",
        model_name="my-model-v1",
        interface="my_provider",
    )],
)
```

---

## AdapterSpec Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interface` | `str` | *(required)* | Interface name used in `ModelConfig.interface` |
| `description` | `str` | *(required)* | Human-readable description |
| `agent_factory` | `Callable \| None` | `None` | Factory for `AgentPort`. `None` if unsupported. |
| `llm_factory` | `Callable \| None` | `None` | Factory for `LLMPort`. `None` if unsupported. |
| `parser_factory` | `Callable \| None` | `None` | Factory for `ParserPort`. `None` if unsupported. |
| `availability_checker` | `Callable \| None` | `None` | Returns `AdapterAvailability`. `None` means always available. |
| `fallback_interface` | `str \| None` | `None` | Interface to fall back to when unavailable |
| `routes_to` | `str \| None` | `None` | Interface this one delegates to (for routing interfaces) |
| `supports_mcp` | `bool` | `False` | Whether the adapter can connect to MCP servers |
| `supports_tools` | `bool` | `False` | Whether the adapter supports tool use |
| `agent_tier` | `str` | `"tool_loop"` | Agent capability tier. `"tool_loop"`: basic tool-calling loop (e.g., LangChain ReAct), the adapter orchestrates each tool call turn. `"deep_agent"`: full agent runtime with built-in tools (e.g., Claude Code, LangChain Deep Agents), the runtime handles tool loops internally and `GenerateAnswer` prefers the `AgentPort` path to capture the full trace. |
| `requires_provider` | `bool` | `True` | If `False`, `model_provider` is not required for this interface. Used by `validate_model_config()` to determine whether to require the provider field. |

---

## Wiring Retries and Error Classification

Every LLM and parser adapter must route its provider calls through a `RetryExecutor` constructed from `model_config.retry_policy`. This is the central retry layer for the entire pipeline; adapters must not add their own retry decorators on top.

For the full rationale and the system's design (categories, budgets, tracking), see [Error Handling and Retries](../advanced-pipeline/error-handling.md). This section covers only the adapter-side wiring.

### Construction

Build the executor in `__init__` from the policy stamped on `ModelConfig`:

```python
from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy


class MyProviderLLMAdapter:
    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config
        retry_policy = model_config.retry_policy or RetryPolicy()
        self._retry_executor = RetryExecutor(retry_policy, ErrorRegistry())
```

The pipeline stamps `model_config.retry_policy` from `VerificationConfig.retry_policy` before tasks run, so a single setting at the top governs all per-model adapter calls. The pipeline also installs custom error patterns onto the shared `ErrorRegistry` it threads through `VerificationContext`. If your adapter needs to participate in that customization, accept a registry from outside; otherwise constructing a fresh `ErrorRegistry()` here is fine for built-in classification.

### Routing Calls Through the Executor

`RetryExecutor` exposes four entry points; pick by sync/async and by whether you want timeout escalation.

| Method | Sync/Async | Timeout escalation |
|--------|------------|--------------------|
| `execute(fn, *args, **kwargs)` | sync | no |
| `aexecute(fn, *args, **kwargs)` | async | no |
| `execute_with_timeout(fn, *args, timeout=..., **kwargs)` | sync | yes |
| `aexecute_with_timeout(fn, *args, timeout=..., **kwargs)` | async | yes |

The `_with_timeout` variants forward `timeout=current_timeout` into the wrapped callable on every attempt and grow it on `TIMEOUT` retries when `policy.timeout_escalation` is set. Use them for streaming and any path where a per-attempt wall-clock guard matters.

```python
async def ainvoke(self, messages: list[Message]) -> LLMResponse:
    return await self._retry_executor.aexecute(self._ainvoke_once, messages)

async def _ainvoke_once(self, messages: list[Message]) -> LLMResponse:
    """Single attempt; called by RetryExecutor on each retry."""
    response = await self._client.chat(self._convert_messages(messages))
    return LLMResponse(content=response.text, ...)


def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
    return self._retry_executor.execute_with_timeout(
        self._stream_invoke_once, messages, timeout=timeout
    )

def _stream_invoke_once(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
    """Single streaming attempt; receives the (possibly escalated) timeout."""
    ...
```

The wrapped `_*_once` callables must be idempotent (they will run multiple times) and must propagate provider exceptions unchanged so the executor's classifier can see them.

### Avoiding Double Retry at the SDK Level

Provider SDKs (Anthropic, OpenAI, ...) ship their own retry loops. With `RetryExecutor` already retrying, those SDK retries multiply the effective budget and cause the provider to back off in ways the executor cannot see. Suppress them.

Two safe approaches exist.

Set the SDK's `max_retries=0` when constructing the client, making `RetryExecutor` the only retry layer:

```python
# Inside the LangChain adapter's _initialize_model
kwargs["max_retries"] = 0
model = init_chat_model_unified(**kwargs)
```

Or, when the SDK's retry path is preferable for some calls (e.g., parsing-only flows), derive a single `max_retries` value from the policy via `RetryPolicy.derive_sdk_max_retries()`:

```python
retry_policy = self._config.retry_policy or RetryPolicy()
kwargs["max_retries"] = retry_policy.derive_sdk_max_retries()
```

`derive_sdk_max_retries` returns `max(connection.max_attempts, timeout.max_attempts, rate_limit.max_attempts, server_error.max_attempts)` so no category is starved by a too-low SDK cap. Pick one of the two approaches per call surface; do not stack them.

### Rule: No Adapter-Side Retry Decorators

Do not wrap your adapter methods (or the underlying SDK client) with `tenacity`, `backoff`, or any other retry decorator. The legacy `tenacity` decorators on the LangChain LLM/parser, deep agents, and the streaming-timeout retry decorator have been removed precisely because they compounded with `RetryExecutor` and produced opaque retry counts. See [Error Handling and Retries](../advanced-pipeline/error-handling.md) for the broader story and what was deleted.

When in doubt, the rule is: if a transient failure should be retried, it should be retried by `RetryExecutor`. Anything else is a bug.

---

## Adapter Lifecycle and Cleanup Tracking

Adapters can hold async resources (`httpx.AsyncClient`, MCP sessions, SDK clients) that must be released before their owning event loop is closed. The registry provides a small lifecycle subsystem so the pipeline can shut adapters down on the right loop, even when verification ran across worker threads.

### What the Registry Tracks

`karenina.adapters.registry` keeps two module-level structures:

| Structure | Type | Purpose |
|-----------|------|---------|
| `_active_adapters` | `list[Any]` | Every adapter handed out by the factory functions, used by `cleanup_all_adapters()` for global teardown. |
| `_adapter_portal_refs` | `list[tuple[weakref.ref, BlockingPortal]]` | Per-portal weak references, recorded only when the creating thread has an active `anyio.from_thread.BlockingPortal`. Used for loop-affine teardown. |

Each structure has its own lock (`_adapters_lock`, `_adapter_portal_lock`). The lock hierarchy is: take `_adapter_portal_lock` only while building a snapshot list, then release it before invoking `aclose()`. This keeps `register_adapter()` calls on other threads from deadlocking against an in-progress shutdown.

### Registration Functions

| Function | Purpose |
|----------|---------|
| `register_adapter(adapter)` | Append to `_active_adapters`; if a `BlockingPortal` is active in the creating thread (via `karenina.benchmark.verification.executor.get_async_portal`), also record a weak `(adapter, portal)` pair for loop-affine cleanup. |
| `unregister_adapter(adapter)` | Remove from `_active_adapters` (used when an adapter is closed manually). |
| `cleanup_all_adapters()` | Async function: snapshot `_active_adapters`, clear the list, and call `aclose()` on every entry that defines it. Exceptions are logged and suppressed so one failing adapter does not block the others. Safe to call multiple times. |
| `snapshot_adapters_for_portal(portal)` | Return the live adapters whose recorded portal matches `portal`. The lock is dropped before the caller invokes `aclose()`. |
| `clear_portal_adapter_refs(portal)` | Drop all `(adapter, portal)` entries for the given portal after its scoped teardown completes; prevents the module-global list from leaking entries across runs. |

### Loop-Affine Teardown

`httpx.AsyncClient.aclose()` (and several SDK clients built on it) raises "Event loop is closed" if it runs on a different loop than the one that opened its transports. Karenina's verification executor runs work through a portal, and the portal's loop is torn down at the end of the run. The portal-keyed weakref list lets the executor:

1. Build the per-portal list with `snapshot_adapters_for_portal(portal)`.
2. Schedule each `aclose()` on the portal's own loop, before the portal exits.
3. Call `clear_portal_adapter_refs(portal)` to drop the entries.

Adapters that do not run inside a portal (sync code paths, single-shot tests) skip the per-portal tracking entirely; only the global `_active_adapters` list applies and `cleanup_all_adapters()` is enough.

### When `aclose()` Runs

Adapters' `aclose()` is invoked in two places:

1. Per-portal teardown inside the verification executor, on the originating loop, before the portal closes.
2. The global `cleanup_all_adapters()` at the end of a run, which closes anything left in `_active_adapters` (e.g., adapters created on the main loop without a portal).

Implementations must be idempotent: each adapter's `aclose()` can run more than once without raising.

### Portal-Affinity Guarantee

When an adapter is recorded with a portal, its `aclose()` is guaranteed to run on the portal's loop before the portal is torn down. Adapters created without a portal are closed on whichever loop runs `cleanup_all_adapters()`. The factory functions (`get_llm`, `get_agent`, `get_parser`) call `register_adapter(adapter)` automatically, so adapters obtained through the public API always participate in this contract.

### Custom Adapters with Lazy Async Resources

The factory's `register_adapter()` call happens once, at construction time. If your adapter constructs `httpx.AsyncClient`, an MCP session, or any other loop-bound resource lazily on the first call (rather than in `__init__`), call `register_adapter(self)` again from the path that creates the resource so the portal binding is correct for the loop that opened it. Adapters that allocate everything in `__init__` do not need to do anything: the factory has already registered them.

For testing utilities related to the registry, see [Testing Utilities](#testing-utilities) below.

---

## Testing Utilities

The registry layer exposes two helpers that exist solely for tests; do not use them in production code.

| Helper | Module | Purpose |
|--------|--------|---------|
| `AdapterRegistry._reset()` | `karenina.adapters.registry` | Clear `_specs`, reset `_initialized` and `_initializing`. Use in fixtures that need to register a custom `AdapterSpec` without colliding with built-ins. |
| `AdapterInstructionRegistry.clear()` | `karenina.ports.adapter_instruction` | Clear all `(interface, task)` factory mappings. Use when a test needs a clean slate for prompt instructions. |

A typical pytest fixture pairs the two:

```python
import pytest

from karenina.adapters.registry import AdapterRegistry
from karenina.ports.adapter_instruction import AdapterInstructionRegistry


@pytest.fixture
def clean_adapter_registry():
    AdapterRegistry._reset()
    AdapterInstructionRegistry.clear()
    yield
    AdapterRegistry._reset()
    AdapterInstructionRegistry.clear()
```

Both helpers are module-global mutators: never call them from library code, and never leave them invoked outside a fixture's setup/teardown.

---

## Key Design Patterns

**Lazy imports** — Import your provider's SDK inside factory functions and adapter methods, not at module level. This prevents import errors when the SDK isn't installed and keeps startup fast.

**Duck typing** — Don't inherit from port protocols. Implement the method signatures and Python's `@runtime_checkable` Protocol system handles the rest. This keeps your adapter independent of karenina's internal types.

**Async-first** — Implement `ainvoke`/`aparse_to_pydantic`/`arun` as the primary API. Sync wrappers (`invoke`/`parse_to_pydantic`/`run`) should call the async version via `asyncio.run()`.

**Message conversion** — Convert between karenina's `Message` type and your provider's message format in a dedicated method. The `Message` class provides `role`, `content` (list of content blocks), and `text` (convenience string property).

**Usage tracking** — Always populate `UsageMetadata` with at least `input_tokens` and `output_tokens`. The pipeline uses these for cost tracking and reporting.

**Adapter cleanup** — `aclose()` is a required protocol method on all three ports. Every adapter must implement it. If the adapter holds no resources, implement it as an empty async method. For adapters that manage MCP sessions, use `AsyncExitStack` to keep sessions alive across the agent loop and clean them up in `aclose()`. The registry calls `cleanup_all_adapters()` at shutdown.

**max_retries varies by adapter** — The `with_structured_output(schema, max_retries=N)` parameter is not universally supported. If your adapter does not support it, emit `logger.warning()` when it is passed so callers know the value is being ignored.

---

## Related

- [Adapter Architecture](index.md) — Hexagonal architecture overview
- [Port Types](ports.md) — Complete protocol signatures for all three ports
- [Available Adapters](available-adapters.md) — Existing adapter implementations for reference
- [Prompt Assembly](../advanced-pipeline/prompt-assembly.md) — How adapter instructions integrate into the prompt pipeline
- [Custom Stages](../advanced-pipeline/custom-stages.md) — Extending the verification pipeline with custom stages
