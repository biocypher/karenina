# Adapters

Adapters are the implementations behind Karenina's LLM backend abstraction. They translate port protocol calls into provider-specific API calls so the [verification pipeline](../advanced-pipeline/index.md) never talks to an LLM provider directly. It calls a port, and the adapter handles the rest.

## 1. What Are Adapters?

The verification pipeline needs LLMs for three distinct purposes: generating answers, parsing responses into structured data, and running quality checks and rubric evaluation. Rather than coding those interactions against a specific provider, the pipeline calls abstract **port protocols**. Adapters implement those protocols for specific backends.

The most important idea is: **the pipeline does not know which LLM backend it is using.** Every pipeline stage sees the same port interface regardless of whether the backend is LangChain, the Anthropic Agent SDK, the Anthropic Python SDK, or a set of pre-recorded traces. Swapping backends means changing a single configuration field (`interface` on `ModelConfig`); no pipeline code changes.

### 1.1. Adapters vs. Ports

These two terms appear together frequently. They serve complementary roles:

| Concept | What It Is | Who Defines It | Who Uses It |
|---------|-----------|----------------|-------------|
| **Port** | A Python `Protocol` class defining methods like `ainvoke()` or `aparse_to_pydantic()` | The framework (`karenina.ports`) | Pipeline stages |
| **Adapter** | A concrete class implementing a port for a specific LLM backend | Each backend package (`karenina.adapters.*`) | The factory creates them; pipeline stages call them through the port interface |

Ports define the contract. Adapters fulfill it. You configure which adapter to use; the pipeline uses only the port interface.

### 1.2. The Abstraction Boundary

What adapters handle:

- Translating port method calls into provider-specific API calls
- Authentication and connection management
- Retry logic and error handling
- Response format conversion into Karenina's unified `Message` and `AgentResult` types
- Structured output enforcement (native API support or JSON parsing fallback)

What adapters do **not** handle:

- Prompt construction (handled by the [PromptAssembler](../advanced-pipeline/prompt-assembly.md))
- Pipeline stage sequencing (handled by the [StageOrchestrator](../advanced-pipeline/index.md))
- Result storage (handled by the results manager)
- Which models to use (configured via [ModelConfig](../reference/configuration/verification-config.md))

### 1.3. Duck Typing

Adapters implement ports via **duck typing**, not class inheritance. Port protocols are Python `Protocol` classes. Any object with matching method signatures satisfies the contract:

```python
# Port protocol (in karenina/ports/llm.py)
class LLMPort(Protocol):
    async def ainvoke(self, messages: list[Message]) -> LLMResponse: ...
    def invoke(self, messages: list[Message]) -> LLMResponse: ...

# Adapter (in karenina/adapters/langchain/llm.py)
class LangChainLLMAdapter:  # No inheritance from LLMPort
    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        # LangChain-specific implementation
        ...
```

This means you can create a new adapter without importing the port classes; just implement the right methods. Type checkers verify compliance statically.

## 2. The Three Ports

Each port protocol defines a specific type of LLM interaction used during verification:

| Port | Purpose | Key Methods | Used During |
|------|---------|-------------|-------------|
| **AgentPort** | Multi-turn agent execution with tools and MCP servers | `arun()` / `run()` | Stage 2: generate answer |
| **ParserPort** | LLM-based structured output parsing (judge LLM fills a template schema) | `aparse_to_pydantic()` / `parse_to_pydantic()` | Stage 7: parse template |
| **LLMPort** | Stateless single-call LLM invocation (no tool loop) | `ainvoke()` / `invoke()` | Stages 5, 6 (quality checks), 11 (rubric evaluation), deep judgment |

All three follow an async-first design: the primary methods are `async` (`ainvoke`, `aparse_to_pydantic`, `arun`), with synchronous wrappers for convenience.

### Capabilities

LLM and Parser adapters expose a `capabilities` property that the [PromptAssembler](../advanced-pipeline/prompt-assembly.md) reads when formatting prompts:

| Capability | Meaning | Effect When `False` |
|------------|---------|---------------------|
| `supports_system_prompt` | Adapter handles separate system messages | System text is prepended to the user message instead |
| `supports_structured_output` | Adapter enforces JSON schema natively (parser only) | JSON schema instructions are added to the prompt text and the text response is parsed manually |

For complete method signatures and detailed protocol documentation, see [Port Types](../advanced-adapters/ports.md).

## 3. Available Interfaces

Karenina ships with six interfaces, configured via the `interface` field on `ModelConfig`. The default is `"langchain"`.

| Interface | Backend | MCP | Tools | Parser: Native Structured Output | Fallback |
|-----------|---------|:---:|:-----:|:-------------------------------:|----------|
| `langchain` | LangChain (multi-provider) | Yes | Yes | No | None |
| `openrouter` | OpenRouter API (routes to `langchain`) | Yes | Yes | No | None |
| `openai_endpoint` | OpenAI-compatible endpoints (routes to `langchain`) | Yes | Yes | No | None |
| `claude_agent_sdk` | Anthropic Agent SDK via Claude CLI | Yes | Yes | Yes | `langchain` |
| `claude_tool` | Anthropic Python SDK (`anthropic` package) | Yes | Yes | Yes | `langchain` |
| `manual` | Pre-recorded traces | No | No | N/A | None |

The **"Parser: Native Structured Output"** column indicates whether the parser adapter uses the LLM API's built-in schema enforcement (tool use or JSON mode) rather than embedding JSON instructions in the prompt and parsing the text response. This distinction matters for parsing reliability on complex schemas. All non-manual LLM adapters support structured output on the LLM port via `with_structured_output()`.

### 3.1. Interface Categories

The six interfaces fall into three categories.

**Native adapters** have their own implementations of all three ports:

- **`langchain`**: The default. Uses LangChain's `init_chat_model` and supports all LangChain-compatible providers: Anthropic, OpenAI, Google, and many more. The broadest-compatibility option.
- **`claude_agent_sdk`**: Uses the Claude CLI (`claude` binary) for agent execution and the Anthropic Agent SDK for LLM and parser operations. Provides native structured output in the parser.
- **`claude_tool`**: Uses the `anthropic` Python package directly with a tool_runner for agent execution. Provides native structured output in the parser without requiring the Claude CLI.
- **`manual`**: Replays pre-recorded traces from `ManualTraceManager`. Only the agent port is functional; LLM and parser adapters raise `ManualInterfaceError` if invoked. See [Manual Interface](../notebooks/core_concepts/manual-interface.ipynb).

**Routing interfaces** share the `langchain` adapter implementation with interface-specific configuration:

- **`openrouter`**: Routes to `langchain` with OpenRouter API configuration. Requires the `OPENROUTER_API_KEY` environment variable.
- **`openai_endpoint`**: Routes to `langchain` with a custom base URL and API key. Requires `endpoint_base_url` and `endpoint_api_key` on `ModelConfig`.

## 4. Choosing an Interface

```
1. Are you replaying pre-recorded traces (no live LLM calls)?
   │
   ├─ YES → manual
   │
   └─ NO
       │
       2. Do you need non-Anthropic models (OpenAI, Google, etc.)?
          │
          ├─ YES
          │   │
          │   3. Access many models through a single API key?
          │      │
          │      ├─ YES → openrouter
          │      │
          │      └─ NO
          │          │
          │          4. Running a local LLM server (Ollama, vLLM, LM Studio)?
          │             │
          │             ├─ YES → openai_endpoint
          │             │
          │             └─ NO → langchain
          │
          └─ NO (Claude models only)
              │
              5. Do you need native structured output in the parser?
                 │
                 ├─ YES
                 │   │
                 │   6. Is the Claude CLI available in PATH?
                 │      │
                 │      ├─ YES → claude_agent_sdk
                 │      │
                 │      └─ NO → claude_tool (requires anthropic package)
                 │
                 └─ NO → langchain (with model_provider="anthropic")
```

### Quick-Reference Table

| Scenario | Recommended |
|----------|-------------|
| Getting started, not sure what to pick | `langchain` |
| Multi-provider evaluation (Anthropic + OpenAI + Google) | `langchain` |
| 200+ models through one API key | `openrouter` |
| Local LLM server (Ollama, vLLM, LM Studio) | `openai_endpoint` |
| Claude-only with full native features | `claude_agent_sdk` |
| Claude-only, lightweight, no CLI dependency | `claude_tool` |
| Replay pre-recorded traces (testing, CI, TaskEval) | `manual` |

### Tradeoffs

| Interface | Advantage | Limitation |
|-----------|-----------|------------|
| `langchain` | Broadest model coverage; stable default | Parser uses JSON text parsing (less reliable on complex schemas) |
| `openrouter` | Single API key for many models | Same parser limitation as `langchain`; depends on OpenRouter availability |
| `openai_endpoint` | Works with any OpenAI-compatible server | Same parser limitation as `langchain`; requires per-model endpoint config |
| `claude_agent_sdk` | Native structured output; thinking support; direct SDK access | Requires Claude CLI binary installed in PATH |
| `claude_tool` | Native structured output without CLI dependency | Requires `anthropic` package; Claude models only |
| `manual` | No live API calls; fully deterministic | Only agent port works; no generation or parsing |

## 5. Routing and Fallback

### Routing Interfaces

`openrouter` and `openai_endpoint` are **routing interfaces**. They resolve to the `langchain` adapter at creation time, sharing the same adapter implementation with interface-specific configuration (API keys, base URLs). From the pipeline's perspective, they behave identically to `langchain`.

### Automatic Fallback

When an adapter's required dependency is missing, the factory transparently falls back to an alternative:

| Interface | Dependency Check | Falls Back To |
|-----------|-----------------|---------------|
| `claude_agent_sdk` | `claude` CLI binary in PATH | `langchain` |
| `claude_tool` | `anthropic` Python package importable | `langchain` |
| `langchain` | `langchain_core` Python package importable | None (raises `AdapterUnavailableError`) |
| `manual` | Always available | None (no dependency) |

Fallback is automatic when `auto_fallback=True` (the default) in the factory functions. When fallback occurs, a warning is logged. Set `auto_fallback=False` to raise an error instead:

```python
from karenina.adapters.factory import get_agent

# Raise an error if claude_agent_sdk is not available (do not fall back)
agent = get_agent(config, auto_fallback=False)
```

## 6. Configuration

### ModelConfig

Adapters are configured through `ModelConfig`. The `interface` field selects the adapter; other fields provide backend-specific settings:

```python
from karenina.schemas.config import ModelConfig

# Default: LangChain with Anthropic (langchain is the only interface
# that requires model_provider)
config = ModelConfig(
    id="haiku",
    model_name="claude-haiku-4-5",
    model_provider="anthropic",
    interface="langchain",  # default; can be omitted
)

# Claude via Agent SDK (native structured output in parser)
config = ModelConfig(
    id="claude",
    model_name="claude-sonnet-4-20250514",
    interface="claude_agent_sdk",
)

# OpenAI-compatible endpoint (local server)
config = ModelConfig(
    id="local",
    model_name="llama-3-70b",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",
    endpoint_api_key="not-needed",
)
```

<div class="admonition note">
<p class="admonition-title">Provider requirements</p>
<p>Only the <code>langchain</code> interface requires <code>model_provider</code> to be set. All other interfaces either assume a specific provider (<code>claude_agent_sdk</code>, <code>claude_tool</code>) or do not need one (<code>openrouter</code>, <code>openai_endpoint</code>, <code>manual</code>).</p>
</div>

### Factory Functions

The three factory functions create port adapters from a `ModelConfig`:

```python
from karenina.adapters.factory import get_llm, get_agent, get_parser

llm = get_llm(config)       # Returns an LLMPort implementation
agent = get_agent(config)    # Returns an AgentPort implementation
parser = get_parser(config)  # Returns a ParserPort implementation
```

All three functions:

1. Read `model_config.interface` to select the adapter
2. Validate required fields (model name; model provider for `langchain` only)
3. Check adapter availability via the registry
4. Fall back automatically if the preferred adapter is unavailable
5. Return a port implementation (never `None`)

In normal usage, you do not call these functions directly. The verification pipeline calls them based on your `VerificationConfig`.

## 7. How Adapters Connect to the Pipeline

The pipeline creates and invokes adapters based on the model roles in your [VerificationConfig](../reference/configuration/verification-config.md). Each role can use a different interface:

```
VerificationConfig
├── answering_models[0]  →  get_agent()  →  Stage 2: generate answer
├── parsing_models[0]    →  get_parser() →  Stage 7: parse template
│                        →  get_llm()    →  Stages 5-6: quality checks
└── rubric_models[0]     →  get_llm()    →  Stage 11: rubric evaluation
```

For example, you might generate answers with `claude_agent_sdk` (for tool use and MCP) while parsing templates with `langchain` (to use a different judge model from a different provider). The pipeline does not care; it calls the same port methods in both cases.

## 8. Adapter Instructions

Each adapter family can register **adapter instructions** that customize the prompts the [PromptAssembler](../advanced-pipeline/prompt-assembly.md) builds. For example:

- The `claude_tool` and `claude_agent_sdk` adapters strip JSON schema instructions from parsing prompts because their native structured output handles schema enforcement automatically
- The `langchain` adapter adds explicit JSON formatting instructions since it parses the LLM's text response manually

This mechanism keeps adapters as pure executors: they receive pre-assembled messages and do not build prompts internally. Adapter-specific prompt tuning is declarative and registered at import time. For the complete prompt assembly system, see [Prompt Assembly](../advanced-pipeline/prompt-assembly.md).

## 9. Next Steps

- [Port Types](../advanced-adapters/ports.md): Complete protocol signatures for LLMPort, ParserPort, and AgentPort
- [Available Adapters](../advanced-adapters/available-adapters.md): Per-adapter features, configuration, and capabilities
- [Writing Custom Adapters](../advanced-adapters/writing-adapters.md): Implementing and registering new adapters
- [MCP Overview](../notebooks/core_concepts/mcp-overview.ipynb): Tool-augmented evaluation with MCP servers
- [Manual Interface](../notebooks/core_concepts/manual-interface.ipynb): Pre-recorded traces for offline evaluation
- [Evaluation Modes](../notebooks/core_concepts/evaluation-modes.ipynb): Controlling which evaluation units run
- [Running Verification](../workflows/running-verification/index.md): End-to-end verification workflow
