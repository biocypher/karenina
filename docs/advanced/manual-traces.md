# Manual Trace System

This guide explains how to use the Manual Trace System to evaluate pre-generated LLM responses without making live API calls during verification.

---

## Overview

The Manual Trace System enables you to provide pre-generated answer traces directly to the Karenina verification engine, bypassing the real-time LLM answer generation step. This feature is designed for standalone backend usage, allowing you to:

- Evaluate pre-recorded LLM responses (from previous experiments, other systems, or manual collection)
- Compare different answer generation approaches without re-running models
- Test verification/rubric systems with controlled answers
- Integrate external LLM outputs into Karenina's evaluation framework

The system supports multiple trace formats:
- Simple string traces (plain text answers)
- Port Message lists (new architecture using `karenina.ports.messages.Message`)
- LangChain message lists (backward compatibility with automatic conversion)

---

## Key Capabilities

- **Programmatic Trace Management**: `ManualTraces` class for managing traces tied to benchmarks
- **Flexible Registration**: Register traces by question hash (MD5) or question text with automatic mapping
- **Multi-Format Support**:
  - Simple string traces (plain text answers)
  - Port Message lists (`karenina.ports.messages.Message`) - native format
  - LangChain message lists (AIMessage, ToolMessage, etc.) - backward compatibility
- **Agent Metrics Extraction**: Automatic tool call counting, failure detection from message lists
- **Batch Registration**: Efficient bulk trace registration with `register_traces()`
- **Post-Config Population**: Populate traces after `ModelConfig` creation for flexible workflows
- **Preset Compatibility**: Manual configs work with preset system (traces excluded from serialization)
- **Session-Based Storage**: Thread-safe, time-bounded trace storage with automatic cleanup
- **Backward Compatible**: Maintains compatibility with existing GUI-based manual trace upload

---

## Quick Start

### Basic Usage with String Traces

```python
from karenina.benchmark import Benchmark
from karenina.adapters.manual import ManualTraces
from karenina.schemas import ModelConfig, VerificationConfig
from pydantic import Field
from karenina.schemas.domain import BaseAnswer

# Define Answer template (must be named "Answer")
class Answer(BaseAnswer):
    value: str = Field(description="The answer value")
    def verify(self) -> bool:
        return len(self.value) > 0

# Create benchmark
benchmark = Benchmark("my_experiment")
benchmark.add_question(
    question="What is 2+2?",
    raw_answer="4",
    answer_template=Answer
)  # Question automatically marked as finished

# Initialize manual traces
manual_traces = ManualTraces(benchmark)

# Register trace by question text
manual_traces.register_trace(
    "What is 2+2?",
    "The answer is 4. I computed this by adding 2 and 2.",
    map_to_id=True
)

# Create manual config
manual_config = ModelConfig(
    interface="manual",
    manual_traces=manual_traces
)

# Create judge config
judge_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain",
    system_prompt="You are an expert judge."
)

# Run verification
config = VerificationConfig(
    answering_models=[manual_config],
    parsing_models=[judge_config]
)

benchmark.run_verification(config)
```

---

## Architecture

### Core Components

#### 1. `ManualTraces` Class
**Location**: `karenina/src/karenina/adapters/manual/traces.py`

**Purpose**: High-level API for managing manual traces for a specific benchmark

**Key Methods**:

- `__init__(benchmark)` - Initialize with benchmark for question mapping
- `register_trace(question_identifier, trace, map_to_id=False)` - Register single trace
- `register_traces(traces_dict, map_to_id=False)` - Batch register traces
- `_question_text_to_hash(question_text)` - Convert text to MD5 hash with validation
- `_preprocess_trace(trace)` - Handle string, port Message, and LangChain formats

#### 2. `ManualTraceManager` Class
**Location**: `karenina/src/karenina/adapters/manual/manager.py`

**Purpose**: Session-based thread-safe storage for manual traces

**Key Features**:

- Thread-safe storage with `threading.RLock()`
- Session timeout (default: 1 hour) with automatic cleanup
- Storage for both traces and agent metrics
- MD5 hash validation

**Key Methods**:

- `set_trace(question_hash, trace, agent_metrics=None)` - Store trace programmatically
- `get_trace(question_hash)` - Retrieve trace
- `get_trace_with_metrics(question_hash)` - Retrieve trace and metrics
- `load_traces_from_json(json_data)` - Load from JSON (GUI upload compatibility)

#### 3. Message Utilities
**Location**: `karenina/src/karenina/adapters/manual/message_utils.py`

**Purpose**: Message processing utilities using port-based architecture

**Key Functions**:

- `convert_langchain_messages(messages)` - Convert LangChain messages to port format
- `harmonize_messages(messages)` - Convert message list to string trace
- `extract_agent_metrics(messages)` - Extract tool call counts, failures, iterations
- `preprocess_message_list(messages)` - Main entry point for message preprocessing
- `is_port_message_list(messages)` - Detect port Message format
- `is_langchain_message_list(messages)` - Detect LangChain format

#### 4. `ModelConfig` Integration
**Location**: `karenina/src/karenina/schemas/workflow/models.py`

**New Field**: `manual_traces: Any = Field(default=None, exclude=True)`

**Validation**:

- Enforces `manual_traces` requirement for manual interface
- Auto-sets `id="manual"` and `model_name="manual"` for manual interface
- Validates that MCP tools are not used with manual interface

---

## User Workflows

### Workflow 1: Basic String Traces

```python
from karenina.benchmark import Benchmark
from karenina.adapters.manual import ManualTraces
from karenina.schemas import ModelConfig, VerificationConfig

# Create benchmark
benchmark = Benchmark("simple_example")
benchmark.add_question(question="What is 2+2?", raw_answer="4")
benchmark.add_question(question="What is 3+3?", raw_answer="6")

# Initialize manual traces
manual_traces = ManualTraces(benchmark)

# Register traces by question text
manual_traces.register_trace(
    "What is 2+2?",
    "The answer is 4. I added 2 and 2 to get 4.",
    map_to_id=True
)

manual_traces.register_trace(
    "What is 3+3?",
    "The answer is 6. I added 3 and 3 to get 6.",
    map_to_id=True
)

# Create manual config
manual_config = ModelConfig(
    interface="manual",
    manual_traces=manual_traces
)

# Create judge config
judge_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

# Run verification
config = VerificationConfig(
    answering_models=[manual_config],
    parsing_models=[judge_config]
)

benchmark.run_verification(config)
```

---

### Workflow 2: LangChain Message Lists with Tool Calls

```python
from langchain_core.messages import AIMessage, ToolMessage
from karenina.adapters.manual import ManualTraces

# Assume benchmark already created
manual_traces = ManualTraces(benchmark)

# Register trace with tool calls
messages = [
    AIMessage(content="I need to calculate this"),
    ToolMessage(
        name="calculator",
        content="Result: 42",
        tool_call_id="call_calc_001"
    ),
    ToolMessage(
        name="validator",
        content="Validation passed",
        tool_call_id="call_valid_002"
    ),
    AIMessage(content="The answer is 42. I verified this using a calculator and validator.")
]

manual_traces.register_trace(
    "What is 6 times 7?",
    messages,
    map_to_id=True
)

# Agent metrics automatically extracted:
# - tool_calls: 2
# - unique_tools_used: 2 (calculator, validator)
# - iterations: 1
```

---

### Workflow 3: Port Message Lists (New Architecture)

```python
from karenina.ports.messages import Message, ToolUseContent
from karenina.adapters.manual import ManualTraces

# Assume benchmark already created
manual_traces = ManualTraces(benchmark)

# Register trace using port Message format (native architecture)
messages = [
    Message.user("What is 6 times 7?"),
    Message.assistant(
        "I'll calculate that.",
        tool_calls=[ToolUseContent(id="calc1", name="calculator", input={"expr": "6*7"})]
    ),
    Message.tool_result("calc1", "42"),
    Message.assistant("The answer is 42.")
]

manual_traces.register_trace(
    "What is 6 times 7?",
    messages,
    map_to_id=True
)

# Works exactly like LangChain messages, with same metrics extraction
```

---

### Workflow 3: Batch Registration

```python
# Prepare traces dictionary
traces = {
    "Question 1?": "Answer 1",
    "Question 2?": [
        AIMessage(content="Thinking..."),
        ToolMessage(name="tool", content="data", tool_call_id="call_1"),
        AIMessage(content="Answer 2")
    ],
    "Question 3?": "Answer 3"
}

# Batch register all at once
manual_traces.register_traces(traces, map_to_id=True)

# All traces now available for verification
```

---

### Workflow 4: Register by Question Hash

```python
import hashlib

# Compute hash manually or get from CSV mapper export
question_hash = hashlib.md5("What is 2+2?".encode("utf-8")).hexdigest()

# Register by hash (map_to_id=False, default)
manual_traces.register_trace(
    question_hash,
    "The answer is 4.",
    map_to_id=False
)
```

---

### Workflow 5: Populate Traces After Config Creation

```python
# 1. Create ManualTraces and ModelConfig upfront
manual_traces = ManualTraces(benchmark)
manual_config = ModelConfig(interface="manual", manual_traces=manual_traces)

# 2. Later, populate traces (e.g., from file, database, API)
for question_text, trace_content in load_traces_from_source():
    manual_traces.register_trace(question_text, trace_content, map_to_id=True)

# 3. Run verification with populated traces
config = VerificationConfig(
    answering_models=[manual_config],
    parsing_models=[judge_config]
)
benchmark.run_verification(config)
```

---

## Implementation Details

### Question Mapping

**Hash Generation**:

- Uses MD5 hash of UTF-8 encoded question text: `hashlib.md5(question_text.encode("utf-8")).hexdigest()`
- Same algorithm as `Question.id` property in `schemas/domain/question.py`
- Results in 32-character hexadecimal string

**Validation**:

- When `map_to_id=True`, question text is searched in benchmark's `_questions_cache`
- Raises `ValueError` if question not found, with computed hash and available count
- Exact text matching (case-sensitive, including whitespace)

**Note**: The `_questions_cache` uses URN format IDs as keys, but `_question_text_to_hash()` searches through all questions by text and returns the MD5 hash for trace storage.

---

### Trace Format Processing

**String Traces**:

- Stored as-is with no preprocessing
- No agent metrics extracted (`metrics = None`)
- Simplest format for basic answer evaluation

**LangChain Message Lists**:

1. **Validation**: Must be list of `BaseMessage` objects (AIMessage, ToolMessage, etc.)
2. **Metrics Extraction**:
   - Calls `_extract_agent_metrics(response)` from `verification_utils.py`
   - Extracts: tool calls, tool failures, iterations
3. **Harmonization**:
   - Calls `harmonize_agent_response(response)` from `mcp_utils.py`
   - Converts message list to unified string format
4. **Storage**: Both harmonized trace and metrics stored together

**Error Handling**:

- `TypeError` raised for invalid trace formats (not str or list)
- `ManualTraceError` raised for preprocessing failures (import errors, etc.)

---

### Agent Metrics Propagation

**Flow**:

1. `ManualTraces._preprocess_trace()` extracts metrics during registration
2. `ManualTraceManager.set_trace()` stores metrics alongside trace
3. `ManualLLM.get_agent_metrics()` retrieves metrics during verification
4. `generate_answer.py` calls `usage_tracker.set_agent_metrics()` to store
5. Metrics appear in `VerificationResult` (tool calls, failures, etc.)

**Metrics Structure**:
```python
{
    "tool_calls": 3,           # Number of tool invocations
    "unique_tools_used": 2,    # Number of unique tools
    "failed_tool_calls": 0,    # Number of failed invocations
    "iterations": 1            # Agent iterations
}
```

---

### Session-Based Storage

**Design**:

- Global singleton `ManualTraceManager` instance
- Thread-safe with `threading.RLock()`
- Session timeout: 1 hour (3600 seconds)
- Automatic cleanup of expired traces

**Cleanup Strategy**:

1. **Timer-Based**: `threading.Timer` triggers cleanup after timeout
2. **Activity-Based**: Timer resets on any trace access
3. **Trace-Level**: Individual traces have timestamps, expired traces removed
4. **Session-Level**: If no activity for timeout period, entire session clears

**Memory Management**:

- `get_memory_usage_info()` provides trace count, character count, estimated bytes
- Useful for monitoring large-scale trace storage

---

### ModelConfig Validation

**Requirements for Manual Interface**:

1. `interface` must be `"manual"`
2. `manual_traces` must not be `None` (raises `ValueError` if missing)
3. `id` defaults to `"manual"` if not provided
4. `model_name` defaults to `"manual"` if not provided
5. `mcp_urls_dict` must be `None` (raises `ValueError` if MCP configured)

**Preset Compatibility**:

- `manual_traces` field marked with `Field(exclude=True)`
- Automatically excluded from Pydantic serialization
- Presets save config structure but not trace data
- Traces must be re-populated when loading preset

**Note on Answer Templates**: Answer template classes must be named `Answer` and questions with templates are auto-marked as finished in backend code.

---

## Best Practices

### 1. Question Text Matching

**Do**:

- Use exact question text from benchmark (case-sensitive, including whitespace)
- Use `map_to_id=True` when working with question text
- Verify question text matches benchmark before registration

**Don't**:

- Modify question text (trim whitespace, change case, etc.)
- Assume approximate matching will work
- Register traces for questions not in benchmark

**Tip**: Export CSV mapper from benchmark to see exact question text and hashes

---

### 2. Trace Format Selection

**Use String Traces When**:

- Answers are simple text without tool calls
- No agent metrics needed
- Simplest workflow sufficient

**Use LangChain Message Lists When**:

- Preserving tool call history is important
- Agent metrics (tool calls, failures) are valuable
- Comparing agent-based vs. non-agent-based approaches
- Debugging tool usage patterns

---

### 3. Error Handling

**Common Errors**:

1. **Question Not Found**:
```python
ValueError: Question not found in benchmark: 'What is 2+2?...'
Computed hash: 936dbc8755f623c951d96ea2b03e13bc
```
**Fix**: Verify exact question text matches benchmark, check for whitespace/case differences

2. **Invalid Hash Format**:
```python
ManualTraceError: Invalid question hash format: 'short'
```
**Fix**: Ensure hash is 32-character hexadecimal MD5

3. **Missing Manual Traces**:
```python
ValueError: manual_traces is required when interface='manual'
```
**Fix**: Pass `manual_traces` to ModelConfig constructor

4. **MCP Configuration Conflict**:
```python
ValueError: MCP tools are not supported with manual interface
```
**Fix**: Remove `mcp_urls_dict` from manual ModelConfig

---

### 4. Performance Optimization

**Batch Registration**:

- Use `register_traces()` instead of multiple `register_trace()` calls
- Reduces overhead for large trace sets
- More readable code

**Memory Management**:

- Monitor trace count with `get_manual_trace_count()`
- Check memory usage with `get_memory_usage_info()`
- Clear traces with `clear_manual_traces()` when done

**Session Cleanup**:

- Traces auto-expire after 1 hour of inactivity
- Manual cleanup with `clear_manual_traces()` if needed
- Activity resets timeout (any trace access)

---

### 5. Testing and Validation

**Before Running Verification**:
```python
from karenina.adapters.manual import has_manual_trace, get_manual_trace
import hashlib

# Verify trace was registered
question_hash = hashlib.md5("What is 2+2?".encode("utf-8")).hexdigest()
assert has_manual_trace(question_hash)

# Retrieve and inspect trace
trace = get_manual_trace(question_hash)
print(f"Registered trace: {trace[:100]}...")
```

**Validate Trace Count**:
```python
from karenina.adapters.manual import get_manual_trace_count

expected_count = len(benchmark._questions_cache)
actual_count = get_manual_trace_count()

if actual_count != expected_count:
    print(f"Warning: Expected {expected_count} traces, have {actual_count}")
```

---

### 6. Preset Workflow

**Saving Presets with Manual Configs**:
```python
# Manual traces automatically excluded
config = VerificationConfig(
    answering_models=[manual_config],  # Contains manual_traces
    parsing_models=[judge_config]
)

# Save preset (manual_traces excluded from serialization)
preset_service.save_preset("my_preset", config)
```

**Loading Presets with Manual Configs**:
```python
# 1. Load preset (manual_traces will be None)
loaded_config = preset_service.load_preset("my_preset")

# 2. Re-populate manual traces
manual_traces = ManualTraces(benchmark)
manual_traces.register_traces(traces_dict, map_to_id=True)

# 3. Update config with traces
loaded_config.answering_models[0].manual_traces = manual_traces

# 4. Run verification
benchmark.run_verification(loaded_config)
```

---

## Complete Examples

### Example 1: Simple String Traces

```python
from karenina.benchmark import Benchmark
from karenina.adapters.manual import ManualTraces
from karenina.schemas import ModelConfig, VerificationConfig

# Create benchmark
benchmark = Benchmark("simple_example")
benchmark.add_question(question="What is 2+2?", raw_answer="4")
benchmark.add_question(question="What is 3+3?", raw_answer="6")

# Initialize manual traces
manual_traces = ManualTraces(benchmark)

# Register traces by question text
manual_traces.register_trace(
    "What is 2+2?",
    "The answer is 4. I added 2 and 2 to get 4.",
    map_to_id=True
)

manual_traces.register_trace(
    "What is 3+3?",
    "The answer is 6. I added 3 and 3 to get 6.",
    map_to_id=True
)

# Create manual config
manual_config = ModelConfig(
    interface="manual",
    manual_traces=manual_traces
)

# Create judge config
judge_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

# Run verification
config = VerificationConfig(
    answering_models=[manual_config],
    parsing_models=[judge_config]
)

benchmark.run_verification(config)
```

---

### Example 2: Batch Registration with Mixed Formats

```python
# Prepare traces dictionary
traces = {
    "Question 1?": "Answer 1",
    "Question 2?": [
        AIMessage(content="Thinking..."),
        ToolMessage(name="tool", content="data", tool_call_id="call_1"),
        AIMessage(content="Answer 2")
    ],
    "Question 3?": "Answer 3"
}

# Batch register all at once
manual_traces.register_traces(traces, map_to_id=True)

# All traces now available for verification
```

---

### Example 3: Delayed Trace Population

```python
# Step 1: Create config structure early
manual_traces = ManualTraces(benchmark)
manual_config = ModelConfig(interface="manual", manual_traces=manual_traces)

# Step 2: Pass config around, set up verification structure
config = VerificationConfig(
    answering_models=[manual_config],
    parsing_models=[judge_config]
)

# Step 3: Later, populate traces (e.g., from file load, API call)
def load_traces_from_file(filepath):
    """Load traces from external file."""
    import json
    with open(filepath) as f:
        traces_data = json.load(f)
    return traces_data

traces_data = load_traces_from_file("experiment_traces.json")

for question_text, trace_content in traces_data.items():
    manual_traces.register_trace(question_text, trace_content, map_to_id=True)

# Step 4: Run verification with populated traces
benchmark.run_verification(config)
```

---

## Troubleshooting

### Issue: "No manual trace found for question hash"

**Error**:
```
ManualTraceNotFoundError: No manual trace found for question hash: '936dbc8755f623c951d96ea2b03e13bc'
```

**Causes**:

1. Trace not registered for that question
2. Question text mismatch during registration
3. Traces cleared or expired

**Solutions**:

1. Verify trace was registered: `has_manual_trace(question_hash)`
2. Check exact question text matches benchmark
3. Re-register traces if session expired
4. Verify `map_to_id=True` was used if registering by text

---

### Issue: Question not found in benchmark

**Error**:
```
ValueError: Question not found in benchmark: 'What is 2+2?...'
```

**Causes**:

1. Question text doesn't match benchmark exactly
2. Question not added to benchmark
3. Whitespace/case differences

**Solutions**:

1. Export CSV mapper to see exact question text
2. Verify question was added to benchmark
3. Check for hidden whitespace or case differences
4. Use question hash directly (`map_to_id=False`)

---

### Issue: Invalid trace format

**Error**:
```
TypeError: Invalid trace format: expected str or list[BaseMessage], got <class 'dict'>
```

**Causes**:

1. Passing invalid trace format (not str or list of messages)

**Solutions**:

1. Use string: `"The answer is 4"`
2. Use message list: `[AIMessage(content="..."), ToolMessage(...)]`
3. Don't use dicts or other formats

---

### Issue: Agent metrics not appearing

**Symptoms**:

- Verification result missing tool call counts
- Metrics are None or 0

**Causes**:

1. Using string trace format (no metrics)
2. Message list doesn't contain tool calls
3. Preprocessing failed silently

**Solutions**:

1. Verify using message list format, not string
2. Ensure messages include ToolMessage objects
3. Check message structure matches LangChain format
4. Inspect preprocessing with `get_manual_trace_with_metrics()`

---

## API Reference

### ManualTraces Class

```python
class ManualTraces:
    """High-level API for managing manual traces for a specific benchmark."""

    def __init__(self, benchmark: Benchmark):
        """Initialize with benchmark for question mapping."""

    def register_trace(
        self,
        question_identifier: str,
        trace: Union[str, List[BaseMessage]],
        map_to_id: bool = False
    ) -> None:
        """Register a single trace.

        Args:
            question_identifier: Question hash or question text
            trace: String or LangChain message list
            map_to_id: If True, treat identifier as question text and map to hash
        """

    def register_traces(
        self,
        traces_dict: Dict[str, Union[str, List[BaseMessage]]],
        map_to_id: bool = False
    ) -> None:
        """Batch register multiple traces.

        Args:
            traces_dict: Dictionary mapping question identifiers to traces
            map_to_id: If True, treat identifiers as question text and map to hashes
        """
```

### Global Functions

```python
def has_manual_trace(question_hash: str) -> bool:
    """Check if a manual trace exists for a question hash."""

def get_manual_trace(question_hash: str) -> str:
    """Retrieve a manual trace for a question hash."""

def get_manual_trace_with_metrics(question_hash: str) -> Tuple[str, Optional[Dict]]:
    """Retrieve a manual trace and its agent metrics."""

def get_manual_trace_count() -> int:
    """Get the number of registered manual traces."""

def clear_manual_traces() -> None:
    """Clear all registered manual traces."""

def get_memory_usage_info() -> Dict[str, Any]:
    """Get memory usage information for manual traces."""
```

---

## Testing

### Unit Tests

**File**: `karenina/tests/test_manual_traces.py`

**Coverage**:

- ManualTraces class initialization
- Trace registration (by hash, by text, batch)
- Format handling (string, message list)
- Question mapping and validation
- ModelConfig validation
- Error cases

**Run**: `uv run pytest karenina/tests/test_manual_traces.py -v`

---

### Integration Tests

**File**: `karenina/tests/test_manual_traces_integration.py`

**Coverage**:

- Complete workflows (string traces, message traces)
- Batch registration
- Delayed trace population
- Mixed trace formats
- ModelConfig validation in verification context

**Run**: `uv run pytest karenina/tests/test_manual_traces_integration.py -v`

---

### Test Summary

**Total**: 31 tests (25 unit + 6 integration)
**Status**: All passing ✅
**Command**: `uv run pytest karenina/tests/test_manual_traces*.py -v`

---

## Related Documentation

- **Quick Start**: Basic verification workflow
- **Verification**: Complete verification documentation
- **Configuration**: Model and provider configuration
- **Troubleshooting**: Common issues and solutions
- **[TaskEval](task-eval.md)**: Evaluate agent workflow traces (similar concept for pre-existing outputs, focused on rubric-based quality assessment)

---

## Summary

The Manual Trace System enables:

1. **Pre-Generated Answer Evaluation** - Evaluate LLM responses without making live API calls
2. **Flexible Trace Formats** - Support for both simple strings and rich message lists
3. **Agent Metrics Extraction** - Automatic tool call and failure tracking
4. **Efficient Workflows** - Batch registration, delayed population, preset compatibility
5. **Production-Ready** - Thread-safe, session-based, with automatic cleanup

**Key Workflow**: Create benchmark → Initialize `ManualTraces` → Register traces → Create `ModelConfig` with `interface="manual"` → Run verification
