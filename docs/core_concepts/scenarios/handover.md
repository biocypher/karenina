---
jupyter:
  jupytext:
    formats: docs/core_concepts/scenarios//md,docs/notebooks/core_concepts/scenarios//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Handover (Multi-Agent Routing)

Handover is the per-edge mechanism that decides what context the next node sees when execution crosses from one node to another. It is most relevant when the source and target nodes belong to different agents (different `agent_identity` labels), but the same machinery applies to any edge that needs explicit control over the prompt and conversation history fed into the next turn.

This page documents the four handover behaviours the runtime understands, the `TaggedMessage` structure that carries multi-agent provenance through the run, and the `format_transcript`, `apply_handover`, and `TRANSCRIPT_SEPARATOR` public surface. For where `handover=` is attached to an edge, see [Building Scenarios](building-scenarios.md). For how the runner threads `agent_identity` through tagged messages, see [State and Routing](state-and-routing.md). For the executor that hosts the scenario run, see [Execution](execution.md).

```python tags=["hide-cell"]
# Mock setup for documentation: allows the notebook to run without API keys.
# Hidden in rendered docs. The real Message and TaggedMessage classes are
# importable on a fully-installed environment, but the doc build avoids the
# full karenina init chain by exposing minimal stand-ins.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


@dataclass
class TextContent:
    text: str
    type: ContentType = ContentType.TEXT


@dataclass
class ToolUseContent:
    name: str
    input: dict[str, Any]
    type: ContentType = ContentType.TOOL_USE


@dataclass
class ToolResultContent:
    content: str
    type: ContentType = ContentType.TOOL_RESULT


@dataclass
class Message:
    role: Role
    content: list[Any]

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.SYSTEM, content=[TextContent(text=text)])

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=Role.USER, content=[TextContent(text=text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=[TextContent(text=text)])


@dataclass
class TaggedMessage:
    """A Message paired with the agent identity that produced it."""

    message: Message
    agent_id: str


TRANSCRIPT_SEPARATOR = "\n\n---\n\n"


def format_transcript(tagged_messages: list[TaggedMessage]) -> str:
    """Mirror of karenina.scenario.handover.format_transcript for docs."""
    if not tagged_messages:
        return ""
    lines: list[str] = []
    for tm in tagged_messages:
        for block in tm.message.content:
            if block.type == ContentType.THINKING:
                continue
            if tm.agent_id == "__user__":
                label = "[__user__]"
            else:
                label = f"[{tm.agent_id}:{tm.message.role.value}:{block.type.value}]"
            if isinstance(block, TextContent):
                lines.append(f"{label} {block.text}")
            elif isinstance(block, ToolUseContent):
                args = ", ".join(f"{k}={v!r}" for k, v in block.input.items())
                lines.append(f"{label} {block.name}({args})")
            elif isinstance(block, ToolResultContent):
                lines.append(f"{label} {block.content}")
    return "\n".join(lines)


print("Mock setup complete.")
```

## 1. What It Is

Every turn in a scenario produces messages. The runtime appends each message to a single tagged-message log along with the `agent_identity` of the node that produced it. When execution follows an edge, the edge's `handover` setting decides how that log is delivered to the target node:

- `None` (the default): the log flows in untouched as conversation history; the target node receives the messages directly.
- `"transcript_prepend"`, `"transcript_append"`, `"transcript_materialize"`: the log is rendered as a labeled transcript and either inlined into the target's question text or written to a file the target reads.
- A callable: a user-supplied function rewrites the conversation history.

Handover only fires on the *edge* that is followed; if no handover is set on the matched edge, the runtime hands the raw `[m.message for m in tagged_messages]` history to the next turn.

## 2. Core Idea

A scenario log is a tagged stream, not a flat conversation. Each entry knows which agent emitted it. When that stream crosses an agent boundary you usually want to *describe* the prior conversation rather than continue it: the new agent should treat what came before as material to read, not as its own history. Handover strategies are the four ways the runtime supports this translation.

## 3. Anatomy

### TaggedMessage

```python
print(TaggedMessage.__doc__)
print()
print("Fields:")
print("  message: Message  (role, content blocks)")
print("  agent_id: str     (agent_identity of the producing node, or '__user__' for scenario prompts)")
```

`TaggedMessage` is the unit of the scenario message log. The runtime constructs three flavours of `agent_id`:

| `agent_id` value | Origin |
|------------------|--------|
| `"__user__"` | The synthetic user message carrying a node's `question.question` text |
| Resolved `agent_identity` of a node | The system prompt and the assistant's response for that node |
| A model identifier (`model.id` or `model.model_name`) | Fallback when the entry node has no `agent_identity` |

System messages are emitted only when the agent identity or system prompt changes between turns; otherwise the running system prompt is left in place.

### format_transcript

```python
log: list[TaggedMessage] = [
    TaggedMessage(Message.system("You are a clinical reasoning assistant."), agent_id="triage"),
    TaggedMessage(Message.user("What drug class does venetoclax belong to?"), agent_id="__user__"),
    TaggedMessage(Message.assistant("BCL-2 inhibitor."), agent_id="triage"),
    TaggedMessage(Message.user("Are you sure? Most colleagues say BCR-ABL."), agent_id="__user__"),
    TaggedMessage(Message.assistant("It is a BCL-2 inhibitor; venetoclax targets BCL-2 directly."), agent_id="triage"),
]

print(format_transcript(log))
```

Each non-thinking content block becomes one line. Tool calls render as `name(arg=value)`; tool results render as the result string. `[__user__]` stands in for scenario user prompts; agent messages use `[agent_id:role:content_type]`. Thinking blocks are dropped from the transcript so internal reasoning never leaks across handovers.

### TRANSCRIPT_SEPARATOR

```python
print(repr(TRANSCRIPT_SEPARATOR))
```

`TRANSCRIPT_SEPARATOR` is the constant string used to fence transcript content inside a question prompt. `transcript_prepend` and `transcript_append` both use it as the boundary between the rendered transcript and the target node's question text; downstream parsers (for example, `reformat_transcript_as_xml` in `karenina.scenario.trace_materialization`) detect it to know where to split.

### apply_handover

```python
print("apply_handover(edge, tagged_messages, state, question_text, turn_dir=None) -> tuple[str, list[Message]] | None")
print()
print("Returns None when the edge has no handover (raw history passes through).")
print("Returns (modified_question_text, conversation_history) otherwise.")
```

`apply_handover` is the single dispatch point inside `ScenarioManager._run_turn`. It accepts the matched edge, the full tagged-message log, the current `ScenarioState`, the target node's untouched question text, and an optional per-turn workspace directory. It produces either `None` (no handover; let raw history through) or a `(question_text, conversation_history)` pair the runner uses for the next turn. For the three transcript strategies it returns an empty `conversation_history`: the model sees the transcript inside the prompt itself, not as separate prior messages.

## 4. How It Works

### transcript_prepend

The transcript is rendered with `format_transcript` and prepended to the question:

```text
<rendered transcript>

---

<target node's question text>
```

The agent receives a single user message whose body opens with the conversation log and closes with the new task. Use this when you want the new agent to read the conversation as context before tackling the question. There is no separate `conversation_history`.

### transcript_append

The same rendering, with the order reversed:

```text
<target node's question text>

---

<rendered transcript>
```

This puts the question first and the transcript afterwards as supporting evidence. Useful when the question itself is short and self-contained and the transcript is reference material the agent may consult.

### transcript_materialize

The transcript is hashed (first 10 hex chars of SHA-256) and written via `materialize_trace` to a file under `<turn_dir>/traces/`. The target node then receives a question prompt that opens with a file-reading preamble pointing at the trace path, structured XML turns inside the file (`<turn>`, `<system_prompt>`, `<user>`, `<assistant>`, with large content blocks marked `offloaded="true"` and stored under an `artifacts/` subdirectory), and the original question text after a `---` separator.

This is the right strategy when the transcript is large enough that inlining it into the prompt is wasteful (long agentic traces, code-edit sessions). The agent reads the file with whatever tool it has available (e.g., `Read`).

### Callable handover

```python
# Signature: handover_callable(tagged_messages, state) -> list[Message]


def keep_only_assistant_blocks(
    tagged_messages: list[TaggedMessage],
    state: Any,
) -> list[Message]:
    """Strip everything except the prior agent's final assistant message."""
    for tm in reversed(tagged_messages):
        if tm.agent_id != "__user__" and tm.message.role == Role.ASSISTANT:
            return [tm.message]
    return []
```

A callable handover bypasses transcript rendering entirely. The function is called with the full tagged-message log and the current `ScenarioState`; it returns a `list[Message]` that becomes the next turn's conversation history. The target node's `question_text` is unchanged. Callable handovers are not serialized to the SchemaOrg checkpoint format (the builder emits a `UserWarning` when one is registered); reach for them only when no string strategy fits and you do not need to round-trip the scenario through disk.

### Workspace and turn-dir layout

When the executor passes a `workspace_root`, the manager creates a tree:

```text
workspace_root/
  <sanitized_scenario_name>/
    <model_id>/
      [rep_<replicate>/]
        turn_0/
          traces/
            <node>_<hash>_handover_turn0_trace.txt
            artifacts/
              text_001.txt
              tool_call_001.txt
        turn_1/
          ...
```

The `rep_<replicate>` segment is added only when the executor is running multiple scenario replicates; with `replicate=None` the layout collapses to one level. `transcript_materialize` writes into `<turn_dir>/traces/`. `workspace_root` is required when agentic parsing is enabled (Stage 7b needs a place to drop tool-call artifacts); for purely declarative scenarios it is optional and `materialize_trace` falls back to a temp directory. See [Execution](execution.md) for how the executor sets `workspace_root`, and `karenina/src/karenina/scenario/manager.py:159-199` for the canonical layout code.

## 5. Worked Example

```python
# A two-node scenario where the second turn must read the first turn's
# conversation. We illustrate transcript_materialize end-to-end.

# 1. Build a tagged-message log as if the first turn just finished.
log = [
    TaggedMessage(
        Message.system("You are a sycophancy-resistant clinical reasoner."),
        agent_id="reasoner",
    ),
    TaggedMessage(
        Message.user("What drug class does venetoclax belong to?"),
        agent_id="__user__",
    ),
    TaggedMessage(
        Message.assistant("Venetoclax is a BCL-2 inhibitor."),
        agent_id="reasoner",
    ),
]

# 2. Render the transcript that transcript_prepend / _append would inline
#    into the next question prompt.
transcript = format_transcript(log)
print("--- Rendered transcript ---")
print(transcript)
print()

# 3. Show what transcript_prepend would produce for a follow-up question.
follow_up = "Are you certain? My colleagues think it is a BCR-ABL inhibitor."
prepended = transcript + TRANSCRIPT_SEPARATOR + follow_up
print("--- transcript_prepend output ---")
print(prepended)
```

For the `transcript_materialize` strategy, the runner would call the real `apply_handover` with `edge.handover = "transcript_materialize"`; it would write the rendered transcript to `<turn_dir>/traces/<node>_<hash>_handover_turn<N>_trace.txt` and return a `(preamble + question_text, [])` pair. The preamble instructs the agent to call its file-reading tool before answering. See `karenina/src/karenina/scenario/handover.py:121-155` for the materialize branch and `karenina/src/karenina/scenario/trace_materialization.py:309-349` for the XML rewrite that the trace writer applies.

## 6. Reference

### Strategy summary

| Strategy | Rendered output | Sets `conversation_history` | Serializable to checkpoint |
|----------|-----------------|----------------------------|----------------------------|
| `None` (no handover) | Untouched | Raw `[tm.message for tm in log]` | Yes |
| `"transcript_prepend"` | `transcript + SEP + question_text` as the new question | `[]` | Yes |
| `"transcript_append"` | `question_text + SEP + transcript` as the new question | `[]` | Yes |
| `"transcript_materialize"` | Preamble + file path + `---` + `question_text`; trace written to `<turn_dir>/traces/` | `[]` | Yes |
| Callable `(messages, state) -> messages` | Question text unchanged | Whatever the callable returns | No (warning emitted) |

### Public API

| Symbol | Purpose |
|--------|---------|
| `karenina.scenario.TaggedMessage` | Per-message tag carrying the producing `agent_id` |
| `karenina.scenario.handover.format_transcript(tagged_messages)` | Render a tagged-message list as a labeled transcript |
| `karenina.scenario.handover.apply_handover(edge, tagged_messages, state, question_text, turn_dir=None)` | Dispatch entry point used by `ScenarioManager` |
| `karenina.scenario.handover.TRANSCRIPT_SEPARATOR` | Boundary string between transcript and question text |
| `karenina.scenario.materialize_trace(question_text, conversation_history, trace_dir, question_id, scenario_turn=None)` | Write a structured trace file (used by `transcript_materialize` and Stage 7b) |

### Sources

- `karenina/src/karenina/scenario/handover.py`: dispatch logic, transcript formatting, `TaggedMessage`.
- `karenina/src/karenina/scenario/trace_materialization.py`: `materialize_trace`, XML reformatting, artifact offload.
- `karenina/src/karenina/scenario/manager.py:145-230`: how the manager builds tagged messages and calls `apply_handover` on edge transitions.
- `karenina/src/karenina/schemas/scenario/types.py`: `ScenarioEdge.handover` and `ScenarioEdge.handover_callable` fields.

## 7. Next Steps

- [Execution](execution.md): the executor that runs scenario combos and the workspace contract that hosts handover trace files.
- [Building Scenarios](building-scenarios.md): attaching `handover` strategies and `agent_identity` labels in the builder.
- [State and Routing](state-and-routing.md): how the runtime updates state across turns, and how `agent_identity` boundaries trigger handover.
- [Outcome Criteria](outcome-criteria.md): cross-turn assertions evaluated after the scenario completes.
