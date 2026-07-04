# Advanced

This section is about *extending*: Karenina's internal architecture for users who need to debug pipeline behavior, customize prompts, write custom stages, or create new LLM adapters. For the conceptual foundations, see [Core Concepts](core_concepts/index.md). For task-oriented guides, see [Workflows](workflows/index.md).

---

## In This Section

| Topic | What It Covers |
|-------|---------------|
| [Pipeline Internals](advanced-pipeline/index.md) | The verification pipeline (13 numbered stages with sub-stages 7a/7b and 11a/11b plus the always-on PlaceholderRetryAutoFail guard), deep judgment, prompt assembly, and custom stages |
| [Adapter Architecture](advanced-adapters/index.md) | Ports and adapters (hexagonal architecture), available adapters, MCP integration, and writing custom adapters |

---

## When You Need This

Most users can work entirely with the interfaces described in [Workflows](workflows/index.md). The advanced documentation is for situations where you need to:

- **Debug failures**: understand why a specific pipeline stage failed or was skipped
- **Tune deep judgment**: configure excerpt extraction, fuzzy matching, and search-enhanced verification
- **Customize prompts**: understand the tri-section prompt assembly system
- **Extend the pipeline**: write custom verification stages
- **Add LLM providers**: create new adapters for unsupported backends
- **Understand port protocols**: work with LLMPort, ParserPort, and AgentPort at the code level
