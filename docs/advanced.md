# Advanced

This section covers Karenina's internal architecture for users who need to debug pipeline behavior, customize prompts, write custom stages, or create new LLM adapters.

---

## In This Section

| Topic | What It Covers |
|-------|---------------|
| [Pipeline Internals](11-advanced-pipeline/index.md) | The 13-stage verification pipeline, deep judgment, prompt assembly, and custom stages |
| [Adapter Architecture](12-advanced-adapters/index.md) | Ports and adapters (hexagonal architecture), available adapters, MCP integration, and writing custom adapters |

---

## When You Need This

Most users can work entirely with the interfaces described in [Workflows](workflows.md). The advanced documentation is for situations where you need to:

- **Debug failures** — Understand why a specific pipeline stage failed or was skipped
- **Tune deep judgment** — Configure excerpt extraction, fuzzy matching, and search-enhanced verification
- **Customize prompts** — Understand the tri-section prompt assembly system
- **Extend the pipeline** — Write custom verification stages
- **Add LLM providers** — Create new adapters for unsupported backends
- **Understand port protocols** — Work with LLMPort, ParserPort, and AgentPort at the code level
