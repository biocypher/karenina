# Configuration Reference

This section provides exhaustive reference documentation for all configuration objects in karenina. For tutorial-style introductions with examples, see [Configuration](../../configuration/index.md) and [Running Verification](../../workflows/running-verification/index.md).

---

## Reference Pages

| Page | What It Covers | Fields |
|------|---------------|--------|
| [Environment Variables](env-vars.md) | All environment variables recognized by karenina | 15+ vars |
| [VerificationConfig](verification-config.md) | Complete verification pipeline configuration | 33 fields |
| [ModelConfig](model-config.md) | Model identity, parameters, MCP, and middleware | 19 fields |
| [PromptConfig](prompt-config.md) | Custom instruction injection into pipeline LLM calls | 6 fields |
| [Preset Schema](preset-schema.md) | JSON format for saved configuration presets | Metadata + config |
| [DBConfig](verification-config.md#dbconfig-fields) | Database connection for auto-saving results | 8 fields |

---

## Quick Lookup

### Which reference page do I need?

| I want to... | See |
|--------------|-----|
| Set an API key or path | [Environment Variables](env-vars.md) |
| Configure answering or parsing models | [ModelConfig](model-config.md) |
| Choose evaluation mode or feature flags | [VerificationConfig](verification-config.md) |
| Customize LLM prompts for parsing or rubric evaluation | [PromptConfig](prompt-config.md) |
| Understand preset file format | [Preset Schema](preset-schema.md) |
| Configure MCP servers or agent middleware | [ModelConfig](model-config.md) |
| Set up async execution | [VerificationConfig](verification-config.md) or [Environment Variables](env-vars.md) |
| Enable deep judgment or embedding checks | [VerificationConfig](verification-config.md) |

---

## Configuration Hierarchy

Configuration values are resolved in this order (highest priority first):

```
CLI arguments / explicit Python args
        ↓ (override)
Preset values
        ↓ (override)
Environment variables
        ↓ (override)
Built-in defaults
```

See [Configuration Hierarchy](../../configuration/index.md) for details.

---

## Import Paths

```python
from karenina.schemas import VerificationConfig, ModelConfig
from karenina.schemas.verification import PromptConfig
```

---

## Related

- [Configuration Tutorial](../../configuration/index.md) — conceptual overview with examples
- [Running Verification](../../workflows/running-verification/index.md) — scenario-based verification tutorials
- [CLI Reference](../cli/index.md) — command-line options that map to these fields
