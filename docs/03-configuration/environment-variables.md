# Environment Variables

Karenina reads environment variables to configure API keys, file paths, execution behavior, and verification features. This page groups variables by purpose and shows practical usage patterns. For an exhaustive reference table, see [Environment Variables Reference](../10-configuration-reference/env-vars.md).

!!! note "Precedence"
    Environment variables are the **third priority** in the configuration hierarchy — they're overridden by CLI arguments and preset values. See [Configuration Hierarchy](index.md) for details.

---

## Setting Environment Variables

The recommended approach is a `.env` file in your project root. Karenina's adapters use `dotenv` to load variables automatically.

```bash
# .env file (add to .gitignore)
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
KARENINA_ASYNC_MAX_WORKERS=4
```

Alternatively, export variables in your shell:

```bash
export OPENAI_API_KEY="sk-..."
export KARENINA_ASYNC_MAX_WORKERS=4
```

Or set them inline for a single command:

```bash
KARENINA_ASYNC_MAX_WORKERS=8 karenina verify checkpoint.jsonld
```

---

## API Keys

API keys authenticate with LLM providers. You only need keys for the providers you actually use.

| Variable | Provider | Required When |
|----------|----------|---------------|
| `OPENAI_API_KEY` | OpenAI | Using `langchain` interface with OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic | Using `claude_agent_sdk`, `claude_tool`, or LangChain with Anthropic models |
| `GOOGLE_API_KEY` | Google | Using `langchain` interface with Google models |
| `OPENROUTER_API_KEY` | OpenRouter | Using `openrouter` interface |
| `TAVILY_API_KEY` | Tavily | Using deep judgment with web search verification |

**Example**: Running verification with OpenAI models requires only `OPENAI_API_KEY`:

```bash
# .env
OPENAI_API_KEY="sk-..."
```

**Example**: Using Claude for answering and OpenAI for parsing requires both:

```bash
# .env
ANTHROPIC_API_KEY="sk-ant-..."
OPENAI_API_KEY="sk-..."
```

The `karenina init` command creates a `.env` template with placeholder entries for all three main providers. See [Workspace Initialization](../02-installation/workspace-init.md) for details.

---

## Path Configuration

These variables control where Karenina stores and looks for files.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DB_PATH` | Directory for SQLite databases | `./dbs/` |
| `KARENINA_PRESETS_DIR` | Directory for preset JSON files | `./presets/` |
| `MCP_PRESETS_DIR` | Directory for MCP configuration presets | `./mcp_presets/` |

All three directories are created automatically by `karenina init`. You typically only set these variables when your project layout differs from the default.

**Example**: Storing databases in a shared location:

```bash
# .env
DB_PATH="/shared/team/karenina/dbs"
```

**Example**: Using a centralized preset library:

```bash
# .env
KARENINA_PRESETS_DIR="/shared/team/presets"
```

The `karenina serve` command sets `DB_PATH`, `KARENINA_PRESETS_DIR`, and `MCP_PRESETS_DIR` relative to the working directory if they aren't already defined.

---

## Async Execution

These variables control parallel verification execution. They apply when running verification across multiple questions.

| Variable | Purpose | Default |
|----------|---------|---------|
| `KARENINA_ASYNC_ENABLED` | Enable parallel execution | `true` |
| `KARENINA_ASYNC_MAX_WORKERS` | Maximum number of parallel workers | `2` |

**Example**: Increasing parallelism for faster runs (with sufficient API rate limits):

```bash
# .env
KARENINA_ASYNC_MAX_WORKERS=8
```

**Example**: Disabling parallelism for debugging:

```bash
# .env
KARENINA_ASYNC_ENABLED=false
```

!!! tip
    These variables are also available as `VerificationConfig` fields (`async_enabled`, `async_max_workers`). Setting them in code or via CLI overrides the environment variable values.

---

## Embedding Check

Embedding similarity checking provides a fallback verification mechanism. When enabled, it compares the LLM's parsed answer to the ground truth using vector similarity. These variables configure the embedding check when it isn't set via `VerificationConfig` or a preset.

| Variable | Purpose | Default |
|----------|---------|---------|
| `EMBEDDING_CHECK` | Enable embedding similarity check | `false` |
| `EMBEDDING_CHECK_MODEL` | SentenceTransformer model name | `all-MiniLM-L6-v2` |
| `EMBEDDING_CHECK_THRESHOLD` | Similarity threshold (0.0–1.0) | `0.85` |

**Example**: Enabling embedding check with a stricter threshold:

```bash
# .env
EMBEDDING_CHECK=true
EMBEDDING_CHECK_THRESHOLD=0.9
```

The embedding check runs after template verification and provides an additional similarity-based signal. See [Answer Templates](../04-core-concepts/answer-templates.md) for how embedding checks integrate with template verification.

---

## Other Settings

| Variable | Purpose | Default |
|----------|---------|---------|
| `AUTOSAVE_DATABASE` | Automatically save verification results to the database | `true` |
| `KARENINA_EXPOSE_GROUND_TRUTH` | Include ground truth values in the judge LLM prompt during template evaluation | `false` |

**`AUTOSAVE_DATABASE`** — When enabled, verification results are automatically persisted to the SQLite database after each run. Disable this if you only want in-memory results:

```bash
# .env
AUTOSAVE_DATABASE=false
```

**`KARENINA_EXPOSE_GROUND_TRUTH`** — When enabled, the ground truth answer is included in the prompt sent to the judge LLM during template evaluation. This is intended for debugging — it helps you understand whether parsing failures are due to ambiguous instructions or genuinely wrong answers. **Do not enable this for production benchmarking**, as it biases the judge.

```bash
# .env
KARENINA_EXPOSE_GROUND_TRUTH=true  # Debugging only
```

---

## Boolean Values

Environment variables that accept boolean values recognize these truthy strings (case-insensitive):

- **True**: `true`, `1`, `yes`, `on`
- **False**: anything else (including `false`, `0`, `no`, `off`)

---

## Next Steps

- [Configuration Hierarchy](index.md) — How environment variables fit into the precedence system
- [Presets](presets.md) — Reusable configuration files that override environment variables
- [Workspace Initialization](../02-installation/workspace-init.md) — `karenina init` creates default directories and `.env` template
- [Environment Variables Reference](../10-configuration-reference/env-vars.md) — Exhaustive table of all variables
- [Running Verification](../06-running-verification/index.md) — Putting configuration into practice
