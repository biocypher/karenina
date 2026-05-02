# CLI Reference

Karenina provides a command-line interface for running verifications, managing presets, initializing workspaces, and starting the web server. All commands are accessed through the `karenina` entry point.

```
karenina [OPTIONS] COMMAND [ARGS]...
```

The only global option is `--help`, which displays the command list.

---

## Commands

| Command | Description | Page |
|---------|-------------|------|
| **verify** | Run verification on a benchmark checkpoint | [verify](verify.md) |
| **verify-status** | Inspect a progressive save state file and show job status | [verify-status](verify-status.md) |
| **preset** | Manage verification presets (list, show, delete) | [preset](preset.md) |
| **serve** | Start the Karenina webapp server | [serve](serve.md) |
| **init** | Initialize Karenina configuration and directories | [init](init.md) |
| **analyze-errors** | Materialize failure cases from a results JSON-LD into per-failure folders for triage and Claude Code launching | — |
| **optimize** | Optimize prompts and instructions using GEPA | [optimize](optimize.md) |
| **optimize-history** | View optimization history | — |
| **optimize-compare** | Compare multiple optimization runs | — |

!!! note "GEPA commands"
    The `optimize`, `optimize-history`, and `optimize-compare` commands are part of the GEPA (Guided Evolutionary Prompt Architecture) module. See [optimize](optimize.md) for details.

---

## Quick Examples

### Run verification with a preset

```bash
karenina verify checkpoint.jsonld --preset default.json
```

### Run verification with CLI overrides

```bash
karenina verify checkpoint.jsonld --preset default.json \
  --answering-model claude-haiku-4-5 \
  --questions 0,1,2
```

### List available presets

```bash
karenina preset list
```

### Initialize a new workspace

```bash
karenina init
```

### Start the web server

```bash
karenina serve --port 8080
```

---

## Common Patterns

### Presets vs CLI arguments

Most users start with a preset file that defines model providers, evaluation modes, and pipeline features. CLI arguments then override specific values as needed:

```bash
# Preset defines everything; CLI overrides just the model
karenina verify checkpoint.jsonld --preset claude-haiku.json --answering-model claude-sonnet-4-5
```

Feature flags (`--abstention`, `--sufficiency`, `--embedding-check`, `--deep-judgment`) are tri-state: `--flag` enables, `--no-flag` disables, and omitting both preserves the preset default. Embedding settings (`--embedding-threshold`, `--embedding-model`) defer to environment variables when not passed explicitly.

See [Configuration Hierarchy](../../workflows/configuration/index.md) for how presets, CLI arguments, and environment variables interact.

### Progressive save and resume

For long-running verification jobs, use progressive save to periodically checkpoint progress:

```bash
# Start with progressive save (saves state every N questions)
karenina verify checkpoint.jsonld --preset default.json --progressive-save

# Check progress of a running or interrupted job
karenina verify-status state_file.json

# Resume an interrupted job
karenina verify checkpoint.jsonld --preset default.json --resume state_file.json
```

---

## Related

- [Configuration](../../workflows/configuration/index.md) — How CLI arguments fit in the configuration hierarchy
- [Running Verification](../../workflows/running-verification/index.md) — Python API and workflow guides
- [Configuration Reference](../configuration/index.md) — Complete reference for all configuration options
