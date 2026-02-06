# karenina serve

Start the Karenina webapp server.

```
karenina serve [OPTIONS]
```

The `serve` command starts the web application server, which serves the karenina-gui frontend and the karenina-server REST API. On first run in a new workspace, it offers interactive setup to create configuration files and directories.

Requires `karenina[webapp]` to be installed (`pip install karenina[webapp]`).

---

## Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--port` | `-p` | `INTEGER` | `8080` | Port to serve on |
| `--host` | | `TEXT` | `localhost` | Host to bind to |
| `--dev` | | flag | `False` | Run in development mode (hot-reloading) |
| `--dir` | `-d` | `PATH` | current directory | Working directory for data files (databases, presets, checkpoints) |
| `--skip-setup` | | flag | `False` | Skip first-time setup even if no configuration exists |

---

## First-Time Setup

When `serve` is run in a directory without a `defaults.json` file (and `--skip-setup` is not set), it offers an interactive setup prompt:

1. **default** — Quick setup with sensible defaults (OpenAI provider, gpt-4.1-mini model, LangChain interface, async enabled with 2 workers)
2. **advanced** — Full guided configuration (equivalent to `karenina init --advanced`)
3. **skip** — Start the server without setup

The setup creates:

- `defaults.json` — Default model configuration
- `.env` — Environment variables (directory paths, async settings, API key placeholders)
- `dbs/`, `presets/`, `mcp_presets/`, `checkpoints/` directories

To set up a workspace without starting the server, use [`karenina init`](init.md) instead.

---

## Development Mode

The `--dev` flag starts the server with hot-reloading enabled, which is useful when developing the karenina-server or karenina-gui packages. In production mode (the default), the server runs with the bundled webapp build.

---

## Environment Variables

When the server starts, it sets the following environment variables for the session (if not already set):

| Variable | Default |
|----------|---------|
| `DB_PATH` | `{working_dir}/dbs` |
| `KARENINA_PRESETS_DIR` | `{working_dir}/presets` |
| `MCP_PRESETS_DIR` | `{working_dir}/mcp_presets` |

These variables control where the server stores databases, presets, and MCP presets. See [Environment Variables](../03-configuration/environment-variables.md) for details.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Server shut down normally |
| `1` | Error — `karenina[webapp]` not installed or startup failure |

---

## Examples

### Start with defaults

```bash
karenina serve
```

Starts the server on `http://localhost:8080`.

### Custom port

```bash
karenina serve --port 9000
```

### Custom host and port

```bash
karenina serve --host 0.0.0.0 --port 3000
```

Binds to all interfaces (useful for remote access or Docker containers).

### Custom working directory

```bash
karenina serve --dir ~/karenina
```

Uses `~/karenina` for databases, presets, and checkpoints.

### Skip first-time setup

```bash
karenina serve --skip-setup
```

Starts the server immediately without interactive setup prompts, even in a new workspace.

### Development mode

```bash
karenina serve --dev
```

Starts with hot-reloading for active development.

---

## Related

- [init](init.md) — Initialize a workspace without starting the server
- [Configuration Hierarchy](../03-configuration/index.md) — How configuration layers interact
- [Environment Variables](../03-configuration/environment-variables.md) — All environment variable settings
- [CLI Reference](index.md) — Overview of all CLI commands
