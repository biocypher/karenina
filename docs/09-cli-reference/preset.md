# karenina preset

Manage verification presets: list available presets, show preset details, and delete presets.

```
karenina preset [OPTIONS] COMMAND [ARGS]...
```

The `preset` command group provides three subcommands for managing verification configuration presets. Presets are JSON files stored in the presets directory that capture a complete `VerificationConfig`.

---

## Subcommands

| Subcommand | Description |
|------------|-------------|
| **list** | List all available presets |
| **show** | Show preset details |
| **delete** | Delete a preset |

---

## Preset Directory Resolution

Preset commands look for preset files in this order:

1. `KARENINA_PRESETS_DIR` environment variable (if set)
2. `./presets/` relative to the current working directory

The `show` and `delete` subcommands accept either a preset **name** (looked up in the presets directory) or a direct **file path**. When a name is given, the command appends `.json` and searches the presets directory.

---

## preset list

```
karenina preset list [OPTIONS]
```

List all `.json` files in the presets directory, showing names and last-modified dates.

### Options

| Option | Description |
|--------|-------------|
| `--help` | Show this message and exit |

### Output

Displays a Rich-formatted table with two columns:

| Column | Description |
|--------|-------------|
| **Name** | Preset filename without `.json` extension |
| **Modified** | Last-modified date (YYYY-MM-DD) |

A total count is shown below the table. If no presets are found, a message explains where presets are expected and how to override the directory with `KARENINA_PRESETS_DIR`.

### Example

```bash
karenina preset list
```

```
         Available Presets
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name            ┃ Modified   ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ default         │ 2026-01-15 │
│ gpt-4o-quick    │ 2026-01-20 │
│ multi-model     │ 2026-02-01 │
└─────────────────┴────────────┘

Total: 3 preset(s)
```

---

## preset show

```
karenina preset show [OPTIONS] NAME_OR_PATH
```

Display the full configuration of a preset, including a JSON dump of all settings and a summary of key parameters.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `NAME_OR_PATH` | `TEXT` | Yes | Preset name (e.g., `default`) or direct file path (e.g., `./presets/default.json`) |

### Options

| Option | Description |
|--------|-------------|
| `--help` | Show this message and exit |

### Output

The command displays four sections:

1. **Header** — Preset name (from filename stem)
2. **Path** — Resolved absolute path to the preset file
3. **Configuration** — Full `VerificationConfig` as syntax-highlighted JSON (excluding `null` fields)
4. **Summary** — Key configuration values:

| Field | Description |
|-------|-------------|
| Answering models | Number of answering model configurations |
| Parsing models | Number of parsing model configurations |
| Replicates | Replicate count |
| Rubric enabled | Whether rubric evaluation is on |
| Abstention enabled | Whether abstention detection is on |
| Sufficiency enabled | Whether sufficiency checking is on |
| Embedding check enabled | Whether embedding similarity check is on |
| Deep judgment enabled | Whether deep judgment is on |

### Name Resolution

The `NAME_OR_PATH` argument is resolved in this order:

1. If it's an existing file path, use it directly
2. Otherwise, look for `{NAME_OR_PATH}.json` (or exact name if it already ends in `.json`) in the presets directory

### Examples

```bash
# By name (looks up presets/default.json)
karenina preset show default

# By explicit path
karenina preset show ./presets/default.json
```

### Error Cases

| Scenario | Message |
|----------|---------|
| Name not found | `Preset 'foo' not found in presets/. Use 'karenina preset list' to see available presets.` |
| Presets dir missing | `Presets directory not found: presets/` |
| Invalid preset JSON | `loading preset: ...` |

---

## preset delete

```
karenina preset delete [OPTIONS] NAME_OR_PATH
```

Delete a preset file after confirmation.

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `NAME_OR_PATH` | `TEXT` | Yes | Preset name or direct file path |

### Options

| Option | Description |
|--------|-------------|
| `--help` | Show this message and exit |

### Behavior

1. Resolves the preset file (same resolution as `show`)
2. Displays the file path and asks for confirmation: `Are you sure? [y/N]`
3. If confirmed, deletes the file and prints a success message
4. If declined, prints `Deletion cancelled.`

### Examples

```bash
# Delete by name
karenina preset delete old-config

# Delete by path
karenina preset delete ./presets/old-config.json
```

### Error Cases

Same resolution errors as `preset show` — name not found or presets directory missing.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Command completed successfully |
| `1` | Error — preset not found, directory missing, or invalid preset file |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KARENINA_PRESETS_DIR` | `./presets/` | Directory where preset files are stored and discovered |

---

## Related

- [Using Presets](../06-running-verification/using-presets.md) — Guide to loading and overriding presets in verification workflows
- [Configuration Hierarchy](../03-configuration/index.md) — How presets fit in the CLI > Preset > Env > Defaults chain
- [Preset Schema Reference](../10-configuration-reference/preset-schema.md) — Full preset file format documentation
- [verify](verify.md) — The `--preset` option for loading presets during verification
- [CLI Reference](index.md) — Overview of all CLI commands
