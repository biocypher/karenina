#!/usr/bin/env bash
#
# Install the karenina agent skills into a coding agent's skills directory.
#
#   ./skills/install.sh                      # Claude Code, this project
#   ./skills/install.sh --agent codex        # Codex / Pi / Gemini CLI / OpenCode
#   ./skills/install.sh --scope user         # available in every project
#   ./skills/install.sh --help
#
set -euo pipefail

SKILLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

AGENT="claude"
SCOPE="project"
DEST=""
MODE="copy"
FORCE=0
WORKFLOW_ONLY=0

usage() {
  cat <<'USAGE'
Install the karenina agent skills.

Usage: skills/install.sh [OPTIONS]

Options:
  -a, --agent NAME    claude | codex | pi | gemini | opencode | agents
                      (default: claude). Everything other than "claude"
                      resolves to the shared .agents/skills convention.
  -s, --scope SCOPE   project | user (default: project)
  -d, --dest DIR      explicit destination, overrides --agent and --scope
  -l, --link          symlink the skills instead of copying them, so they
                      track this checkout
  -w, --workflow-only install only the usage skills, skip the six
                      karenina-adapter-* development skills
  -f, --force         replace skills that are already installed
  -h, --help          show this message

Destinations:
  claude   project .claude/skills   user ~/.claude/skills
  others   project .agents/skills   user ~/.agents/skills
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    -a|--agent)         AGENT="${2:?--agent needs a value}"; shift 2 ;;
    -s|--scope)         SCOPE="${2:?--scope needs a value}"; shift 2 ;;
    -d|--dest)          DEST="${2:?--dest needs a value}"; shift 2 ;;
    -l|--link)          MODE="link"; shift ;;
    -w|--workflow-only) WORKFLOW_ONLY=1; shift ;;
    -f|--force)         FORCE=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$DEST" ]; then
  case "$SCOPE" in
    project) root="." ;;
    user)    root="$HOME" ;;
    *) echo "Unknown scope: $SCOPE (expected project or user)" >&2; exit 2 ;;
  esac
  case "$AGENT" in
    claude)                              DEST="$root/.claude/skills" ;;
    codex|pi|gemini|opencode|agents)     DEST="$root/.agents/skills" ;;
    *) echo "Unknown agent: $AGENT (expected claude, codex, pi, gemini, opencode or agents)" >&2; exit 2 ;;
  esac
fi

mkdir -p "$DEST"
DEST="$(cd -- "$DEST" && pwd -P)"

if [ "$DEST" = "$SKILLS_DIR" ]; then
  echo "Destination is the source directory, nothing to do: $DEST"
  exit 0
fi

installed=0
skipped=0

for skill_path in "$SKILLS_DIR"/*/; do
  skill="$(basename -- "$skill_path")"
  [ -f "$skill_path/SKILL.md" ] || continue
  if [ "$WORKFLOW_ONLY" -eq 1 ]; then
    case "$skill" in karenina-adapter-*) continue ;; esac
  fi

  target="$DEST/$skill"
  if [ -e "$target" ] || [ -L "$target" ]; then
    if [ "$FORCE" -eq 0 ]; then
      echo "  skip     $skill (already installed, use --force to replace)"
      skipped=$((skipped + 1))
      continue
    fi
    if [ ! -L "$target" ] && [ ! -f "$target/SKILL.md" ]; then
      echo "  refuse   $skill ($target exists and is not a skill directory)" >&2
      skipped=$((skipped + 1))
      continue
    fi
    rm -rf -- "$target"
  fi

  if [ "$MODE" = "link" ]; then
    ln -s -- "${skill_path%/}" "$target"
    echo "  link     $skill"
  else
    cp -R -- "${skill_path%/}" "$target"
    echo "  install  $skill"
  fi
  installed=$((installed + 1))
done

echo
echo "$installed skill(s) installed into $DEST ($skipped skipped)."
if [ "$AGENT" = "codex" ]; then
  echo "Codex also reads ~/.codex/skills. The shared .agents/skills location is preferred."
fi
echo "Restart your agent if it does not pick the skills up automatically."
