# Claude Code CLI image for live tests

Minimal Docker image used by the claude_agent_sdk container backend
(`src/karenina/adapters/claude_agent_sdk/docker_cli_wrapper.py`) when running
live tests against a vLLM endpoint.

## Why the pin

The CLI is pinned to 2.1.146. Newer versions (2.1.170 and later) append
system-role messages to the Anthropic messages array, and vLLM's
`/v1/messages` rejects any system role inside messages with HTTP 400.
2.1.146 is the version the retained May CSDK runs used. See
`paper_examples/bix_bench/build_claude_sif.sh` for the original rationale.
Bump the pin only when a CLI upgrade is intended and revalidated against vLLM.

## Build

```bash
docker build -t karenina-live-claude-cli:2.1.146 \
  --build-arg CLAUDE_CODE_VERSION=2.1.146 \
  -f karenina/tests/live/docker/Dockerfile.claude-cli \
  karenina/tests/live/docker/
```

Image tag: `karenina-live-claude-cli:2.1.146`

## Docker daemon on macOS

If no daemon is running, bring one up with colima:

```bash
colima start --runtime docker --cpu 4 --memory 8
```

## Validate

```bash
docker run --rm karenina-live-claude-cli:2.1.146 claude --version
docker run --rm karenina-live-claude-cli:2.1.146 /bin/sh -lc 'claude --version'
```

Both must print 2.1.146. The second form mirrors how the wrapper launches the
CLI (a login shell, so `claude` must resolve on the login-shell PATH, which it
does from /usr/local/bin).
