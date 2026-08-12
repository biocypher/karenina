# Deferred Live Checks

Smoke checks that need live endpoints, deferred while those endpoints are
unavailable. Sweep top to bottom when the endpoint returns and mark each
line with date and outcome.

Codon was unavailable on 2026-08-12 and is expected back 2026-08-13.

| Check | Needs | Status |
|---|---|---|
| live_gate.sh adapter + save/resume groups for fix/paper-foundations (MCP timeouts, sink resume) | Codon vLLM qwen endpoint | pending |
| Manual MCP smoke: one QA question against a local OTP MCP server with mcp_http_timeout=240 set, confirming the timeout reaches the transport | Codon GPT-OSS judge (or any live judge) | pending |
