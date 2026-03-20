# Vulture whitelist - false positives and intentional unused code
# This file is used by vulture to suppress false positives.
# See: https://github.com/jendrikseipp/vulture#ignoring-unused-code

# TYPE_CHECKING imports used for type annotations
from anyio.from_thread import BlockingPortal  # noqa: F401

from karenina.ports.usage import UsageMetadata as PortUsageMetadata  # noqa: F401

BlockingPortal  # Used in type hints within TYPE_CHECKING block
PortUsageMetadata  # Used in type hints within TYPE_CHECKING block (trace_usage_tracker.py)

# Hook method parameters that must match interface signature
def _whitelist_runtime(runtime):
    """Whitelist for LangGraph hook interface parameter (middleware.py)."""
    _ = runtime  # Tell vulture this parameter name is intentionally used


# Protocol method parameters that define interface signatures
def _whitelist_schema(schema):
    """Whitelist for ParserPort/LLMPort protocol method parameter (ports/parser.py, ports/llm.py)."""
    _ = schema  # Tell vulture this parameter name is intentionally used

# create_verbose_logger parameters kept for API backwards compatibility
# These are consumed via _ = param pattern to avoid vulture warnings


# Pytest fixtures used as side-effect-only parameters in conformance tests.
# The fixture sets up mocks before arun() is called; the parameter name in the
# test method signature is required by pytest to inject it, but the test body
# does not reference it directly.
def _whitelist_conformance_fixtures(mock_deep_agents_agent_result):
    """Whitelist for conformance test fixture parameters (test_agent_port.py)."""
    _ = mock_deep_agents_agent_result
