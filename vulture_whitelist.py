# Vulture whitelist - false positives and intentional unused code
# This file is used by vulture to suppress false positives.
# See: https://github.com/jendrikseipp/vulture#ignoring-unused-code

# TYPE_CHECKING imports used for type annotations
from anyio.from_thread import BlockingPortal  # noqa: F401

from karenina.ports.usage import UsageMetadata as PortUsageMetadata  # noqa: F401

BlockingPortal  # Used in type hints within TYPE_CHECKING block
PortUsageMetadata  # Used in type hints within TYPE_CHECKING block (trace_usage_tracker.py)

# Hook method parameters that must match interface signature
runtime  # LangGraph hook interface parameter (interface.py)

# Protocol method parameters that define interface signatures
schema  # LLMPort protocol method parameter (ports/llm.py)

# create_verbose_logger parameters kept for API backwards compatibility
# These are consumed via _ = param pattern to avoid vulture warnings
