# Vulture whitelist - false positives and intentional unused code
# This file is used by vulture to suppress false positives.
# See: https://github.com/jendrikseipp/vulture#ignoring-unused-code

# TYPE_CHECKING imports used for type annotations
from anyio.from_thread import BlockingPortal  # noqa: F401

BlockingPortal  # Used in type hints within TYPE_CHECKING block

# Hook method parameters that must match interface signature
_runtime  # LangGraph hook interface parameter (interface.py)

# create_verbose_logger parameters kept for API backwards compatibility
# These are consumed via _ = param pattern to avoid vulture warnings
