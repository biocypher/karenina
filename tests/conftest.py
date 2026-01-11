# pytest configuration for karenina tests

import pytest


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Pure logic tests - no I/O, no LLM calls")
    config.addinivalue_line("markers", "integration: Multiple components working together")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests taking > 1 second")
    config.addinivalue_line("markers", "pipeline: Verification pipeline tests")
    config.addinivalue_line("markers", "rubric: Rubric evaluation tests")
    config.addinivalue_line("markers", "storage: Checkpoint I/O tests")
    config.addinivalue_line("markers", "cli: CLI command tests")
