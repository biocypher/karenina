"""Shared adapter conformance tests.

These tests validate that any adapter implementation correctly satisfies
the port protocols (AgentPort, LLMPort, ParserPort). Each adapter provides
its own conftest.py with fixtures; the conformance tests are parametrized
across all available adapters.
"""
