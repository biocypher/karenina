"""Shared fixtures for streaming JSON exporter tests.

Exposes deterministic builders (fixture_builders.py) used by
- unit tests in karenina/tests/unit/benchmark/verification/stages/
- the regeneration script (regen_fixtures.py) that materializes the
  byte-exact golden files in this directory.

On-disk golden files (results_export_full.json, results_export_empty.json)
are generated; do not hand-edit.
"""
