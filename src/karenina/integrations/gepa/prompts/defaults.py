"""Default seed prompts for GEPA optimization.

These prompts are used as initial seeds when no custom prompts
are provided for optimization targets.
"""

DEFAULT_ANSWERING_SYSTEM_PROMPT = "You are a helpful assistant."
"""Default system prompt for the answering model when optimizing ANSWERING_SYSTEM_PROMPT."""

DEFAULT_PARSING_INSTRUCTIONS = "Extract the answer from the response following the schema."
"""Default instructions for the parsing model when optimizing PARSING_INSTRUCTIONS."""
