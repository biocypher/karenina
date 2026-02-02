"""Prompt construction for template evaluation.

Canonical location for TemplatePromptBuilder.

This module provides the TemplatePromptBuilder class for constructing
prompts used in template parsing operations. It is used by:
- TemplateEvaluator (evaluators/template/evaluator.py) for standard parsing
- TemplateEvaluator._parse_with_deep_judgment() for deep judgment combined prompt

The old location (evaluators/template/prompts.py) re-exports from here
for backwards compatibility.
"""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class TemplatePromptBuilder:
    """Builds format-agnostic prompts for template parsing operations.

    This class encapsulates base prompt construction logic for template evaluation.
    It produces format-agnostic prompts — format-specific sections (output format,
    JSON schema, parsing notes) are appended by each adapter via AdapterInstruction.

    Example:
        builder = TemplatePromptBuilder(answer_class=MyAnswer)
        system_prompt = builder.build_system_prompt(has_tool_traces=True)
        user_prompt = builder.build_user_prompt(
            question_text="What gene is targeted?",
            response_to_parse="BCL2 is the primary target...",
        )
    """

    answer_class: type[BaseModel]
    """The Answer class for extracting JSON schema."""

    def build_system_prompt(
        self,
        has_tool_traces: bool = False,
        ground_truth: dict[str, Any] | None = None,
    ) -> str:
        """Build format-agnostic system prompt for template parsing.

        This function creates a composable system prompt with:
        1. Base guidelines (always included) - extraction protocol and critical rules
        2. Tool trace verification section (conditional - only when MCP/tools present)
        3. Ground truth section (optional - for semantic matching assistance)

        Format-specific sections (output format, JSON schema) are NOT included here.
        Each adapter appends its own format instructions via AdapterInstruction.

        User instructions are injected by PromptAssembler (via PromptConfig),
        not passed directly to this builder.

        Args:
            has_tool_traces: Whether the response includes tool call traces (MCP context)
            ground_truth: Optional ground truth for disambiguation assistance

        Returns:
            Composed system prompt string
        """
        # === BASE GUIDELINES (always included) ===
        base_guidelines = """You are an evaluator that extracts structured information from responses.

You will receive:
1. A response to parse (either a final answer or a complete trace)
2. A JSON schema with descriptive fields indicating what information to extract

# Extraction Protocol

## 1. Focus on the Final Answer
- Your primary extraction source is the **final answer** given to the user
- Extract information from this answer according to the schema field descriptions

## 2. Extract According to Schema
- Each field description specifies WHAT to extract from the answer and HOW
- Follow field descriptions precisely
- Use `null` for information not present in the final answer (if field allows null)

## 3. Validate Structure
- Return valid JSON matching the provided schema exactly
- Use correct data types for each field

# Critical Rules

**Answer-First**: Extract primarily from the final answer content.

**Description Adherence**: Each field's description is authoritative for what and how to extract.

**Fidelity**: Extract only what's actually stated. Don't infer or add information not present.

**JSON Only**: Return ONLY the JSON object - no explanations, no markdown fences, no surrounding text."""

        # === TOOL TRACE SECTION (conditional - only when MCP/tools present) ===
        tool_trace_section = """

# Tool Trace Verification (when traces are present)

When the response includes tool calls and results:

## Verify Grounding in Tool Results
- Cross-reference claims in the final answer against tool results in the trace
- Check that factual statements are supported by retrieved data
- Note if the answer includes information not present in tool results

**Grounding Check**: Use the trace to verify the answer's claims are supported by tool calls/results."""

        # === GROUND TRUTH SECTION (optional) ===
        ground_truth_section = ""
        if ground_truth is not None:
            ground_truth_str = json.dumps(ground_truth, indent=2, default=str)
            ground_truth_section = f"""

# Ground Truth Reference

The following ground truth information is provided as reference to help with semantic matching and disambiguation.
Use this information carefully - do not blindly copy it, but it may help resolve ambiguities when the trace
and template are semantically close but differ in exact wording.

Ground Truth:
{ground_truth_str}"""

        # Compose final prompt
        sections = [base_guidelines]
        if has_tool_traces:
            sections.append(tool_trace_section)
        if ground_truth_section:
            sections.append(ground_truth_section)

        return "".join(sections)

    def build_user_prompt(
        self,
        question_text: str,
        response_to_parse: str | Any,
    ) -> str:
        """Build format-agnostic user prompt for template parsing.

        Includes the original question and response to parse. Format-specific
        sections (JSON schema, parsing notes, response trailer) are NOT included
        here — each adapter appends its own via AdapterInstruction.

        Args:
            question_text: The original question that was asked
            response_to_parse: The LLM response to parse into structured format

        Returns:
            Formatted user prompt string
        """
        return f"""Parse the following response and extract structured information.

**ORIGINAL QUESTION:**
{question_text}

**RESPONSE TO PARSE:**
{response_to_parse}"""
