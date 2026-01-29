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
    """Builds prompts for template parsing operations.

    This class encapsulates all prompt construction logic for template evaluation,
    providing composable system and user prompts for the judge LLM.

    Example:
        builder = TemplatePromptBuilder(answer_class=MyAnswer)
        system_prompt = builder.build_system_prompt(
            format_instructions=format_instr,
            has_tool_traces=True,
        )
        user_prompt = builder.build_user_prompt(
            question_text="What gene is targeted?",
            response_to_parse="BCL2 is the primary target...",
        )
    """

    answer_class: type[BaseModel]
    """The Answer class for extracting JSON schema."""

    def build_system_prompt(
        self,
        format_instructions: str,
        user_system_prompt: str | None = None,
        has_tool_traces: bool = False,
        ground_truth: dict[str, Any] | None = None,
    ) -> str:
        """Build enhanced composable system prompt for template parsing.

        This function creates a composable system prompt with:
        1. Base guidelines (always included) - extraction protocol and critical rules
        2. Tool trace verification section (conditional - only when MCP/tools present)
        3. User customizations (merged as "Additional Instructions")
        4. Ground truth section (optional - for semantic matching assistance)
        5. Output format section with format instructions

        Args:
            format_instructions: Pydantic format instructions from PydanticOutputParser
            user_system_prompt: Optional user-provided system prompt to merge
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

        # === USER CUSTOMIZATIONS SECTION ===
        user_section = ""
        if user_system_prompt:
            user_section = f"""

# Additional Instructions

{user_system_prompt}"""

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

        # === OUTPUT FORMAT ===
        output_section = f"""

# Output Format

Return only the completed JSON object - no surrounding text, no markdown fences:

<format_instructions>
{format_instructions}
</format_instructions>"""

        # Compose final prompt
        sections = [base_guidelines]
        if has_tool_traces:
            sections.append(tool_trace_section)
        if user_section:
            sections.append(user_section)
        if ground_truth_section:
            sections.append(ground_truth_section)
        sections.append(output_section)

        return "".join(sections)

    def build_user_prompt(
        self,
        question_text: str,
        response_to_parse: str | Any,
    ) -> str:
        """Build enhanced user prompt for template parsing.

        Includes the original question, response to parse, and JSON schema
        to help the LLM understand the expected output structure.

        Args:
            question_text: The original question that was asked
            response_to_parse: The LLM response to parse into structured format

        Returns:
            Formatted user prompt string
        """
        # Generate JSON schema from the Answer class
        json_schema = json.dumps(self.answer_class.model_json_schema(), indent=2)

        return f"""Parse the following response and extract structured information.

**ORIGINAL QUESTION:**
{question_text}

**RESPONSE TO PARSE:**
{response_to_parse}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Extract values for each field based on its description in the schema
- If information for a field is not present, use null (if field allows null) or your best inference
- Return ONLY the JSON object - no surrounding text

**YOUR JSON RESPONSE:**"""
