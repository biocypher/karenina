"""Shared helpers for agentic answer parsing."""

import json
import logging
from typing import Any

from karenina.adapters import get_agent, get_parser
from karenina.adapters.agent_runtime import workspace_path_for_prompt
from karenina.adapters.registry import close_adapter
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.utils.schema_builder import (
    build_extraction_relaxed_class,
    rebuild_strict_answer_with_null_fields,
)
from karenina.benchmark.verification.utils.trace_parsing import extract_final_ai_message
from karenina.ports import AgentConfig, UsageMetadata
from karenina.schemas.entities.answer import BaseAnswer

logger = logging.getLogger(__name__)

_MAX_EXTRACTION_REPORT_CHARS = 60_000


def run_investigation(
    context: VerificationContext,
    clean_schema: dict[str, Any],
    *,
    screening_reasoning: str | None = None,
) -> tuple[str, bool, UsageMetadata]:
    """Run investigation agent with tools.

    Args:
        context: Verification context.
        clean_schema: Pre-built JSON schema for the answer template.
        screening_reasoning: Optional upstream screening note to include in
            the investigation prompt.

    Returns:
        Raw trace from the investigation agent, whether it hit the turn
        limit, and token usage from the investigation agent call.
    """
    agent = get_agent(context.parsing_model)
    schema_json = json.dumps(clean_schema, indent=2)
    capabilities = agent.capabilities
    if capabilities.supports_code_execution:
        execution_text = (
            "You have access to file tools. Code execution may be available, but do not use it for this check."
            if capabilities.uses_sandboxed_execution
            else "You have access to file tools. Code execution may be available, but do not use it for this check."
        )
    else:
        execution_text = "You have access to file tools, but command execution is not available."

    system_text = (
        "You are a verification agent evaluating whether an AI coding "
        f"assistant correctly completed a task. {execution_text}\n\n"
        "Be parsimonious. Look only for artifacts that appear to contain final "
        "reported results, such as result files, summaries, reports, tables, "
        "or final answer JSON/CSV/TXT/Markdown outputs. Read those artifacts "
        "and extract the values they report. Do not run scripts, notebooks, "
        "or commands. Do not re-compute the task, repair code, or create new "
        "files. If you cannot find a usable final-results artifact, report "
        "that you could not verify the results.\n\n"
        "Report your findings as a JSON object matching this schema:\n"
        f"{schema_json}"
    )

    # Build user prompt based on judge context mode
    user_parts: list[str] = [f"Question: {context.question_text}"]

    if screening_reasoning:
        user_parts.append(f"\n--- SCREENING REASONING ---\n{screening_reasoning}\n--- END SCREENING REASONING ---")

    if context.agentic_judge_context in (
        "trace_and_workspace",
        "trace_only",
    ):
        raw_trace = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{raw_trace}\n--- END TRACE ---")

    if context.workspace_path and context.agentic_judge_context != "trace_only":
        prompt_workspace = workspace_path_for_prompt(context.parsing_model, context.workspace_path)
        user_parts.append(
            f"\nWorkspace directory: {prompt_workspace}",
        )

    user_text = "\n".join(user_parts)

    # Assemble with adapter + user instructions
    assembler = PromptAssembler(
        task=PromptTask.AGENTIC_PARSING_INVESTIGATION,
        interface=context.parsing_model.interface,
        capabilities=capabilities,
    )
    user_instructions = (
        context.prompt_config.get_for_task(PromptTask.AGENTIC_PARSING_INVESTIGATION.value)
        if context.prompt_config
        else None
    )
    messages = assembler.assemble(
        system_text=system_text,
        user_text=user_text,
        user_instructions=user_instructions,
        instruction_context={"json_schema": clean_schema},
    )

    agent_config = AgentConfig(
        max_turns=context.agentic_parsing_max_turns,
        timeout=context.agentic_parsing_timeout,
        workspace_path=context.workspace_path,
    )

    try:
        result = agent.run(messages=messages, config=agent_config)
        logger.info(
            "Investigation completed in %d turns (limit_reached=%s)",
            result.turns,
            result.limit_reached,
        )
        return result.raw_trace, result.limit_reached, result.usage
    finally:
        close_adapter(agent)


def run_extraction(
    context: VerificationContext,
    answer_class: type[BaseAnswer],
    investigation_trace: str,
    clean_schema: dict[str, Any],
) -> tuple[BaseAnswer, UsageMetadata]:
    """Extract structured answer from investigation findings.

    Args:
        context: Verification context.
        answer_class: The answer template class.
        investigation_trace: Raw trace from investigation agent.
        clean_schema: Pre-built JSON schema for the answer template.

    Returns:
        Parsed answer instance and token usage from the extraction call.
    """
    parser = get_parser(context.parsing_model)
    schema_json = json.dumps(clean_schema, indent=2)
    extraction_input = prepare_extraction_input(investigation_trace)

    system_text = (
        "You are a structured data extraction assistant. "
        "Extract the findings from the investigation report into "
        "the exact JSON schema provided.\n\n"
        f"Schema:\n{schema_json}"
    )
    user_text = f"Investigation report:\n\n{extraction_input}"

    # Assemble with adapter + user instructions
    assembler = PromptAssembler(
        task=PromptTask.AGENTIC_PARSING_EXTRACTION,
        interface=context.parsing_model.interface,
        capabilities=parser.capabilities,
    )
    user_instructions = (
        context.prompt_config.get_for_task(PromptTask.AGENTIC_PARSING_EXTRACTION.value)
        if context.prompt_config
        else None
    )
    messages = assembler.assemble(
        system_text=system_text,
        user_text=user_text,
        user_instructions=user_instructions,
        instruction_context={
            "json_schema": clean_schema,
            "format_instructions": "",
        },
    )

    try:
        # Parse against a relaxed sibling class where each VerifiedField is
        # Optional[T] = None. This lets the LLM return null for individual
        # sub-questions ("couldn't compute X") without the whole record
        # failing pydantic validation. After parsing, we reconstruct the
        # strict template via model_construct (skipping null fields) and
        # carry the set of null fields as private metadata so downstream
        # verification reports them as None rather than False, preventing
        # accidental matches against a False ground truth.
        extraction_class = build_extraction_relaxed_class(answer_class)
        parse_result = parser.parse_to_pydantic(messages, extraction_class)
        strict_instance = rebuild_strict_answer_with_null_fields(
            answer_class,
            parse_result.parsed,
        )

        return strict_instance, parse_result.usage
    finally:
        close_adapter(parser)


def prepare_extraction_input(investigation_trace: str) -> str:
    """Prepare a compact investigation report for structured extraction.

    The investigation trace can include large file reads and tool outputs.
    The extractor only needs the investigator's final report, so prefer the
    final AI message. If the trace ended on a tool call or timeout before a
    final report was produced, fall back to a bounded head/tail excerpt so
    the parser call remains inside provider context limits.
    """
    final_report, extraction_error = extract_final_ai_message(investigation_trace)
    if final_report is not None and final_report.strip():
        return truncate_extraction_input(final_report)

    logger.warning(
        "Could not extract final investigation report for agentic parsing (%s); using bounded trace excerpt",
        extraction_error,
    )
    return truncate_extraction_input(investigation_trace)


def truncate_extraction_input(text: str, max_chars: int = _MAX_EXTRACTION_REPORT_CHARS) -> str:
    """Bound extraction input with a middle-truncation marker."""
    if len(text) <= max_chars:
        return text

    marker = f"\n\n[... investigation report truncated; omitted {len(text) - max_chars} characters ...]\n\n"
    if len(marker) >= max_chars:
        return text[:max_chars]

    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars - len(marker)
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


__all__ = [
    "prepare_extraction_input",
    "run_extraction",
    "run_investigation",
    "truncate_extraction_input",
]
