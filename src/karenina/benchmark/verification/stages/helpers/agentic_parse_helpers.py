"""Shared helpers for agentic answer parsing."""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from karenina.adapters import get_agent, get_parser
from karenina.adapters.agent_runtime import map_path_for_prompt, workspace_path_for_prompt
from karenina.adapters.registry import close_adapter
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.utils.parser_resilience import parse_to_pydantic_resilient
from karenina.benchmark.verification.utils.schema_builder import (
    build_extraction_relaxed_class,
    rebuild_strict_answer_with_null_fields,
)
from karenina.benchmark.verification.utils.trace_parsing import extract_final_ai_message
from karenina.ports import AgentConfig, UsageMetadata
from karenina.schemas.entities.answer import BaseAnswer
from karenina.utils.json_extraction import strip_markdown_fences

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

    partial_timeout_response = bool(
        context.get_result_field(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL, False)
        and context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
    )
    trace_in_context = context.agentic_judge_context in (
        "trace_and_workspace",
        "trace_only",
    )
    materialize_trace = context.agentic_parsing_materialize_trace and context.has_artifact(
        ArtifactKeys.RAW_LLM_RESPONSE
    )
    trace_file_path: Path | None = None

    if partial_timeout_response and trace_in_context:
        trace_location = (
            "materialized answering trace file referenced in the prompt"
            if materialize_trace
            else "answering trace included in the prompt"
        )
        evidence_text = (
            f"You may inspect the workspace artifacts and the {trace_location}. "
            "The answering agent hit a wall-clock timeout, so final results may appear only in the trace, especially "
            "in recent tool output, stdout, or partial final text. Treat both workspace artifacts and the trace as "
            "evidence of what the answerer reported; do not rerun code."
        )
    elif partial_timeout_response:
        evidence_text = (
            "You may inspect the workspace artifacts. The answering agent hit a wall-clock timeout, so final results "
            "may be incomplete and may appear only in partial workspace outputs produced before the timeout. Treat "
            "those artifacts as evidence of what the answerer reported; do not rerun code."
        )
    else:
        evidence_text = (
            "You may inspect the workspace artifacts. If the prompt includes an answering trace, you may also use it "
            "as evidence of what the answerer reported."
        )

    system_text = (
        "You are a verification agent evaluating whether an AI coding "
        f"assistant correctly completed a task. {execution_text} {evidence_text}\n\n"
        "Be parsimonious. Look only for artifacts that appear to contain final "
        "reported results, such as result files, summaries, reports, tables, "
        "or final answer JSON/CSV/TXT/Markdown outputs. Read those artifacts "
        "and extract the values they report. Do not run scripts, notebooks, "
        "or commands. Do not re-compute the task, repair code, or create new "
        "files. Check file sizes before reading. Be mindful of the context window. "
        "Read large files carefully: avoid loading raw data files, package build "
        "logs, or wide result tables wholesale unless that is the only way to "
        "recover final answers. Prefer targeted reads of final summaries and small "
        "machine-readable result files; when an artifact is large, inspect only "
        "headers, relevant rows, or concise excerpts needed to answer the schema. "
        "If the answerer explicitly reported a value, extract that value. "
        "If the answerer did not report a value for a field, use JSON null for that field. "
        "Do not use 0, false, empty strings, or empty lists as placeholders for missing answers. "
        "For boolean equivalence fields, use false only when the answerer reported an answer "
        "and it is not equivalent to the ideal; use null when no answer was reported. "
        "When any field is unanswered, also include an internal '_unanswered_fields' array listing those field names.\n\n"
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
        if materialize_trace:
            trace_file_path = write_agentic_trace_file(
                workspace_path=context.workspace_path,
                trace=raw_trace,
                question_id=context.question_id,
                scenario_turn=context.scenario_turn,
            )
            prompt_trace_path = map_path_for_prompt(context.parsing_model, trace_file_path, context.workspace_path)
            trace_content = (
                f"The full answering agent trace is saved to: {prompt_trace_path}\n"
                "Use file tools (grep, search, read) to examine it carefully."
            )
        else:
            trace_content = raw_trace
        user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{trace_content}\n--- END TRACE ---")

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
        if trace_file_path is not None and not context.agentic_parsing_persist_trace:
            try:
                trace_file_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to clean up materialized trace file: %s", trace_file_path, exc_info=True)
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
    unanswered_fields = extract_unanswered_fields_from_report(extraction_input, answer_class)

    system_text = (
        "You are a structured data extraction assistant. "
        "Extract the findings from the investigation report into "
        "the exact JSON schema provided. If the investigation report lists "
        "'_unanswered_fields', set those fields to JSON null in the schema output; "
        "do not turn missing values into 0, false, empty strings, or empty lists.\n\n"
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
        parse_result = parse_to_pydantic_resilient(
            parser,
            messages,
            extraction_class,
            retry_policy=context.parsing_model.retry_policy,
            error_registry=context.error_registry,
        )
        strict_instance = rebuild_strict_answer_with_null_fields(
            answer_class,
            parse_result.parsed,
            unanswered_fields=unanswered_fields,
        )

        return strict_instance, parse_result.usage
    finally:
        close_adapter(parser)


def recover_extraction_from_investigation(
    answer_class: type[BaseAnswer],
    investigation_trace: str,
) -> BaseAnswer:
    """Recover structured answers from machine-readable investigation output.

    This is intentionally deterministic and local. It only runs after the
    agentic investigation completed; it does not inspect the original answerer
    trace or call another model.
    """
    extraction_input = prepare_extraction_input(investigation_trace)
    cleaned = strip_markdown_fences(extraction_input)
    if not cleaned:
        raise ValueError("No JSON-like content found in investigation report")

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Recovered investigation JSON must be an object, got {type(data).__name__}")

    unanswered_fields = extract_unanswered_fields_from_data(data, answer_class)
    extraction_class = build_extraction_relaxed_class(answer_class)
    relaxed_instance = extraction_class.model_validate(data)
    return rebuild_strict_answer_with_null_fields(answer_class, relaxed_instance, unanswered_fields=unanswered_fields)


def extract_unanswered_fields_from_report(
    report: str,
    answer_class: type[BaseAnswer],
) -> set[str]:
    """Extract verified field names explicitly marked as unanswered."""
    cleaned = strip_markdown_fences(report)
    if not cleaned:
        return set()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, dict):
        return set()
    return extract_unanswered_fields_from_data(data, answer_class)


def extract_unanswered_fields_from_data(
    data: dict[str, Any],
    answer_class: type[BaseAnswer],
) -> set[str]:
    """Return verified names from an internal _unanswered_fields list."""
    raw = data.get("_unanswered_fields")
    if not isinstance(raw, list):
        return set()
    verified_names = set(answer_class._get_verified_fields().keys())
    return {name for name in raw if isinstance(name, str) and name in verified_names}


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


def write_agentic_trace_file(
    *,
    workspace_path: Path | None,
    trace: str,
    question_id: str,
    scenario_turn: int | None = None,
) -> Path:
    """Write an answering trace for file-tool access during agentic parsing."""
    if workspace_path is None:
        trace_dir = Path(tempfile.mkdtemp(prefix="karenina_traces_"))
    else:
        trace_dir = Path(workspace_path) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", question_id)
    filename = f"{safe_id}_turn{scenario_turn}_trace.txt" if scenario_turn is not None else f"{safe_id}_trace.txt"
    trace_path = trace_dir / filename
    trace_path.write_text(trace, encoding="utf-8")
    return trace_path


__all__ = [
    "prepare_extraction_input",
    "recover_extraction_from_investigation",
    "extract_unanswered_fields_from_data",
    "extract_unanswered_fields_from_report",
    "run_extraction",
    "run_investigation",
    "truncate_extraction_input",
    "write_agentic_trace_file",
]
