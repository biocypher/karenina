"""Agentic template parsing stage.

Two-step parsing for coding tasks: an investigation agent with tools
verifies artifacts in the workspace, then a parser extracts structured
data from the investigation findings.
"""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from karenina.adapters import get_agent, get_parser
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.utils.schema_builder import build_parsing_schema
from karenina.ports import AgentConfig, PortCapabilities
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.verification.model_identity import ModelIdentity

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class AgenticParseTemplateStage(BaseVerificationStage):
    """Parse template via agentic investigation + structured extraction.

    This stage replaces ParseTemplateStage when agentic_parsing is enabled.
    It runs in two steps:
      1. Investigation: AgentPort.run() with tools examines the workspace
      2. Extraction: ParserPort.parse_to_pydantic() produces the answer

    Requires:
        - "RawAnswer": Validated Answer class
        - "Answer": Answer class with question ID injected
        - "raw_llm_response": Raw answering agent trace

    Produces:
        - "parsed_answer": Parsed Pydantic object
        - "parsing_model_str": Model string for result
        - "investigation_trace": Raw trace from investigation agent
        - "agentic_parsing_performed": True
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "AgenticParseTemplate"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [
            ArtifactKeys.RAW_ANSWER,
            ArtifactKeys.ANSWER,
            ArtifactKeys.RAW_LLM_RESPONSE,
        ]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.PARSED_ANSWER,
            ArtifactKeys.PARSING_MODEL_STR,
            ArtifactKeys.INVESTIGATION_TRACE,
            ArtifactKeys.AGENTIC_PARSING_PERFORMED,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only when agentic parsing is enabled and no prior errors."""
        if not super().should_run(context):
            return False
        if not context.agentic_parsing:
            return False
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        if context.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL, False):
            return False
        if context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED, False):
            return False
        if context.get_artifact(ArtifactKeys.ABSTENTION_DETECTED, False):
            return False
        if context.get_artifact(ArtifactKeys.SUFFICIENCY_DETECTED) is False:
            return False
        return context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE) and context.has_artifact(ArtifactKeys.ANSWER)

    def execute(self, context: VerificationContext) -> None:
        """Run investigation then extraction.

        Args:
            context: Verification context with workspace and config.
        """
        answer_class = context.get_artifact(ArtifactKeys.ANSWER)
        parsing_model = context.parsing_model

        # Build schema once for both investigation and extraction
        clean_schema = build_parsing_schema(answer_class)

        try:
            # Step 1: Investigation
            try:
                investigation_trace = self._run_investigation(context, clean_schema)
            except Exception as e:
                context.mark_error(f"Agentic investigation failed: {e}")
                return

            context.set_artifact(ArtifactKeys.INVESTIGATION_TRACE, investigation_trace)

            # Step 2: Extraction
            try:
                parsed_answer = self._run_extraction(
                    context,
                    answer_class,
                    investigation_trace,
                    clean_schema,
                )
            except Exception as e:
                context.mark_error(f"Agentic extraction failed: {e}")
                return

            # Store results
            model_str = ModelIdentity.from_model_config(
                parsing_model,
                role="parsing",
            ).display_string

            context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed_answer)
            context.set_artifact(ArtifactKeys.PARSING_MODEL_STR, model_str)
            context.set_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED, True)
            context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)
            context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)

            # Result builder fields
            context.set_result_field(
                ArtifactKeys.INVESTIGATION_TRACE,
                investigation_trace,
            )
            context.set_result_field(
                ArtifactKeys.AGENTIC_PARSING_PERFORMED,
                True,
            )

            logger.info("Agentic parsing completed successfully")
        finally:
            # Read the artifact inside finally so cleanup runs even when
            # _run_investigation raised after writing the file (see the
            # set_artifact call inside _run_investigation).
            materialized_trace_path = context.get_artifact("_materialized_trace_path")
            if materialized_trace_path is not None and not context.agentic_parsing_persist_trace:
                try:
                    materialized_trace_path.unlink(missing_ok=True)
                    logger.debug(
                        "Cleaned up materialized trace file: %s",
                        materialized_trace_path,
                    )
                except Exception:
                    logger.warning(
                        "Failed to clean up materialized trace file: %s",
                        materialized_trace_path,
                        exc_info=True,
                    )

    def _run_investigation(
        self,
        context: VerificationContext,
        clean_schema: dict[str, Any],
    ) -> str:
        """Run investigation agent with tools.

        Args:
            context: Verification context.
            clean_schema: Pre-built JSON schema for the answer template.

        Returns:
            Raw trace from the investigation agent.
        """
        agent = get_agent(context.parsing_model)
        schema_json = json.dumps(clean_schema, indent=2)

        system_text = (
            "You are a verification agent evaluating whether an AI coding "
            "assistant correctly completed a task. You have access to the "
            "file system and can execute code.\n\n"
            "Your job is to evaluate the artifacts the assistant left in the "
            "workspace (scripts, output files, logs, data). Read and inspect "
            "these artifacts to determine the results. You may re-run existing "
            "scripts to confirm their output, but do NOT re-implement the task "
            "from scratch or write your own solution. If the workspace contains "
            "no usable artifacts, report that you could not verify the results.\n\n"
            "Report your findings as a JSON object matching this schema:\n"
            f"{schema_json}"
        )

        # Build user prompt based on judge context mode
        user_parts: list[str] = [f"Question: {context.question_text}"]

        materialized_trace_path: Path | None = None
        if context.agentic_judge_context in (
            "trace_and_workspace",
            "trace_only",
        ):
            raw_trace = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
            if context.agentic_parsing_materialize_trace and raw_trace:
                materialized_trace_path = self._write_trace_file(
                    workspace_path=context.workspace_path,
                    trace=raw_trace,
                    question_id=context.question_id,
                    scenario_turn=context.scenario_turn,
                )
                # Stash the path early so execute()'s finally can clean it up
                # even if agent.run raises before we return from this method.
                context.set_artifact(
                    "_materialized_trace_path",
                    materialized_trace_path,
                )
                user_parts.append(
                    f"\nThe full answering agent trace is saved to: "
                    f"{materialized_trace_path}\n"
                    f"Use file tools (Read, Grep, Glob) to examine it."
                )
            else:
                user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{raw_trace}\n--- END TRACE ---")

        if context.workspace_path and context.agentic_judge_context != "trace_only":
            user_parts.append(
                f"\nWorkspace directory: {context.workspace_path}",
            )

        user_text = "\n".join(user_parts)

        # Assemble with adapter + user instructions
        assembler = PromptAssembler(
            task=PromptTask.AGENTIC_PARSING_INVESTIGATION,
            interface=context.parsing_model.interface,
            capabilities=PortCapabilities(),
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

        result = agent.run(messages=messages, config=agent_config)
        logger.info(
            "Investigation completed in %d turns (limit_reached=%s)",
            result.turns,
            result.limit_reached,
        )
        return result.raw_trace

    @staticmethod
    def _write_trace_file(
        workspace_path: Path | None,
        trace: str,
        question_id: str,
        scenario_turn: int | None = None,
    ) -> Path:
        """Write the answering agent trace to a file for filesystem-tool access.

        Mirrors the pattern from
        karenina.benchmark.verification.stages.pipeline.agentic_rubric_evaluation
        so Stage 7b and Stage 11b share trace-materialization semantics. The
        ``scenario_turn`` suffix prevents multi-turn scenarios (where the same
        question_id can be reached via different nodes) from overwriting each
        other's trace file within one workspace.

        When ``workspace_path`` is None, falls back to a tempdir. The file is
        placed under ``<workspace>/.karenina/traces/<safe_qid>[_turn<N>]_trace.txt``.

        Args:
            workspace_path: Resolved workspace directory, or None.
            trace: The full answering agent trace text.
            question_id: Question identifier (sanitized for filesystem safety).
            scenario_turn: Optional turn index for multi-turn scenarios.

        Returns:
            Path to the written trace file.
        """
        if workspace_path is None:
            trace_dir = Path(tempfile.mkdtemp(prefix="karenina_traces_"))
        else:
            trace_dir = Path(workspace_path) / ".karenina" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", question_id)
        filename = f"{safe_id}_turn{scenario_turn}_trace.txt" if scenario_turn is not None else f"{safe_id}_trace.txt"
        trace_path = trace_dir / filename
        trace_path.write_text(trace, encoding="utf-8")
        return trace_path

    def _run_extraction(
        self,
        context: VerificationContext,
        answer_class: type[BaseAnswer],
        investigation_trace: str,
        clean_schema: dict[str, Any],
    ) -> BaseAnswer:
        """Extract structured answer from investigation findings.

        Args:
            context: Verification context.
            answer_class: The answer template class.
            investigation_trace: Raw trace from investigation agent.
            clean_schema: Pre-built JSON schema for the answer template.

        Returns:
            Parsed answer instance.
        """
        parser = get_parser(context.parsing_model)
        schema_json = json.dumps(clean_schema, indent=2)

        system_text = (
            "You are a structured data extraction assistant. "
            "Extract the findings from the investigation report into "
            "the exact JSON schema provided.\n\n"
            f"Schema:\n{schema_json}"
        )
        user_text = f"Investigation report:\n\n{investigation_trace}"

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

        parse_result = parser.parse_to_pydantic(messages, answer_class)
        return parse_result.parsed
