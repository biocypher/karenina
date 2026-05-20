"""Agentic template parsing stage.

Two-step parsing for coding tasks: an investigation agent with tools
verifies artifacts in the workspace, then a parser extracts structured
data from the investigation findings.
"""

import json
import logging
from typing import Any

from karenina.adapters import get_agent, get_parser
from karenina.adapters.agent_runtime import workspace_path_for_prompt
from karenina.adapters.registry import close_adapter
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.utils.schema_builder import build_parsing_schema
from karenina.ports import AgentConfig, UsageMetadata
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
        """Run investigation then extraction."""
        answer_class = context.get_artifact(ArtifactKeys.ANSWER)
        parsing_model = context.parsing_model
        parsing_model_str = ModelIdentity.from_model_config(
            parsing_model,
            role="parsing",
        ).display_string
        usage_tracker = self.get_or_create_usage_tracker(context)

        clean_schema = build_parsing_schema(answer_class)

        # Step 1: Investigation
        try:
            investigation_trace, investigation_limit_reached, investigation_usage = self._run_investigation(
                context, clean_schema
            )
        except Exception as e:
            context.mark_error(f"Agentic investigation failed: {e}")
            return

        self._track_usage(
            usage_tracker,
            "agentic_parsing_investigation",
            parsing_model_str,
            investigation_usage,
        )
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)
        context.set_artifact(ArtifactKeys.INVESTIGATION_TRACE, investigation_trace)
        context.set_result_field(
            ArtifactKeys.INVESTIGATION_TRACE,
            investigation_trace,
        )
        if investigation_limit_reached:
            context.mark_error("Agentic investigation hit the turn limit before producing a reliable report")
            return

        # Step 2: Extraction
        try:
            parsed_answer, extraction_usage = self._run_extraction(
                context,
                answer_class,
                investigation_trace,
                clean_schema,
            )
        except Exception as e:
            context.mark_error(f"Agentic extraction failed: {e}")
            return

        self._track_usage(
            usage_tracker,
            "agentic_parsing_extraction",
            parsing_model_str,
            extraction_usage,
        )
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

        # Store results
        context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed_answer)
        context.set_artifact(ArtifactKeys.PARSING_MODEL_STR, parsing_model_str)
        context.set_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED, True)
        context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)
        context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)

        context.set_result_field(
            ArtifactKeys.AGENTIC_PARSING_PERFORMED,
            True,
        )

        logger.info("Agentic parsing completed successfully")

    def _run_investigation(
        self,
        context: VerificationContext,
        clean_schema: dict[str, Any],
    ) -> tuple[str, bool, UsageMetadata]:
        """Run investigation agent with tools.

        Args:
            context: Verification context.
            clean_schema: Pre-built JSON schema for the answer template.

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

    def _run_extraction(
        self,
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

        try:
            parse_result = parser.parse_to_pydantic(messages, answer_class)
            return parse_result.parsed, parse_result.usage
        finally:
            close_adapter(parser)

    @staticmethod
    def _track_usage(
        usage_tracker: Any,
        stage_name: str,
        model_str: str,
        usage: UsageMetadata,
    ) -> None:
        """Track non-empty token usage for one agentic parsing substage."""
        usage_dict = usage.to_dict()
        if usage_dict.get("input_tokens", 0) > 0 or usage_dict.get("output_tokens", 0) > 0:
            usage_tracker.track_call(stage_name, model_str, usage_dict)
