"""Dynamic template parsing stage.

First tries to parse the final answer message directly. If the final message
is malformed or insufficient, it escalates to the shared agentic investigation
and extraction flow.
"""

import json
import logging
from typing import Any

from pydantic import ValidationError

from karenina.adapters import get_llm
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.prompts.parsing import (
    DYNAMIC_PARSING_DECISION_SYS,
    DYNAMIC_PARSING_DECISION_USER,
)
from karenina.benchmark.verification.stages.helpers.agentic_parse_helpers import (
    run_extraction,
    run_investigation,
)
from karenina.benchmark.verification.utils.schema_builder import (
    build_extraction_relaxed_class,
    build_parsing_schema,
    rebuild_strict_answer_with_null_fields,
)
from karenina.benchmark.verification.utils.trace_parsing import prepare_evaluation_input
from karenina.ports import LLMResponse, UsageMetadata
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.json_extraction import strip_markdown_fences

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class DynamicParseTemplateStage(BaseVerificationStage):
    """Parse templates directly from the final message, escalating when needed."""

    @property
    def name(self) -> str:
        """Stage name."""
        return "DynamicParseTemplate"

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
            ArtifactKeys.DYNAMIC_PARSING_PERFORMED,
            ArtifactKeys.DYNAMIC_PARSE_DECISION,
            ArtifactKeys.DYNAMIC_DECISION_REASONING,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only for dynamic agentic parsing when parsing is still allowed."""
        if not super().should_run(context) and not context.can_score_partial_timeout():
            return False
        if not context.agentic_parsing:
            return False
        if context.agentic_parsing_trigger != "dynamic":
            return False
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        if (
            context.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL, False)
            and not context.allow_partial_trace_scoring
        ):
            return False
        entry = context.get_artifact(ArtifactKeys.REPLAY_ENTRY)
        if (
            entry is not None
            and getattr(entry, "verify_result", None) is not None
            and getattr(entry, "parsed_answer_fields", None) is None
        ):
            return False
        if context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED, False):
            return False
        if context.get_artifact(ArtifactKeys.ABSTENTION_DETECTED, False):
            return False
        if context.get_artifact(ArtifactKeys.SUFFICIENCY_DETECTED) is False:
            return False
        return context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE) and context.has_artifact(ArtifactKeys.ANSWER)

    def execute(self, context: VerificationContext) -> None:
        """Run direct final-message parsing, escalating to agentic parsing as needed."""
        answer_class = context.get_artifact(ArtifactKeys.ANSWER)
        raw_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        parsing_model_str = ModelIdentity.from_model_config(
            context.parsing_model,
            role="parsing",
        ).display_string
        usage_tracker = self.get_or_create_usage_tracker(context)
        clean_schema = build_parsing_schema(answer_class)

        self.set_artifact_and_result(context, ArtifactKeys.DYNAMIC_PARSING_PERFORMED, True)

        final_message, extraction_error = prepare_evaluation_input(raw_response, use_full_trace=False)
        context.set_result_field(ArtifactKeys.USED_FULL_TRACE, False)
        context.set_result_field(ArtifactKeys.EVALUATION_INPUT, None)
        context.set_artifact(ArtifactKeys.TRACE_EXTRACTION_ERROR, extraction_error)
        context.set_result_field(ArtifactKeys.TRACE_EXTRACTION_ERROR, extraction_error)

        if extraction_error:
            self._escalate(
                context,
                answer_class,
                clean_schema,
                parsing_model_str,
                usage_tracker,
                screening_reasoning=extraction_error,
            )
            return

        try:
            decision_response = self._run_decision(context, final_message, clean_schema)
        except Exception as e:
            context.mark_error(
                f"Dynamic parsing decision failed: {e}",
                category=context.error_registry.classify(e),
            )
            return

        self._track_usage(
            usage_tracker,
            "dynamic_parsing_decision",
            parsing_model_str,
            decision_response.usage,
        )
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

        decision = self._parse_decision(decision_response.content)
        reasoning = decision.get("reasoning") if isinstance(decision, dict) else None
        screening_reasoning = reasoning if isinstance(reasoning, str) else None
        if screening_reasoning:
            self.set_artifact_and_result(context, ArtifactKeys.DYNAMIC_DECISION_REASONING, screening_reasoning)

        if not isinstance(decision, dict):
            self._record_malformed_decision(context)
            self._escalate(
                context,
                answer_class,
                clean_schema,
                parsing_model_str,
                usage_tracker,
                screening_reasoning=screening_reasoning,
            )
            return

        sufficient = decision.get("sufficient")
        if sufficient is False:
            self._escalate(
                context,
                answer_class,
                clean_schema,
                parsing_model_str,
                usage_tracker,
                screening_reasoning=screening_reasoning,
            )
            return

        answer_payload = decision.get("answer")
        if sufficient is not True or not isinstance(answer_payload, dict):
            self._record_malformed_decision(context)
            self._escalate(
                context,
                answer_class,
                clean_schema,
                parsing_model_str,
                usage_tracker,
                screening_reasoning=screening_reasoning,
            )
            return

        try:
            extraction_class = build_extraction_relaxed_class(answer_class)
            relaxed_instance = extraction_class.model_validate(answer_payload)
            parsed_answer = rebuild_strict_answer_with_null_fields(answer_class, relaxed_instance)
        except ValidationError:
            logger.warning("Dynamic direct parse validation failed; escalating to agentic parsing", exc_info=True)
            self._record_malformed_decision(context)
            self._escalate(
                context,
                answer_class,
                clean_schema,
                parsing_model_str,
                usage_tracker,
                screening_reasoning=screening_reasoning,
            )
            return

        context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed_answer)
        context.set_artifact(ArtifactKeys.PARSING_MODEL_STR, parsing_model_str)
        context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)
        context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)
        context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)
        self.set_artifact_and_result(context, ArtifactKeys.DYNAMIC_PARSE_DECISION, "direct")

        logger.info("Dynamic direct parsing completed successfully")

    def _run_decision(
        self,
        context: VerificationContext,
        final_message: str,
        clean_schema: dict[str, Any],
    ) -> LLMResponse:
        """Run the direct parsing decision prompt."""
        llm = get_llm(context.parsing_model)
        schema_json = json.dumps(clean_schema, indent=2)
        user_text = DYNAMIC_PARSING_DECISION_USER.format(
            question=context.question_text,
            response=final_message,
            schema=schema_json,
        )
        assembler = PromptAssembler(
            task=PromptTask.AGENTIC_PARSING_DECISION,
            interface=context.parsing_model.interface,
            capabilities=llm.capabilities,
        )
        user_instructions = (
            context.prompt_config.get_for_task(PromptTask.AGENTIC_PARSING_DECISION.value)
            if context.prompt_config
            else None
        )
        messages = assembler.assemble(
            system_text=DYNAMIC_PARSING_DECISION_SYS,
            user_text=user_text,
            user_instructions=user_instructions,
            instruction_context={
                "json_schema": clean_schema,
                "format_instructions": "",
            },
        )
        return llm.invoke(messages)

    @staticmethod
    def _parse_decision(content: str) -> dict[str, Any] | None:
        """Parse a dynamic decision response, accepting fenced or mixed JSON."""
        cleaned = strip_markdown_fences(content)
        if not isinstance(cleaned, str):
            return None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Dynamic parsing decision was not valid JSON: %r", content)
            return None
        return parsed if isinstance(parsed, dict) else None

    def _record_malformed_decision(self, context: VerificationContext) -> None:
        """Record that the dynamic decision was malformed."""
        self.set_artifact_and_result(context, ArtifactKeys.PARSE_DECISION_MALFORMED, True)

    def _escalate(
        self,
        context: VerificationContext,
        answer_class: type[BaseAnswer],
        clean_schema: dict[str, Any],
        parsing_model_str: str,
        usage_tracker: Any,
        *,
        screening_reasoning: str | None = None,
    ) -> None:
        """Escalate to shared agentic investigation and extraction."""
        self.set_artifact_and_result(context, ArtifactKeys.DYNAMIC_PARSE_DECISION, "escalated")

        try:
            investigation_trace, investigation_limit_reached, investigation_usage = run_investigation(
                context,
                clean_schema,
                screening_reasoning=screening_reasoning,
            )
        except Exception as e:
            context.mark_error(
                f"Dynamic agentic investigation failed: {e}",
                category=context.error_registry.classify(e),
            )
            return

        self._track_usage(
            usage_tracker,
            "agentic_parsing_investigation",
            parsing_model_str,
            investigation_usage,
        )
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)
        context.set_artifact(ArtifactKeys.INVESTIGATION_TRACE, investigation_trace)
        context.set_result_field(ArtifactKeys.INVESTIGATION_TRACE, investigation_trace)

        if investigation_limit_reached:
            context.mark_error("Dynamic agentic investigation hit the turn limit before producing a reliable report")
            return

        try:
            parsed_answer, extraction_usage = run_extraction(
                context,
                answer_class,
                investigation_trace,
                clean_schema,
            )
        except Exception as e:
            context.mark_error(
                f"Dynamic agentic extraction failed: {e}",
                category=context.error_registry.classify(e),
            )
            return

        self._track_usage(
            usage_tracker,
            "agentic_parsing_extraction",
            parsing_model_str,
            extraction_usage,
        )
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

        context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed_answer)
        context.set_artifact(ArtifactKeys.PARSING_MODEL_STR, parsing_model_str)
        self.set_artifact_and_result(context, ArtifactKeys.AGENTIC_PARSING_PERFORMED, True)
        context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)
        context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)
        context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)

        logger.info("Dynamic escalated agentic parsing completed successfully")

    @staticmethod
    def _track_usage(
        usage_tracker: Any,
        stage_name: str,
        model_str: str,
        usage: UsageMetadata,
    ) -> None:
        """Track non-empty token usage for one dynamic parsing substage."""
        usage_dict = usage.to_dict()
        if usage_dict.get("input_tokens", 0) > 0 or usage_dict.get("output_tokens", 0) > 0:
            usage_tracker.track_call(stage_name, model_str, usage_dict)
