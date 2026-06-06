"""Agentic template parsing stage.

Two-step parsing for coding tasks: an investigation agent with tools
verifies artifacts in the workspace, then a parser extracts structured
data from the investigation findings.
"""

import logging
from typing import Any

from karenina.benchmark.verification.utils.parser_resilience import classify_parser_exception
from karenina.benchmark.verification.utils.schema_builder import (
    build_parsing_schema,
)
from karenina.ports import UsageMetadata
from karenina.schemas.verification.model_identity import ModelIdentity

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext
from ..helpers.agentic_parse_helpers import (
    recover_extraction_from_investigation,
    run_extraction,
    run_investigation,
)

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
        if not super().should_run(context) and not context.can_score_partial_timeout():
            return False
        if not context.agentic_parsing:
            return False
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        if (
            context.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL, False)
            and not context.allow_partial_trace_scoring
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
            investigation_trace, investigation_limit_reached, investigation_usage = run_investigation(
                context, clean_schema
            )
        except Exception as e:
            context.mark_error(
                f"Agentic investigation failed: {e}",
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
        context.set_result_field(
            ArtifactKeys.INVESTIGATION_TRACE,
            investigation_trace,
        )
        if investigation_limit_reached:
            context.mark_error("Agentic investigation hit the turn limit before producing a reliable report")
            return

        # Step 2: Extraction
        try:
            parsed_answer, extraction_usage = run_extraction(
                context,
                answer_class,
                investigation_trace,
                clean_schema,
            )
        except Exception as e:
            context.set_result_field(ArtifactKeys.AGENTIC_EXTRACTION_ERROR, str(e))
            try:
                parsed_answer = recover_extraction_from_investigation(answer_class, investigation_trace)
            except Exception as recovery_error:
                context.mark_error(
                    f"Agentic extraction failed: {e}; local JSON recovery failed: {recovery_error}",
                    category=classify_parser_exception(e, context.error_registry),
                )
                return
            context.set_result_field(ArtifactKeys.AGENTIC_EXTRACTION_RECOVERY, "local_json")
            context.add_warning(f"Agentic extraction recovered locally after parser error: {e}")
        else:
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
