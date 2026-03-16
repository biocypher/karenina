"""Agentic trait evaluator: investigate then extract.

Two-step evaluation for agentic rubric traits:
1. An agent investigates the response/workspace using tools.
2. A parser extracts the final score from the investigation trace.

This mirrors the Stage 7b (AgenticParseTemplate) pattern.
"""

import logging
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from karenina.adapters import get_agent, get_parser
from karenina.ports import AgentConfig, Message
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait
from karenina.schemas.outputs.rubric import (
    SingleBooleanScore,
    SingleLiteralClassification,
    SingleNumericScore,
)

logger = logging.getLogger(__name__)


class AgenticTraitEvaluator:
    """Evaluates a single agentic rubric trait via investigation and extraction.

    Args:
        model_config: The resolved ModelConfig for this evaluation
            (either trait.model_override or the inherited parsing model).
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._model_config = model_config

    def evaluate_trait(
        self,
        trait: AgenticRubricTrait,
        question_text: str,
        raw_llm_response: str | None,
        workspace_path: Path | str | None,
    ) -> tuple[int | bool | dict[str, Any] | None, str | None]:
        """Evaluate a single agentic rubric trait.

        Args:
            trait: The agentic rubric trait to evaluate.
            question_text: The original question text.
            raw_llm_response: The answering model's raw response (may be None).
            workspace_path: Path to the workspace directory (may be None).

        Returns:
            Tuple of (score, investigation_trace).
            score is None if evaluation failed; a dict for template kind traits.
            investigation_trace is None if the agent failed before producing output.
        """
        # Step 1: Investigation
        try:
            investigation_trace = self._run_investigation(
                trait,
                question_text,
                raw_llm_response,
                workspace_path,
            )
        except Exception:
            logger.warning(
                "Agentic investigation failed for trait '%s'",
                trait.name,
                exc_info=True,
            )
            return None, None

        # Step 2: Extraction
        try:
            score = self.run_extraction(trait, investigation_trace)
        except Exception:
            logger.warning(
                "Score extraction failed for trait '%s', preserving trace",
                trait.name,
                exc_info=True,
            )
            return None, investigation_trace

        return score, investigation_trace

    def _run_investigation(
        self,
        trait: AgenticRubricTrait,
        question_text: str,
        raw_llm_response: str | None,
        workspace_path: Path | str | None,
    ) -> str:
        """Launch agent to investigate the response/workspace.

        Returns:
            Raw investigation trace string.
        """
        agent = get_agent(self._model_config)

        system_prompt = (
            "You are an evaluation agent investigating the quality of an LLM response. "
            "You have access to tools and can examine files, run code, and navigate "
            "the workspace.\n\n"
            f"Your task: {trait.description}\n\n"
            "After investigating, summarize your findings clearly. Your investigation "
            f"trace will be parsed into a {trait.kind} result."
        )

        # Build user message based on context_mode
        user_parts: list[str] = [f"Question: {question_text}"]

        if trait.context_mode in ("trace_and_workspace", "trace_only") and raw_llm_response:
            user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{raw_llm_response}\n--- END TRACE ---")

        if workspace_path and trait.context_mode != "trace_only":
            user_parts.append(f"\nWorkspace directory: {workspace_path}")

        messages = [
            Message.system(system_prompt),
            Message.user("\n".join(user_parts)),
        ]

        agent_config = AgentConfig(
            max_turns=trait.max_turns,
            timeout=float(trait.timeout_seconds),
            workspace_path=Path(workspace_path) if workspace_path else None,
        )

        result = agent.run(messages=messages, config=agent_config)
        logger.info(
            "Agentic rubric investigation for '%s' completed in %d turns (limit_reached=%s)",
            trait.name,
            result.turns,
            result.limit_reached,
        )
        return result.raw_trace

    def run_extraction(
        self,
        trait: AgenticRubricTrait,
        investigation_trace: str,
    ) -> int | bool | dict[str, Any]:
        """Extract score from investigation trace.

        This method is public because the shared strategy in
        AgenticRubricEvaluationStage needs to call it directly
        (shared investigation, per-trait extraction).

        Returns:
            The extracted score (bool for boolean traits, int for score/literal),
            or a dict for template kind traits.
        """
        if trait.is_template_kind:
            return self._extract_template(trait, investigation_trace)

        parser = get_parser(self._model_config)
        messages = self._build_extraction_messages(trait, investigation_trace)

        if trait.kind == "boolean":
            bool_result = parser.parse_to_pydantic(messages, SingleBooleanScore)
            return bool_result.parsed.result

        if trait.kind == "score":
            score_result = parser.parse_to_pydantic(messages, SingleNumericScore)
            return score_result.parsed.score

        if trait.kind == "literal":
            literal_result = parser.parse_to_pydantic(
                messages,
                SingleLiteralClassification,
            )
            return self._resolve_literal_index(
                literal_result.parsed.classification,
                trait,
            )

        raise ValueError(f"Unknown trait kind: {trait.kind}")

    def _extract_template(
        self,
        trait: AgenticRubricTrait,
        investigation_trace: str,
    ) -> dict[str, Any]:
        """Extract structured findings into the user's Pydantic class.

        Args:
            trait: The agentic rubric trait with a template kind.
            investigation_trace: Raw text from the agent investigation.

        Returns:
            Dict of extracted field values (model_dump of the parsed model).
        """
        parser = get_parser(self._model_config)
        messages = [
            Message.system(
                "You are extracting structured findings from an investigation. "
                "Based on the investigation output below, fill in every field "
                "of the requested format with evidence from the investigation."
            ),
            Message.user(investigation_trace),
        ]
        kind_class = cast(type[BaseModel], trait.kind)
        parse_result = parser.parse_to_pydantic(messages, kind_class)
        return parse_result.parsed.model_dump()

    def _build_extraction_messages(
        self,
        trait: AgenticRubricTrait,
        investigation_trace: str,
    ) -> list[Message]:
        """Build prompt messages for score extraction from investigation trace."""
        system_parts = [
            "You are a structured data extraction assistant. "
            "Extract the final evaluation result from the investigation report below."
        ]

        if trait.kind == "score":
            system_parts.append(f"\nScore range: {trait.min_score} to {trait.max_score}.")
        elif trait.kind == "literal" and trait.classes:
            class_desc = ", ".join(f"'{k}': {v}" for k, v in trait.classes.items())
            system_parts.append(f"\nClassify into one of: {class_desc}")

        return [
            Message.system("\n".join(system_parts)),
            Message.user(f"Investigation report:\n\n{investigation_trace}"),
        ]

    @staticmethod
    def _resolve_literal_index(
        classification: str,
        trait: AgenticRubricTrait,
    ) -> int:
        """Convert a literal classification name to its integer index.

        Returns -1 if the classification does not match any defined class.
        """
        if not trait.classes:
            logger.warning(
                "No classes defined for literal trait '%s', returning -1",
                trait.name,
            )
            return -1
        class_names = list(trait.classes.keys())
        try:
            return class_names.index(classification)
        except ValueError:
            logger.warning(
                "Invalid classification '%s' for trait '%s', returning -1",
                classification,
                trait.name,
            )
            return -1
