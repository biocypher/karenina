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

from karenina.adapters import get_agent, get_llm, get_parser
from karenina.adapters.agent_runtime import map_path_for_prompt, workspace_path_for_prompt
from karenina.adapters.registry import close_adapter
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.ports import AgentConfig, Message, PortCapabilities
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait
from karenina.schemas.outputs.rubric import (
    SingleBooleanScore,
    SingleLiteralClassification,
    SingleNumericScore,
)
from karenina.schemas.verification.prompt_config import PromptConfig

logger = logging.getLogger(__name__)


class AgenticTraitEvaluator:
    """Evaluates a single agentic rubric trait via investigation and extraction.

    Args:
        model_config: The resolved ModelConfig for this evaluation
            (either trait.model_override or the inherited parsing model).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_config: PromptConfig | None = None,
    ) -> None:
        self._model_config = model_config
        self._prompt_config = prompt_config

    def evaluate_trait(
        self,
        trait: AgenticRubricTrait,
        question_text: str,
        raw_llm_response: str | None,
        workspace_path: Path | str | None,
        trace_file_path: Path | None = None,
    ) -> tuple[int | bool | dict[str, Any] | None, str | None]:
        """Evaluate a single agentic rubric trait.

        Args:
            trait: The agentic rubric trait to evaluate.
            question_text: The original question text.
            raw_llm_response: The answering model's raw response (may be None).
            workspace_path: Path to the workspace directory (may be None).
            trace_file_path: If provided and trait.materialize_trace is True,
                the investigation prompt references this file path instead of
                inlining the full trace.

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
                trace_file_path=trace_file_path,
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
        trace_file_path: Path | None = None,
    ) -> str:
        """Launch agent to investigate the response/workspace.

        For ``trace_only`` mode without a workspace, uses LLMPort (single
        call) instead of AgentPort, since no tool access is needed.

        Args:
            trait: The agentic rubric trait to evaluate.
            question_text: The original question text.
            raw_llm_response: The answering model's raw response (may be None).
            workspace_path: Path to the workspace directory (may be None).
            trace_file_path: If provided and trait.materialize_trace is True,
                the prompt references this file instead of inlining the trace.

        Returns:
            Raw investigation trace string.
        """
        agent = None
        capabilities = PortCapabilities()
        needs_tools = (
            trait.context_mode != "trace_only"
            or workspace_path is not None
            or (trait.materialize_trace and trace_file_path is not None)
        )
        if needs_tools:
            agent = get_agent(self._model_config)
            capabilities = agent.capabilities

        if capabilities.supports_code_execution:
            execution_text = (
                "You may inspect files and execute commands in the sandboxed workspace."
                if capabilities.uses_sandboxed_execution
                else "You may inspect files and execute commands in the configured workspace."
            )
        elif capabilities.supports_file_tools:
            execution_text = "You may inspect files, but command execution is not available."
        else:
            execution_text = "Use only the response text supplied in the prompt."
        system_text = (
            "You are an evaluation agent investigating the quality of an LLM response. "
            f"Your task: {trait.description}\n\n"
            f"{execution_text}\n\n"
            "After investigating, summarize your findings clearly. Your investigation "
            f"trace will be parsed into a {trait.kind} result."
        )

        # Build user message based on context_mode
        user_parts: list[str] = [f"Question: {question_text}"]

        if trait.context_mode in ("trace_and_workspace", "trace_only") and raw_llm_response:
            if trait.materialize_trace and trace_file_path:
                prompt_trace_path = map_path_for_prompt(
                    self._model_config,
                    trace_file_path,
                    Path(workspace_path) if workspace_path else None,
                )
                trace_content = (
                    f"The full agent trace is saved to: {prompt_trace_path}\n"
                    "Use file tools (grep, search, read) to examine it."
                )
            else:
                trace_content = raw_llm_response
            user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{trace_content}\n--- END TRACE ---")

        if workspace_path and trait.context_mode != "trace_only":
            prompt_workspace = workspace_path_for_prompt(self._model_config, Path(workspace_path))
            user_parts.append(f"\nWorkspace directory: {prompt_workspace}")

        user_text = "\n".join(user_parts)

        # Assemble with adapter + user instructions
        assembler = PromptAssembler(
            task=PromptTask.RUBRIC_AGENTIC_TRAIT_INVESTIGATION,
            interface=self._model_config.interface,
            capabilities=capabilities,
        )
        user_instructions = (
            self._prompt_config.get_for_task(PromptTask.RUBRIC_AGENTIC_TRAIT_INVESTIGATION.value)
            if self._prompt_config
            else None
        )
        messages = assembler.assemble(
            system_text=system_text,
            user_text=user_text,
            user_instructions=user_instructions,
        )

        if not needs_tools:
            return self._run_llm_investigation(trait, messages)

        assert agent is not None
        return self._run_agent_investigation(trait, messages, workspace_path, agent=agent)

    def _run_llm_investigation(
        self,
        trait: AgenticRubricTrait,
        messages: list[Message],
    ) -> str:
        """Run investigation via LLMPort (single call, no tools).

        Used for trace_only mode where the trace is inlined in the prompt
        and no workspace tools are needed.
        """
        llm = get_llm(self._model_config)
        try:
            result = llm.invoke(messages)
            logger.info(
                "Agentic rubric investigation for '%s' completed via LLMPort (trace_only, no tools)",
                trait.name,
            )
            return result.content or ""
        finally:
            close_adapter(llm)

    def _run_agent_investigation(
        self,
        trait: AgenticRubricTrait,
        messages: list[Message],
        workspace_path: Path | str | None,
        *,
        agent: Any | None = None,
    ) -> str:
        """Run investigation via AgentPort (multi-turn with tools)."""
        if agent is None:
            agent = get_agent(self._model_config)

        agent_config = AgentConfig(
            max_turns=trait.max_turns,
            timeout=float(trait.timeout_seconds),
            workspace_path=Path(workspace_path) if workspace_path else None,
        )

        try:
            result = agent.run(messages=messages, config=agent_config)
            logger.info(
                "Agentic rubric investigation for '%s' completed in %d turns (limit_reached=%s)",
                trait.name,
                result.turns,
                result.limit_reached,
            )
            return result.raw_trace
        finally:
            close_adapter(agent)

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
        system_text, user_text = self._build_extraction_texts(trait, investigation_trace)

        # Assemble with adapter + user instructions
        assembler = PromptAssembler(
            task=PromptTask.RUBRIC_AGENTIC_TRAIT_EXTRACTION,
            interface=self._model_config.interface,
            capabilities=parser.capabilities,
        )
        user_instructions = (
            self._prompt_config.get_for_task(PromptTask.RUBRIC_AGENTIC_TRAIT_EXTRACTION.value)
            if self._prompt_config
            else None
        )
        messages = assembler.assemble(
            system_text=system_text,
            user_text=user_text,
            user_instructions=user_instructions,
        )

        try:
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
        finally:
            close_adapter(parser)

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

        system_text = (
            "You are extracting structured findings from an investigation. "
            "Based on the investigation output below, fill in every field "
            "of the requested format with evidence from the investigation."
        )
        user_text = investigation_trace

        # Assemble with adapter + user instructions
        assembler = PromptAssembler(
            task=PromptTask.RUBRIC_AGENTIC_TRAIT_EXTRACTION,
            interface=self._model_config.interface,
            capabilities=parser.capabilities,
        )
        user_instructions = (
            self._prompt_config.get_for_task(PromptTask.RUBRIC_AGENTIC_TRAIT_EXTRACTION.value)
            if self._prompt_config
            else None
        )
        messages = assembler.assemble(
            system_text=system_text,
            user_text=user_text,
            user_instructions=user_instructions,
        )

        try:
            kind_class = cast(type[BaseModel], trait.kind)
            parse_result = parser.parse_to_pydantic(messages, kind_class)
            return parse_result.parsed.model_dump()
        finally:
            close_adapter(parser)

    @staticmethod
    def _build_extraction_texts(
        trait: AgenticRubricTrait,
        investigation_trace: str,
    ) -> tuple[str, str]:
        """Build base prompt texts for score extraction from investigation trace.

        Returns:
            Tuple of (system_text, user_text) before assembly.
        """
        system_parts = [
            "You are a structured data extraction assistant. "
            "Extract the final evaluation result from the investigation report below."
        ]

        if trait.kind == "score":
            system_parts.append(f"\nScore range: {trait.min_score} to {trait.max_score}.")
        elif trait.kind == "literal" and trait.classes:
            class_desc = ", ".join(f"'{k}': {v}" for k, v in trait.classes.items())
            system_parts.append(f"\nClassify into one of: {class_desc}")

        return "\n".join(system_parts), f"Investigation report:\n\n{investigation_trace}"

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
