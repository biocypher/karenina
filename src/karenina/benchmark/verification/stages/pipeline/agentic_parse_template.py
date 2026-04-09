"""Agentic template parsing stage.

Two-step parsing for coding tasks: an investigation agent with tools
verifies artifacts in the workspace, then a parser extracts structured
data from the investigation findings.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from karenina.adapters import get_agent, get_parser
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.utils.schema_builder import build_parsing_schema
from karenina.ports import AgentConfig, PortCapabilities
from karenina.ports.messages import Message
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.verification.model_identity import ModelIdentity

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)

_DEFAULT_TRACE_TRUNCATION_THRESHOLD = 2000


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
                conv_history = context.get_artifact("conversation_history")
                materialized_trace_path = self._write_trace_file(
                    workspace_path=context.workspace_path,
                    trace=raw_trace,
                    question_id=context.question_id,
                    scenario_turn=context.scenario_turn,
                    question_text=context.question_text,
                    conversation_history=conv_history if conv_history else None,
                )
                # Stash the path early so execute()'s finally can clean it up
                # even if agent.run raises before we return from this method.
                context.set_artifact(
                    "_materialized_trace_path",
                    materialized_trace_path,
                )
                user_parts.append(
                    "\nThe full answering agent trace has been saved as a "
                    f"plain-text file at:\n  {materialized_trace_path}\n\n"
                    "The file starts with a header line reading "
                    "`# KARENINA RAW ANSWERING AGENT TRACE`. Everything "
                    "below that header is the VERBATIM conversation the "
                    "answering agent produced, serialized for file-tool "
                    "access. It contains raw `--- Thinking ---`, "
                    "`--- AI Message ---`, and `--- Tool Message ---` "
                    "blocks. Any reasoning, hedging, or meta-commentary "
                    "you see inside those blocks is the answering agent's "
                    "OWN output, not a pre-existing analysis by anyone "
                    "else. Do NOT dismiss the file as an analysis or "
                    "summary; treat every block as raw evidence.\n\n"
                    "Recommended workflow:\n"
                    "1. First, Read the entire file so you have full "
                    "context. If the file is large and Read truncates, "
                    "keep calling Read with increasing offset (and an "
                    "explicit limit) until you have walked all the way to "
                    "the end. Verify you reached the last block.\n"
                    "2. Then use the Grep tool with targeted patterns "
                    "against the file path to pinpoint salient passages. "
                    "Useful patterns include:\n"
                    "   - `Tool Calls` to find where the agent called tools\n"
                    "   - `Tool Message` to find the tool responses\n"
                    "   - `AI Message` to find each assistant utterance\n"
                    "   - free-text patterns tied to the specific question "
                    "(entity IDs, numeric values, specific claims under "
                    "review, or any term that might reveal a reversal)\n"
                    "3. Cross-check the agent's final answer against its "
                    "earlier answers and against its own tool outputs. "
                    "Base your judgement on the evidence you find in the "
                    "file itself, not on prior assumptions.\n\n"
                    "Note: some large content blocks (tool results, long "
                    "text) may have been offloaded to separate files in "
                    "the `artifacts/` subdirectory next to the trace file. "
                    'When you see an XML element with `offloaded="true"`, '
                    "the element body contains the file path. Use Read to "
                    "access the full content when needed."
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

    # Regex matching tagged-message lines produced by format_transcript().
    # Captures: group 1 = full tag (e.g. "__user__" or "agent:assistant:text"),
    # group 2 = content after the tag on the same line.
    _TAG_RE = re.compile(
        r"^\[(__user__|[^\]]+:(?:system|user|assistant|tool):(?:text|tool_use|tool_result))\] ?(.*)",
    )

    @staticmethod
    def _serialize_conversation_history(history: list[Message]) -> str:
        """Serialize a list of Message objects into labeled text blocks.

        Uses ``Message.text`` which joins TextContent blocks only.
        ToolUseContent and ToolResultContent are not rendered; this is
        acceptable for the current scenario guardrail use case where
        conversation history contains user/assistant text turns.

        Args:
            history: Prior-turn messages from the scenario conversation.

        Returns:
            Formatted string with ``--- Role Message ---`` delimiters.
        """
        parts: list[str] = []
        for msg in history:
            role_label = msg.role.value.title()
            parts.append(f"--- {role_label} Message ---")
            parts.append(msg.text)
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _reformat_transcript_as_xml(
        question_text: str,
        artifacts_dir: Path | None = None,
        truncation_threshold: int = _DEFAULT_TRACE_TRUNCATION_THRESHOLD,
    ) -> str:
        """Reformat transcript_prepend question text into XML-structured turns.

        Detects whether ``question_text`` contains a ``transcript_prepend``
        pattern (``[__user__]`` tags followed by the ``\\n\\n---\\n\\n``
        separator). If so, parses the tagged-message format produced by
        ``format_transcript()`` and reformats into nested XML with
        ``<turn>``, ``<user>``, ``<assistant>``, ``<text>``, ``<tool_call>``,
        and ``<tool_result>`` elements.

        Content blocks exceeding ``truncation_threshold`` characters are
        offloaded to separate files in ``artifacts_dir`` when provided.

        If the text does not match the transcript pattern, returns it
        unchanged.

        Args:
            question_text: The question prompt, potentially containing a
                prepended transcript.
            artifacts_dir: Directory for offloaded content files. When None,
                all content stays inline.
            truncation_threshold: Character count above which content blocks
                are offloaded to separate files.

        Returns:
            Reformatted text with XML structure, or the original text.
        """
        separator = "\n\n---\n\n"
        if separator not in question_text or "[__user__]" not in question_text:
            return question_text

        transcript_part, eval_part = question_text.split(separator, 1)

        # Parse tagged lines into entries
        entries = AgenticParseTemplateStage._parse_transcript_entries(
            transcript_part,
        )
        if not entries:
            return question_text

        # Group into turns and format as XML
        turns = AgenticParseTemplateStage._group_entries_into_turns(entries)
        xml = AgenticParseTemplateStage._format_turns_as_xml(
            turns,
            artifacts_dir=artifacts_dir,
            truncation_threshold=truncation_threshold,
        )

        return xml + "\n\n" + eval_part

    @staticmethod
    def _parse_transcript_entries(
        transcript: str,
    ) -> list[dict[str, str | None]]:
        """Parse tagged-message transcript into a list of entry dicts.

        Each entry has keys: role, agent_id, content_type, content.
        Multi-line content (continuation lines without a tag prefix)
        is appended to the previous entry.

        Args:
            transcript: Raw transcript text from format_transcript().

        Returns:
            List of parsed entries.
        """
        tag_re = AgenticParseTemplateStage._TAG_RE
        entries: list[dict[str, str | None]] = []
        current: dict[str, str | None] | None = None

        for line in transcript.split("\n"):
            match = tag_re.match(line)
            if match:
                if current is not None:
                    entries.append(current)
                tag, content = match.group(1), match.group(2)
                if tag == "__user__":
                    current = {
                        "role": "user",
                        "agent_id": None,
                        "content_type": "text",
                        "content": content,
                    }
                else:
                    # 3-part tag: agent_id:role:content_type
                    parts = tag.rsplit(":", 2)
                    agent_id, role, content_type = parts[0], parts[1], parts[2]
                    current = {
                        "role": role,
                        "agent_id": agent_id,
                        "content_type": content_type,
                        "content": content,
                    }
            elif current is not None:
                current["content"] = (current["content"] or "") + "\n" + line

        if current is not None:
            entries.append(current)

        return entries

    @staticmethod
    def _group_entries_into_turns(
        entries: list[dict[str, str | None]],
    ) -> list[dict[str, Any]]:
        """Group parsed entries into turns.

        Each ``[__user__]`` entry starts a new turn. All subsequent
        non-user entries become the assistant response blocks for
        that turn.

        Args:
            entries: Parsed transcript entries.

        Returns:
            List of turn dicts with ``user_content``, ``agent_id``,
            and ``blocks`` (list of assistant entries).
        """
        turns: list[dict[str, Any]] = []
        current_turn: dict[str, Any] | None = None

        for entry in entries:
            if entry["role"] == "user":
                if current_turn is not None:
                    turns.append(current_turn)
                current_turn = {
                    "user_content": entry["content"],
                    "agent_id": None,
                    "blocks": [],
                }
            elif current_turn is not None:
                if current_turn["agent_id"] is None and entry["agent_id"]:
                    current_turn["agent_id"] = entry["agent_id"]
                current_turn["blocks"].append(entry)

        if current_turn is not None:
            turns.append(current_turn)

        return turns

    @staticmethod
    def _format_turns_as_xml(
        turns: list[dict[str, Any]],
        artifacts_dir: Path | None = None,
        truncation_threshold: int = _DEFAULT_TRACE_TRUNCATION_THRESHOLD,
    ) -> str:
        """Format grouped turns into XML with nested elements.

        Produces ``<turn>``, ``<user>``, ``<assistant>``, ``<text>``,
        ``<tool_call>``, and ``<tool_result>`` elements. Tool results
        carry a ``name`` attribute from the most recent preceding
        tool call.

        Content blocks exceeding ``truncation_threshold`` are offloaded
        to numbered files in ``artifacts_dir`` when provided.

        Args:
            turns: Grouped turn dicts from ``_group_entries_into_turns``.
            artifacts_dir: Directory for offloaded content. None disables
                offloading.
            truncation_threshold: Char count above which blocks are
                offloaded.

        Returns:
            XML-formatted string.
        """
        artifact_counter = 0

        def _maybe_offload(content: str, tag_type: str) -> tuple[str, bool]:
            """Offload content to a file if it exceeds the threshold.

            Returns (content_or_reference, was_offloaded).
            """
            nonlocal artifact_counter
            if artifacts_dir is None or len(content) <= truncation_threshold:
                return content, False

            artifact_counter += 1
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{tag_type}_{artifact_counter:03d}.txt"
            artifact_path = artifacts_dir / filename
            artifact_path.write_text(content, encoding="utf-8")

            reference = f"[Content offloaded: {len(content):,} chars]\nFile: {artifact_path}"
            return reference, True

        def _emit_block(
            parts: list[str],
            tag: str,
            attrs: str,
            content: str,
            tag_type: str,
        ) -> None:
            """Emit an XML block, offloading content if needed."""
            body, offloaded = _maybe_offload(content.strip(), tag_type)
            offload_attr = ' offloaded="true"' if offloaded else ""
            parts.append(f"    <{tag}{attrs}{offload_attr}>")
            for line in body.split("\n"):
                parts.append(f"      {line}")
            parts.append(f"    </{tag}>")

        parts: list[str] = []

        for idx, turn in enumerate(turns, 1):
            parts.append(f'<turn number="{idx}">')
            parts.append("  <user>")
            for line in (turn["user_content"] or "").strip().split("\n"):
                parts.append(f"    {line}")
            parts.append("  </user>")

            if turn["blocks"]:
                agent_id = turn["agent_id"] or "unknown"
                parts.append(f'  <assistant agent="{agent_id}">')

                last_tool_name = "unknown"
                for block in turn["blocks"]:
                    ctype = block["content_type"]
                    raw = (block["content"] or "").strip()

                    if ctype == "text":
                        _emit_block(parts, "text", "", raw, "text")
                    elif ctype == "tool_use":
                        paren_idx = raw.find("(")
                        if paren_idx > 0 and raw.endswith(")"):
                            tool_name = raw[:paren_idx]
                            tool_args = raw[paren_idx + 1 : -1]
                        else:
                            tool_name = "unknown"
                            tool_args = raw
                        last_tool_name = tool_name
                        _emit_block(
                            parts,
                            "tool_call",
                            f' name="{tool_name}"',
                            tool_args,
                            "tool_call",
                        )
                    elif ctype == "tool_result":
                        _emit_block(
                            parts,
                            "tool_result",
                            f' name="{last_tool_name}"',
                            raw,
                            "tool_result",
                        )

                parts.append("  </assistant>")
            parts.append("</turn>")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _write_trace_file(
        workspace_path: Path | None,
        trace: str,
        question_id: str,
        scenario_turn: int | None = None,
        question_text: str | None = None,
        conversation_history: list[Message] | None = None,
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

        When ``question_text`` or ``conversation_history`` are provided, they
        are included as structured sections before the answering model response.
        This gives the investigation agent full context for multi-turn scenario
        nodes (where the raw trace alone is just the current turn's output).

        Args:
            workspace_path: Resolved workspace directory, or None.
            trace: The full answering agent trace text.
            question_id: Question identifier (sanitized for filesystem safety).
            scenario_turn: Optional turn index for multi-turn scenarios.
            question_text: The question (or transcript-prepended prompt) the
                answering model received. Included as a "QUESTION CONTEXT"
                section when provided.
            conversation_history: Prior-turn Message objects from the scenario
                conversation. Included as a "CONVERSATION HISTORY" section
                when the list is non-empty.

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

        # Distinctive header so the investigation agent cannot misread the
        # file as a pre-existing analysis. The raw --- AI Message --- blocks
        # (including the agent's own inline reasoning like "let me verify")
        # have historically been confused for someone else's commentary.
        header = (
            "# =============================================================\n"
            "# KARENINA RAW ANSWERING AGENT TRACE\n"
            "# -------------------------------------------------------------\n"
            "# This file is the VERBATIM conversation the answering agent\n"
            "# produced, serialized for file-tool access. It contains raw\n"
            "# blocks of the form:\n"
            "#\n"
            "#   --- Thinking ---      (the agent's private reasoning)\n"
            "#   --- AI Message ---    (assistant utterances; may include\n"
            "#                          Tool Calls: <name> lines with args)\n"
            "#   --- Tool Message --- (tool responses, JSON or text)\n"
            "#\n"
            "# When QUESTION CONTEXT or CONVERSATION HISTORY sections are\n"
            "# present, they show what the answering model received as input.\n"
            "# The ANSWERING MODEL RESPONSE section is its output.\n"
            "#\n"
            "# It is NOT an analysis, evaluation, or summary written by\n"
            "# anyone else. Any reasoning or hedging text you see is the\n"
            "# answering agent's OWN output. Treat every block as evidence.\n"
            "# =============================================================\n\n"
        )

        sections: list[str] = [header]

        # Resolve truncation threshold from env var with fallback.
        threshold = int(
            os.environ.get(
                "KARENINA_TRACE_TRUNCATION_THRESHOLD",
                str(_DEFAULT_TRACE_TRUNCATION_THRESHOLD),
            ),
        )
        artifacts_dir = trace_dir / "artifacts" if workspace_path else None

        if question_text:
            formatted_text = AgenticParseTemplateStage._reformat_transcript_as_xml(
                question_text,
                artifacts_dir=artifacts_dir,
                truncation_threshold=threshold,
            )
            sections.append(f"# QUESTION CONTEXT\n# =================\n{formatted_text}\n\n")

        if conversation_history:
            serialized = AgenticParseTemplateStage._serialize_conversation_history(
                conversation_history,
            )
            sections.append(
                "# CONVERSATION HISTORY (prior scenario turns)\n"
                "# ============================================\n"
                f"{serialized}\n"
            )

        # Label the response section when context sections are present,
        # so the boundary between input and output is unambiguous.
        if question_text or conversation_history:
            sections.append("# ANSWERING MODEL RESPONSE\n# ========================\n")

        sections.append(trace)

        trace_path.write_text("".join(sections), encoding="utf-8")
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
