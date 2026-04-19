"""Answer generation stage.

Generates LLM responses to questions using the configured answering model.
Uses a unified adapter path for ALL interfaces including manual.
"""

import hashlib
import logging
import os
import shutil
import time
import traceback
from typing import Any

from karenina.adapters import get_agent, get_llm
from karenina.benchmark.verification.utils.llm_invocation import _construct_few_shot_prompt
from karenina.benchmark.verification.utils.trace_agent_metrics import extract_agent_metrics_from_messages
from karenina.benchmark.verification.utils.trace_usage_tracker import UsageTracker
from karenina.ports import AgentConfig, AgentPort, LLMPort, Message, Role
from karenina.replay.exceptions import ReplayMissError
from karenina.replay.ports_message_hydration import hydrate_trace_messages
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.errors import ErrorCategory

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


def _try_replay_hit(context: VerificationContext) -> Any:
    """Look up a ReplayEntry for the current turn.

    Returns the ReplayEntry on hit, None on fall-through miss, and
    raises ReplayMissError on strict miss. Uses getattr defaults so
    unit tests with SimpleNamespace context fakes that pre-date the
    replay fields still work without modification.
    """
    store = getattr(context, "replay_store", None)
    if store is None:
        return None

    answering_display = ModelIdentity.from_model_config(context.answering_model, role="answering").display_string

    return store.lookup(
        question_id=context.question_id,
        scenario_id=getattr(context, "scenario_id", None),
        scenario_node=getattr(context, "scenario_node", None),
        answering_model_id=answering_display,
        visit_index=getattr(context, "scenario_node_visit_index", None),
        replicate=getattr(context, "replicate", None),
    )


def _populate_artifacts_from_replay_entry(
    context: VerificationContext,
    entry: Any,
) -> None:
    """Populate GenerateAnswer output artifacts from a ReplayEntry.

    Mirrors the artifact set a live GenerateAnswer would produce so
    downstream stages see indistinguishable input. Fields that
    FinalizeResultStage reads from result fields (raw_llm_response,
    recursion_limit_reached) are written to both the artifact map and
    the result field map; see issue 198.
    """
    answering_model_str = (
        ModelIdentity.from_model_config(context.answering_model, role="answering").display_string + " (replay)"
    )

    context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, entry.raw_trace)
    context.set_result_field(ArtifactKeys.RAW_LLM_RESPONSE, entry.raw_trace)

    if entry.trace_messages:
        try:
            hydrated = hydrate_trace_messages(entry.trace_messages)
        except Exception as e:  # noqa: BLE001
            logger.warning("Replay trace_messages hydration failed: %s", e, exc_info=True)
            hydrated = None
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, hydrated)
    else:
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, None)

    recursion_limit_reached = bool((entry.agent_metrics or {}).get("limit_reached", False))
    context.set_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, recursion_limit_reached)
    context.set_result_field(ArtifactKeys.RECURSION_LIMIT_REACHED, recursion_limit_reached)
    context.set_artifact(ArtifactKeys.ANSWERING_MODEL_STR, answering_model_str)
    context.set_artifact(ArtifactKeys.REPLAY_ENTRY, entry)


class GenerateAnswerStage(BaseVerificationStage):
    """
    Generates LLM answer to the question using unified adapter path.

    This stage uses a single code path for ALL interfaces including manual:
    1. Computes question hash (for manual trace lookup, ignored by other adapters)
    2. Gets adapter via unified factory (get_agent)
    3. Constructs prompt messages (system prompt + few-shot examples)
    4. Invokes adapter with AgentConfig containing question_hash
    5. Handles recursion limit errors gracefully
    6. Extracts raw response text from AgentResult

    Requires:
        - "Answer": Validated Answer class (to check if we should proceed)

    Produces:
        - "raw_llm_response": Raw text response from LLM
        - "recursion_limit_reached": Whether agent hit recursion limit (bool)
        - "answering_model_str": Model string for result (provider/model)
        - "answering_mcp_servers": List of MCP server names (if using MCP)

    Error Handling:
        If adapter call fails with non-recoverable error, marks context.error
        and sets completed_without_errors=False. Recursion limit errors
        are handled gracefully and allow pipeline to continue.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "GenerateAnswer"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        # Answer artifact is optional - not needed in rubric_only mode
        return []

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.RAW_LLM_RESPONSE,
            ArtifactKeys.RECURSION_LIMIT_REACHED,
            ArtifactKeys.ANSWERING_MODEL_STR,
            ArtifactKeys.ANSWERING_MCP_SERVERS,
            ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run if either:
        - Answer class was successfully validated (template modes), OR
        - No Answer artifact exists (rubric_only mode - template validation skipped)

        Inherits error-checking from BaseVerificationStage.
        In rubric_only mode, Answer artifact won't exist but we still need to generate response.
        In template modes, Answer artifact should exist.
        """
        # Base class handles error checking via super().should_run()
        return super().should_run(context)

    def _resolve_workspace(self, context: VerificationContext) -> None:
        """Resolve and set the effective workspace for this question.

        Creates or copies the workspace directory as needed. Sets
        context.workspace_path and context.workspace_is_copy.

        When ``workspace_path`` is already set (e.g. by ScenarioManager
        which pre-creates per-turn directories), this method is a no-op.

        Args:
            context: Verification context with workspace config.

        Raises:
            RuntimeError: If a referenced source directory does not exist.
        """
        if context.workspace_path is not None:
            logger.debug("Workspace already set: %s", context.workspace_path)
            return

        if not context.agentic_parsing:
            return

        root = context.workspace_root
        if root is None:
            raise RuntimeError("workspace_root must be set when agentic_parsing is True")

        # Build unique suffix (replicate-safe for parallel execution)
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        suffix = f"run_{timestamp}_pid{os.getpid()}"
        if context.replicate is not None:
            suffix += f"_rep{context.replicate}"

        if context.question_workspace_path:
            # Question references a pre-existing directory
            source = root / context.question_workspace_path
            if not source.is_dir():
                raise RuntimeError(
                    f"Question workspace not found: {source} "
                    f"(workspace_root={root}, question_workspace_path={context.question_workspace_path})"
                )

            if context.workspace_copy:
                working = root / f"{context.question_workspace_path}_{suffix}"
                shutil.copytree(source, working)
                context.workspace_path = working
                context.workspace_is_copy = True
                logger.info("Copied workspace %s to %s", source, working)
            else:
                context.workspace_path = source
                context.workspace_is_copy = False
                logger.info("Using workspace in place: %s", source)
        else:
            # No pre-existing workspace; create empty directory
            working = root / f"{context.question_id}_{suffix}"
            working.mkdir(parents=True, exist_ok=True)
            context.workspace_path = working
            context.workspace_is_copy = True
            logger.info("Created workspace: %s", working)

    def execute(self, context: VerificationContext) -> None:
        """
        Generate answer using configured LLM or use cached answer.

        This stage checks if cached answer data is available in the context.
        If so, it injects the cached data and skips LLM invocation. This
        optimization allows multiple judges to share the same answering trace.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["raw_llm_response"]
            - Sets context.artifacts["recursion_limit_reached"]
            - Sets context.artifacts["answering_model_str"]
            - Sets context.artifacts["answering_mcp_servers"]
            - Sets context.error if LLM call fails fatally
        """
        # --- Replay short-circuit ---
        # Runs ahead of workspace resolution, cached_answer_data, and adapter
        # selection. On a hit we populate the same artifacts a live run would
        # and return early. On a fall-through miss we continue to the cache
        # path. On a strict miss we mark_error and return.
        try:
            replay_entry = _try_replay_hit(context)
        except ReplayMissError as exc:
            logger.info("Replay miss: %s (strict)", exc)
            context.mark_error(
                f"Replay strict miss: {exc}",
                category=ErrorCategory.PERMANENT,
            )
            return

        if replay_entry is not None:
            logger.info(
                "Replay hit: scenario=%s node=%s model=%s visit=%s",
                context.scenario_id,
                context.scenario_node,
                context.answering_model.id or context.answering_model.model_name,
                context.scenario_node_visit_index,
            )
            _populate_artifacts_from_replay_entry(context, replay_entry)
            return

        # Resolve workspace for agentic parsing (before any LLM calls)
        self._resolve_workspace(context)

        # Check if cached answer data is available
        if context.cached_answer_data is not None:
            logger.info(f"Using cached answer for question {context.question_id}")

            # Extract cached data
            raw_llm_response = context.cached_answer_data.get("raw_llm_response", "")
            recursion_limit_reached = context.cached_answer_data.get("recursion_limit_reached", False)
            answering_mcp_servers = context.cached_answer_data.get("answering_mcp_servers")
            usage_metadata = context.cached_answer_data.get("usage_metadata")
            agent_metrics = context.cached_answer_data.get("agent_metrics")

            # Build model string for result via ModelIdentity
            answering_model = context.answering_model
            answering_model_str = ModelIdentity.from_model_config(answering_model, role="answering").display_string

            # Store cached data in context (both artifact and result field)
            self.set_artifact_and_result(context, "raw_llm_response", raw_llm_response)
            self.set_artifact_and_result(context, "recursion_limit_reached", recursion_limit_reached)
            self.set_artifact_and_result(context, "answering_mcp_servers", answering_mcp_servers)
            context.set_artifact(ArtifactKeys.ANSWERING_MODEL_STR, answering_model_str)

            # Extract trace_messages from cached data (e.g. from TaskEval)
            trace_messages_data = context.cached_answer_data.get("trace_messages")
            if trace_messages_data:
                from karenina.ports.messages import Message as PortMessage

                if isinstance(trace_messages_data[0], dict):
                    trace_msgs = [PortMessage.from_dict(m) for m in trace_messages_data]
                else:
                    trace_msgs = trace_messages_data
                context.set_artifact(ArtifactKeys.TRACE_MESSAGES, trace_msgs)

            # Handle usage tracking for cached answers
            usage_tracker = UsageTracker()
            if usage_metadata:
                usage_tracker.track_call("answer_generation", answering_model_str, usage_metadata)
            if agent_metrics:
                usage_tracker.set_agent_metrics(agent_metrics)
            context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

            # Prefer cached conversation_context (symmetric to trace_messages
            # rehydration above). Fall back to reconstruction from
            # conversation_history only when the cache did not carry it.
            conversation_context_data = context.cached_answer_data.get("conversation_context")
            if conversation_context_data:
                from karenina.ports.messages import Message as PortMessage

                if isinstance(conversation_context_data[0], dict):
                    ctx_msgs = [PortMessage.from_dict(m) for m in conversation_context_data]
                else:
                    ctx_msgs = conversation_context_data
                context.set_artifact(ArtifactKeys.CONVERSATION_CONTEXT, ctx_msgs)
            else:
                # Reconstruct conversation context for cached results (scenario turns)
                conversation_history = context.get_artifact("conversation_history")
                if conversation_history:
                    context_messages = list(conversation_history)
                    context_messages.append(Message.user(context.question_text))
                    context.set_artifact(ArtifactKeys.CONVERSATION_CONTEXT, context_messages)

            return  # Skip LLM invocation

        # No cached answer - proceed with normal answer generation
        answering_model = context.answering_model

        # Build model string for result via ModelIdentity
        answering_model_str = ModelIdentity.from_model_config(answering_model, role="answering").display_string
        context.set_artifact(ArtifactKeys.ANSWERING_MODEL_STR, answering_model_str)

        # Extract MCP server names if configured
        answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None
        context.set_artifact(ArtifactKeys.ANSWERING_MCP_SERVERS, answering_mcp_servers)

        # Log MCP configuration if present
        if answering_mcp_servers:
            logger.info(f"Answering model MCP servers: {answering_mcp_servers}")

        # Step 1: Compute question hash (needed for manual, ignored by other adapters)
        question_hash = hashlib.md5(context.question_text.encode("utf-8")).hexdigest()
        if answering_model.interface == "manual":
            logger.info(f"Manual interface: Using MD5 hash '{question_hash}' from question text for trace lookup")

        # Step 2: Determine whether to use AgentPort or LLMPort.
        # Use AgentPort when MCP servers are configured, OR when the adapter
        # is a deep_agent (e.g. Claude Code), OR when the interface is manual
        # (ManualLLMAdapter raises ManualInterfaceError; manual must use
        # ManualAgentAdapter via AgentPort). Deep agent runtimes execute tools
        # internally; the LLMPort path would lose the tool call trace.
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec(answering_model.interface)
        use_agent = (
            bool(answering_model.mcp_urls_dict)
            or (spec is not None and spec.agent_tier == "deep_agent")
            or answering_model.interface == "manual"
        )
        answering_agent: AgentPort | None = None
        answering_llm: LLMPort | None = None

        try:
            if use_agent:
                # Use AgentPort for MCP-enabled models
                answering_agent = get_agent(answering_model, auto_fallback=True)
                logger.info(f"Using AgentPort ({answering_model.interface}) for {answering_model_str} with MCP")
            else:
                # Use LLMPort for simple LLM calls without tools
                answering_llm = get_llm(answering_model, auto_fallback=True)
                logger.info(f"Using LLMPort ({answering_model.interface}) for {answering_model_str} (no MCP)")
        except Exception as e:
            error_msg = f"Failed to initialize answering model: {type(e).__name__}: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg, category=context.error_registry.classify(e))
            return

        # Step 3: Construct prompt text
        constructed_prompt = _construct_few_shot_prompt(
            context.question_text, context.few_shot_examples, context.few_shot_enabled
        )

        # Step 4: Invoke adapter
        recursion_limit_reached = False
        raw_llm_response = ""

        # Initialize usage tracker for this stage
        usage_tracker = UsageTracker()

        try:
            # Construct messages in unified Message format
            adapter_messages: list[Message] = []

            # Inject conversation history from prior scenario turns (if present)
            conversation_history = context.get_artifact("conversation_history")

            # Build system prompt. Skip the model's system_prompt if
            # conversation_history already carries a system message (same-agent
            # continuation). But ALWAYS apply prompt_config generation
            # instructions, since those are pipeline-level and not part of
            # the conversation history.
            has_system_in_history = conversation_history and any(m.role == Role.SYSTEM for m in conversation_history)
            system_parts: list[str] = []
            if answering_model.system_prompt and not has_system_in_history:
                system_parts.append(answering_model.system_prompt)
            if context.prompt_config:
                gen_instructions = context.prompt_config.get_for_task("generation")
                if gen_instructions:
                    system_parts.append(gen_instructions)
            if system_parts:
                adapter_messages.append(Message.system("\n\n".join(system_parts)))

            if conversation_history:
                adapter_messages.extend(conversation_history)

            # Append workspace location to the user prompt so the agent
            # knows where its files are instead of searching the filesystem.
            user_prompt = constructed_prompt
            if context.workspace_path:
                user_prompt += (
                    f"\n\nWorkspace directory: {context.workspace_path}\n"
                    "All input data files are in this directory. "
                    "Save any output files here as well."
                )
            adapter_messages.append(Message.user(user_prompt))

            # Store the full conversation input for curation trace display
            context.set_artifact(
                ArtifactKeys.CONVERSATION_CONTEXT,
                list(adapter_messages),
            )

            if use_agent and answering_agent is not None:
                # AgentPort path: Use for MCP-enabled models with tool calling
                # Build MCP server config
                mcp_servers: dict[str, Any] = {}
                if answering_model.mcp_urls_dict:
                    for name, url in answering_model.mcp_urls_dict.items():
                        mcp_servers[name] = {"type": "http", "url": url}

                # Build agent config with question_hash (used by manual, ignored by others)
                agent_config = AgentConfig(
                    max_turns=answering_model.agent_middleware.limits.model_call_limit
                    if answering_model.agent_middleware
                    else 25,
                    timeout=answering_model.agent_timeout or 180,
                    question_hash=question_hash,
                    workspace_path=context.workspace_path,
                )

                # Run the agent
                result = answering_agent.run(
                    messages=adapter_messages,
                    mcp_servers=mcp_servers,
                    config=agent_config,
                )

                # Extract results from AgentResult
                raw_llm_response = result.raw_trace
                recursion_limit_reached = result.limit_reached

                # Handle agent timeout with partial trace
                if result.timeout_reached:
                    if result.raw_trace:
                        self.set_artifact_and_result(context, "response_timeout_partial", True)
                        self.set_artifact_and_result(context, ArtifactKeys.USAGE_UNAVAILABLE, True)
                        error_msg = (
                            f"Agent timed out with partial trace ({len(result.raw_trace)} chars, {result.turns} turns)"
                        )
                        context.mark_error(error_msg, category=ErrorCategory.TIMEOUT)
                        logger.warning("Question %s: %s", context.question_id, error_msg)
                    else:
                        error_msg = "Agent timed out with no trace messages"
                        context.mark_error(error_msg, category=ErrorCategory.TIMEOUT)
                        context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "")
                        context.set_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False)
                        return

                # Track usage metadata from adapter
                if result.usage:
                    inner_usage: dict[str, int | float] = {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                    if result.usage.cost_usd is not None:
                        inner_usage["cost_usd"] = result.usage.cost_usd
                    if result.usage.cache_read_tokens is not None:
                        inner_usage["cache_read_input_tokens"] = result.usage.cache_read_tokens
                    if result.usage.cache_creation_tokens is not None:
                        inner_usage["cache_creation_input_tokens"] = result.usage.cache_creation_tokens

                    usage_metadata = {answering_model_str: inner_usage}
                    usage_tracker.track_call("answer_generation", answering_model_str, usage_metadata)

                # Track agent metrics — extract full tool metrics from trace
                if result.trace_messages:
                    agent_metrics = extract_agent_metrics_from_messages(result.trace_messages)
                    agent_metrics["limit_reached"] = result.limit_reached
                else:
                    agent_metrics = {
                        "iterations": result.turns,
                        "limit_reached": result.limit_reached,
                    }
                usage_tracker.set_agent_metrics(agent_metrics)

                # Store trace_messages for future use (PR5a)
                if result.trace_messages:
                    context.set_artifact(ArtifactKeys.TRACE_MESSAGES, result.trace_messages)

            else:
                # LLMPort path: Use for simple LLM calls without tools
                assert answering_llm is not None

                # Use streaming when available to capture partial output on timeout.
                # The adapter handles retries internally via RetryExecutor.
                # StreamingTimeoutError propagates to the outer exception handler.
                if answering_llm.capabilities.supports_streaming:
                    llm_response = answering_llm.stream_invoke(
                        adapter_messages,
                        timeout=context.answering_model.request_timeout,
                    )
                else:
                    llm_response = answering_llm.invoke(adapter_messages)

                # Format response as trace (AI message only, question is not part of the trace)
                raw_llm_response = f"--- AI Message ---\n{llm_response.content}"

                # Mark partial response if streaming timed out with some content
                if llm_response.is_partial:
                    self.set_artifact_and_result(context, "response_timeout_partial", True)
                    error_msg = f"Response truncated by streaming timeout ({len(llm_response.content)} chars captured)"
                    context.mark_error(error_msg, category=ErrorCategory.TIMEOUT)
                    logger.warning("Question %s: %s", context.question_id, error_msg)

                # Propagate usage_unavailable flag if present
                if llm_response.usage_unavailable:
                    self.set_artifact_and_result(context, ArtifactKeys.USAGE_UNAVAILABLE, True)

                # Build trace_messages for the LLM path too
                llm_trace_messages = [
                    Message.assistant(llm_response.content),
                ]
                context.set_artifact(ArtifactKeys.TRACE_MESSAGES, llm_trace_messages)

                # Track usage metadata
                if llm_response.usage:
                    inner_usage = {
                        "input_tokens": llm_response.usage.input_tokens,
                        "output_tokens": llm_response.usage.output_tokens,
                        "total_tokens": llm_response.usage.total_tokens,
                    }
                    if llm_response.usage.cost_usd is not None:
                        inner_usage["cost_usd"] = llm_response.usage.cost_usd

                    usage_metadata = {answering_model_str: inner_usage}
                    usage_tracker.track_call("answer_generation", answering_model_str, usage_metadata)

                # No agent metrics for simple LLM calls
                usage_tracker.set_agent_metrics({"iterations": 1, "limit_reached": False})

        except Exception as e:
            # Adapters handle recursion limits internally and return AgentResult with
            # limit_reached=True. If we reach here, it's an unexpected error.
            # Log detailed error information for debugging
            error_details = traceback.format_exc()
            logger.error(
                f"Adapter call failed for question {context.question_id}\n"
                f"Exception type: {type(e).__name__}\n"
                f"Exception message: {e}\n"
                f"Full traceback:\n{error_details}"
            )

            # StreamingTimeoutError with partial content: salvage what we have
            # instead of discarding it. The pipeline can continue with truncated
            # output (downstream stages handle partial responses).
            from karenina.exceptions import StreamingTimeoutError

            if isinstance(e, StreamingTimeoutError) and e.partial_content:
                raw_llm_response = f"--- AI Message ---\n{e.partial_content}"
                self.set_artifact_and_result(context, "raw_llm_response", raw_llm_response)
                self.set_artifact_and_result(context, "response_timeout_partial", True)
                self.set_artifact_and_result(context, "recursion_limit_reached", False)
                context.add_warning(
                    f"Response truncated by streaming timeout after retries ({len(e.partial_content)} chars captured)"
                )
                return

            # Mark error (category classification determines if scenario retries)
            error_msg = f"Adapter call failed: {type(e).__name__}: {e}"
            context.mark_error(error_msg, category=context.error_registry.classify(e))
            context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "")
            context.set_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False)
            return

        # Store results (both artifact and result field)
        self.set_artifact_and_result(context, "raw_llm_response", raw_llm_response)
        self.set_artifact_and_result(context, "recursion_limit_reached", recursion_limit_reached)
        self.set_artifact_and_result(context, "answering_mcp_servers", answering_mcp_servers)

        # Store usage tracker for next stages to continue tracking (artifact only)
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)
