"""GLM-5.1 failure-burden rubric and trace preparation."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field, StrictInt

from karenina.benchmark import ModelConfig, format_trace_messages
from karenina.schemas.entities import AgenticRubricTrait, Rubric
from karenina.schemas.verification import VerificationResult
from paper.bixbench_harness_comparison.config import failure_burden_model

TRAIT_NAME = "trace_failure_burdens_v5"
BURDEN_FIELDS = (
    "environment_failure_burden",
    "data_ingestion_failure_burden",
    "tool_api_failure_burden",
    "analysis_code_failure_burden",
    "repetition_churn_burden",
    "incompletion_burden",
)


class TraceFailureBurdens(BaseModel):
    """Six independent ordinal severities and their trace evidence."""

    environment_failure_burden: StrictInt = Field(
        ge=0,
        le=3,
        description=(
            "Severity of missing tools, failed installs, runtime/path, permission, "
            "or filesystem friction."
        ),
    )
    data_ingestion_failure_burden: StrictInt = Field(
        ge=0,
        le=3,
        description=(
            "Severity of data locating, loading, parsing, schema, cleaning, or "
            "alignment friction."
        ),
    )
    tool_api_failure_burden: StrictInt = Field(
        ge=0,
        le=3,
        description=(
            "Severity of tool/library/CLI/API usage confusion or interface "
            "trial-and-error."
        ),
    )
    analysis_code_failure_burden: StrictInt = Field(
        ge=0,
        le=3,
        description=(
            "Severity of bugs or invalid results in analysis/statistical/modeling code."
        ),
    )
    repetition_churn_burden: StrictInt = Field(
        ge=0,
        le=3,
        description=(
            "Severity of loops, redundant retries, or repeated actions that do not "
            "add new information."
        ),
    )
    incompletion_burden: StrictInt = Field(
        ge=0,
        le=3,
        description=(
            "Severity of timeout, missing final answer, missing requested artifact, "
            "or partial completion."
        ),
    )
    evidence: str = Field(
        description="Short trace-grounded evidence for nonzero burdens.",
        default="",
    )


RUBRIC_DESCRIPTION = r"""
Inspect the supplied agent answering trace and score six independent
failure-burden severities. This v5 taxonomy is failure-centered: it is meant to
explain robustness costs, not to estimate how much time the agent spent in each
normal work phase.

The final structured output must contain only these seven fields:

{
  "environment_failure_burden": <0-3 integer>,
  "data_ingestion_failure_burden": <0-3 integer>,
  "tool_api_failure_burden": <0-3 integer>,
  "analysis_code_failure_burden": <0-3 integer>,
  "repetition_churn_burden": <0-3 integer>,
  "incompletion_burden": <0-3 integer>,
  "evidence": "<short concrete evidence from the trace>"
}

Do not emit derived fields such as dominant burden, trace format, section counts,
model, harness, replicate, or extraction method. Those are handled downstream.

=================================================================
GLOBAL SCORING RULES
=================================================================
Score burden severity, not section fraction. The scores are independent ordinal
levels and do not sum to 1. Use only the integer levels 0, 1, 2, and 3:

- 0: absent or negligible
- 1: minor friction, quickly resolved, little effect on task trajectory
- 2: material slowdown or repeated localized recovery, but substantive progress remains
- 3: severe/dominant/blocking burden, causing timeout, no useful result, or consuming most of the trace

Inspect the raw answering trace, not any previous annotation. Deterministic
anchors may be supplied before the raw trace; use them as hints, but verify them
against the trace. Do not penalize a trace merely for exposing explicit thinking or planning sections. Do not treat normal first-pass exploration as failure. If a burden is nonzero, cite concrete trace evidence in the evidence field. Cite evidence such as error messages, repeated commands, failed installs, API misuse, parse/schema mismatches, timeout markers, lack of final answer, or similar.

If the same episode has multiple independent failure causes, score each relevant
burden. Example: a failed import fixed by installing a package is environment;
a wrong function call after the package imports is tool/API; a traceback in a
custom analysis script is analysis-code.

=================================================================
environment_failure_burden
=================================================================
Measures friction from making the execution environment usable.

Score this when the trace shows missing tools, missing packages, broken installs,
bad paths, runtime availability problems, permissions, or filesystem constraints.

Include:
- Missing system tools or binaries: R, Rscript, python, unzip, wget, git, mafft,
  hmmer, busco, CLI tools.
- Failed package/dependency setup: failed pip, conda, mamba, apt, source builds,
  missing wheels, incompatible versions.
- Runtime/path problems: bad PYTHONPATH, wrong interpreter, virtualenv issues,
  package installed but not importable.
- Filesystem constraints: read-only filesystem, permission denied, unwritable
  install paths.
- Repeated checks after an environment error: which, --version, import checks,
  path inspection, environment-variable fixes.
- Workarounds whose purpose is to make the runtime usable.

Exclude:
- Clean first-pass workspace listing or file discovery.
- Reading docs/source to learn an installed tool's API; that is
  tool_api_failure_burden.
- Fixing analysis script logic after the runtime works; that is
  analysis_code_failure_burden.

Anchors:
- 0: no visible environment friction.
- 1: one missing package/tool fixed immediately.
- 2: several setup retries, but task proceeds.
- 3: environment blocks meaningful analysis, dominates the trace, or consumes the run until timeout.

=================================================================
data_ingestion_failure_burden
=================================================================
Measures friction from finding, opening, parsing, aligning, or cleaning task data.

Score this when the agent struggles with input data structure or data preparation
before analysis.

Include:
- File-format problems: delimiter errors, encoding issues, compressed archives,
  Excel sheet confusion, malformed CSV/TSV, RDS/HDF5/binary loading problems.
- Schema problems: wrong column names, missing fields, row/column orientation
  errors.
- Sample/data alignment problems: mismatched sample IDs, metadata/count matrix
  mismatch, inconsistent identifiers.
- Data loading/coercion failures: dtype errors, empty data due to wrong loader
  assumptions, parse errors.
- Repeated data exploration because earlier assumptions about the input were
  wrong.
- Fixes to preprocessing logic where the central issue is understanding or
  aligning the input data.

Exclude:
- Normal successful first-pass inspection of files/columns.
- Statistical/modeling bugs after data are loaded; that is
  analysis_code_failure_burden.
- Learning a library API used to read the data; that is tool_api_failure_burden.

Anchors:
- 0: data are read and understood cleanly.
- 1: one small data-format correction.
- 2: multiple parse/schema corrections, but analysis proceeds.
- 3: data cannot be loaded/aligned enough for meaningful analysis, or data ingestion dominates the trace.

=================================================================
tool_api_failure_burden
=================================================================
Measures friction from not knowing how to use a tool, package, CLI, library, or
API correctly.

Score this when the trace shows tool-interface trial and error, API discovery,
docs/source probing, invalid flags, or wrong function usage.

Include:
- Reading docs, source, README, --help, signatures, dir(), help(), package
  internals because usage is unclear or prior usage failed.
- CLI/API misuse: wrong flag, wrong command, wrong function, wrong argument
  shape, invalid mode, deprecated parameter.
- Switching tools because the initial API could not be used correctly.
- Trial-and-error learning of a package interface after errors.
- Fixes where the main issue is "how do I call this tool correctly?"

Exclude:
- Installing the tool or making it importable; that is environment_failure_burden.
- Data schema/format misunderstanding; that is data_ingestion_failure_burden.
- Bugs in custom analysis code once the API is known; that is
  analysis_code_failure_burden.

Anchors:
- 0: no visible tool/API confusion.
- 1: one minor help lookup or API correction.
- 2: repeated API corrections but progress continues.
- 3: API/tool confusion prevents completion, causes a dead end, or dominates the trace.

=================================================================
analysis_code_failure_burden
=================================================================
Measures friction from bugs or invalid results in analysis/statistical/modeling
code.

Score this when the trace shows the agent writing, editing, and rerunning
analysis code because the code or computation fails.

Include:
- Script failures: syntax errors, tracebacks, serialization errors, wrong imports
  inside analysis code.
- Modeling/statistical failures: invalid coefficient names, wrong contrast,
  formula errors, NaNs, empty model output, failed enrichment/model fitting.
- Logic errors detected after running analysis: wrong grouping, wrong direction,
  bad filters, invalid result requiring code revision.
- Edit-rerun-debug loops where the main object being fixed is analysis code.
- Verification that reveals computed results are wrong and require code changes.

Exclude:
- Package absence or path/runtime failures; that is environment_failure_burden.
- Input data mismatch/loading failures; that is data_ingestion_failure_burden.
- Learning how to call a library before analysis can be written; that is
  tool_api_failure_burden.

Anchors:
- 0: analysis code runs cleanly or no code is needed.
- 1: one small script/model fix.
- 2: several corrections but final result is reached.
- 3: analysis code never reaches a usable result, or analysis debugging dominates with only partial/fragile completion.

=================================================================
repetition_churn_burden
=================================================================
Measures nonproductive repetition: repeated actions that do not add new
information or advance the task.

Score this when the trace shows looping, repeated commands, redundant checks, or
retry behavior without meaningful strategy change.

Include:
- Identical or near-identical commands repeated many times.
- Retry loops with no changed hypothesis, command, data, or code.
- Repeated file listings/reads/checks after the relevant information is already
  known.
- Cycling between failed approaches without incorporating new evidence.
- Long stretches where commands succeed but the agent makes no progress.
- Stuck behavior continuing until timeout.

Exclude:
- Legitimate iterative debugging where each retry changes the code or hypothesis
  meaningfully; score under the relevant failure burden.
- Purposeful repeated analysis across different files/samples.
- Limited verification after saving outputs.

Anchors:
- 0: no obvious nonproductive repetition.
- 1: minor redundant checks.
- 2: noticeable repeated work, but task still progresses.
- 3: trace is dominated by a loop or repeated unproductive action.

=================================================================
incompletion_burden
=================================================================
Measures failure to reach a complete benchmark-relevant endpoint.

Score this when the trace does not produce a complete answer, complete requested
artifact, or complete substantive analysis.

Include:
- Timeout before final answer.
- Trace cut off mid-command, mid-analysis, or mid-saving.
- No final answer, no interpretation, no saved artifact when required.
- Final response explicitly says the task could not be completed.
- Agent only performs setup/exploration and never reaches requested analysis.
- Structurally unusable or empty answer caused by the answerer run.
- Partial answer missing important requested components.

Exclude:
- A completed but scientifically wrong answer; correctness is evaluated elsewhere.
- This v5 judge/extraction failure; that belongs in extraction metadata.
- Minor missing polish when the benchmark answer is otherwise complete.

Anchors:
- 0: complete answer/artifact state reached.
- 1: mostly complete, minor missing verification/finalization.
- 2: partial result with important requested component missing.
- 3: no meaningful task result, timeout before core analysis, substantial work without a complete benchmark answer, or structurally
  unusable answer.
"""


def repeated_command_fingerprints(
    trace_text: str,
    *,
    min_count: int = 3,
) -> list[str]:
    """Return repeated command-like lines as deterministic churn anchors."""

    candidates: list[str] = []
    for raw_line in trace_text.splitlines():
        line = raw_line.strip()
        if len(line) >= 12 and any(
            token in line for token in ("command", "execute", "Bash", "python", "Rscript")
        ):
            candidates.append(re.sub(r"\s+", " ", line)[:240])
    counts = Counter(candidates)
    return [f"{count}x {line}" for line, count in counts.most_common(10) if count >= min_count]


def build_trace_anchors(trace_text: str) -> dict[str, Any]:
    """Compute the deterministic calibration anchors."""

    lower = trace_text.lower()
    has_thinking = "--- Thinking ---" in trace_text
    has_tool_message = "--- Tool Message" in trace_text
    has_tool_result = "--- Tool Result" in trace_text
    if has_thinking and not has_tool_message:
        marker_style = "explicit_thinking_sections"
    elif has_tool_message and not has_thinking:
        marker_style = "tool_message_sections"
    elif has_thinking or has_tool_message or has_tool_result:
        marker_style = "mixed_markers"
    else:
        marker_style = "unknown"
    return {
        "marker_style": marker_style,
        "ai_message_sections": trace_text.count("--- AI Message ---"),
        "thinking_sections": trace_text.count("--- Thinking ---"),
        "tool_message_sections": trace_text.count("--- Tool Message"),
        "tool_result_sections": trace_text.count("--- Tool Result"),
        "timeout_mentions": lower.count("timeout") + lower.count("timed out"),
        "traceback_mentions": lower.count("traceback"),
        "error_mentions": lower.count("error"),
        "permission_or_readonly_mentions": lower.count("permission denied")
        + lower.count("read-only")
        + lower.count("readonly"),
        "module_not_found_mentions": lower.count("modulenotfounderror"),
        "repeated_command_fingerprints": repeated_command_fingerprints(trace_text),
    }


def trace_text(result: VerificationResult, max_trace_chars: int = 300_000) -> str:
    """Render and calibrate one stored answer trace for the burden judge."""

    if result.template is None:
        raw_trace = ""
    elif result.template.trace_messages:
        raw_trace = format_trace_messages(result.template.trace_messages)
    else:
        raw_trace = result.template.raw_llm_response
    anchors = build_trace_anchors(raw_trace)
    if max_trace_chars > 0 and len(raw_trace) > max_trace_chars:
        head_chars = (max_trace_chars * 6) // 10
        tail_chars = max_trace_chars - head_chars
        dropped = len(raw_trace) - max_trace_chars
        raw_trace = (
            raw_trace[:head_chars]
            + f"\n\n... [TRUNCATED {dropped} chars; anchors cover the full trace] ...\n\n"
            + raw_trace[-tail_chars:]
        )
    return (
        "=== Deterministic trace anchors for calibration (verify against raw trace) ===\n"
        f"{json.dumps(anchors, indent=2)}\n\n"
        "=== Raw BixBench answering trace ===\n"
        f"{raw_trace}"
    )


def failure_burden_rubric(
    timeout: int = 600,
    *,
    model: ModelConfig | None = None,
) -> Rubric:
    """Build the six-axis agentic failure-burden rubric."""

    model = model or failure_burden_model(timeout)
    return Rubric(
        agentic_traits=[
            AgenticRubricTrait(
                name=TRAIT_NAME,
                summary=None,
                description=RUBRIC_DESCRIPTION,
                kind=TraceFailureBurdens,
                higher_is_better=None,
                min_score=None,
                max_score=None,
                classes=None,
                context_mode="trace_only",
                materialize_trace=False,
                persist_trace=False,
                max_turns=8,
                timeout_seconds=timeout,
                model_override=model,
            )
        ]
    )


def reconstruct_burdens(scores: dict[str, object]) -> TraceFailureBurdens:
    """Validate flattened TaskEval scores as one burden judgment."""

    prefix = f"{TRAIT_NAME}."
    values = {
        key.removeprefix(prefix): value
        for key, value in scores.items()
        if key.startswith(prefix)
    }
    if not values:
        raise ValueError("Failure-burden judge returned no verdict")
    return TraceFailureBurdens.model_validate(values)
