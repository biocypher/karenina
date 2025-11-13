"""Data models for TaskEval."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# Import VerificationResult for use in StepEval
from ...schemas.workflow import VerificationResult


class LogEvent(BaseModel):
    """Single log event in TaskEval."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    level: Literal["debug", "info", "warn", "error"]
    text: str
    tags: list[str] | None = None
    payload: dict[str, Any] | None = None
    # New fields for agent output logging
    question_id: str | None = Field(default=None, description="Question this log answers")
    is_agent_output: bool = Field(default=False, description="Whether this is agent output to be evaluated")
    output_type: str | None = Field(default=None, description="Type of output: answer, reasoning, analysis, etc.")
    # Dict trace support
    is_dict_structured: bool = Field(default=False, description="Whether this log is from a dict trace")
    dict_keys: list[str] | None = Field(default=None, description="Keys from dict trace for quick access")


class StepEval(BaseModel):
    """Evaluation results for a single step or global evaluation."""

    verification_results: dict[str, list[VerificationResult]] = Field(
        default_factory=dict,
        description="Full verification results per question: {question_id: [VerificationResult, ...]}",
    )

    def format_rubric_scores(self, indent: str = "  ") -> str:
        """Format verification results including rubric scores."""
        return self.format_verification_results(indent)

    def format_verification_results(self, indent: str = "  ") -> str:
        """Format verification results using VerificationResult data."""
        if not self.verification_results:
            return f"{indent}No verification results"

        lines = []
        for question_id, results in self.verification_results.items():
            lines.append(f"{indent}Question: {question_id}")

            for i, result in enumerate(results):
                result_num = f"[{i + 1}]" if len(results) > 1 else ""

                # Verification status
                if result.verify_result:
                    status = "✓ PASSED"
                elif result.verify_result is False:
                    status = "✗ FAILED"
                else:
                    status = "⚠ NO RESULT"
                lines.append(f"{indent}  {result_num} Status: {status}")

                # Show full output (no truncation)
                output = result.raw_llm_response
                lines.append(f'{indent}  {result_num} Output: "{output}"')

                # Show LLM and manual rubric traits
                if result.verify_rubric:
                    llm_manual_traits = []
                    for k, v in result.verify_rubric.items():
                        if isinstance(v, bool):
                            llm_manual_traits.append(f"{k}={'✓' if v else '✗'}")
                        else:
                            llm_manual_traits.append(f"{k}={v}")
                    lines.append(f"{indent}  {result_num} Rubric: {', '.join(llm_manual_traits)}")

                # Show metric traits separately with confusion matrix and metrics
                if result.metric_trait_confusion_lists:
                    for trait_name, confusion in result.metric_trait_confusion_lists.items():
                        counts = []
                        for bucket in ["tp", "fp", "fn", "tn"]:
                            if bucket in confusion:
                                counts.append(f"{bucket.upper()}={len(confusion[bucket])}")
                        lines.append(f"{indent}  {result_num} Metric [{trait_name}]: {', '.join(counts)}")

                        # Show computed metrics
                        if result.metric_trait_metrics and trait_name in result.metric_trait_metrics:
                            metrics = result.metric_trait_metrics[trait_name]
                            metric_strs = [f"{k}={v:.3f}" for k, v in metrics.items()]
                            lines.append(f"{indent}      {result_num} Metrics: {', '.join(metric_strs)}")

                # Show special verification features
                if result.abstention_detected:
                    lines.append(f"{indent}  {result_num} ⚠ Abstention detected")
                if result.embedding_override_applied:
                    lines.append(
                        f"{indent}  {result_num} ✓ Embedding check overrode failure (similarity: {result.embedding_similarity_score:.3f})"
                    )

                # Show error if present
                if result.error:
                    lines.append(f"{indent}  {result_num} Error: {result.error}")

                if i < len(results) - 1:
                    lines.append("")  # Separator between multiple results

            lines.append("")  # Separator between questions

        return "\n".join(lines).rstrip()

    def get_summary_stats(self) -> dict[str, Any]:
        """
        Get summary statistics for this evaluation.

        Distinguishes between template verification (structural validation)
        and rubric evaluation (qualitative assessment).

        Returns:
            dict: Summary statistics with keys:
                - traces_total: Number of traces evaluated
                - traces_passed: Number of traces with at least one passing result
                - results_total: Total number of verification results (across all replicates)
                - template_verification_total: Number of results with template verification
                - template_verification_passed: Number passing template verification
                - rubric_traits_total: Total rubric traits evaluated
                - rubric_traits_passed: Number of rubric traits passing
                - success_rate: Percentage of traces passing (0-100)
        """
        total_traces = len(self.verification_results)
        passed_traces = 0
        total_results = 0

        # Template verification stats
        template_verification_total = 0
        template_verification_passed = 0

        # Rubric evaluation stats
        rubric_traits_total = 0
        rubric_traits_passed = 0

        for _trace_id, results in self.verification_results.items():
            total_results += len(results)
            if any(result.verify_result for result in results):
                passed_traces += 1

            # Count template verification and rubric traits separately
            for result in results:
                # Template verification (structural validation)
                if result.template_verification_performed:
                    template_verification_total += 1
                    if result.verify_result:
                        template_verification_passed += 1

                # Rubric evaluation (qualitative assessment)
                if result.rubric_evaluation_performed:
                    # Count LLM and manual rubric traits
                    if result.verify_rubric:
                        for score in result.verify_rubric.values():
                            rubric_traits_total += 1
                            if score is True or (isinstance(score, int) and score > 0):
                                rubric_traits_passed += 1

                    # Count metric traits
                    if result.metric_trait_metrics:
                        rubric_traits_total += len(result.metric_trait_metrics)
                        rubric_traits_passed += len(
                            result.metric_trait_metrics
                        )  # All computed metrics count as "passed"

        return {
            "traces_total": total_traces,
            "traces_passed": passed_traces,
            "results_total": total_results,
            "template_verification_total": template_verification_total,
            "template_verification_passed": template_verification_passed,
            "rubric_traits_total": rubric_traits_total,
            "rubric_traits_passed": rubric_traits_passed,
            "success_rate": (passed_traces / total_traces * 100) if total_traces > 0 else 0,
        }

    def aggregate_rubric_results(self) -> dict[str, Any]:
        """
        Aggregate rubric results across replicates for all traces.

        Aggregation rules:
        - LLM traits (scores): Averaged across successful replicates
        - Manual traits (booleans): Pass rate (0.0 to 1.0)
        - Metric traits: Metrics averaged, confusion matrices omitted
        - Single replicate: Returned as-is without modification
        - Failed replicates: Excluded from aggregation, count tracked

        Returns:
            dict: Mapping trace_id to aggregated results:
                {
                    "trace_id": {
                        "llm": {"clarity": 4.5, "analysis_quality": 3.2},
                        "manual": {"has_citation": 0.75},  # 75% pass rate
                        "metric": {
                            "entity_extraction": {
                                "metrics": {"precision": 0.85, "recall": 0.92}
                            }
                        },
                        "failed_replicate_count": 1  # Only if > 0
                    },
                    ...
                }

        Example:
            >>> step_eval = StepEval()
            >>> step_eval.verification_results = {
            ...     "trace_1": [result1, result2, result3]
            ... }
            >>> aggregated = step_eval.aggregate_rubric_results()
            >>> aggregated["trace_1"]["llm"]["clarity"]  # Averaged score
            4.333
        """
        aggregated = {}

        for trace_id, results in self.verification_results.items():
            if not results:
                continue

            # Aggregate replicates for this trace
            aggregated[trace_id] = self._aggregate_trace_replicates(results)

        return aggregated

    def _aggregate_trace_replicates(self, results: list[VerificationResult]) -> dict[str, Any]:
        """
        Aggregate replicates for a single trace.

        Args:
            results: List of VerificationResult objects (one per replicate)

        Returns:
            dict: Aggregated rubric results with failed replicate count
        """
        # Filter out failed replicates
        successful_results = [r for r in results if r.completed_without_errors]
        failed_count = len(results) - len(successful_results)

        if not successful_results:
            # All replicates failed - return empty result with failure count
            return {"failed_replicate_count": failed_count} if failed_count > 0 else {}

        # Single replicate: return as-is without modification
        if len(successful_results) == 1 and failed_count == 0:
            return successful_results[0].rubric_results

        # Multiple replicates: aggregate by trait type
        aggregated: dict[str, Any] = {}

        llm_aggregated = self._aggregate_llm_traits(successful_results)
        if llm_aggregated:
            aggregated["llm"] = llm_aggregated

        manual_aggregated = self._aggregate_manual_traits(successful_results)
        if manual_aggregated:
            aggregated["manual"] = manual_aggregated

        metric_aggregated = self._aggregate_metric_traits(successful_results)
        if metric_aggregated:
            aggregated["metric"] = metric_aggregated

        # Add failed replicate count if any failed
        if failed_count > 0:
            aggregated["failed_replicate_count"] = failed_count

        return aggregated

    def _aggregate_llm_traits(self, results: list[VerificationResult]) -> dict[str, float]:
        """
        Average LLM trait scores across replicates.

        Args:
            results: List of successful VerificationResult objects

        Returns:
            dict: Trait names mapped to averaged scores
        """
        trait_scores: dict[str, list[int]] = {}

        for result in results:
            rubric_data = result.rubric_results
            if "llm" in rubric_data:
                for trait_name, score in rubric_data["llm"].items():
                    if trait_name not in trait_scores:
                        trait_scores[trait_name] = []
                    trait_scores[trait_name].append(score)

        # Average each trait
        return {trait_name: sum(scores) / len(scores) for trait_name, scores in trait_scores.items()}

    def _aggregate_manual_traits(self, results: list[VerificationResult]) -> dict[str, float]:
        """
        Calculate pass rate for manual traits (booleans -> 0.0 to 1.0).

        Args:
            results: List of successful VerificationResult objects

        Returns:
            dict: Trait names mapped to pass rates (0.0 to 1.0)
        """
        trait_values: dict[str, list[bool]] = {}

        for result in results:
            rubric_data = result.rubric_results
            if "manual" in rubric_data:
                for trait_name, value in rubric_data["manual"].items():
                    if trait_name not in trait_values:
                        trait_values[trait_name] = []
                    trait_values[trait_name].append(value)

        # Calculate pass rate
        return {trait_name: sum(values) / len(values) for trait_name, values in trait_values.items()}

    def _aggregate_metric_traits(self, results: list[VerificationResult]) -> dict[str, dict[str, Any]]:
        """
        Average metric trait metrics across replicates.

        Confusion matrices are omitted from aggregation as per design decision.

        Args:
            results: List of successful VerificationResult objects

        Returns:
            dict: Trait names mapped to aggregated metrics (confusion matrices omitted)
        """
        trait_metrics: dict[str, list[dict[str, float]]] = {}

        for result in results:
            rubric_data = result.rubric_results
            if "metric" in rubric_data:
                for trait_name, trait_data in rubric_data["metric"].items():
                    # Collect metrics
                    if "metrics" in trait_data:
                        if trait_name not in trait_metrics:
                            trait_metrics[trait_name] = []
                        trait_metrics[trait_name].append(trait_data["metrics"])

        # Average metrics for each trait
        aggregated = {}
        for trait_name, metrics_list in trait_metrics.items():
            aggregated[trait_name] = {"metrics": self._average_metrics(metrics_list)}

        return aggregated

    def _average_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """
        Average metrics across replicates.

        Args:
            metrics_list: List of metric dictionaries from different replicates

        Returns:
            dict: Averaged metrics
        """
        if not metrics_list:
            return {}

        # Collect all metric names
        all_metric_names: set[str] = set()
        for metrics in metrics_list:
            all_metric_names.update(metrics.keys())

        # Average each metric
        averaged = {}
        for metric_name in all_metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                averaged[metric_name] = sum(values) / len(values)

        return averaged


class TaskEvalResult(BaseModel):
    """Complete task evaluation results."""

    task_id: str | None = None
    metadata: dict[str, Any] | None = None
    global_eval: StepEval | None = None
    per_step: dict[str, StepEval] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    logs: dict[str, list[LogEvent]] = Field(default_factory=dict, description="Optional: include logs in results")

    def display(self) -> str:
        """Display a clean, formatted representation of the evaluation results."""
        lines = []

        # Header
        lines.append("═" * 80)
        lines.append("                        TASK EVALUATION RESULTS")
        lines.append("═" * 80)

        # Basic info
        if self.task_id:
            lines.append(f"Task ID: {self.task_id}")

        if self.metadata:
            for key, value in self.metadata.items():
                lines.append(f"{key.replace('_', ' ').title()}: {value}")

        # Timestamp
        if self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"Timestamp: {formatted_time}")
            except Exception:
                lines.append(f"Timestamp: {self.timestamp}")

        lines.append("")

        # Global evaluation
        if self.global_eval:
            lines.append("─" * 60)
            lines.append("GLOBAL EVALUATION")
            lines.append("─" * 60)

            if self.global_eval.verification_results:
                lines.append("Verification Results:")
                lines.append(self.global_eval.format_verification_results())
                lines.append("")

        # Step evaluations
        if self.per_step:
            for step_id, step_eval in self.per_step.items():
                lines.append("─" * 60)
                lines.append(f"STEP EVALUATION: {step_id}")
                lines.append("─" * 60)

                if step_eval.verification_results:
                    lines.append(f"Verification Results ({step_id}):")
                    lines.append(step_eval.format_verification_results())
                    lines.append("")

        # Summary
        lines.append("═" * 80)
        summary_line = self.summary()
        lines.append(f"SUMMARY: {summary_line}")
        lines.append("═" * 80)

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a concise summary of the evaluation results."""
        total_traces = 0
        passed_traces = 0
        total_template_verifications = 0
        passed_template_verifications = 0
        total_rubric_traits = 0
        passed_rubric_traits = 0

        # Global stats
        if self.global_eval:
            stats = self.global_eval.get_summary_stats()
            total_traces += stats["traces_total"]
            passed_traces += stats["traces_passed"]
            total_template_verifications += stats["template_verification_total"]
            passed_template_verifications += stats["template_verification_passed"]
            total_rubric_traits += stats["rubric_traits_total"]
            passed_rubric_traits += stats["rubric_traits_passed"]

        # Step stats
        for step_eval in self.per_step.values():
            stats = step_eval.get_summary_stats()
            total_traces += stats["traces_total"]
            passed_traces += stats["traces_passed"]
            total_template_verifications += stats["template_verification_total"]
            passed_template_verifications += stats["template_verification_passed"]
            total_rubric_traits += stats["rubric_traits_total"]
            passed_rubric_traits += stats["rubric_traits_passed"]

        parts = []
        if total_template_verifications > 0:
            parts.append(
                f"{passed_template_verifications}/{total_template_verifications} template verifications passed"
            )
        if total_rubric_traits > 0:
            parts.append(f"{passed_rubric_traits}/{total_rubric_traits} rubric traits passed")
        if not parts and total_traces > 0:
            parts.append(f"{passed_traces}/{total_traces} traces passed")

        return " | ".join(parts) if parts else "No evaluations performed"

    def summary_compact(self) -> str:
        """Return a very compact one-line summary."""
        if self.global_eval:
            stats = self.global_eval.get_summary_stats()
            parts = []
            if stats["template_verification_total"] > 0:
                parts.append(
                    f"{stats['template_verification_passed']}/{stats['template_verification_total']} template verifications"
                )
            if stats["rubric_traits_total"] > 0:
                parts.append(f"{stats['rubric_traits_passed']}/{stats['rubric_traits_total']} rubric traits")
            if not parts:
                parts.append(f"{stats['traces_passed']}/{stats['traces_total']} traces")

            return f"TaskEval [{self.task_id or 'Unknown'}]: {', '.join(parts)} ({stats['success_rate']:.0f}% success)"
        return f"TaskEval [{self.task_id or 'Unknown'}]: No evaluation data"

    def to_dict_clean(self) -> dict[str, Any]:
        """Return a simplified dictionary representation."""
        result = {"task_id": self.task_id, "metadata": self.metadata, "timestamp": self.timestamp, "summary": {}}

        # Add global summary
        if self.global_eval:
            global_stats = self.global_eval.get_summary_stats()
            result["global"] = global_stats
            if isinstance(result["summary"], dict):
                result["summary"].update(global_stats)

        # Add step summaries
        if self.per_step:
            result["steps"] = {}
            for step_id, step_eval in self.per_step.items():
                if isinstance(result["steps"], dict):
                    result["steps"][step_id] = step_eval.get_summary_stats()

        return result

    def export_json(self, include_logs: bool = False, indent: int = 2) -> str:
        """Export results as JSON string."""
        import json

        # Create a comprehensive dictionary representation
        result_dict = {
            "task_id": self.task_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "summary": self.summary(),
            "global_evaluation": None,
            "step_evaluations": {},
        }

        # Add global evaluation details
        if self.global_eval:
            result_dict["global_evaluation"] = {
                "verification_results": {
                    qid: [vr.model_dump() for vr in results]
                    for qid, results in self.global_eval.verification_results.items()
                },
                "summary_stats": self.global_eval.get_summary_stats(),
            }

        # Add step evaluations
        for step_id, step_eval in self.per_step.items():
            step_evaluations = result_dict["step_evaluations"]
            if isinstance(step_evaluations, dict):
                step_evaluations[step_id] = {
                    "verification_results": {
                        qid: [vr.model_dump() for vr in results]
                        for qid, results in step_eval.verification_results.items()
                    },
                    "summary_stats": step_eval.get_summary_stats(),
                }

        # Optionally include logs
        if include_logs and self.logs:
            result_dict["logs"] = {
                step_id: [log.model_dump() for log in log_list] for step_id, log_list in self.logs.items()
            }

        return json.dumps(result_dict, indent=indent, default=str)

    def export_markdown(self) -> str:
        """Export results as markdown format."""
        lines = []

        lines.append("# Task Evaluation Results")
        lines.append("")

        if self.task_id:
            lines.append(f"**Task ID:** {self.task_id}")
        if self.metadata:
            for key, value in self.metadata.items():
                lines.append(f"**{key.replace('_', ' ').title()}:** {value}")

        lines.append(f"**Status:** {self._get_status_emoji()}")
        lines.append("")

        # Global evaluation - show summary stats
        if self.global_eval and self.global_eval.verification_results:
            lines.append("## Verification Results")
            lines.append("")

            stats = self.global_eval.get_summary_stats()
            lines.append(f"- **Traces Passed**: {stats['traces_passed']}/{stats['traces_total']}")
            if stats["template_verification_total"] > 0:
                lines.append(
                    f"- **Template Verifications Passed**: {stats['template_verification_passed']}/{stats['template_verification_total']}"
                )
            if stats["rubric_traits_total"] > 0:
                lines.append(
                    f"- **Rubric Traits Passed**: {stats['rubric_traits_passed']}/{stats['rubric_traits_total']}"
                )
            lines.append(f"- **Success Rate**: {stats['success_rate']:.1f}%")
            lines.append("")

            # Show per-trace results
            for trace_id, results in self.global_eval.verification_results.items():
                lines.append(f"### Trace: {trace_id}")
                for i, result in enumerate(results):
                    status_emoji = "✅" if result.verify_result else "❌"
                    lines.append(f"- {status_emoji} Result {i + 1}: {'PASSED' if result.verify_result else 'FAILED'}")

                    # Show rubric traits
                    if result.verify_rubric:
                        lines.append(f"  - Rubric: {', '.join(f'{k}={v}' for k, v in result.verify_rubric.items())}")

                    # Show metric traits
                    if result.metric_trait_metrics:
                        for trait_name, metrics in result.metric_trait_metrics.items():
                            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                            lines.append(f"  - Metric [{trait_name}]: {metrics_str}")
                lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- {self.summary()}")

        return "\n".join(lines)

    def _get_status_emoji(self) -> str:
        """Get emoji status based on results."""
        if not self.global_eval:
            return "❓ No Data"

        stats = self.global_eval.get_summary_stats()
        if stats["success_rate"] == 100:
            return "✅ All Passed"
        elif stats["success_rate"] >= 80:
            return "⚠️ Mostly Passed"
        elif stats["success_rate"] > 0:
            return "❌ Some Failed"
        else:
            return "❌ All Failed"
