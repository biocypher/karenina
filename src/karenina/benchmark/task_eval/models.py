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
        """Get summary statistics for this evaluation."""
        total_questions = len(self.verification_results)
        passed_questions = 0
        total_results = 0
        rubric_passed = 0
        rubric_total = 0

        for _question_id, results in self.verification_results.items():
            total_results += len(results)
            if any(result.verify_result for result in results):
                passed_questions += 1

            # Count rubric traits from verification results
            for result in results:
                if result.verify_rubric:
                    for score in result.verify_rubric.values():
                        rubric_total += 1
                        if score is True or (isinstance(score, int) and score > 0):
                            rubric_passed += 1

                # Also count metric traits
                if result.metric_trait_metrics:
                    rubric_total += len(result.metric_trait_metrics)
                    rubric_passed += len(result.metric_trait_metrics)  # All computed metrics count as "passed"

        return {
            "questions_total": total_questions,
            "questions_passed": passed_questions,
            "results_total": total_results,
            "rubric_passed": rubric_passed,
            "rubric_total": rubric_total,
            "success_rate": (passed_questions / total_questions * 100) if total_questions > 0 else 0,
        }


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
        total_questions = 0
        passed_questions = 0
        total_rubric = 0
        passed_rubric = 0

        # Global stats
        if self.global_eval:
            stats = self.global_eval.get_summary_stats()
            total_questions += stats["questions_total"]
            passed_questions += stats["questions_passed"]
            total_rubric += stats["rubric_total"]
            passed_rubric += stats["rubric_passed"]

        # Step stats
        for step_eval in self.per_step.values():
            stats = step_eval.get_summary_stats()
            total_questions += stats["questions_total"]
            passed_questions += stats["questions_passed"]
            total_rubric += stats["rubric_total"]
            passed_rubric += stats["rubric_passed"]

        parts = []
        if total_questions > 0:
            parts.append(f"{passed_questions}/{total_questions} questions passed")
        if total_rubric > 0:
            parts.append(f"{passed_rubric}/{total_rubric} rubric traits passed")

        return " | ".join(parts) if parts else "No evaluations performed"

    def summary_compact(self) -> str:
        """Return a very compact one-line summary."""
        if self.global_eval:
            stats = self.global_eval.get_summary_stats()
            return (
                f"TaskEval [{self.task_id or 'Unknown'}]: "
                f"{stats['questions_passed']}/{stats['questions_total']} questions, "
                f"{stats['rubric_passed']}/{stats['rubric_total']} rubric traits "
                f"({stats['success_rate']:.0f}% success)"
            )
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
            lines.append(f"- **Questions Passed**: {stats['questions_passed']}/{stats['questions_total']}")
            lines.append(f"- **Rubric Traits Passed**: {stats['rubric_passed']}/{stats['rubric_total']}")
            lines.append(f"- **Success Rate**: {stats['success_rate']:.1f}%")
            lines.append("")

            # Show per-question results
            for question_id, results in self.global_eval.verification_results.items():
                lines.append(f"### Question: {question_id}")
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
