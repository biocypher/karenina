"""Data models for TaskEval."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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


class StepEval(BaseModel):
    """Evaluation results for a single step or global evaluation."""

    rubric_scores: dict[str, int | bool | None] = Field(
        default_factory=dict, description="Rubric trait evaluations with same structure as verification"
    )
    question_verification: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Question verification results: question_id -> list of results for multiple responses",
    )

    def format_rubric_scores(self, indent: str = "  ") -> str:
        """Format rubric scores as a readable table."""
        if not self.rubric_scores:
            return f"{indent}No rubric scores available"

        lines = []
        for trait_name, score in self.rubric_scores.items():
            if isinstance(score, bool):
                status = "✓" if score else "✗"
                display_score = "PASS" if score else "FAIL"
            elif isinstance(score, int):
                status = "✓" if score > 0 else "✗"
                display_score = str(score)
            else:
                status = "?"
                display_score = str(score) if score is not None else "N/A"

            lines.append(f"{indent}{status} {trait_name:<20}: {display_score}")

        return "\n".join(lines)

    def format_question_results(self, indent: str = "  ") -> str:
        """Format question verification results clearly."""
        if not self.question_verification:
            return f"{indent}No question verification results"

        lines = []
        for question_id, results in self.question_verification.items():
            lines.append(f"{indent}Question: {question_id}")

            for i, result in enumerate(results):
                result_num = f"[{i + 1}]" if len(results) > 1 else ""

                # Overall status
                if result.get("correct"):
                    status = "✓ PASSED"
                elif result.get("success", True):
                    status = "⚠ COMPLETED"
                else:
                    status = "✗ FAILED"

                lines.append(f"{indent}  {result_num} Status: {status}")

                # Agent output (truncated)
                agent_output = result.get("agent_output", "N/A")
                if len(agent_output) > 100:
                    agent_output = agent_output[:100] + "..."
                lines.append(f'{indent}  {result_num} Output: "{agent_output}"')

                # Ground truth and parsed responses
                details = result.get("details", {})
                if isinstance(details, dict):
                    # Show ground truth (expected answer)
                    if "parsed_gt_response" in details:
                        gt_response = details["parsed_gt_response"]
                        lines.append(f"{indent}  {result_num} Expected: {gt_response}")

                    # Show LLM parsed response
                    if "parsed_llm_response" in details:
                        llm_response = details["parsed_llm_response"]
                        lines.append(f"{indent}  {result_num} Parsed: {llm_response}")

                    # Execution time if available
                    if "execution_time" in details:
                        exec_time = details["execution_time"]
                        lines.append(f"{indent}  {result_num} Time: {exec_time:.3f}s")

                # Error information
                if result.get("error"):
                    lines.append(f"{indent}  {result_num} Error: {result['error']}")

                # Rubric scores for this question
                question_rubric = result.get("rubric_scores", {})
                if question_rubric:
                    passed = sum(1 for v in question_rubric.values() if v is True or (isinstance(v, int) and v > 0))
                    total = len(question_rubric)
                    lines.append(f"{indent}  {result_num} Rubric: {passed}/{total} traits passed")

                if i < len(results) - 1:
                    lines.append("")  # Separator between multiple results

            lines.append("")  # Separator between questions

        return "\n".join(lines).rstrip()

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for this evaluation."""
        total_questions = len(self.question_verification)
        passed_questions = 0
        total_results = 0

        for _question_id, results in self.question_verification.items():
            total_results += len(results)
            if any(result.get("correct", False) for result in results):
                passed_questions += 1

        rubric_passed = sum(
            1 for score in self.rubric_scores.values() if score is True or (isinstance(score, int) and score > 0)
        )
        rubric_total = len(self.rubric_scores)

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

    def display(self, show_details: bool = True) -> str:
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

            if self.global_eval.rubric_scores:
                lines.append("Rubric Scores:")
                lines.append(self.global_eval.format_rubric_scores())
                lines.append("")

            if show_details and self.global_eval.question_verification:
                lines.append("Question Verification:")
                lines.append(self.global_eval.format_question_results())
                lines.append("")

        # Step evaluations
        if self.per_step:
            for step_id, step_eval in self.per_step.items():
                lines.append("─" * 60)
                lines.append(f"STEP EVALUATION: {step_id}")
                lines.append("─" * 60)

                if step_eval.rubric_scores:
                    lines.append(f"Rubric Scores ({step_id}):")
                    lines.append(step_eval.format_rubric_scores())
                    lines.append("")

                if show_details and step_eval.question_verification:
                    lines.append(f"Question Verification ({step_id}):")
                    lines.append(step_eval.format_question_results())
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
                "rubric_scores": self.global_eval.rubric_scores,
                "question_verification": self.global_eval.question_verification,
                "summary_stats": self.global_eval.get_summary_stats(),
            }

        # Add step evaluations
        for step_id, step_eval in self.per_step.items():
            step_evaluations = result_dict["step_evaluations"]
            if isinstance(step_evaluations, dict):
                step_evaluations[step_id] = {
                    "rubric_scores": step_eval.rubric_scores,
                    "question_verification": step_eval.question_verification,
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

        # Global evaluation
        if self.global_eval and self.global_eval.rubric_scores:
            lines.append("## Rubric Scores")
            lines.append("")
            lines.append("| Trait | Result | Status |")
            lines.append("|-------|--------|--------|")

            for trait_name, score in self.global_eval.rubric_scores.items():
                if isinstance(score, bool):
                    result_str = "✅ Pass" if score else "❌ Fail"
                    status = "PASS" if score else "FAIL"
                elif isinstance(score, int):
                    result_str = f"📊 {score}"
                    status = "PASS" if score > 0 else "FAIL"
                else:
                    result_str = "❓ N/A"
                    status = "N/A"

                lines.append(f"| {trait_name} | {result_str} | {status} |")

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
