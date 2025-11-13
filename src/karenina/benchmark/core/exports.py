"""Export and reporting functionality for benchmark structure and metadata.

This module provides the ExportManager class for exporting benchmark STRUCTURE
and METADATA - the definition of the benchmark itself (questions, templates,
rubrics, statistics), not verification execution results.

Key Methods:
- to_dict(): Export benchmark as dictionary
- to_markdown(): Export benchmark as markdown document
- to_csv(): Export questions as CSV
- get_summary(): Get benchmark statistics
- get_statistics(): Get detailed statistics
- check_readiness(): Check if benchmark is ready for verification
- get_health_report(): Get comprehensive health report
- export_to_file(): Export to file in various formats

Note: This module is distinct from benchmark/exporter.py, which handles
exporting verification EXECUTION RESULTS (what happened during verification),
not benchmark structure/metadata.

Usage:
    from karenina.benchmark import Benchmark

    benchmark = Benchmark.load("path/to/benchmark.jsonld")
    # ExportManager is accessed via Benchmark.export_manager
    summary = benchmark.export_manager.get_summary()
    benchmark_dict = benchmark.export_manager.to_dict()
"""

import csv
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase
    from .rubrics import RubricManager
    from .templates import TemplateManager


class ExportManager:
    """Manager for exporting benchmark structure, metadata, and reporting.

    This class handles exporting the benchmark STRUCTURE and METADATA:
    - Questions and their definitions
    - Templates and their definitions
    - Rubrics and their definitions
    - Statistics and health reports
    - Benchmark metadata (name, version, creator, etc.)

    This is distinct from benchmark/exporter.py functions which export
    verification EXECUTION RESULTS (what happened when questions were verified).

    Examples:
        >>> benchmark = Benchmark.load("benchmark.jsonld")
        >>> # Export benchmark structure
        >>> summary = benchmark.export_manager.get_summary()
        >>> markdown = benchmark.export_manager.to_markdown()
        >>> # Check readiness
        >>> readiness = benchmark.export_manager.check_readiness()
        >>> # Export to file
        >>> benchmark.export_manager.export_to_file("export.json", format="json")
    """

    def __init__(
        self, base: "BenchmarkBase", templates_manager: "TemplateManager", rubrics_manager: "RubricManager"
    ) -> None:
        """Initialize with reference to benchmark base and other managers."""
        self.base = base
        self.templates_manager = templates_manager
        self.rubrics_manager = rubrics_manager

    def to_dict(self) -> dict[str, Any]:
        """Export benchmark as a plain dictionary."""
        return {
            "metadata": {
                "name": self.base.name,
                "description": self.base.description,
                "version": self.base.version,
                "creator": self.base.creator,
                "created_at": self.base.created_at,
                "modified_at": self.base.modified_at,
            },
            "statistics": self.get_summary(),
            "questions": list(self.base._questions_cache.values()),
            "global_rubric": (
                global_rubric.model_dump() if (global_rubric := self.rubrics_manager.get_global_rubric()) else None
            ),
        }

    def to_markdown(self) -> str:
        """Export benchmark as markdown document."""
        lines = []

        # Header
        lines.append(f"# {self.base.name}")
        lines.append("")
        if self.base.description:
            lines.append(self.base.description)
            lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append(f"- **Version**: {self.base.version}")
        lines.append(f"- **Creator**: {self.base.creator}")
        lines.append(f"- **Created**: {self.base.created_at}")
        lines.append(f"- **Modified**: {self.base.modified_at}")
        lines.append("")

        # Statistics
        stats = self.get_summary()
        lines.append("## Statistics")
        lines.append(f"- **Questions**: {stats['question_count']}")
        lines.append(f"- **Finished**: {stats['finished_count']}")
        lines.append(f"- **Progress**: {stats['progress_percentage']:.1f}%")
        lines.append(f"- **Has Templates**: {stats['has_template_count']}")
        lines.append("")

        # Global rubric
        global_rubric = self.rubrics_manager.get_global_rubric()
        if global_rubric:
            lines.append("## Global Rubric")
            for trait in global_rubric.traits:
                lines.append(f"- **{trait.name}**: {trait.description}")
            lines.append("")

        # Questions
        lines.append("## Questions")
        for i, q_data in enumerate(self.base._questions_cache.values(), 1):
            status = "✅" if q_data.get("finished", False) else "❌"
            template_status = "📝" if q_data.get("answer_template") else "❌"

            lines.append(f"### {i}. {q_data['question']}")
            lines.append(f"**Status**: {status} | **Template**: {template_status}")
            lines.append("")

            if q_data.get("raw_answer"):
                lines.append(f"**Expected Answer**: {q_data['raw_answer']}")
                lines.append("")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export questions as CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            ["Question ID", "Question", "Raw Answer", "Has Template", "Finished", "Author", "Created", "Modified"]
        )

        # Data rows
        for q_id, q_data in self.base._questions_cache.items():
            author = ""
            if q_data.get("author"):
                author = q_data["author"].get("name", "")

            writer.writerow(
                [
                    q_id,
                    q_data["question"],
                    q_data.get("raw_answer", ""),
                    "Yes" if q_data.get("answer_template") else "No",
                    "Yes" if q_data.get("finished", False) else "No",
                    author,
                    q_data.get("date_created", ""),
                    q_data.get("date_modified", ""),
                ]
            )

        return output.getvalue()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive benchmark statistics."""
        has_template_count = sum(1 for q_id in self.base._questions_cache if self.templates_manager.has_template(q_id))
        has_rubric_count = sum(1 for q in self.base._questions_cache.values() if q.get("question_rubric"))

        global_rubric = self.rubrics_manager.get_global_rubric()

        return {
            "name": self.base.name,
            "version": self.base.version,
            "creator": self.base.creator,
            "created_at": self.base.created_at,
            "modified_at": self.base.modified_at,
            "question_count": float(self.base.question_count),
            "finished_count": self.base.finished_count,
            "has_template_count": has_template_count,
            "has_rubric_count": has_rubric_count,
            "progress_percentage": self.base.get_progress(),
            "is_complete": self.base.is_complete,
            "has_global_rubric": global_rubric is not None,
            "global_rubric_traits": len(global_rubric.traits) if global_rubric else 0,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about the benchmark."""
        templates = [
            q.get("answer_template", "")
            for q_id, q in self.base._questions_cache.items()
            if self.templates_manager.has_template(q_id)
        ]

        avg_template_length = 0
        if templates:
            avg_template_length = int(sum(len(t) for t in templates) / len(templates))

        return {
            **self.get_summary(),
            "avg_template_length": round(avg_template_length, 1),
            "min_template_length": min(len(t) for t in templates) if templates else 0,
            "max_template_length": max(len(t) for t in templates) if templates else 0,
            "questions_with_custom_metadata": sum(
                1
                for q in self.base._questions_cache.values()
                if q.get("custom_metadata") or q.get("author") or q.get("sources")
            ),
        }

    def check_readiness(self) -> dict[str, Any]:
        """
        Comprehensive readiness check for verification.

        Returns:
            Dictionary with readiness status and details
        """
        missing_templates = self.templates_manager.get_missing_templates(ids_only=True)
        unfinished = [q_id for q_id, q_data in self.base._questions_cache.items() if not q_data.get("finished", False)]
        template_valid, template_errors = self.templates_manager.validate_templates()
        rubric_valid, rubric_errors = self.rubrics_manager.validate_rubrics()

        # Check if questions exist
        has_questions = self.base.question_count > 0

        # Check if all questions have templates
        all_have_templates = len(missing_templates) == 0

        # Check if all questions are finished
        all_finished = len(unfinished) == 0

        # Overall readiness
        ready_for_verification = (
            has_questions and all_have_templates and all_finished and template_valid and rubric_valid
        )

        return {
            "ready_for_verification": ready_for_verification,
            "has_questions": has_questions,
            "all_have_templates": all_have_templates,
            "all_finished": all_finished,
            "templates_valid": template_valid,
            "rubrics_valid": rubric_valid,
            "missing_templates_count": len(missing_templates),
            "unfinished_count": float(len(unfinished)),
            "template_errors_count": len(template_errors),
            "rubric_errors_count": len(rubric_errors),
            "missing_templates": missing_templates,
            "unfinished_questions": unfinished,
            "template_errors": template_errors,
            "rubric_errors": rubric_errors,
        }

    def get_health_report(self) -> dict[str, Any]:
        """
        Get comprehensive health/status report.

        Returns:
            Detailed health report with all aspects of benchmark status
        """
        readiness = self.check_readiness()
        stats = self.get_statistics()

        # Calculate health score (0-100)
        score = 0
        max_score = 100

        # For empty benchmarks, score is 0
        if not readiness["has_questions"]:
            health_score = 0.0
        else:
            # Questions exist (20 points)
            score += 20

            # Progress (30 points based on completion percentage)
            progress = self.base.get_progress()
            score += int((progress / 100) * 30)

            # Templates valid (25 points)
            if readiness["templates_valid"]:
                score += 25

            # Rubrics valid (15 points)
            if readiness["rubrics_valid"]:
                score += 15

            # All finished (10 points)
            if readiness["all_finished"]:
                score += 10

            health_score = min(score, max_score)

        # Status levels
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"

        return {
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "timestamp": datetime.now().isoformat(),
            "readiness": readiness,
            "statistics": stats,
            "recommendations": self._get_recommendations(readiness),
        }

    def _get_recommendations(self, readiness: dict[str, Any]) -> list[str]:
        """Get recommendations for improving benchmark health."""
        recommendations = []

        if not readiness["has_questions"]:
            recommendations.append("Add questions to the benchmark")

        if readiness["missing_templates_count"] > 0:
            recommendations.append(f"Add templates to {readiness['missing_templates_count']} questions")

        if readiness["unfinished_count"] > 0:
            recommendations.append(f"Mark {readiness['unfinished_count']} questions as finished")

        if not readiness["templates_valid"]:
            recommendations.append("Fix template syntax errors")

        if not readiness["rubrics_valid"]:
            recommendations.append("Fix rubric configuration issues")

        if self.base.question_count < 5:
            recommendations.append("Consider adding more questions for a robust benchmark")

        global_rubric = self.rubrics_manager.get_global_rubric()
        if not global_rubric:
            recommendations.append("Consider adding a global rubric for consistent evaluation")

        return recommendations

    def clone(self) -> "BenchmarkBase":
        """Create a deep copy of the benchmark."""
        # Create a temporary file to save/load from
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            temp_path = Path(f.name)

        try:
            self.base.save(temp_path)
            from .base import BenchmarkBase

            cloned = BenchmarkBase.load(temp_path)
            return cloned
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def export_to_file(self, path: Path, format: str = "auto") -> None:
        """
        Export benchmark to file in specified format.

        Args:
            path: Output file path
            format: Export format ('auto', 'json', 'csv', 'markdown', 'jsonld')
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix == ".csv":
                format = "csv"
            elif suffix in [".md", ".markdown"]:
                format = "markdown"
            elif suffix == ".jsonld":
                format = "jsonld"
            else:
                format = "json"  # default

        # Generate content based on format
        if format == "json":
            content = self._to_json()
        elif format == "csv":
            content = self.to_csv()
        elif format == "markdown":
            content = self.to_markdown()
        elif format == "jsonld":
            # Use the base save method for JSON-LD
            self.base.save(path)
            return
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _to_json(self) -> str:
        """Convert benchmark to JSON format."""
        import json

        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_progress_report(self) -> dict[str, Any]:
        """Get a progress report suitable for status displays."""
        return {
            "name": self.base.name,
            "progress_percentage": self.base.get_progress(),
            "questions_total": self.base.question_count,
            "questions_finished": self.base.finished_count,
            "questions_with_templates": sum(
                1 for q_id in self.base._questions_cache if self.templates_manager.has_template(q_id)
            ),
            "is_ready_for_verification": self.check_readiness()["ready_for_verification"],
            "last_modified": self.base.modified_at,
        }
