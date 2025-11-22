"""
VerificationResultSet class - main container for verification results.

This module provides the top-level container returned by run_verification,
with accessor methods for specialized result views (rubrics, templates, judgments).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .judgment_results import JudgmentResults
from .rubric_judgment_results import RubricJudgmentResults
from .rubric_results import RubricResults
from .template_results import TemplateResults

if TYPE_CHECKING:
    from .verification import VerificationResult


class VerificationResultSet(BaseModel):
    """
    Container for all verification results from a run_verification call.

    This is the main class returned by Benchmark.run_verification(). It stores
    all individual VerificationResult objects and provides three specialized
    accessor methods for working with different evaluation types:
    - get_rubrics_results(): Access rubric evaluation data
    - get_template_results(): Access template verification data
    - get_judgment_results(): Access deep judgment data

    The class also provides filtering, grouping, and iteration capabilities.

    Attributes:
        results: List of all VerificationResult objects from the verification run
    """

    results: list[VerificationResult] = Field(description="List of all verification results from the run")

    model_config = {"arbitrary_types_allowed": True}

    # ========================================================================
    # Accessor Methods (Main Interface)
    # ========================================================================

    def get_rubrics_results(self, include_deep_judgment: bool = False) -> RubricResults:
        """
        Get rubric evaluation results.

        Returns a RubricResults object that provides access to:
        - LLM trait scores (1-5 scale)
        - Manual trait scores (boolean)
        - Metric trait scores (precision, recall, f1, etc.)
        - Confusion matrices for metric traits
        - Aggregation and grouping capabilities
        - Deep judgment columns (optional, when include_deep_judgment=True)

        Args:
            include_deep_judgment: Whether to include deep judgment columns in DataFrame exports.
                When True, LLM traits will have three additional columns:
                - trait_reasoning: Reasoning text for the trait score
                - trait_excerpts: JSON-serialized list of excerpts
                - trait_hallucination_risk: Hallucination risk assessment
                Default: False (backward compatible)

        Returns:
            RubricResults instance wrapping all verification results

        Example:
            ```python
            result_set = benchmark.run_verification(config)

            # Standard rubric results (backward compatible)
            rubric_results = result_set.get_rubrics_results()
            df = rubric_results.to_dataframe()

            # With deep judgment columns
            rubric_results = result_set.get_rubrics_results(include_deep_judgment=True)
            df = rubric_results.to_dataframe()
            # df now has trait_reasoning, trait_excerpts, trait_hallucination_risk columns

            # Aggregate by question
            aggregated = rubric_results.aggregate_llm_traits(
                strategy="mean",
                by="question"
            )

            # Filter and group
            filtered = rubric_results.filter(question_ids=["q1", "q2"])
            by_model = filtered.group_by_model()
            ```
        """
        return RubricResults(results=self.results, include_deep_judgment=include_deep_judgment)

    def get_template_results(self) -> TemplateResults:
        """
        Get template verification results.

        Returns a TemplateResults object that provides access to:
        - Template verification pass/fail status
        - Embedding similarity scores
        - Regex validation results
        - Abstention detection
        - MCP tool usage metrics
        - Parsed responses (LLM vs ground truth)

        Returns:
            TemplateResults instance wrapping all verification results

        Example:
            ```python
            result_set = benchmark.run_verification(config)
            template_results = result_set.get_template_results()

            # Get pass rate by question
            pass_rates = template_results.aggregate_pass_rate(by="question")

            # Get embedding scores
            embedding_scores = template_results.get_embedding_scores()

            # Aggregate embedding similarity
            avg_similarity = template_results.aggregate_embedding_scores(
                strategy="mean",
                by="model"
            )

            # Filter to only passed results
            passed = template_results.filter(passed_only=True)
            ```
        """
        return TemplateResults(results=self.results)

    def get_judgment_results(self) -> JudgmentResults:
        """
        Get deep judgment evaluation results.

        Returns a JudgmentResults object that provides access to:
        - Extracted excerpts per attribute
        - Reasoning traces for each attribute
        - Hallucination risk assessments
        - Search validation results
        - Processing metrics (model calls, retries, etc.)

        Returns:
            JudgmentResults instance wrapping all verification results

        Example:
            ```python
            result_set = benchmark.run_verification(config)
            judgment_results = result_set.get_judgment_results()

            # Get extracted excerpts
            excerpts = judgment_results.get_extracted_excerpts(
                attribute_name="location"
            )

            # Get hallucination risk distribution
            risk_dist = judgment_results.aggregate_hallucination_risk_distribution(
                by="question"
            )

            # Get processing metrics
            metrics = judgment_results.get_processing_metrics()

            # Filter to results with search enabled
            with_search = judgment_results.filter(with_search_only=True)
            ```
        """
        return JudgmentResults(results=self.results)

    def get_rubric_judgments_results(self) -> RubricJudgmentResults:
        """
        Get detailed deep judgment rubric results with excerpt-level explosion.

        Returns a RubricJudgmentResults object that provides fine-grained access to:
        - Per-excerpt data (text, confidence, similarity scores)
        - Per-trait metadata (model calls, retry counts, stages completed)
        - Reasoning for each trait evaluation
        - Hallucination risk assessments per excerpt
        - Search validation results (if search enabled)

        Unlike get_rubrics_results() which returns one row per trait, this method
        explodes each trait into multiple rows (one per excerpt) for detailed analysis.

        Key Differences:
            - get_rubrics_results(): Standard export, one row per trait
            - get_rubric_judgments_results(): Detailed export, one row per (trait × excerpt)

        Returns:
            RubricJudgmentResults instance with excerpt-exploded data

        Example:
            ```python
            result_set = benchmark.run_verification(config)
            rubric_judgments = result_set.get_rubric_judgments_results()

            # Get detailed DataFrame with excerpt explosion
            df = rubric_judgments.to_dataframe()

            # Analyze per-excerpt confidence
            avg_confidence = df.groupby("trait_name")["excerpt_confidence"].mean()

            # Examine traits with validation failures
            failed = df[df["trait_validation_failed"] == True]
            print(failed[["trait_name", "trait_excerpt_retries", "excerpt_text"]])

            # Study hallucination risks
            risky = df[df["excerpt_hallucination_risk"].notna()]
            print(risky[["excerpt_text", "excerpt_hallucination_risk", "excerpt_hallucination_justification"]])

            # Get summary statistics
            excerpt_summary = rubric_judgments.get_excerpt_count_summary()
            retry_summary = rubric_judgments.get_retry_summary()
            ```
        """
        return RubricJudgmentResults(results=self.results)

    # ========================================================================
    # Filtering
    # ========================================================================

    def filter(
        self,
        question_ids: list[str] | None = None,
        answering_models: list[str] | None = None,
        parsing_models: list[str] | None = None,
        replicates: list[int] | None = None,
        completed_only: bool = False,
        has_template: bool = False,
        has_rubric: bool = False,
        has_judgment: bool = False,
    ) -> VerificationResultSet:
        """
        Filter results by various criteria.

        Args:
            question_ids: Filter by question IDs
            answering_models: Filter by answering model names
            parsing_models: Filter by parsing model names
            replicates: Filter by replicate numbers
            completed_only: Only include results that completed without errors
            has_template: Only include results with template verification data
            has_rubric: Only include results with rubric evaluation data
            has_judgment: Only include results with deep judgment data

        Returns:
            New VerificationResultSet with filtered results

        Example:
            ```python
            # Filter to specific questions
            filtered = result_set.filter(question_ids=["q1", "q2"])

            # Filter to successful results with rubric data
            filtered = result_set.filter(
                completed_only=True,
                has_rubric=True
            )

            # Filter to specific model and replicate
            filtered = result_set.filter(
                answering_models=["gpt-4"],
                replicates=[1]
            )
            ```
        """
        filtered = self.results

        if question_ids:
            filtered = [r for r in filtered if r.metadata.question_id in question_ids]

        if answering_models:
            filtered = [r for r in filtered if r.metadata.answering_model in answering_models]

        if parsing_models:
            filtered = [r for r in filtered if r.metadata.parsing_model in parsing_models]

        if replicates:
            filtered = [
                r
                for r in filtered
                if r.metadata.answering_replicate in replicates or r.metadata.parsing_replicate in replicates
            ]

        if completed_only:
            filtered = [r for r in filtered if r.metadata.completed_without_errors]

        if has_template:
            filtered = [r for r in filtered if r.template is not None and r.template.template_verification_performed]

        if has_rubric:
            filtered = [r for r in filtered if r.rubric is not None and r.rubric.rubric_evaluation_performed]

        if has_judgment:
            filtered = [r for r in filtered if r.deep_judgment is not None and r.deep_judgment.deep_judgment_performed]

        return VerificationResultSet(results=filtered)

    # ========================================================================
    # Grouping
    # ========================================================================

    def group_by_question(self) -> dict[str, VerificationResultSet]:
        """
        Group results by question ID.

        Returns:
            Dictionary mapping question IDs to VerificationResultSet instances

        Example:
            ```python
            by_question = result_set.group_by_question()
            for question_id, question_results in by_question.items():
                print(f"Question {question_id}: {len(question_results)} results")

                # Get rubric results for this question
                rubric_results = question_results.get_rubrics_results()
            ```
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            qid = result.metadata.question_id
            if qid not in grouped:
                grouped[qid] = []
            grouped[qid].append(result)

        return {qid: VerificationResultSet(results=results) for qid, results in grouped.items()}

    def group_by_model(
        self, by: Literal["answering", "parsing", "both"] = "answering"
    ) -> dict[str, VerificationResultSet]:
        """
        Group results by model(s).

        Args:
            by: How to group results:
                - "answering": Group by answering model (includes MCP servers if attached)
                - "parsing": Group by parsing model
                - "both": Group by both answering and parsing models

        Returns:
            Dictionary mapping model identifier(s) to VerificationResultSet instances

        Notes:
            - When grouping by answering model, MCP servers are included in the key
              to distinguish between the same model with different MCP configurations
            - Format examples:
              - answering: "gpt-4" or "gpt-4 + MCP[server1,server2]"
              - parsing: "gpt-4o-mini"
              - both: "gpt-4 / gpt-4o-mini" or "gpt-4 + MCP[server1] / gpt-4o-mini"

        Example:
            ```python
            # Group by answering model (with MCP distinction)
            by_answering = result_set.group_by_model(by="answering")
            for model_key, model_results in by_answering.items():
                print(f"Model {model_key}: {len(model_results)} results")

            # Group by parsing model
            by_parsing = result_set.group_by_model(by="parsing")

            # Group by both
            by_both = result_set.group_by_model(by="both")
            for combo_key, combo_results in by_both.items():
                print(f"Combination {combo_key}: {len(combo_results)} results")
            ```
        """
        grouped: dict[str, list[VerificationResult]] = {}

        for result in self.results:
            answering_model = result.metadata.answering_model
            parsing_model = result.metadata.parsing_model

            # Build answering model key with MCP if present
            if by in ("answering", "both"):
                mcp_servers = result.answering_mcp_servers
                if mcp_servers and len(mcp_servers) > 0:
                    answering_key = f"{answering_model} + MCP[{','.join(sorted(mcp_servers))}]"
                else:
                    answering_key = answering_model

            # Determine grouping key based on mode
            if by == "answering":
                key = answering_key
            elif by == "parsing":
                key = parsing_model
            elif by == "both":
                key = f"{answering_key} / {parsing_model}"
            else:
                raise ValueError(f"Invalid grouping mode: {by}. Must be 'answering', 'parsing', or 'both'")

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        return {key: VerificationResultSet(results=results) for key, results in grouped.items()}

    def group_by_replicate(self) -> dict[int, VerificationResultSet]:
        """
        Group results by replicate number.

        Returns:
            Dictionary mapping replicate numbers to VerificationResultSet instances

        Example:
            ```python
            by_replicate = result_set.group_by_replicate()
            for rep_num, rep_results in by_replicate.items():
                print(f"Replicate {rep_num}: {len(rep_results)} results")
            ```
        """
        grouped: dict[int, list[VerificationResult]] = {}
        for result in self.results:
            rep = result.metadata.answering_replicate or 0
            if rep not in grouped:
                grouped[rep] = []
            grouped[rep].append(result)

        return {rep: VerificationResultSet(results=results) for rep, results in grouped.items()}

    # ========================================================================
    # Summary and Statistics
    # ========================================================================

    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive summary statistics for the entire result set.

        Returns:
            Dictionary with summary statistics including:

            **Basic Counts**:
            - num_results: Total number of results
            - num_completed: Number of results that completed without errors
            - num_with_template: Number with template verification
            - num_with_rubric: Number with rubric evaluation
            - num_with_judgment: Number with deep judgment
            - num_questions: Number of unique questions
            - num_models: Number of unique answering models
            - num_parsing_models: Number of unique parsing models
            - num_replicates: Number of unique replicates

            **Execution**:
            - total_execution_time: Total execution time in seconds

            **Token Usage**:
            - tokens: Dict with total_input, total_output, template_input, template_output,
                     rubric_input, rubric_output

            **Completion Status**:
            - completion_by_combo: Dict mapping (answering, parsing, mcp_tuple) to
                                  {total, completed, completion_pct}

            **Evaluation Types** (if rubrics enabled):
            - rubric_traits: Dict with trait breakdowns (global/question-specific by type)

            **Template Pass Rates**:
            - template_pass_by_combo: Dict mapping combos to {total, passed, pass_pct}
            - template_pass_overall: Dict with {total, passed, pass_pct}

            **Replicate Statistics** (if replicates > 1):
            - replicate_pass_rates: Dict mapping replicate number to {total, passed, pass_pct, pass_rate}
            - replicate_summary: Dict with {mean, std}

        Example:
            ```python
            summary = result_set.get_summary()
            print(f"Total results: {summary['num_results']}")
            print(f"Questions: {summary['num_questions']}")
            print(f"Total tokens: {summary['tokens']['total_input']} input")
            ```
        """
        from collections import defaultdict

        # Basic counts
        num_completed = 0
        num_with_template = 0
        num_with_rubric = 0
        num_with_judgment = 0
        questions = set()
        models = set()
        parsing_models = set()
        replicates = set()

        for result in self.results:
            if result.metadata.completed_without_errors:
                num_completed += 1

            if result.template and result.template.template_verification_performed:
                num_with_template += 1

            if result.rubric and result.rubric.rubric_evaluation_performed:
                num_with_rubric += 1

            if result.deep_judgment and result.deep_judgment.deep_judgment_performed:
                num_with_judgment += 1

            questions.add(result.metadata.question_id)
            models.add(result.metadata.answering_model)
            parsing_models.add(result.metadata.parsing_model)
            if result.metadata.answering_replicate is not None:
                replicates.add(result.metadata.answering_replicate)

        # Execution time
        total_execution_time = sum(
            r.metadata.execution_time for r in self.results if r.metadata.execution_time is not None
        )

        # Token usage
        total_input_tokens = 0
        total_output_tokens = 0
        template_input_tokens = 0
        template_output_tokens = 0
        rubric_input_tokens = 0
        rubric_output_tokens = 0
        deep_judgment_input_tokens = 0
        deep_judgment_output_tokens = 0

        for result in self.results:
            if result.template and hasattr(result.template, "usage_metadata") and result.template.usage_metadata:
                usage_metadata = result.template.usage_metadata

                if "total" in usage_metadata:
                    total_usage = usage_metadata["total"]
                    total_input_tokens += total_usage.get("input_tokens", 0)
                    total_output_tokens += total_usage.get("output_tokens", 0)

                if "answer_generation" in usage_metadata:
                    answer_usage = usage_metadata["answer_generation"]
                    template_input_tokens += answer_usage.get("input_tokens", 0)
                    template_output_tokens += answer_usage.get("output_tokens", 0)

                if "parsing" in usage_metadata:
                    parsing_usage = usage_metadata["parsing"]
                    template_input_tokens += parsing_usage.get("input_tokens", 0)
                    template_output_tokens += parsing_usage.get("output_tokens", 0)

                if "rubric_evaluation" in usage_metadata:
                    rubric_usage = usage_metadata["rubric_evaluation"]
                    rubric_input_tokens += rubric_usage.get("input_tokens", 0)
                    rubric_output_tokens += rubric_usage.get("output_tokens", 0)

            if (
                result.deep_judgment
                and hasattr(result.deep_judgment, "usage_metadata")
                and result.deep_judgment.usage_metadata
            ):
                dj_usage = result.deep_judgment.usage_metadata
                if "total" in dj_usage:
                    dj_total = dj_usage["total"]
                    deep_judgment_input_tokens += dj_total.get("input_tokens", 0)
                    deep_judgment_output_tokens += dj_total.get("output_tokens", 0)

        # Completion status by model combination
        combo_stats: dict[tuple[str, str, tuple[str, ...] | None], dict[str, int]] = defaultdict(
            lambda: {"total": 0, "completed": 0}
        )

        for result in self.results:
            mcp_servers = (
                result.template.answering_mcp_servers
                if result.template and result.template.answering_mcp_servers
                else None
            )
            mcp_key = tuple(sorted(mcp_servers)) if mcp_servers else None
            combo_key = (result.metadata.answering_model, result.metadata.parsing_model, mcp_key)

            combo_stats[combo_key]["total"] += 1
            if result.metadata.completed_without_errors:
                combo_stats[combo_key]["completed"] += 1

        # Add completion percentages
        completion_by_combo = {}
        for combo_key, stats in combo_stats.items():
            completion_by_combo[combo_key] = {
                "total": stats["total"],
                "completed": stats["completed"],
                "completion_pct": (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            }

        # Rubric trait breakdown (if rubrics enabled)
        rubric_traits = None
        if num_with_rubric > 0:
            trait_questions = defaultdict(set)
            trait_types = {}

            for result in self.results:
                if result.rubric and result.rubric.rubric_evaluation_performed:
                    q_id = result.metadata.question_id

                    if result.rubric.llm_trait_scores:
                        for trait_name in result.rubric.llm_trait_scores:
                            trait_questions[trait_name].add(q_id)
                            trait_types[trait_name] = "llm"

                    if result.rubric.regex_trait_scores:
                        for trait_name in result.rubric.regex_trait_scores:
                            trait_questions[trait_name].add(q_id)
                            trait_types[trait_name] = "regex"

                    if result.rubric.callable_trait_scores:
                        for trait_name in result.rubric.callable_trait_scores:
                            trait_questions[trait_name].add(q_id)
                            trait_types[trait_name] = "callable"

                    if result.rubric.metric_trait_confusion_lists:
                        for trait_name in result.rubric.metric_trait_confusion_lists:
                            trait_questions[trait_name].add(q_id)
                            trait_types[trait_name] = "metric"

            num_questions = len(questions)
            global_traits = {trait for trait, qs in trait_questions.items() if len(qs) == num_questions}

            # Count evaluations by type
            global_llm = global_regex = global_callable = global_metric = 0
            qs_llm = qs_regex = qs_callable = qs_metric = 0

            for result in self.results:
                if result.rubric and result.rubric.rubric_evaluation_performed:
                    if result.rubric.llm_trait_scores:
                        for trait in result.rubric.llm_trait_scores:
                            if trait in global_traits:
                                global_llm += 1
                            else:
                                qs_llm += 1

                    if result.rubric.regex_trait_scores:
                        for trait in result.rubric.regex_trait_scores:
                            if trait in global_traits:
                                global_regex += 1
                            else:
                                qs_regex += 1

                    if result.rubric.callable_trait_scores:
                        for trait in result.rubric.callable_trait_scores:
                            if trait in global_traits:
                                global_callable += 1
                            else:
                                qs_callable += 1

                    if result.rubric.metric_trait_confusion_lists:
                        for trait in result.rubric.metric_trait_confusion_lists:
                            if trait in global_traits:
                                global_metric += 1
                            else:
                                qs_metric += 1

            rubric_traits = {
                "global_traits": {
                    "llm": {"count": global_llm},
                    "regex": {"count": global_regex},
                    "callable": {"count": global_callable},
                    "metric": {"count": global_metric},
                },
                "question_specific_traits": {
                    "llm": {"count": qs_llm},
                    "regex": {"count": qs_regex},
                    "callable": {"count": qs_callable},
                    "metric": {"count": qs_metric},
                },
            }

        # Template pass rates
        template_pass_by_combo = None
        template_pass_overall = None

        if num_with_template > 0:
            combo_pass_stats: dict[tuple[str, str, tuple[str, ...] | None], dict[str, int]] = defaultdict(
                lambda: {"total": 0, "passed": 0}
            )

            for result in self.results:
                if result.template and result.template.template_verification_performed:
                    mcp_servers = (
                        result.template.answering_mcp_servers if result.template.answering_mcp_servers else None
                    )
                    mcp_key = tuple(sorted(mcp_servers)) if mcp_servers else None
                    combo_key = (result.metadata.answering_model, result.metadata.parsing_model, mcp_key)

                    combo_pass_stats[combo_key]["total"] += 1
                    if result.template.verify_result:
                        combo_pass_stats[combo_key]["passed"] += 1

            # Add pass percentages
            template_pass_by_combo = {}
            for combo_key, stats in combo_pass_stats.items():
                template_pass_by_combo[combo_key] = {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "pass_pct": (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0,
                }

            # Overall pass rate
            overall_passed = sum(s["passed"] for s in combo_pass_stats.values())
            overall_total = sum(s["total"] for s in combo_pass_stats.values())
            template_pass_overall = {
                "total": overall_total,
                "passed": overall_passed,
                "pass_pct": (overall_passed / overall_total * 100) if overall_total > 0 else 0,
            }

        # Replicate statistics
        replicate_pass_rates = None
        replicate_summary = None

        if len(replicates) > 1 and num_with_template > 0:
            rep_stats_dict: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})

            for result in self.results:
                if result.template and result.template.template_verification_performed:
                    rep_num = result.metadata.answering_replicate
                    if rep_num is not None:
                        rep_stats_dict[rep_num]["total"] += 1
                        if result.template.verify_result:
                            rep_stats_dict[rep_num]["passed"] += 1

            replicate_pass_rates = {}
            pass_rates_list = []

            for rep_num, stats in rep_stats_dict.items():
                passed = stats["passed"]
                total = stats["total"]
                pct = (passed / total * 100) if total > 0 else 0
                pass_rate = passed / total if total > 0 else 0
                pass_rates_list.append(pass_rate)

                replicate_pass_rates[rep_num] = {
                    "total": total,
                    "passed": passed,
                    "pass_pct": pct,
                    "pass_rate": pass_rate,
                }

            # Calculate mean and std
            if len(pass_rates_list) > 0:
                import statistics

                mean = statistics.mean(pass_rates_list)
                std = statistics.stdev(pass_rates_list) if len(pass_rates_list) > 1 else 0.0
                replicate_summary = {"mean": mean, "std": std}

        # Build replicate_stats only if we have replicates
        replicate_stats: dict[str, dict[int, dict[str, float | int]] | dict[str, float]] | None = None
        if replicate_pass_rates is not None and replicate_summary is not None:
            replicate_stats = {
                "replicate_pass_rates": replicate_pass_rates,
                "replicate_summary": replicate_summary,
            }

        return {
            # Basic counts
            "num_results": len(self.results),
            "num_completed": num_completed,
            "num_with_template": num_with_template,
            "num_with_rubric": num_with_rubric,
            "num_with_judgment": num_with_judgment,
            "num_questions": len(questions),
            "num_models": len(models),
            "num_parsing_models": len(parsing_models),
            "num_replicates": len(replicates),
            # Execution
            "total_execution_time": total_execution_time,
            # Token usage
            "tokens": {
                "total_input": total_input_tokens,
                "total_output": total_output_tokens,
                "template_input": template_input_tokens,
                "template_output": template_output_tokens,
                "rubric_input": rubric_input_tokens,
                "rubric_output": rubric_output_tokens,
                "deep_judgment_input": deep_judgment_input_tokens if num_with_judgment > 0 else None,
                "deep_judgment_output": deep_judgment_output_tokens if num_with_judgment > 0 else None,
            },
            # Completion status
            "completion_by_combo": completion_by_combo,
            # Evaluation types
            "rubric_traits": rubric_traits,
            # Template pass rates
            "template_pass_by_combo": template_pass_by_combo,
            "template_pass_overall": template_pass_overall,
            # Replicate statistics
            "replicate_stats": replicate_stats,
        }

    def get_question_ids(self) -> list[str]:
        """
        Get list of all unique question IDs in the result set.

        Returns:
            Sorted list of question IDs

        Example:
            ```python
            question_ids = result_set.get_question_ids()
            print(f"Questions: {', '.join(question_ids)}")
            ```
        """
        return sorted({r.metadata.question_id for r in self.results})

    def get_model_names(self) -> list[str]:
        """
        Get list of all unique answering model names in the result set.

        Returns:
            Sorted list of model names

        Example:
            ```python
            models = result_set.get_model_names()
            print(f"Models: {', '.join(models)}")
            ```
        """
        return sorted({r.metadata.answering_model for r in self.results})

    # ========================================================================
    # Special Methods
    # ========================================================================

    def __len__(self) -> int:
        """
        Return number of results in the set.

        Example:
            ```python
            print(f"Result set contains {len(result_set)} results")
            ```
        """
        return len(self.results)

    def __iter__(self) -> Any:
        """
        Iterate over individual VerificationResult objects.

        Example:
            ```python
            for result in result_set:
                print(f"Question: {result.metadata.question_id}")
                if result.template:
                    print(f"  Passed: {result.template.verify_result}")
            ```
        """
        return iter(self.results)

    def __getitem__(self, index: int) -> VerificationResult:
        """
        Access individual result by index.

        Example:
            ```python
            first_result = result_set[0]
            last_result = result_set[-1]
            ```
        """
        return self.results[index]

    def __repr__(self) -> str:
        """
        Enhanced string representation with detailed statistics.

        Fetches comprehensive data from get_summary() and formats it for display.
        Shows overview, completion status, evaluation types, template pass rates,
        and replicate statistics.
        """
        if not self.results:
            return "VerificationResultSet(empty)"

        # Helper function to format time
        def format_time(seconds: float) -> str:
            """Format seconds as H:M:S, omitting hours/minutes if zero."""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60

            if hours > 0:
                return f"{hours}h {minutes}m {secs:.1f}s"
            elif minutes > 0:
                return f"{minutes}m {secs:.1f}s"
            else:
                return f"{secs:.1f}s"

        # Helper function to format token counts with dot separators
        def format_tokens(count: int) -> str:
            """Format token count with dot separators (thousands)."""
            return f"{count:,}".replace(",", ".")

        # Get comprehensive summary
        summary = self.get_summary()

        lines = ["VerificationResultSet("]

        # === OVERVIEW ===
        lines.append("  === OVERVIEW ===")
        lines.append(f"  Total Results: {summary['num_results']}")
        lines.append(f"  Questions: {summary['num_questions']}")
        lines.append(f"  Models: {summary['num_models']} answering x {summary['num_parsing_models']} parsing")

        if summary["num_replicates"] > 0:
            lines.append(f"  Replicates: {summary['num_replicates']}")

        if summary["total_execution_time"] > 0:
            lines.append(f"  Total Execution Time: {format_time(summary['total_execution_time'])}")

        # Token usage breakdown
        tokens = summary["tokens"]
        if tokens["total_input"] > 0 or tokens["total_output"] > 0:
            lines.append(
                f"  Total Tokens: {format_tokens(tokens['total_input'])} input, "
                f"{format_tokens(tokens['total_output'])} output"
            )

            if tokens["template_input"] > 0 or tokens["template_output"] > 0:
                lines.append(
                    f"    └─ Templates: {format_tokens(tokens['template_input'])} input, "
                    f"{format_tokens(tokens['template_output'])} output"
                )

            if tokens["rubric_input"] > 0 or tokens["rubric_output"] > 0:
                lines.append(
                    f"    └─ Rubrics: {format_tokens(tokens['rubric_input'])} input, "
                    f"{format_tokens(tokens['rubric_output'])} output"
                )

        # === COMPLETION STATUS ===
        lines.append("")
        lines.append("  === COMPLETION STATUS ===")
        lines.append("  By Model Combination:")

        for combo_key, stats in sorted(summary["completion_by_combo"].items()):
            answering, parsing, mcp_key = combo_key
            # Format MCP suffix
            mcp_suffix = f" + {', '.join(mcp_key)}" if mcp_key else ""

            combo_str = f"{answering} / {parsing}{mcp_suffix}"
            lines.append(
                f"    {combo_str:40} {stats['completed']}/{stats['total']} completed ({stats['completion_pct']:.1f}%)"
            )

        lines.append(
            f"  Overall: {summary['num_completed']}/{summary['num_results']} "
            f"completed ({(summary['num_completed'] / summary['num_results'] * 100) if summary['num_results'] > 0 else 0:.1f}%)"
        )

        # === EVALUATION TYPES ===
        lines.append("")
        lines.append("  === EVALUATION TYPES ===")

        if summary["num_with_template"] > 0:
            lines.append(f"  Template Verification: {summary['num_with_template']} results")

        if summary["num_with_rubric"] > 0 and summary["rubric_traits"]:
            traits = summary["rubric_traits"]
            lines.append(f"  Rubric Evaluation: {summary['num_with_rubric']} results")
            lines.append(f"    Total Trait Evaluations: {traits['total']}")

            # Global traits breakdown
            if traits["global_total"] > 0:
                lines.append(f"    Global: {traits['global_total']}")
                breakdown = []
                if traits["global_llm"] > 0:
                    breakdown.append(f"LLM: {traits['global_llm']}")
                if traits["global_regex"] > 0:
                    breakdown.append(f"Regex: {traits['global_regex']}")
                if traits["global_callable"] > 0:
                    breakdown.append(f"Callable: {traits['global_callable']}")
                if traits["global_metric"] > 0:
                    breakdown.append(f"Metric: {traits['global_metric']}")
                if breakdown:
                    lines.append(f"      └─ {', '.join(breakdown)}")

            # Question-specific traits breakdown
            if traits["qs_total"] > 0:
                lines.append(f"    Question-Specific: {traits['qs_total']}")
                breakdown = []
                if traits["qs_llm"] > 0:
                    breakdown.append(f"LLM: {traits['qs_llm']}")
                if traits["qs_regex"] > 0:
                    breakdown.append(f"Regex: {traits['qs_regex']}")
                if traits["qs_callable"] > 0:
                    breakdown.append(f"Callable: {traits['qs_callable']}")
                if traits["qs_metric"] > 0:
                    breakdown.append(f"Metric: {traits['qs_metric']}")
                if breakdown:
                    lines.append(f"      └─ {', '.join(breakdown)}")

        if summary["num_with_judgment"] > 0:
            lines.append(f"  Deep Judgment: {summary['num_with_judgment']} results")

        # === TEMPLATE PASS RATES ===
        if summary["template_pass_by_combo"]:
            lines.append("")
            lines.append("  === TEMPLATE PASS RATES ===")
            lines.append("  By Model Combination (+ MCP if used):")

            for combo_key, stats in sorted(summary["template_pass_by_combo"].items()):
                answering, parsing, mcp_key = combo_key
                mcp_suffix = f" + {', '.join(mcp_key)}" if mcp_key else ""

                combo_str = f"{answering} / {parsing}{mcp_suffix}"
                lines.append(f"    {combo_str:40} {stats['passed']}/{stats['total']} passed ({stats['pass_pct']:.1f}%)")

            if summary["template_pass_overall"]:
                overall = summary["template_pass_overall"]
                lines.append(f"  Overall: {overall['passed']}/{overall['total']} passed ({overall['pass_pct']:.1f}%)")

        # === REPLICATE STATISTICS ===
        if summary["replicate_pass_rates"]:
            lines.append("")
            lines.append("  === REPLICATE STATISTICS ===")
            lines.append("  Template Pass Rate by Replicate:")

            for rep_num in sorted(summary["replicate_pass_rates"].keys()):
                stats = summary["replicate_pass_rates"][rep_num]
                lines.append(
                    f"    Replicate {rep_num}: {stats['passed']}/{stats['total']} passed ({stats['pass_pct']:.1f}%)"
                )

            if summary["replicate_summary"]:
                rep_sum = summary["replicate_summary"]
                lines.append(f"  Summary: mean={rep_sum['mean']:.3f}, std={rep_sum['std']:.3f}")

        lines.append(")")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (same as repr for developer-friendly output)."""
        return self.__repr__()

    # ========================================================================
    # Legacy Compatibility
    # ========================================================================

    def to_legacy_dict(self) -> dict[str, VerificationResult]:
        """
        Convert to legacy dict format for backward compatibility.

        This method recreates the dictionary format used by the old run_verification
        API where keys were in the format:
        {question_id}_{answering_model}_{parsing_model}_rep{N}_{timestamp}

        Returns:
            Dictionary mapping result keys to VerificationResult objects

        Example:
            ```python
            result_set = benchmark.run_verification(config)

            # Convert to old dict format for legacy code
            legacy_dict = result_set.to_legacy_dict()

            # Keys like: "q1_gpt-4_gpt-4_rep1_1699123456789"
            for result_key, result in legacy_dict.items():
                print(f"{result_key}: {result.metadata.question_id}")
            ```
        """
        import time

        result_dict = {}
        for result in self.results:
            # Generate key matching the old format:
            # {question_id}_{answering}_{parsing}_rep{N}_{timestamp}
            key_parts = [
                result.metadata.question_id,
                result.metadata.answering_model,
                result.metadata.parsing_model,
            ]

            # Add replicate if present
            if result.metadata.answering_replicate is not None:
                key_parts.append(f"rep{result.metadata.answering_replicate}")

            # Use the result's timestamp if available, otherwise current time
            if result.metadata.timestamp:
                # Try to parse ISO timestamp to milliseconds
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(result.metadata.timestamp.replace("Z", "+00:00"))
                    timestamp_ms = str(int(dt.timestamp() * 1000))
                except (ValueError, AttributeError):
                    # Fallback to current time
                    timestamp_ms = str(int(time.time() * 1000))
            else:
                timestamp_ms = str(int(time.time() * 1000))

            key_parts.append(timestamp_ms)

            result_key = "_".join(key_parts)
            result_dict[result_key] = result

        return result_dict
