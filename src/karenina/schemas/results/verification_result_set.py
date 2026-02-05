"""
VerificationResultSet class - main container for verification results.

This module provides the top-level container returned by run_verification,
with accessor methods for specialized result views (rubrics, templates, judgments).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .judgment import JudgmentResults
from .rubric import RubricResults
from .rubric_judgment import RubricJudgmentResults
from .template import TemplateResults

if TYPE_CHECKING:
    from ..verification import VerificationResult


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
            filtered = [r for r in filtered if r.metadata.replicate in replicates]

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
                mcp_servers = result.template.answering_mcp_servers if result.template else None
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
            rep = result.metadata.replicate or 0
            if rep not in grouped:
                grouped[rep] = []
            grouped[rep].append(result)

        return {rep: VerificationResultSet(results=results) for rep, results in grouped.items()}

    # ========================================================================
    # Summary and Statistics - Private Helper Methods
    # ========================================================================

    def _calculate_basic_counts(self) -> dict[str, Any]:
        """Calculate basic counts: completed, template, rubric, judgment, unique sets."""
        num_completed = 0
        num_with_template = 0
        num_with_rubric = 0
        num_with_judgment = 0
        questions: set[str] = set()
        models: set[str] = set()
        parsing_models: set[str] = set()
        replicates: set[int] = set()

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
            if result.metadata.replicate is not None:
                replicates.add(result.metadata.replicate)

        total_execution_time = sum(
            r.metadata.execution_time for r in self.results if r.metadata.execution_time is not None
        )

        return {
            "num_completed": num_completed,
            "num_with_template": num_with_template,
            "num_with_rubric": num_with_rubric,
            "num_with_judgment": num_with_judgment,
            "questions": questions,
            "models": models,
            "parsing_models": parsing_models,
            "replicates": replicates,
            "total_execution_time": total_execution_time,
        }

    def _calculate_token_stats(
        self, num_with_judgment: int
    ) -> tuple[dict[str, Any], dict[tuple[str, str, tuple[str, ...] | None], dict[str, int]]]:
        """Calculate token usage statistics.

        Returns:
            Tuple of (tokens_dict, tokens_by_combo_dict)
        """
        import numpy as np

        total_input_tokens_list: list[int] = []
        total_output_tokens_list: list[int] = []
        template_input_tokens_list: list[int] = []
        template_output_tokens_list: list[int] = []
        rubric_input_tokens_list: list[int] = []
        rubric_output_tokens_list: list[int] = []
        deep_judgment_input_tokens_list: list[int] = []
        deep_judgment_output_tokens_list: list[int] = []

        per_question_input_tokens: list[int] = []
        per_question_output_tokens: list[int] = []

        def is_valid_number(val: Any) -> bool:
            """Check if value is a valid non-NaN number."""
            return val is not None and isinstance(val, int | float) and not (isinstance(val, float) and np.isnan(val))

        for result in self.results:
            if result.template and hasattr(result.template, "usage_metadata") and result.template.usage_metadata:
                usage_metadata = result.template.usage_metadata

                # Total tokens
                if "total" in usage_metadata:
                    total_usage = usage_metadata["total"]
                    inp = total_usage.get("input_tokens", 0)
                    out = total_usage.get("output_tokens", 0)
                    if is_valid_number(inp):
                        total_input_tokens_list.append(int(inp))
                        per_question_input_tokens.append(int(inp))
                    if is_valid_number(out):
                        total_output_tokens_list.append(int(out))
                        per_question_output_tokens.append(int(out))

                # Template tokens (answer_generation + parsing)
                template_inp = 0
                template_out = 0
                if "answer_generation" in usage_metadata:
                    answer_usage = usage_metadata["answer_generation"]
                    ans_inp = answer_usage.get("input_tokens", 0)
                    ans_out = answer_usage.get("output_tokens", 0)
                    if is_valid_number(ans_inp):
                        template_inp += int(ans_inp)
                    if is_valid_number(ans_out):
                        template_out += int(ans_out)

                if "parsing" in usage_metadata:
                    parsing_usage = usage_metadata["parsing"]
                    parse_inp = parsing_usage.get("input_tokens", 0)
                    parse_out = parsing_usage.get("output_tokens", 0)
                    if is_valid_number(parse_inp):
                        template_inp += int(parse_inp)
                    if is_valid_number(parse_out):
                        template_out += int(parse_out)

                if template_inp > 0 or template_out > 0:
                    template_input_tokens_list.append(template_inp)
                    template_output_tokens_list.append(template_out)

                # Rubric tokens
                if "rubric_evaluation" in usage_metadata:
                    rubric_usage = usage_metadata["rubric_evaluation"]
                    rubric_inp = rubric_usage.get("input_tokens", 0)
                    rubric_out = rubric_usage.get("output_tokens", 0)
                    if is_valid_number(rubric_inp) and rubric_inp > 0:
                        rubric_input_tokens_list.append(int(rubric_inp))
                    if is_valid_number(rubric_out) and rubric_out > 0:
                        rubric_output_tokens_list.append(int(rubric_out))

            # Deep judgment tokens
            if (
                result.deep_judgment
                and hasattr(result.deep_judgment, "usage_metadata")
                and result.deep_judgment.usage_metadata
            ):
                dj_usage = result.deep_judgment.usage_metadata
                if "total" in dj_usage:
                    dj_total = dj_usage["total"]
                    dj_inp = dj_total.get("input_tokens", 0)
                    dj_out = dj_total.get("output_tokens", 0)
                    if is_valid_number(dj_inp) and dj_inp > 0:
                        deep_judgment_input_tokens_list.append(int(dj_inp))
                    if is_valid_number(dj_out) and dj_out > 0:
                        deep_judgment_output_tokens_list.append(int(dj_out))

        def compute_stats(values: list[int]) -> tuple[float, float]:
            """Compute median and std, returning (median, std)"""
            if not values:
                return 0.0, 0.0
            arr = np.array([v for v in values if not (isinstance(v, float) and np.isnan(v))])
            if len(arr) == 0:
                return 0.0, 0.0
            median_val = float(np.median(arr))
            std_val = float(np.std(arr))
            if np.isnan(median_val):
                median_val = 0.0
            if np.isnan(std_val):
                std_val = 0.0
            return median_val, std_val

        # Compute actual totals (sum) for all token types
        total_input_sum = sum(total_input_tokens_list)
        total_output_sum = sum(total_output_tokens_list)
        template_input_sum = sum(template_input_tokens_list)
        template_output_sum = sum(template_output_tokens_list)
        rubric_input_sum = sum(rubric_input_tokens_list)
        rubric_output_sum = sum(rubric_output_tokens_list)
        dj_input_sum = sum(deep_judgment_input_tokens_list)
        dj_output_sum = sum(deep_judgment_output_tokens_list)

        # Compute median and std for variance information
        total_input_median, total_input_std = compute_stats(total_input_tokens_list)
        total_output_median, total_output_std = compute_stats(total_output_tokens_list)
        template_input_median, template_input_std = compute_stats(template_input_tokens_list)
        template_output_median, template_output_std = compute_stats(template_output_tokens_list)
        rubric_input_median, rubric_input_std = compute_stats(rubric_input_tokens_list)
        rubric_output_median, rubric_output_std = compute_stats(rubric_output_tokens_list)
        dj_input_median, dj_input_std = compute_stats(deep_judgment_input_tokens_list)
        dj_output_median, dj_output_std = compute_stats(deep_judgment_output_tokens_list)

        # Median tokens per question
        per_q_input_median, per_q_input_std = compute_stats(per_question_input_tokens)
        per_q_output_median, per_q_output_std = compute_stats(per_question_output_tokens)

        # Token usage by model combination
        from collections import defaultdict

        combo_token_stats: dict[tuple[str, str, tuple[str, ...] | None], dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )

        for result in self.results:
            mcp_servers = (
                result.template.answering_mcp_servers
                if result.template and result.template.answering_mcp_servers
                else None
            )
            mcp_key = tuple(sorted(mcp_servers)) if mcp_servers else None
            combo_key = (result.metadata.answering_model, result.metadata.parsing_model, mcp_key)

            # Add tokens from template verification
            if result.template and hasattr(result.template, "usage_metadata") and result.template.usage_metadata:
                usage_metadata = result.template.usage_metadata
                if "total" in usage_metadata:
                    total_usage = usage_metadata["total"]
                    inp = total_usage.get("input_tokens", 0)
                    out = total_usage.get("output_tokens", 0)
                    if is_valid_number(inp):
                        combo_token_stats[combo_key]["input"] += int(inp)
                    if is_valid_number(out):
                        combo_token_stats[combo_key]["output"] += int(out)

            # Add tokens from deep judgment
            if (
                result.deep_judgment
                and hasattr(result.deep_judgment, "usage_metadata")
                and result.deep_judgment.usage_metadata
            ):
                dj_usage = result.deep_judgment.usage_metadata
                if "total" in dj_usage:
                    dj_total = dj_usage["total"]
                    dj_inp = dj_total.get("input_tokens", 0)
                    dj_out = dj_total.get("output_tokens", 0)
                    if is_valid_number(dj_inp):
                        combo_token_stats[combo_key]["input"] += int(dj_inp)
                    if is_valid_number(dj_out):
                        combo_token_stats[combo_key]["output"] += int(dj_out)

        tokens_by_combo = {
            combo_key: {
                "input": stats["input"],
                "output": stats["output"],
                "total": stats["input"] + stats["output"],
            }
            for combo_key, stats in combo_token_stats.items()
        }

        tokens = {
            # Total tokens (sum across all results)
            "total_input": total_input_sum,
            "total_input_std": total_input_std,
            "total_output": total_output_sum,
            "total_output_std": total_output_std,
            # Template tokens (sum)
            "template_input": template_input_sum,
            "template_input_std": template_input_std,
            "template_output": template_output_sum,
            "template_output_std": template_output_std,
            # Rubric tokens (sum)
            "rubric_input": rubric_input_sum,
            "rubric_input_std": rubric_input_std,
            "rubric_output": rubric_output_sum,
            "rubric_output_std": rubric_output_std,
            # Deep judgment tokens (sum)
            "deep_judgment_input": dj_input_sum if num_with_judgment > 0 else None,
            "deep_judgment_input_std": dj_input_std if num_with_judgment > 0 else None,
            "deep_judgment_output": dj_output_sum if num_with_judgment > 0 else None,
            "deep_judgment_output_std": dj_output_std if num_with_judgment > 0 else None,
            # Median tokens per question (aggregated over questions and replicates)
            "median_per_question_input": per_q_input_median,
            "median_per_question_input_std": per_q_input_std,
            "median_per_question_output": per_q_output_median,
            "median_per_question_output_std": per_q_output_std,
        }

        return tokens, tokens_by_combo

    def _calculate_completion_by_combo(
        self,
    ) -> dict[tuple[str, str, tuple[str, ...] | None], dict[str, Any]]:
        """Calculate completion status by model combination."""
        from collections import defaultdict

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

        return {
            combo_key: {
                "total": stats["total"],
                "completed": stats["completed"],
                "completion_pct": (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            }
            for combo_key, stats in combo_stats.items()
        }

    def _calculate_rubric_traits(self, questions: set[str], num_with_rubric: int) -> dict[str, Any] | None:
        """Calculate rubric trait breakdown (global vs question-specific by type)."""
        if num_with_rubric == 0:
            return None

        from collections import defaultdict

        trait_questions: dict[str, set[str]] = defaultdict(set)
        trait_types: dict[str, str] = {}

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

        return {
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

    def _calculate_template_pass_rates(
        self, num_with_template: int
    ) -> tuple[dict[tuple[str, str, tuple[str, ...] | None], dict[str, Any]] | None, dict[str, Any] | None]:
        """Calculate template pass rates by combo and overall.

        Returns:
            Tuple of (pass_by_combo, pass_overall) or (None, None) if no templates.
        """
        if num_with_template == 0:
            return None, None

        from collections import defaultdict

        combo_pass_stats: dict[tuple[str, str, tuple[str, ...] | None], dict[str, int]] = defaultdict(
            lambda: {"total": 0, "passed": 0}
        )

        for result in self.results:
            if result.template and result.template.template_verification_performed:
                mcp_servers = result.template.answering_mcp_servers if result.template.answering_mcp_servers else None
                mcp_key = tuple(sorted(mcp_servers)) if mcp_servers else None
                combo_key = (result.metadata.answering_model, result.metadata.parsing_model, mcp_key)

                combo_pass_stats[combo_key]["total"] += 1
                if result.template.verify_result:
                    combo_pass_stats[combo_key]["passed"] += 1

        template_pass_by_combo = {
            combo_key: {
                "total": stats["total"],
                "passed": stats["passed"],
                "pass_pct": (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            }
            for combo_key, stats in combo_pass_stats.items()
        }

        overall_passed = sum(s["passed"] for s in combo_pass_stats.values())
        overall_total = sum(s["total"] for s in combo_pass_stats.values())
        template_pass_overall = {
            "total": overall_total,
            "passed": overall_passed,
            "pass_pct": (overall_passed / overall_total * 100) if overall_total > 0 else 0,
        }

        return template_pass_by_combo, template_pass_overall

    def _calculate_replicate_stats(self, replicates: set[int], num_with_template: int) -> dict[str, Any] | None:
        """Calculate replicate statistics for template pass rates.

        Returns:
            Dict with replicate_pass_rates and replicate_summary, or None.
        """
        if len(replicates) <= 1 or num_with_template == 0:
            return None

        import statistics
        from collections import defaultdict

        rep_stats_dict: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})

        for result in self.results:
            if result.template and result.template.template_verification_performed:
                rep_num = result.metadata.replicate
                if rep_num is not None:
                    rep_stats_dict[rep_num]["total"] += 1
                    if result.template.verify_result:
                        rep_stats_dict[rep_num]["passed"] += 1

        replicate_pass_rates: dict[int, dict[str, Any]] = {}
        pass_rates_list: list[float] = []

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

        replicate_summary: dict[str, float] | None = None
        if len(pass_rates_list) > 0:
            mean = statistics.mean(pass_rates_list)
            std = statistics.stdev(pass_rates_list) if len(pass_rates_list) > 1 else 0.0
            replicate_summary = {"mean": mean, "std": std}

        if replicate_pass_rates and replicate_summary is not None:
            return {
                "replicate_pass_rates": replicate_pass_rates,
                "replicate_summary": replicate_summary,
            }
        return None

    def _calculate_tool_usage_stats(self) -> dict[str, Any] | None:
        """Calculate tool usage aggregation across all results."""
        from collections import defaultdict

        tool_total_calls: dict[str, int] = defaultdict(int)
        tool_trace_counts: dict[str, int] = defaultdict(int)
        results_with_tools = 0

        for result in self.results:
            if result.template and result.template.agent_metrics:
                tool_counts = result.template.agent_metrics.get("tool_call_counts")
                if tool_counts:
                    results_with_tools += 1
                    for tool_name, count in tool_counts.items():
                        tool_total_calls[tool_name] += count
                        tool_trace_counts[tool_name] += 1

        if results_with_tools == 0:
            return None

        return {
            "tools": {
                name: {
                    "total_calls": tool_total_calls[name],
                    "traces_using": tool_trace_counts[name],
                    "avg_calls_per_trace": tool_total_calls[name] / results_with_tools,
                }
                for name in sorted(tool_total_calls.keys())
            },
            "total_traces_with_tools": results_with_tools,
            "total_tool_calls": sum(tool_total_calls.values()),
        }

    def _calculate_trace_length_stats(self) -> dict[str, Any] | None:
        """Calculate trace length statistics (iterations = AI message cycles)."""
        iteration_counts: list[int] = []

        for result in self.results:
            if result.template and result.template.agent_metrics:
                iterations = result.template.agent_metrics.get("iterations")
                if iterations is not None:
                    iteration_counts.append(iterations)

        if not iteration_counts:
            return None

        iterations_sorted = sorted(iteration_counts)
        n = len(iterations_sorted)
        median_iterations = (
            iterations_sorted[n // 2] if n % 2 == 1 else (iterations_sorted[n // 2 - 1] + iterations_sorted[n // 2]) / 2
        )
        mean_iterations = sum(iteration_counts) / n
        std_iterations = (sum((x - mean_iterations) ** 2 for x in iteration_counts) / n) ** 0.5

        return {
            "median_iterations": median_iterations,
            "mean_iterations": mean_iterations,
            "std_iterations": std_iterations,
            "min_iterations": min(iteration_counts),
            "max_iterations": max(iteration_counts),
            "num_traces": n,
        }

    # ========================================================================
    # Summary and Statistics - Public Methods
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
            - tokens_by_combo: Dict mapping (answering, parsing, mcp_tuple) to
                              {input, output, total}

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

            **Tool Usage Statistics** (if agents used):
            - tool_usage_stats: Dict with {tools, total_traces_with_tools, total_tool_calls}
              where tools maps tool name to {total_calls, traces_using, avg_calls_per_trace}

            **Trace Length Statistics** (if agents used):
            - trace_length_stats: Dict with {median_iterations, mean_iterations, std_iterations, min_iterations, max_iterations, num_traces}
              Iteration counts (AI message cycles) for agent traces

        Example:
            ```python
            summary = result_set.get_summary()
            print(f"Total results: {summary['num_results']}")
            print(f"Questions: {summary['num_questions']}")
            print(f"Total tokens: {summary['tokens']['total_input']} input")
            ```
        """
        # Calculate all components using helper methods
        basic_counts = self._calculate_basic_counts()

        num_completed = basic_counts["num_completed"]
        num_with_template = basic_counts["num_with_template"]
        num_with_rubric = basic_counts["num_with_rubric"]
        num_with_judgment = basic_counts["num_with_judgment"]
        questions = basic_counts["questions"]
        models = basic_counts["models"]
        parsing_models = basic_counts["parsing_models"]
        replicates = basic_counts["replicates"]
        total_execution_time = basic_counts["total_execution_time"]

        # Token statistics
        tokens, tokens_by_combo = self._calculate_token_stats(num_with_judgment)

        # Completion status by model combination
        completion_by_combo = self._calculate_completion_by_combo()

        # Rubric trait breakdown
        rubric_traits = self._calculate_rubric_traits(questions, num_with_rubric)

        # Template pass rates
        template_pass_by_combo, template_pass_overall = self._calculate_template_pass_rates(num_with_template)

        # Replicate statistics
        replicate_stats = self._calculate_replicate_stats(replicates, num_with_template)

        # Tool usage statistics
        tool_usage_stats = self._calculate_tool_usage_stats()

        # Trace length statistics
        trace_length_stats = self._calculate_trace_length_stats()

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
            "tokens": tokens,
            # Token usage by model combination
            "tokens_by_combo": tokens_by_combo,
            # Completion status
            "completion_by_combo": completion_by_combo,
            # Evaluation types
            "rubric_traits": rubric_traits,
            # Template pass rates
            "template_pass_by_combo": template_pass_by_combo,
            "template_pass_overall": template_pass_overall,
            # Replicate statistics
            "replicate_stats": replicate_stats,
            # Tool usage statistics (only present when agents are used)
            "tool_usage_stats": tool_usage_stats,
            # Trace length statistics
            "trace_length_stats": trace_length_stats,
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

            # Global traits breakdown
            global_traits = traits.get("global_traits", {})
            global_llm = global_traits.get("llm", {}).get("count", 0)
            global_regex = global_traits.get("regex", {}).get("count", 0)
            global_callable = global_traits.get("callable", {}).get("count", 0)
            global_metric = global_traits.get("metric", {}).get("count", 0)
            global_total = global_llm + global_regex + global_callable + global_metric

            if global_total > 0:
                lines.append(f"    Global: {global_total}")
                breakdown = []
                if global_llm > 0:
                    breakdown.append(f"LLM: {global_llm}")
                if global_regex > 0:
                    breakdown.append(f"Regex: {global_regex}")
                if global_callable > 0:
                    breakdown.append(f"Callable: {global_callable}")
                if global_metric > 0:
                    breakdown.append(f"Metric: {global_metric}")
                if breakdown:
                    lines.append(f"      └─ {', '.join(breakdown)}")

            # Question-specific traits breakdown
            qs_traits = traits.get("question_specific_traits", {})
            qs_llm = qs_traits.get("llm", {}).get("count", 0)
            qs_regex = qs_traits.get("regex", {}).get("count", 0)
            qs_callable = qs_traits.get("callable", {}).get("count", 0)
            qs_metric = qs_traits.get("metric", {}).get("count", 0)
            qs_total = qs_llm + qs_regex + qs_callable + qs_metric

            if qs_total > 0:
                lines.append(f"    Question-Specific: {qs_total}")
                breakdown = []
                if qs_llm > 0:
                    breakdown.append(f"LLM: {qs_llm}")
                if qs_regex > 0:
                    breakdown.append(f"Regex: {qs_regex}")
                if qs_callable > 0:
                    breakdown.append(f"Callable: {qs_callable}")
                if qs_metric > 0:
                    breakdown.append(f"Metric: {qs_metric}")
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
        replicate_stats = summary.get("replicate_stats")
        if replicate_stats and replicate_stats.get("replicate_pass_rates"):
            lines.append("")
            lines.append("  === REPLICATE STATISTICS ===")
            lines.append("  Template Pass Rate by Replicate:")

            for rep_num in sorted(replicate_stats["replicate_pass_rates"].keys()):
                stats = replicate_stats["replicate_pass_rates"][rep_num]
                lines.append(
                    f"    Replicate {rep_num}: {stats['passed']}/{stats['total']} passed ({stats['pass_pct']:.1f}%)"
                )

            if replicate_stats.get("replicate_summary"):
                rep_sum = replicate_stats["replicate_summary"]
                lines.append(f"  Summary: mean={rep_sum['mean']:.3f}, std={rep_sum['std']:.3f}")

        lines.append(")")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (same as repr for developer-friendly output)."""
        return self.__repr__()
