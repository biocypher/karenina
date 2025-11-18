# Rubric-Related Test Fixes Checklist

**Status**: Phase 9 complete, major rubric test fixes applied
**Progress**:
- Started: 1246/1430 passing (87.1%), 132 failures, ~42 rubric-related
- After Phase 9a: 1284/1430 passing (89.8%), 94 failures, 35 rubric-related
- **Current: 1312/1430 passing (91.7%), 66 failures, ~20 rubric-related**
- **Fixed: 66 failures total (132→66), 38 rubric-specific tests**

**Goal**: Fix all rubric-related test failures

## Categories of Failures

### 1. Merge Rubrics Function Issues (13 tests) ✅ FIXED
Tests for the `merge_rubrics()` function that need to handle the new split trait types.

- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_global_only`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_question_only`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_both_present`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_preserves_trait_properties`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_empty_rubrics`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_mixed_empty`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_name_conflicts`
- [x] `test_question_rubrics.py::TestQuestionRubrics::test_merge_rubrics_both_none`
- [x] `test_regex_trait.py::TestMergeRubricsWithRegexTraits::test_merge_rubrics_with_regex_traits`
- [x] `test_regex_trait.py::TestMergeRubricsWithRegexTraits::test_merge_rubrics_name_conflicts_across_types`
- [x] `test_regex_trait.py::TestMergeRubricsWithRegexTraits::test_merge_empty_rubrics_with_regex_traits`
- [x] `test_callable_trait.py::TestMergeRubricsWithCallableTraits::test_merge_rubrics_with_callable_traits`
- [x] `test_callable_trait.py::TestMergeRubricsWithCallableTraits::test_merge_rubrics_name_conflicts_across_types`

**Fix Applied**: Replaced `.traits` with `.llm_traits` in all test assertions

### 2. Rubric Evaluator Issues (7 tests) ✅ MOSTLY FIXED
Tests for RubricEvaluator that may reference old trait structure or manual_traits.

- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluate_rubric_success`
- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluate_rubric_partial_response`
- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluate_rubric_mixed_types`
- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluate_rubric_integration`
- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluator_initialization`
- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluate_empty_rubric`
- [x] `test_rubric_evaluator.py::TestRubricEvaluator::test_evaluator_with_different_providers`
- [ ] `test_rubric_evaluator.py::TestRubricEvaluatorEdgeCases::test_evaluator_initialization_manual_no_provider` (NOT RUBRIC-RELATED)
- [ ] `test_rubric_evaluator.py::TestRubricEvaluatorEdgeCases::test_evaluator_handles_different_interface_types` (NOT RUBRIC-RELATED)

**Fix Applied**: Updated tests to unpack tuple return value `(results, usage)`

### 3. Rubric Schema Tests (2 tests) ✅ FIXED
Schema round-trip and workflow tests.

- [x] `test_rubric_schemas.py::TestRubricIntegration::test_complete_rubric_workflow`
- [x] `test_rubric_schemas.py::TestRubricIntegration::test_rubric_json_round_trip`

**Fix Applied**: Replaced `.traits` with `.llm_traits` in test assertions

### 4. TaskEval Simplified Interface (3 tests) ✅ PARTIALLY FIXED
Tests using `manual_traits` parameter in `Rubric()`.

- [x] `test_task_eval_simplified_interface.py::TestTaskEvalSimplifiedInterface::test_rubric_only_mode_string_trace`
- [ ] `test_task_eval_simplified_interface.py::TestTaskEvalSimplifiedInterface::test_rubric_only_mode_dict_trace`
- [x] `test_task_eval_simplified_interface.py::TestTaskEvalSimplifiedInterface::test_step_specific_dict_traces`
- [ ] `test_task_eval_simplified_interface.py::TestTaskEvalSimplifiedInterface::test_template_and_rubric_mode`

**Fix Applied**: Replaced `manual_traits=` with `regex_traits=`

### 5. Export/DataFrame Tests (6 tests) ✅ FIXED
Tests for exporting rubric data to CSV/JSON/DataFrame.

- [x] `test_exporter.py::TestExportVerificationResultsCSV::test_consolidate_question_specific_rubrics_with_global_rubric`
- [x] `test_exporter.py::TestExportVerificationResultsCSV::test_all_global_rubrics_when_no_question_specific_exist`
- [x] `test_exporter.py::TestExportVerificationResultsCSV::test_no_global_rubric_all_rubrics_become_question_specific`
- [x] `test_exporter.py::TestExportVerificationResultsJSON::test_json_export_preserves_all_rubric_data`
- [x] `test_rubric_results_dataframe.py::test_to_dataframe_manual_traits`
- [x] `test_rubric_results_dataframe.py::test_to_dataframe_all_trait_types`

**Fix Applied**: Replaced `manual_trait_scores` with `regex_trait_scores` in all test fixtures

### 6. Evaluation Mode Tests (3 tests) - REMAINING
Tests for different evaluation modes (template_and_rubric, rubric_only).

- [ ] `test_evaluation_modes.py::TestTemplateAndRubricMode::test_template_and_rubric_performs_both`
- [ ] `test_evaluation_modes.py::TestRubricOnlyMode::test_rubric_only_skips_template_verification`
- [ ] `test_evaluation_modes.py::TestRubricOnlyMode::test_rubric_only_with_abstention_check`

**Root Cause**: Tests may need verification result field updates

### 7. TaskEval Tests (9 tests) ✅ FIXED
TaskEval rubric-related tests.

- [x] `test_task_eval.py::TestTaskEvalRubrics::test_merge_rubrics_single`
- [x] `test_task_eval.py::TestTaskEvalRubrics::test_merge_rubrics_multiple`
- [x] `test_task_eval.py::TestTaskEvalRubricEvaluation::test_rubric_evaluation_with_answer_template`
- [x] `test_task_eval.py::TestTaskEvalRubricEvaluation::test_rubric_trait_extraction_from_template`
- [x] `test_task_eval.py::TestTaskEvalRubricEvaluation::test_multiple_questions_with_same_rubric`
- [x] `test_task_eval_aggregation.py::test_aggregate_rubric_results_all_trait_types`
- [x] `test_task_eval_aggregation.py::test_aggregate_rubric_results_single_replicate`
- [x] `test_task_eval_metric_traits.py::TestTaskEvalMetricTraits::test_metric_trait_with_other_rubric_types`
- [x] `test_taskeval_formatting.py::TestTaskEvalFormatting::test_step_eval_format_rubric_scores`

**Fix Applied**:
- Replaced `.traits` with `.llm_traits` in merge tests
- Updated evaluation_rubric dicts (traits→llm_traits, manual_traits→regex_traits, added callable_traits)
- Updated test_taskeval_formatting to use VerificationResult structure
- Fixed manual_traits parameter to regex_traits in metric traits test

### 8. Verification Tests (5 tests) - REMAINING
Verification stage and result tests.

- [ ] `test_verification/test_rubric_finalize_stages.py::TestRubricEvaluationStage::test_metric_trait_evaluation`
- [ ] `test_verification/test_rubric_finalize_stages.py::TestFinalizeResultStage::test_finalize_with_deep_judgment`
- [ ] `test_verification/test_rubric_finalize_stages.py::TestFinalizeResultStage::test_template_and_rubric_mode_flags`
- [ ] `test_verification/test_rubric_finalize_stages.py::TestFinalizeResultStage::test_rubric_only_mode_flags`
- [ ] `test_verification/test_stage_orchestrator.py::TestStageOrchestratorConfiguration::test_empty_rubric_not_included`
- [ ] `test_verification/test_verification_regression.py::TestBasicRegression::test_template_with_rubric_equivalence`

**Root Cause**: Verification result field structure changes

### 9. Other Integration Tests (5 tests) - REMAINING
Misc integration tests.

- [ ] `test_benchmark_verification_integration.py::TestBenchmarkVerificationIntegration::test_rubric_merging`
- [ ] `test_answer_template_class_support.py::TestAnswerClassIntegration::test_answer_class_with_rubrics`
- [ ] `storage/test_queries.py::TestRubricScoresAggregate::test_get_rubric_scores_aggregate_no_scores`
- [ ] `test_verification_result_set/test_rubric_results.py::TestRubricResultsDataAccess::test_get_manual_trait_scores`
- [ ] `test_verification_result_set/test_rubric_results.py::TestRubricResultsDataAccess::test_get_all_trait_scores`
- [ ] `test_verification_result_set/test_rubric_results.py::TestRubricResultsAggregation::test_aggregate_manual_traits_majority_vote`
- [ ] `test_verification_result_set/test_rubric_results.py::TestRubricResultsExtensibility::test_register_custom_aggregator`

**Root Cause**: Various integration issues

---

## Fix Strategy

### Priority 1: Fix Core Functions
1. Fix `merge_rubrics()` function in `schemas/domain/rubric.py`
2. Fix RubricEvaluator to handle split trait types
3. Fix TaskEval simplified interface tests (remove manual_traits usage)

### Priority 2: Fix Schema Tests
4. Update schema workflow tests to use `.llm_traits`

### Priority 3: Fix Export/DataFrame
5. Update export code to handle split trait types
6. Update DataFrame conversion for split traits

### Priority 4: Fix Integration Tests
7. Fix evaluation mode tests
8. Fix remaining integration tests

---

## Investigation Commands

```bash
# Run specific category
uv run pytest tests/test_question_rubrics.py -xvs

# Check merge_rubrics implementation
grep -A20 "def merge_rubrics" src/karenina/schemas/domain/rubric.py

# Check RubricEvaluator
grep -A10 "class RubricEvaluator" src/karenina/benchmark/verification/evaluators/rubric_evaluator.py
```

---

## Notes
- The `Rubric` model now has separate fields: `llm_traits`, `regex_traits`, `callable_traits`, `metric_traits`
- Old `traits` field renamed to `llm_traits`
- Old `manual_traits` parameter no longer exists - split into `regex_traits` and `callable_traits`
- All serialized rubrics will have the new field names in their JSON representation
