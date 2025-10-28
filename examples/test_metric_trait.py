"""Test metric trait evaluation with anus disease child terms question.

This example tests the metric trait evaluation functionality end-to-end:
1. Creates a question with a metric rubric trait (tp_only mode)
2. Defines TP instructions for the four child terms
3. Runs verification with gpt-4o-mini
4. Prints the confusion matrix and computed metrics
"""

from karenina.benchmark.benchmark import Benchmark
from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.verification.orchestrator import run_question_verification
from karenina.schemas.question_class import Question
from karenina.schemas.rubric_class import MetricRubricTrait, Rubric

# Create question
question_text = "What are the child terms of anus disease (EFO_0009660)? Provide names."
correct_answer = "proctitis, anal neoplasm, anal polyp, imperforate anus"

question = Question(
    question=question_text,
    raw_answer=correct_answer,
    tags=["Disease"],
)

# Define the answer template (Pydantic verification)
template_code = """class Answer(BaseAnswer):
    mentions_proctitis: bool = Field(
        description="Answer with true if the response mentions proctitis; otherwise, answer false."
    )
    mentions_anal_neoplasm: bool = Field(
        description="Answer with true if the response mentions anal neoplasm; otherwise, answer false."
    )
    mentions_anal_polyp: bool = Field(
        description="Answer with true if the response mentions anal polyp; otherwise, answer false."
    )
    mentions_imperforate_anus: bool = Field(
        description="Answer with true if the response mentions imperforate anus; otherwise, answer false."
    )

    def model_post_init(self, __context):
        self.correct = {
            "mentions_proctitis": True,
            "mentions_anal_neoplasm": True,
            "mentions_anal_polyp": True,
            "mentions_imperforate_anus": True,
        }

    def verify(self) -> bool:
        return (
            self.mentions_proctitis == self.correct["mentions_proctitis"]
            and self.mentions_anal_neoplasm == self.correct["mentions_anal_neoplasm"]
            and self.mentions_anal_polyp == self.correct["mentions_anal_polyp"]
            and self.mentions_imperforate_anus == self.correct["mentions_imperforate_anus"]
        )
"""

# Create metric rubric trait (TP-only mode)
metric_trait = MetricRubricTrait(
    name="Child Terms Extraction",
    description="Evaluate recall of child term extraction",
    evaluation_mode="tp_only",  # Only TP instructions, FP = anything else
    metrics=["recall", "precision", "f1"],  # Compute recall, precision, and F1
    tp_instructions=[
        "proctitis",
        "anal neoplasm",
        "anal polyp",
        "imperforate anus",
    ],
    tn_instructions=[],  # Not used in tp_only mode
    repeated_extraction=True,
)

# Create rubric with the metric trait
rubric = Rubric(
    traits=[],  # No LLM traits
    manual_traits=[],  # No manual traits
    metric_traits=[metric_trait],  # Only metric trait
)

# Create benchmark
benchmark = Benchmark.create(
    name="Anus Disease Child Terms Test",
    description="Test metric trait evaluation with anus disease child terms",
    version="1.0.0",
    creator="Test",
)

# Add question with template and rubric
question_id = benchmark.add_question(
    question=question,
    answer_template=template_code,
    finished=True,
)

# Set question-specific rubric
benchmark.set_question_rubric(question_id, rubric)

# Configure models first
answering_model = ModelConfig(
    id="gpt41mini-answering",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain",
    system_prompt="You are a helpful assistant that answers questions accurately and concisely.",
)

parsing_model = ModelConfig(
    id="gpt41mini-parsing",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain",
    system_prompt="You are an expert at extracting structured information from text.",
)

# Configure verification
config = VerificationConfig(
    answering_models=[answering_model],
    parsing_models=[parsing_model],
    replicate_count=1,
    rubric_enabled=True,
    embedding_check_enabled=False,
    regex_validation_enabled=False,
    abstention_detection_enabled=False,
)

# Get the finished template
templates = benchmark.get_finished_templates()
if not templates:
    print("❌ No finished templates found!")
    exit(1)

template = templates[0]
print(f"📋 Question: {question_text}")
print(f"✅ Correct answer: {correct_answer}")
print(f"\n🎯 Metric Trait: {metric_trait.name}")
print(f"   Mode: {metric_trait.evaluation_mode}")
print(f"   Metrics: {', '.join(metric_trait.metrics)}")
print(f"   TP Instructions ({len(metric_trait.tp_instructions)}):")
for i, tp in enumerate(metric_trait.tp_instructions, 1):
    print(f"     {i}. {tp}")

print("\n" + "=" * 80)
print("🚀 Running verification with gpt-4.1-mini...")
print("=" * 80 + "\n")

# Run verification
results_dict = run_question_verification(
    question_id=question_id,
    question_text=question_text,
    template_code=template_code,
    config=config,
    rubric=rubric,
)

# Get the first (and only) result
if not results_dict:
    print("❌ No results returned!")
    exit(1)

result_key = list(results_dict.keys())[0]
result = results_dict[result_key]

print("\n" + "=" * 80)
print("📊 VERIFICATION RESULTS")
print("=" * 80 + "\n")

print(f"Result Key: {result_key}")
print(f"✅ Completed without errors: {result.completed_without_errors}")
print(f"🔍 Verification result: {result.verify_result}")
print(f"📈 Granular result: {result.verify_granular_result}")

# Print the raw LLM response
print(f"\n💬 Raw LLM Response:")
print(f"   {result.raw_llm_response}")

# Print metric trait results
if result.verify_metric_confusion_lists and result.verify_metric_results:
    print("\n" + "=" * 80)
    print("🎯 METRIC TRAIT EVALUATION")
    print("=" * 80 + "\n")

    trait_name = metric_trait.name
    confusion = result.verify_metric_confusion_lists.get(trait_name, {})
    metrics = result.verify_metric_results.get(trait_name, {})

    print(f"Trait: {trait_name}")
    print(f"Mode: {metric_trait.evaluation_mode}")
    print()

    # Print confusion matrix
    print("📊 Confusion Matrix:")
    print(f"   ✅ True Positives (TP):  {len(confusion.get('tp', []))} items")
    for item in confusion.get("tp", []):
        print(f"      - {item}")

    print(f"   ❌ False Negatives (FN): {len(confusion.get('fn', []))} items")
    for item in confusion.get("fn", []):
        print(f"      - {item}")

    print(f"   ⚠️  False Positives (FP): {len(confusion.get('fp', []))} items")
    for item in confusion.get("fp", []):
        print(f"      - {item}")

    if metric_trait.evaluation_mode == "full_matrix":
        print(f"   ✓  True Negatives (TN):  {len(confusion.get('tn', []))} items")
        for item in confusion.get("tn", []):
            print(f"      - {item}")

    # Print computed metrics
    print("\n📈 Computed Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name.capitalize()}: {metric_value:.3f}")

else:
    print("\n⚠️  No metric trait results found!")
    print("   This means the metric trait evaluation was not performed.")
    print("   Check if rubric.metric_traits is populated and being passed correctly.")

print("\n" + "=" * 80)
print("✨ Test completed!")
print("=" * 80)
