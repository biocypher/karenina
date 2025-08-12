#!/usr/bin/env python3
"""
Example usage of the Karenina benchmark system.

This example demonstrates how to:
1. Create benchmarks programmatically
2. Add questions and answer templates
3. Save and load JSON-LD benchmark files
4. Full compatibility with GUI exports
"""

from pathlib import Path

from karenina.benchmark.benchmark import Benchmark
from karenina.schemas.rubric_class import RubricTrait


def main():
    # ========================================
    # 1. Create a new benchmark from scratch
    # ========================================
    print("Creating a new benchmark...")
    benchmark = Benchmark.create(
        name="Python Programming Knowledge Test",
        description="A benchmark to test understanding of Python programming concepts",
        version="1.0.0",
        creator="Karenina Example",
    )

    # ========================================
    # 2. Add questions manually
    # ========================================
    print("Adding questions...")

    # Simple question with auto-generated template
    q1_id = benchmark.add_question(
        question="What is a Python decorator?",
        raw_answer="A decorator is a design pattern that allows you to modify or extend the behavior of functions or classes without permanently modifying them.",
    )
    print(f"  Added question 1: {q1_id}")

    # Question with custom answer template
    # Note: No imports needed - they are provided by the execution environment
    template_code = '''class Answer(BaseAnswer):
    """Answer template for list comprehension question."""

    definition: str = Field(description="Definition of list comprehension")
    syntax_example: str = Field(description="Example of list comprehension syntax")
    advantages: List[str] = Field(description="Advantages over traditional loops")
    use_cases: List[str] = Field(description="Common use cases")

    def verify(self) -> bool:
        # Check that key concepts are mentioned
        has_definition = len(self.definition) > 20
        has_syntax = "[" in self.syntax_example and "]" in self.syntax_example
        has_advantages = len(self.advantages) >= 2
        has_use_cases = len(self.use_cases) >= 1

        return has_definition and has_syntax and has_advantages and has_use_cases
'''

    q2_id = benchmark.add_question(
        question="Explain Python list comprehensions with examples",
        raw_answer="List comprehensions provide a concise way to create lists. They consist of brackets containing an expression followed by a for clause.",
        answer_template=template_code,
        finished=True,  # Mark as ready for verification
        author={"name": "Example Author", "email": "author@example.com"},
        custom_metadata={"difficulty": "intermediate", "topic": "data structures"},
    )
    print(f"  Added question 2: {q2_id}")

    # ========================================
    # 3. Add rubrics for evaluation
    # ========================================
    print("Adding rubrics...")

    # Global rubric traits (apply to all questions)
    benchmark.add_global_rubric_trait(
        RubricTrait(name="clarity", description="Is the explanation clear and easy to understand?", kind="boolean")
    )

    benchmark.add_global_rubric_trait(
        RubricTrait(
            name="completeness",
            description="How complete is the answer on a scale of 1-5?",
            kind="score",
            min_score=1,
            max_score=5,
        )
    )
    print("  Added global rubric traits: clarity, completeness")

    # Question-specific rubric trait
    benchmark.add_question_rubric_trait(
        q2_id,
        RubricTrait(name="code_examples", description="Does the answer include working code examples?", kind="boolean"),
    )
    print(f"  Added question-specific rubric for {q2_id}")

    # ========================================
    # 4. Save benchmark to JSON-LD file
    # ========================================
    output_path = Path("example_benchmark.jsonld")
    benchmark.save(output_path)
    print(f"\n✅ Saved benchmark to {output_path}")
    print("   This file can be loaded directly in the Karenina GUI!")

    # ========================================
    # 5. Load benchmark from file
    # ========================================
    print(f"\nLoading benchmark from {output_path}...")
    loaded_benchmark = Benchmark.load(output_path)

    # Display loaded data
    print(f"Loaded benchmark: {loaded_benchmark._checkpoint.name}")
    print(f"  Version: {loaded_benchmark._checkpoint.version}")
    print(f"  Questions: {len(loaded_benchmark.get_question_ids())}")

    global_rubric = loaded_benchmark.get_global_rubric()
    if global_rubric:
        print(f"  Global rubric traits: {len(global_rubric.traits)}")

    # ========================================
    # 6. Get finished templates for verification
    # ========================================
    print("\nFinished templates ready for verification:")
    templates = loaded_benchmark.get_finished_templates()
    for template in templates:
        print(f"  - {template.question_preview}")

    # ========================================
    # 7. Example: Run verification (if configured)
    # ========================================
    print("\nTo run verification, use:")
    print("""
    from karenina.benchmark.models import VerificationConfig

    config = VerificationConfig(
        answering_model_provider="openai",
        answering_model_name="gpt-4",
        parsing_model_provider="openai",
        parsing_model_name="gpt-4",
        temperature=0.7
    )

    results = loaded_benchmark.run_verification(config)
    """)

    # ========================================
    # 8. Validate benchmark
    # ========================================
    is_valid, msg = loaded_benchmark.validate()
    print(f"\nBenchmark validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if not is_valid:
        print(f"  Error: {msg}")

    print("\n" + "=" * 50)
    print("Example complete! The benchmark file can be:")
    print("  - Loaded in the Karenina GUI for visual editing")
    print("  - Used for benchmark verification with LLMs")
    print("  - Shared with others as a standard JSON-LD file")
    print("  - Version controlled with Git")
    print("=" * 50)


if __name__ == "__main__":
    main()
