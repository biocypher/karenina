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

    # ========================================
    # 9. Metadata management
    # ========================================
    print("\nMetadata management:")

    # Set benchmark metadata
    loaded_benchmark.id = "python-knowledge-benchmark-v1"
    loaded_benchmark.version = "1.1.0"
    loaded_benchmark.creator = "Enhanced Example"

    # Set custom properties
    loaded_benchmark.set_multiple_custom_properties(
        {
            "domain": "computer_science",
            "difficulty_level": "intermediate",
            "estimated_duration_minutes": 30,
            "requires_python_version": "3.8+",
            "tags": ["python", "programming", "fundamentals"],
            "license": "CC-BY-4.0",
        }
    )

    print(f"  Benchmark ID: {loaded_benchmark.id}")
    print(f"  Custom properties: {len(loaded_benchmark.get_all_custom_properties())}")
    print(f"  Domain: {loaded_benchmark.get_custom_property('domain')}")
    print(f"  Tags: {loaded_benchmark.get_custom_property('tags')}")

    # Question metadata management
    print("\nQuestion metadata management:")
    q_ids = loaded_benchmark.get_question_ids()
    if q_ids:
        first_q_id = q_ids[0]

        # Get comprehensive question metadata
        q_metadata = loaded_benchmark.get_question_metadata(first_q_id)
        print(f"  Question ID: {q_metadata['id']}")
        print(f"  Question: {q_metadata['question'][:50]}...")
        print(f"  Has custom author: {q_metadata['author'] is not None}")
        print(f"  Finished: {q_metadata['finished']}")

        # Update question metadata
        loaded_benchmark.update_question_metadata(
            first_q_id,
            author={"name": "Enhanced Example", "role": "Demonstration"},
            custom_metadata={"reviewed": True, "review_date": "2024-01-15", "complexity_score": 8.5},
        )

        # Access specific metadata
        author = loaded_benchmark.get_question_author(first_q_id)
        complexity = loaded_benchmark.get_question_custom_property(first_q_id, "complexity_score")
        timestamps = loaded_benchmark.get_question_timestamps(first_q_id)

        print(f"  Updated author: {author['name']} ({author['role']})")
        print(f"  Complexity score: {complexity}")
        print(f"  Created: {timestamps['created'][:19]}")  # Show date/time only

    # ========================================
    # 10. Demonstrate enhanced features
    # ========================================
    print("\n" + "=" * 50)
    print("ENHANCED FEATURES DEMO")
    print("=" * 50)

    # Magic methods and properties
    print(f"String representation: {loaded_benchmark}")
    print(f"Length (question count): {len(loaded_benchmark)}")
    print(f"Is complete: {loaded_benchmark.is_complete}")
    print(f"Progress: {loaded_benchmark.get_progress():.1f}%")

    # Filtering and search
    finished_questions = loaded_benchmark.filter_questions(finished=True)
    questions_with_templates = loaded_benchmark.filter_questions(has_template=True)
    python_questions = loaded_benchmark.search_questions("Python")

    print("\nFiltering results:")
    print(f"  Finished questions: {len(finished_questions)}")
    print(f"  Questions with templates: {len(questions_with_templates)}")
    print(f"  Questions mentioning 'Python': {len(python_questions)}")

    # Health report
    health = loaded_benchmark.get_health_report()
    print("\nHealth Report:")
    print(f"  Score: {health['health_score']}/100 ({health['health_status']})")
    print(f"  Recommendations: {len(health['recommendations'])}")
    for i, rec in enumerate(health["recommendations"][:3], 1):
        print(f"    {i}. {rec}")

    # Export capabilities
    print("\nExport capabilities:")
    summary = loaded_benchmark.get_summary()
    print(f"  Summary keys: {list(summary.keys())}")

    csv_export = loaded_benchmark.to_csv()
    print(f"  CSV export: {len(csv_export)} characters")

    # Template management
    q_ids = loaded_benchmark.get_question_ids()
    if q_ids:
        first_q_id = q_ids[0]
        print("\nTemplate management:")
        print(f"  First question has template: {loaded_benchmark.has_template(first_q_id)}")
        if loaded_benchmark.has_template(first_q_id):
            template = loaded_benchmark.get_template(first_q_id)
            print(f"  Template length: {len(template)} characters")

    print("\n" + "=" * 50)
    print("Example complete! The enhanced benchmark class provides:")
    print("  ✓ Intuitive magic methods (__str__, __len__, __iter__, etc.)")
    print("  ✓ Property accessors for common attributes")
    print("  ✓ Comprehensive metadata management (benchmark + question level)")
    print("  ✓ Custom properties system for extensible metadata")
    print("  ✓ Author and source tracking for questions")
    print("  ✓ Powerful filtering and search capabilities")
    print("  ✓ Comprehensive health and readiness checks")
    print("  ✓ Multiple export formats (dict, CSV, markdown)")
    print("  ✓ Bulk operations and advanced template management")
    print("  ✓ Template validation and syntax checking")
    print("  ✓ Timestamp tracking and modification history")
    print("  ✓ Full compatibility with GUI JSON-LD format")
    print("=" * 50)


if __name__ == "__main__":
    main()
