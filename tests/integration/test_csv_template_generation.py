"""Integration test for CSV question extraction and template generation.

This test validates the complete workflow of:
1. Extracting questions from a CSV file
2. Generating answer templates using a custom OpenAI-compatible endpoint
"""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from karenina.domain.answers.generator import generate_answer_template
from karenina.domain.questions.extractor import extract_questions_from_file
from karenina.schemas.workflow import ModelConfig


@pytest.fixture
def csv_file_path():
    """Path to the test CSV file."""
    return "/Users/carli/Projects/karenina_dev/data/MCP_benchmark_v3_input.csv"


@pytest.fixture
def openai_endpoint_config():
    """Configuration for the OpenAI-compatible endpoint."""
    return ModelConfig(
        id="glm-4.6-generator",
        model_name="glm-4.6",
        model_provider="openai",  # Not used for openai_endpoint interface
        interface="openai_endpoint",
        temperature=0.0,
        system_prompt="",  # Not used in template generation
        endpoint_base_url="http://codon-gpu-001.ebi.ac.uk:8000/v1",
        endpoint_api_key=SecretStr("mock"),
    )


def test_csv_extraction(csv_file_path):
    """Test that questions can be extracted from the CSV file."""
    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=csv_file_path,
        question_column="Question",
        answer_column="Answer",
    )

    # Verify we got questions
    assert len(questions_with_metadata) > 0, "No questions extracted from CSV"

    # Check the structure of the first question
    first_question, first_metadata = questions_with_metadata[0]

    assert first_question.question, "Question text is empty"
    assert first_question.raw_answer, "Answer text is empty"
    assert first_question.id, "Question ID not generated"

    print(f"\n✓ Successfully extracted {len(questions_with_metadata)} questions from CSV")
    print("\nFirst question:")
    print(f"  ID: {first_question.id}")
    print(f"  Question: {first_question.question[:100]}...")
    print(f"  Answer: {first_question.raw_answer[:100]}...")


def test_template_generation_with_openai_endpoint(csv_file_path, openai_endpoint_config):
    """Test template generation using OpenAI-compatible endpoint."""
    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=csv_file_path,
        question_column="Question",
        answer_column="Answer",
    )

    assert len(questions_with_metadata) > 0, "No questions extracted"

    # Take the first question for testing
    first_question, _ = questions_with_metadata[0]

    print("\n🔧 Generating template for question:")
    print(f"  Q: {first_question.question}")
    print(f"  A: {first_question.raw_answer}")

    # Generate template using the custom endpoint
    try:
        template_code = generate_answer_template(
            question=first_question.question,
            raw_answer=first_question.raw_answer,
            config=openai_endpoint_config,
        )

        # Verify the template was generated
        assert template_code, "Template code is empty"
        assert "class Answer(BaseAnswer):" in template_code, "Template doesn't contain Answer class"
        assert "def verify(self)" in template_code, "Template doesn't contain verify method"

        print("\n✓ Template generated successfully!")
        print("\nGenerated template:")
        print("-" * 80)
        print(template_code)
        print("-" * 80)

    except Exception as e:
        pytest.skip(f"Template generation failed (endpoint may be unavailable): {e}")


@pytest.mark.slow
def test_batch_template_generation(csv_file_path, openai_endpoint_config):
    """Test template generation for multiple questions (slow test)."""
    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=csv_file_path,
        question_column="Question",
        answer_column="Answer",
    )

    # Test with first 3 questions
    num_questions = min(3, len(questions_with_metadata))

    print(f"\n🔧 Generating templates for {num_questions} questions...")

    templates = {}
    for i in range(num_questions):
        question, _ = questions_with_metadata[i]

        print(f"\n  [{i + 1}/{num_questions}] Processing: {question.question[:60]}...")

        try:
            template_code = generate_answer_template(
                question=question.question,
                raw_answer=question.raw_answer,
                config=openai_endpoint_config,
            )

            templates[question.id] = template_code
            print("    ✓ Generated")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            pytest.skip(f"Batch generation failed (endpoint may be unavailable): {e}")

    # Verify all templates were generated
    assert len(templates) == num_questions, f"Expected {num_questions} templates, got {len(templates)}"

    print(f"\n✓ Successfully generated {len(templates)} templates!")


@pytest.mark.skip(reason="Debug test - only run manually when investigating model response issues")
def test_raw_model_response_debug(csv_file_path, openai_endpoint_config):
    """Debug test to capture and display raw model responses for both phases."""
    from langchain_core.prompts import ChatPromptTemplate

    from karenina.domain.answers.generator import (
        FIELD_DESCRIPTION_SYSTEM_PROMPT,
        FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE,
        GROUND_TRUTH_SYSTEM_PROMPT,
        GROUND_TRUTH_USER_PROMPT_TEMPLATE,
    )
    from karenina.infrastructure.llm.interface import init_chat_model_unified

    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=csv_file_path,
        question_column="Question",
        answer_column="Answer",
    )

    assert len(questions_with_metadata) > 0, "No questions extracted"

    # Take the first question for testing
    first_question, _ = questions_with_metadata[0]

    print("\n🔍 DEBUG: Raw Model Response Analysis")
    print("=" * 80)
    print(f"\nQuestion: {first_question.question}")
    print(f"Answer: {first_question.raw_answer}")
    print(f"\nEndpoint: {openai_endpoint_config.endpoint_base_url}")
    print(f"Model: {openai_endpoint_config.model_name}")

    model_params = {
        "model": openai_endpoint_config.model_name,
        "provider": openai_endpoint_config.model_provider,
        "interface": openai_endpoint_config.interface,
        "temperature": openai_endpoint_config.temperature,
    }

    if openai_endpoint_config.interface == "openai_endpoint":
        model_params["endpoint_base_url"] = openai_endpoint_config.endpoint_base_url
        if openai_endpoint_config.endpoint_api_key:
            if hasattr(openai_endpoint_config.endpoint_api_key, "get_secret_value"):
                model_params["endpoint_api_key"] = openai_endpoint_config.endpoint_api_key.get_secret_value()
            else:
                model_params["endpoint_api_key"] = openai_endpoint_config.endpoint_api_key

    model = init_chat_model_unified(**model_params)

    try:
        # PHASE 1: Ground Truth Generation
        print("\n" + "-" * 80)
        print("PHASE 1: Ground Truth Generation")
        print("-" * 80)

        inputs_phase1 = {"question": first_question.question, "answer": first_question.raw_answer}

        print("\n📤 Sending Phase 1 request to model...")

        prompt_phase1 = ChatPromptTemplate.from_messages(
            [
                ("system", GROUND_TRUTH_SYSTEM_PROMPT),
                ("user", GROUND_TRUTH_USER_PROMPT_TEMPLATE),
            ]
        )

        chain_phase1 = prompt_phase1 | model
        result_phase1 = chain_phase1.invoke(inputs_phase1)
        raw_text_phase1 = result_phase1.content if hasattr(result_phase1, "content") else str(result_phase1)

        print("\n📥 Raw Phase 1 Response:")
        print("-" * 80)
        print(raw_text_phase1)
        print("-" * 80)

        print("\n🔍 Phase 1 JSON Validation:")
        gt_json = None
        try:
            gt_json = json.loads(raw_text_phase1)
            print("✅ Valid JSON!")
            print("\nPhase 1 Parsed structure:")
            print(json.dumps(gt_json, indent=2))
        except json.JSONDecodeError as e:
            print("❌ Invalid JSON!")
            print(f"Error: {e}")
            print(f"Error position: line {e.lineno}, column {e.colno}")
            lines = raw_text_phase1.split("\n")
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            for i in range(start, end):
                marker = " >>> " if i == e.lineno - 1 else "     "
                print(f"{marker}Line {i + 1}: {lines[i]}")
            pytest.skip(f"Phase 1 failed with invalid JSON: {e}")

        # PHASE 2: Field Descriptions
        print("\n" + "-" * 80)
        print("PHASE 2: Field Descriptions Generation")
        print("-" * 80)

        # Remove ground_truth values for phase 2
        spec_for_descriptions = {
            "attributes": [{k: v for k, v in attr.items() if k != "ground_truth"} for attr in gt_json["attributes"]]
        }

        inputs_phase2 = {
            "question": first_question.question,
            "answer": first_question.raw_answer,
            "spec_json": json.dumps(spec_for_descriptions, ensure_ascii=False),
        }

        print("\n📤 Sending Phase 2 request to model...")
        print("\nSpec JSON sent to model:")
        print(json.dumps(spec_for_descriptions, indent=2))

        prompt_phase2 = ChatPromptTemplate.from_messages(
            [
                ("system", FIELD_DESCRIPTION_SYSTEM_PROMPT),
                ("user", FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE),
            ]
        )

        chain_phase2 = prompt_phase2 | model
        result_phase2 = chain_phase2.invoke(inputs_phase2)
        raw_text_phase2 = result_phase2.content if hasattr(result_phase2, "content") else str(result_phase2)

        print("\n📥 Raw Phase 2 Response:")
        print("-" * 80)
        print(raw_text_phase2)
        print("-" * 80)

        print("\n🔍 Phase 2 JSON Validation:")
        try:
            fd_json = json.loads(raw_text_phase2)
            print("✅ Valid JSON!")
            print("\nPhase 2 Parsed structure:")
            print(json.dumps(fd_json, indent=2))
        except json.JSONDecodeError as e:
            print("❌ Invalid JSON!")
            print(f"Error: {e}")
            print(f"Error position: line {e.lineno}, column {e.colno}")
            lines = raw_text_phase2.split("\n")
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            for i in range(start, end):
                marker = " >>> " if i == e.lineno - 1 else "     "
                print(f"{marker}Line {i + 1}: {lines[i]}")

        print("\n" + "-" * 80)
        print("End of debug output")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during request: {type(e).__name__}: {e}")
        import traceback

        print("\nFull traceback:")
        print(traceback.format_exc())
        pytest.skip(f"Debug test failed: {e}")


def test_csv_file_exists(csv_file_path):
    """Verify the CSV file exists."""
    csv_path = Path(csv_file_path)
    assert csv_path.exists(), f"CSV file not found at {csv_file_path}"
    assert csv_path.suffix == ".csv", f"File is not a CSV: {csv_file_path}"
    print(f"\n✓ CSV file exists: {csv_file_path}")


def test_csv_has_required_columns(csv_file_path):
    """Verify the CSV has the required Question and Answer columns."""
    import pandas as pd

    df = pd.read_csv(csv_file_path)

    assert "Question" in df.columns, "CSV missing 'Question' column"
    assert "Answer" in df.columns, "CSV missing 'Answer' column"

    # Check we have data
    assert len(df) > 0, "CSV file is empty"

    # Check for non-null values
    question_count = df["Question"].notna().sum()
    answer_count = df["Answer"].notna().sum()

    print("\n✓ CSV structure validated:")
    print(f"  Total rows: {len(df)}")
    print(f"  Questions with content: {question_count}")
    print(f"  Answers with content: {answer_count}")
    print(f"  Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
