# Karenina Usage Examples

This guide provides comprehensive examples for using Karenina in various scenarios, from basic usage to advanced workflows.

## Table of Contents

- [Basic Workflow](#basic-workflow)
- [Question Extraction Examples](#question-extraction-examples)
- [Answer Template Generation](#answer-template-generation)
- [Benchmark Verification](#benchmark-verification)
- [Rubric-Based Evaluation](#rubric-based-evaluation)
- [Multi-Model Testing](#multi-model-testing)
- [Session Management](#session-management)
- [Manual Verification](#manual-verification)
- [Advanced Configurations](#advanced-configurations)
- [Error Handling](#error-handling)
- [Real-World Use Cases](#real-world-use-cases)

## Basic Workflow

### Complete End-to-End Example

This example shows a complete workflow from question extraction to verification results.

```python
import os
from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.models import VerificationConfig, ModelConfiguration
from karenina.benchmark.verification.orchestrator import run_question_verification

# Set up environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Step 1: Extract questions from Excel file
questions = extract_and_generate_questions(
    file_path="data/geography_questions.xlsx",
    output_path="geography_questions.py",
    question_column="Question",
    answer_column="Expected Answer",
    sheet_name="Questions"
)

print(f"Extracted {len(questions)} questions")

# Step 2: Generate answer templates
templates, template_blocks = generate_answer_templates_from_questions_file(
    questions_py_path="geography_questions.py",
    model="gpt-4",
    model_provider="openai",
    return_blocks=True
)

print(f"Generated {len(templates)} templates")

# Step 3: Configure verification
config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="gpt4-answerer",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are a knowledgeable geography expert. Answer questions accurately and concisely."
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35-parser",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse and validate the response against the given template."
        )
    ],
    replicate_count=1
)

# Step 4: Run verification for all questions
all_results = {}
for question in questions[:5]:  # Test first 5 questions
    template_code = template_blocks.get(question.id, "")
    if template_code:
        results = run_question_verification(
            question_id=question.id,
            question_text=question.question,
            template_code=template_code,
            config=config
        )
        all_results.update(results)

        # Print result summary
        for result_id, result in results.items():
            print(f"Question: {question.question}")
            print(f"Result: {'✓ Correct' if result.is_correct else '✗ Incorrect'}")
            if result.granular_score:
                print(f"Score: {result.granular_score:.2f}")
            print("-" * 50)
```

## Question Extraction Examples

### Excel File with Multiple Sheets

```python
from karenina.questions.extractor import extract_and_generate_questions

# Extract from specific sheet
questions_easy = extract_and_generate_questions(
    file_path="data/science_questions.xlsx",
    output_path="science_easy.py",
    question_column="Question",
    answer_column="Answer",
    sheet_name="Easy",
    tags_column="Category"
)

# Extract from another sheet
questions_hard = extract_and_generate_questions(
    file_path="data/science_questions.xlsx",
    output_path="science_hard.py",
    question_column="Question",
    answer_column="Answer",
    sheet_name="Hard",
    tags_column="Category"
)
```

### CSV File with Custom Columns

```python
# CSV file with different column names
questions = extract_and_generate_questions(
    file_path="data/medical_qa.csv",
    output_path="medical_questions.py",
    question_column="query",
    answer_column="expected_response",
    tags_column="domain"
)
```

### File Preview Before Extraction

```python
from karenina.questions.extractor import get_file_preview

# Preview file contents to understand structure
preview = get_file_preview(
    file_path="data/unknown_format.xlsx",
    max_rows=10
)

print("File Info:")
print(f"Total rows: {preview['total_rows']}")
print(f"Columns: {preview['columns']}")
print("\nSample data:")
for row in preview['data'][:3]:
    print(row)

# Now extract with correct column names
questions = extract_and_generate_questions(
    file_path="data/unknown_format.xlsx",
    output_path="questions.py",
    question_column=preview['columns'][0],  # Use first column as question
    answer_column=preview['columns'][1]     # Use second column as answer
)
```

## Answer Template Generation

### Single Question Template

```python
from karenina.answers.generator import generate_answer_template

# Simple factual question
template_code = generate_answer_template(
    question="What is the population of Tokyo?",
    raw_answer="37.4 million",
    model="gpt-4",
    model_provider="openai",
    temperature=0.0
)

print("Generated template:")
print(template_code)

# Expected output:
# class Answer(BaseAnswer):
#     population: float = Field(description="Population of Tokyo in millions")
#
#     def model_post_init(self, __context):
#         self.correct = {"population": 37.4}
#
#     def verify(self) -> bool:
#         return abs(self.population - self.correct["population"]) < 0.1
```

### Complex Multi-Part Question

```python
# Medical question with multiple components
complex_template = generate_answer_template(
    question="What are the three most common symptoms of Type 2 diabetes and their typical severity levels?",
    raw_answer="Increased thirst (moderate to severe), frequent urination (moderate to severe), fatigue (mild to moderate)",
    model="gpt-4",
    model_provider="openai"
)

# This generates a template that validates:
# - symptom names
# - severity levels
# - completeness of the answer
```

### Custom System Prompt for Domain Expertise

```python
# Custom prompt for financial questions
financial_prompt = """
You are an expert in financial analysis and accounting. Generate Pydantic classes
that validate financial answers with appropriate constraints:
- Monetary values should be positive floats
- Percentages should be between 0 and 100
- Date fields should validate proper formats
- Include appropriate business logic in verification methods
"""

template = generate_answer_template(
    question="What is the current P/E ratio of Apple Inc. and is it considered overvalued?",
    raw_answer="P/E ratio: 28.5, Status: Fairly valued",
    model="gpt-4",
    model_provider="openai",
    custom_system_prompt=financial_prompt
)
```

### Batch Template Generation with Progress

```python
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.questions.reader import read_questions_from_file
import time

# Read questions first to show progress
questions = read_questions_from_file("large_question_set.py")
print(f"Starting template generation for {len(questions)} questions...")

start_time = time.time()
templates, blocks = generate_answer_templates_from_questions_file(
    questions_py_path="large_question_set.py",
    model="gemini-2.0-flash",
    model_provider="google_genai",
    return_blocks=True
)

elapsed_time = time.time() - start_time
print(f"Generated {len(templates)} templates in {elapsed_time:.1f} seconds")
print(f"Average time per template: {elapsed_time/len(templates):.2f} seconds")

# Save templates to file for reuse
with open("generated_templates.py", "w") as f:
    f.write("# Generated answer templates\n\n")
    for question_id, template_code in blocks.items():
        f.write(f"# Template for question {question_id}\n")
        f.write(template_code)
        f.write("\n\n")
```

## Benchmark Verification

### Single Question Verification

```python
from karenina.benchmark.models import VerificationConfig, ModelConfiguration
from karenina.benchmark.verification.orchestrator import run_question_verification

# Define models for testing
config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="claude",
            model_provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            temperature=0.2,
            interface="langchain",
            system_prompt="You are a precise and knowledgeable assistant."
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35-parser",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse responses accurately according to the template."
        )
    ]
)

# Template code for a geography question
template_code = """
class Answer(BaseAnswer):
    capital: str = Field(description="Name of the capital city")
    country: str = Field(description="Name of the country")

    def model_post_init(self, __context):
        self.correct = {"capital": "Paris", "country": "France"}

    def verify(self) -> bool:
        return (self.capital.lower() == self.correct["capital"].lower() and
                self.country.lower() == self.correct["country"].lower())

    def verify_granular(self) -> float:
        score = 0
        if self.capital.lower() == self.correct["capital"].lower():
            score += 1
        if self.country.lower() == self.correct["country"].lower():
            score += 1
        return score / 2
"""

# Run verification
results = run_question_verification(
    question_id="geo_001",
    question_text="What is the capital of France and which country is it in?",
    template_code=template_code,
    config=config
)

# Analyze results
for result_id, result in results.items():
    print(f"Result ID: {result_id}")
    print(f"Raw Response: {result.raw_response}")
    print(f"Parsed Response: {result.parsed_response}")
    print(f"Correct: {result.is_correct}")
    print(f"Granular Score: {result.granular_score}")
    if result.error:
        print(f"Error: {result.error}")
```

### Batch Verification with Error Handling

```python
from karenina.questions.reader import read_questions_from_file
from karenina.benchmark.verification.orchestrator import run_question_verification
import json

def run_batch_verification(questions_file, templates_file, config, output_file="results.json"):
    """Run verification for all questions with error handling."""

    # Load questions and templates
    questions = read_questions_from_file(questions_file)

    with open(templates_file, 'r') as f:
        templates_content = f.read()

    all_results = {}
    errors = []

    for i, question in enumerate(questions):
        try:
            print(f"Processing question {i+1}/{len(questions)}: {question.id}")

            # Extract template for this question (simplified)
            # In practice, you'd have a more sophisticated template lookup
            template_start = templates_content.find(f"# Template for question {question.id}")
            if template_start == -1:
                errors.append(f"No template found for question {question.id}")
                continue

            # Run verification
            results = run_question_verification(
                question_id=question.id,
                question_text=question.question,
                template_code="",  # Would extract from templates_content
                config=config
            )

            all_results.update(results)

        except Exception as e:
            error_msg = f"Error processing question {question.id}: {str(e)}"
            errors.append(error_msg)
            print(error_msg)

    # Save results
    with open(output_file, 'w') as f:
        json.dump({
            "results": all_results,
            "errors": errors,
            "summary": {
                "total_questions": len(questions),
                "successful_verifications": len(all_results),
                "errors": len(errors)
            }
        }, f, indent=2, default=str)

    print(f"Verification complete. Results saved to {output_file}")
    return all_results, errors

# Run batch verification
results, errors = run_batch_verification(
    questions_file="questions.py",
    templates_file="templates.py",
    config=config
)
```

## Rubric-Based Evaluation

### Creating and Using Rubrics

```python
from karenina.schemas.rubric_class import Rubric, RubricTrait
from karenina.benchmark.verification.orchestrator import run_question_verification

# Define evaluation rubric
rubric = Rubric(traits=[
    RubricTrait(
        name="accuracy",
        description="Factual correctness of the response",
        kind="score",
        min_score=1,
        max_score=5
    ),
    RubricTrait(
        name="completeness",
        description="Response addresses all parts of the question",
        kind="boolean"
    ),
    RubricTrait(
        name="clarity",
        description="Response is clear and well-structured",
        kind="score",
        min_score=1,
        max_score=5
    ),
    RubricTrait(
        name="conciseness",
        description="Response is appropriately concise without unnecessary information",
        kind="boolean"
    )
])

# Enable rubric evaluation in config
config = VerificationConfig(
    answering_models=[...],  # Your model configurations
    parsing_models=[...],
    rubric_enabled=True,
    rubric_trait_names=["accuracy", "completeness"]  # Optional: filter traits
)

# Run verification with rubric
results = run_question_verification(
    question_id="complex_001",
    question_text="Explain the process of photosynthesis and its importance to ecosystems.",
    template_code=template_code,
    config=config,
    rubric=rubric
)

# Analyze rubric results
for result_id, result in results.items():
    print(f"Question: {result.question_id}")
    print(f"Correctness: {result.is_correct}")

    if result.rubric_evaluation:
        print("Rubric Scores:")
        for trait_name, score in result.rubric_evaluation.items():
            trait = next(t for t in rubric.traits if t.name == trait_name)
            if trait.kind == "score":
                print(f"  {trait_name}: {score}/5")
            else:
                print(f"  {trait_name}: {'Yes' if score else 'No'}")
```

### Domain-Specific Rubrics

```python
# Medical diagnosis rubric
medical_rubric = Rubric(traits=[
    RubricTrait(
        name="diagnostic_accuracy",
        description="Accuracy of the medical diagnosis or assessment",
        kind="score",
        min_score=1,
        max_score=5
    ),
    RubricTrait(
        name="mentions_differential",
        description="Mentions differential diagnoses or alternative possibilities",
        kind="boolean"
    ),
    RubricTrait(
        name="safety_considerations",
        description="Includes appropriate safety warnings or precautions",
        kind="boolean"
    ),
    RubricTrait(
        name="evidence_based",
        description="Response is based on current medical evidence and guidelines",
        kind="score",
        min_score=1,
        max_score=5
    )
])

# Programming rubric
programming_rubric = Rubric(traits=[
    RubricTrait(
        name="code_correctness",
        description="Code is syntactically correct and runs without errors",
        kind="boolean"
    ),
    RubricTrait(
        name="algorithm_efficiency",
        description="Algorithm uses appropriate time and space complexity",
        kind="score",
        min_score=1,
        max_score=5
    ),
    RubricTrait(
        name="code_readability",
        description="Code is well-structured and easy to understand",
        kind="score",
        min_score=1,
        max_score=5
    ),
    RubricTrait(
        name="best_practices",
        description="Code follows language-specific best practices",
        kind="boolean"
    )
])
```

## Multi-Model Testing

### Comprehensive Model Comparison

```python
from karenina.benchmark.models import VerificationConfig, ModelConfiguration

# Define multiple models for comparison
config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="gpt4",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are an expert assistant. Provide accurate, concise answers."
        ),
        ModelConfiguration(
            id="claude-sonnet",
            model_provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            temperature=0.1,
            interface="langchain",
            system_prompt="You are a knowledgeable and precise assistant."
        ),
        ModelConfiguration(
            id="gemini-pro",
            model_provider="google_genai",
            model_name="gemini-2.0-flash",
            temperature=0.1,
            interface="langchain",
            system_prompt="Provide accurate and detailed responses."
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35-parser",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse responses accurately according to the template."
        )
    ],
    replicate_count=3  # Run each combination 3 times for statistical significance
)

# This will test all answering models against all parsing models
# with 3 replicates each, for a total of 3 * 1 * 3 = 9 verification runs
```

### Temperature Sensitivity Testing

```python
# Test how temperature affects model performance
temperature_configs = []

for temp in [0.0, 0.3, 0.7, 1.0]:
    config = VerificationConfig(
        answering_models=[
            ModelConfiguration(
                id=f"gpt4-temp-{temp}",
                model_provider="openai",
                model_name="gpt-4",
                temperature=temp,
                interface="langchain",
                system_prompt="Answer the question accurately."
            )
        ],
        parsing_models=[...],  # Same parser for all
        replicate_count=5
    )
    temperature_configs.append((temp, config))

# Run verification with different temperatures
temp_results = {}
for temp, config in temperature_configs:
    results = run_question_verification(
        question_id="temp_test_001",
        question_text="What is the primary cause of global warming?",
        template_code=template_code,
        config=config
    )
    temp_results[temp] = results

# Analyze temperature effects
for temp, results in temp_results.items():
    correct_count = sum(1 for r in results.values() if r.is_correct)
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Temperature {temp}: {accuracy:.2%} accuracy ({correct_count}/{total_count})")
```

## Session Management

### Interactive Chat Sessions

```python
from karenina.llm.interface import create_chat_session, chat_with_session

# Create a session for ongoing conversation
session_id = create_chat_session(
    model="gpt-4",
    provider="openai",
    temperature=0.7
)

print(f"Created session: {session_id}")

# Have a multi-turn conversation
conversations = [
    "What is machine learning?",
    "Can you give me an example of supervised learning?",
    "How does it differ from unsupervised learning?",
    "What are some common algorithms used in each?"
]

for message in conversations:
    response = chat_with_session(
        session_id=session_id,
        message=message,
        system_message="You are a helpful AI tutor explaining machine learning concepts."
    )

    print(f"User: {message}")
    print(f"Assistant: {response.message}")
    print("-" * 50)
```

### Session-Based Question Generation

```python
from karenina.llm.interface import create_chat_session, chat_with_session

def interactive_template_refinement(question, raw_answer):
    """Interactively refine answer templates with a session."""

    session_id = create_chat_session(
        model="gpt-4",
        provider="openai",
        temperature=0.1
    )

    # Initial template generation
    initial_prompt = f"""
    Generate a Pydantic answer template for this question:
    Question: {question}
    Expected Answer: {raw_answer}

    Make it as comprehensive as possible for validation.
    """

    response = chat_with_session(
        session_id=session_id,
        message=initial_prompt,
        system_message="You are an expert in creating Pydantic validation schemas."
    )

    print("Initial template:")
    print(response.message)

    # Refinement loop
    refinements = [
        "Add more detailed field descriptions",
        "Include edge case handling in the verify method",
        "Add type hints for better validation"
    ]

    for refinement in refinements:
        response = chat_with_session(
            session_id=session_id,
            message=f"Please improve the template: {refinement}"
        )
        print(f"\nAfter refinement - {refinement}:")
        print(response.message)

    return response.message

# Use interactive refinement
refined_template = interactive_template_refinement(
    question="What are the three branches of the U.S. government and their primary functions?",
    raw_answer="Legislative (makes laws), Executive (enforces laws), Judicial (interprets laws)"
)
```

## Manual Verification

### Human-in-the-Loop Verification

```python
from karenina.llm.manual_llm import create_manual_llm
from karenina.llm.manual_traces import save_manual_traces, load_manual_traces
from karenina.benchmark.models import ModelConfiguration, VerificationConfig

# Create manual traces (human responses)
manual_traces = {
    "q1": "The capital of France is Paris, located in the Île-de-France region.",
    "q2": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
    "q3": "The three branches of US government are: Legislative (Congress), Executive (President), and Judicial (Courts)."
}

# Save traces for reuse
save_manual_traces(manual_traces, "human_responses.json")

# Load traces
loaded_traces = load_manual_traces("human_responses.json")

# Create manual LLM
manual_llm = create_manual_llm(loaded_traces)

# Configure verification with manual model
config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="human-expert",
            model_provider="manual",
            model_name="human-expert-v1",
            interface="manual",
            system_prompt="N/A"  # Not used for manual
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="gpt35-parser",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the human response accurately."
        )
    ]
)

# Run verification with human responses
results = run_question_verification(
    question_id="q1",
    question_text="What is the capital of France?",
    template_code=template_code,
    config=config
)
```

### Collecting Manual Traces Interactively

```python
def collect_manual_traces_interactive(questions_file, output_file):
    """Interactively collect human responses for questions."""

    from karenina.questions.reader import read_questions_from_file

    questions = read_questions_from_file(questions_file)
    manual_traces = {}

    print(f"Collecting manual responses for {len(questions)} questions...")
    print("Press Ctrl+C to stop and save current progress.\n")

    try:
        for i, question in enumerate(questions):
            print(f"Question {i+1}/{len(questions)} (ID: {question.id}):")
            print(f"Q: {question.question}")
            print(f"Expected: {question.raw_answer}")

            response = input("Your response: ").strip()
            if response:
                manual_traces[question.id] = response
                print("✓ Saved\n")
            else:
                print("⚠ Skipped\n")

    except KeyboardInterrupt:
        print("\nStopping collection...")

    # Save collected traces
    save_manual_traces(manual_traces, output_file)
    print(f"Saved {len(manual_traces)} manual traces to {output_file}")

    return manual_traces

# Collect traces
traces = collect_manual_traces_interactive(
    questions_file="geography_questions.py",
    output_file="manual_geography_traces.json"
)
```

## Advanced Configurations

### Custom Provider Integration

```python
from karenina.llm.interface import init_chat_model_unified

# Using OpenRouter for additional models
openrouter_config = ModelConfiguration(
    id="llama-70b",
    model_provider="meta-llama/llama-2-70b-chat",  # OpenRouter model ID
    model_name="meta-llama/llama-2-70b-chat",
    temperature=0.2,
    interface="openrouter",
    system_prompt="You are a helpful assistant."
)

# Custom headers for OpenRouter
import os
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-key"
os.environ["OPENROUTER_HTTP_REFERER"] = "https://your-domain.com"
```

### Verification with Custom Validation

```python
def custom_verification_pipeline(question_id, question_text, expected_answer, llm_models):
    """Custom verification with additional validation steps."""

    results = []

    for model_config in llm_models:
        try:
            # Initialize LLM
            llm = init_chat_model_unified(
                model=model_config.model_name,
                provider=model_config.model_provider,
                interface=model_config.interface,
                temperature=model_config.temperature
            )

            # Get response
            response = chat_with_llm(
                llm=llm,
                message=question_text,
                system_message=model_config.system_prompt
            )

            # Custom validation logic
            is_correct = custom_validate_response(response, expected_answer)
            confidence_score = calculate_confidence(response, expected_answer)

            # Additional checks
            toxicity_score = check_toxicity(response)  # Hypothetical function
            relevance_score = check_relevance(response, question_text)

            result = {
                "question_id": question_id,
                "model_id": model_config.id,
                "response": response,
                "is_correct": is_correct,
                "confidence": confidence_score,
                "toxicity": toxicity_score,
                "relevance": relevance_score,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

        except Exception as e:
            print(f"Error with model {model_config.id}: {e}")
            results.append({
                "question_id": question_id,
                "model_id": model_config.id,
                "error": str(e)
            })

    return results

def custom_validate_response(response, expected):
    """Custom validation logic."""
    # Implement your own validation
    # Could include fuzzy matching, semantic similarity, etc.
    pass

def calculate_confidence(response, expected):
    """Calculate confidence score."""
    # Implement confidence calculation
    pass
```

### Parallel Processing Setup

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from karenina.benchmark.verification.orchestrator import run_question_verification

async def parallel_verification(questions, template_codes, config, max_workers=4):
    """Run verifications in parallel for better performance."""

    def run_single_verification(args):
        question, template_code = args
        return run_question_verification(
            question_id=question.id,
            question_text=question.question,
            template_code=template_code,
            config=config
        )

    # Prepare arguments
    verification_args = []
    for question in questions:
        template_code = template_codes.get(question.id, "")
        if template_code:
            verification_args.append((question, template_code))

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, run_single_verification, args)
            for args in verification_args
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    all_results = {}
    for i, result in enumerate(results):
        if isinstance(result, dict):
            all_results.update(result)
        else:
            print(f"Error in verification {i}: {result}")

    return all_results

# Usage
import asyncio

all_results = asyncio.run(parallel_verification(
    questions=questions,
    template_codes=template_codes,
    config=config,
    max_workers=4
))
```

## Error Handling

### Comprehensive Error Handling

```python
from karenina.llm.interface import LLMError, LLMNotAvailableError, SessionError
from karenina.benchmark.verification.orchestrator import run_question_verification

def robust_verification(question_id, question_text, template_code, config, max_retries=3):
    """Run verification with comprehensive error handling."""

    last_error = None

    for attempt in range(max_retries):
        try:
            results = run_question_verification(
                question_id=question_id,
                question_text=question_text,
                template_code=template_code,
                config=config
            )

            # Validate results
            if not results:
                raise ValueError("No results returned from verification")

            # Check for errors in results
            for result_id, result in results.items():
                if result.error:
                    print(f"Warning: Result {result_id} has error: {result.error}")

            return results

        except LLMNotAvailableError as e:
            print(f"LLM not available: {e}")
            return {"error": "LLM_NOT_AVAILABLE", "details": str(e)}

        except SessionError as e:
            print(f"Session error on attempt {attempt + 1}: {e}")
            last_error = e
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            last_error = e
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)

    # All retries failed
    return {
        "error": "MAX_RETRIES_EXCEEDED",
        "details": str(last_error),
        "question_id": question_id
    }

# Usage with error handling
result = robust_verification(
    question_id="test_001",
    question_text="What is 2+2?",
    template_code=simple_template,
    config=config
)

if "error" in result:
    print(f"Verification failed: {result['error']}")
else:
    print(f"Verification successful: {len(result)} results")
```

### Validation and Recovery

```python
def validate_and_fix_config(config):
    """Validate configuration and attempt to fix common issues."""

    # Check for required environment variables
    required_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google_genai": "GOOGLE_API_KEY"
    }

    missing_keys = []
    for model in config.answering_models + config.parsing_models:
        if model.interface == "langchain":
            env_var = required_vars.get(model.model_provider)
            if env_var and not os.getenv(env_var):
                missing_keys.append(f"{env_var} for {model.model_provider}")

    if missing_keys:
        print(f"Warning: Missing environment variables: {', '.join(missing_keys)}")

        # Attempt to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("Loaded environment variables from .env file")
        except ImportError:
            print("python-dotenv not installed. Consider installing it for .env support")

    # Validate model configurations
    for model in config.answering_models + config.parsing_models:
        if not model.model_name:
            raise ValueError(f"Model {model.id} missing model_name")
        if not model.model_provider and model.interface == "langchain":
            raise ValueError(f"Model {model.id} missing model_provider")
        if model.temperature < 0 or model.temperature > 2:
            print(f"Warning: Model {model.id} has unusual temperature {model.temperature}")

    return config

# Use validation
try:
    validated_config = validate_and_fix_config(config)
    results = run_question_verification(...)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Real-World Use Cases

### Medical Q&A Benchmarking

```python
# Medical question benchmarking setup
def setup_medical_benchmark():
    """Setup for medical domain benchmarking."""

    # Extract medical questions
    medical_questions = extract_and_generate_questions(
        file_path="data/medical_board_questions.xlsx",
        output_path="medical_questions.py",
        question_column="Clinical_Question",
        answer_column="Correct_Answer",
        tags_column="Medical_Specialty"
    )

    # Medical-specific system prompts
    medical_answerer_prompt = """
    You are a board-certified physician with expertise across multiple medical specialties.
    Provide accurate, evidence-based medical information. Always include:
    1. Primary diagnosis or answer
    2. Key differential diagnoses when relevant
    3. Appropriate clinical reasoning
    4. Safety considerations and limitations
    """

    medical_parser_prompt = """
    You are a medical knowledge validator. Parse responses carefully to ensure:
    1. Medical terminology is used correctly
    2. Clinical reasoning is sound
    3. Safety considerations are addressed
    4. Information aligns with current medical guidelines
    """

    # Configure medical evaluation rubric
    medical_rubric = Rubric(traits=[
        RubricTrait(
            name="clinical_accuracy",
            description="Medical information is factually correct and up-to-date",
            kind="score",
            min_score=1,
            max_score=5
        ),
        RubricTrait(
            name="includes_differentials",
            description="Mentions relevant differential diagnoses",
            kind="boolean"
        ),
        RubricTrait(
            name="safety_awareness",
            description="Includes appropriate safety warnings and limitations",
            kind="boolean"
        ),
        RubricTrait(
            name="evidence_based",
            description="Response is grounded in current medical evidence",
            kind="score",
            min_score=1,
            max_score=5
        )
    ])

    return medical_questions, medical_rubric, medical_answerer_prompt, medical_parser_prompt

# Run medical benchmark
questions, rubric, answerer_prompt, parser_prompt = setup_medical_benchmark()

medical_config = VerificationConfig(
    answering_models=[
        ModelConfiguration(
            id="medical-gpt4",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            interface="langchain",
            system_prompt=answerer_prompt
        ),
        ModelConfiguration(
            id="medical-claude",
            model_provider="anthropic",
            model_name="claude-3-opus-20240229",
            temperature=0.1,
            interface="langchain",
            system_prompt=answerer_prompt
        )
    ],
    parsing_models=[
        ModelConfiguration(
            id="medical-parser",
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            interface="langchain",
            system_prompt=parser_prompt
        )
    ],
    replicate_count=2,
    rubric_enabled=True
)
```

### Code Generation Benchmarking

```python
def setup_coding_benchmark():
    """Setup for programming benchmark."""

    coding_questions = extract_and_generate_questions(
        file_path="data/leetcode_problems.csv",
        output_path="coding_questions.py",
        question_column="problem_statement",
        answer_column="optimal_solution",
        tags_column="difficulty"
    )

    coding_prompt = """
    You are an expert software engineer. For programming questions:
    1. Provide working, efficient code
    2. Include complexity analysis (time and space)
    3. Explain your approach
    4. Consider edge cases
    5. Use best practices and clean code principles
    """

    coding_rubric = Rubric(traits=[
        RubricTrait(
            name="correctness",
            description="Code is syntactically correct and produces expected output",
            kind="boolean"
        ),
        RubricTrait(
            name="efficiency",
            description="Algorithm uses optimal time and space complexity",
            kind="score",
            min_score=1,
            max_score=5
        ),
        RubricTrait(
            name="readability",
            description="Code is well-structured and easy to understand",
            kind="score",
            min_score=1,
            max_score=5
        ),
        RubricTrait(
            name="edge_cases",
            description="Code handles edge cases appropriately",
            kind="boolean"
        )
    ])

    return coding_questions, coding_rubric, coding_prompt

# Custom template for code problems
def generate_coding_template(problem_statement, solution_code):
    """Generate specialized template for coding problems."""

    coding_template_prompt = """
    Generate a Pydantic class that validates programming solutions. Include:
    - code: str field for the solution code
    - time_complexity: str field for Big O time complexity
    - space_complexity: str field for Big O space complexity
    - explanation: str field for approach explanation

    In the verify method, check if the code produces the expected output.
    In verify_granular, award points for correctness, efficiency, and explanation quality.
    """

    return generate_answer_template(
        question=problem_statement,
        raw_answer=solution_code,
        custom_system_prompt=coding_template_prompt
    )
```

### Multi-Language Benchmarking

```python
def setup_multilingual_benchmark():
    """Setup for testing multilingual capabilities."""

    # Questions in multiple languages
    languages = {
        "english": "data/questions_en.csv",
        "spanish": "data/questions_es.csv",
        "french": "data/questions_fr.csv",
        "german": "data/questions_de.csv"
    }

    multilingual_questions = {}
    for lang, file_path in languages.items():
        questions = extract_and_generate_questions(
            file_path=file_path,
            output_path=f"questions_{lang}.py",
            question_column="question",
            answer_column="answer",
            tags_column="language"
        )
        multilingual_questions[lang] = questions

    # Language-specific prompts
    def get_language_prompt(language):
        prompts = {
            "spanish": "Responda en español de manera precisa y completa.",
            "french": "Répondez en français de manière précise et complète.",
            "german": "Antworten Sie auf Deutsch präzise und vollständig.",
            "english": "Respond in English accurately and completely."
        }
        return prompts.get(language, prompts["english"])

    # Test model performance across languages
    multilingual_results = {}

    for language, questions in multilingual_questions.items():
        config = VerificationConfig(
            answering_models=[
                ModelConfiguration(
                    id=f"gpt4-{language}",
                    model_provider="openai",
                    model_name="gpt-4",
                    temperature=0.1,
                    interface="langchain",
                    system_prompt=get_language_prompt(language)
                )
            ],
            parsing_models=[...],
            replicate_count=2
        )

        # Run verification for subset of questions
        language_results = {}
        for question in questions[:10]:  # Test first 10 per language
            results = run_question_verification(
                question_id=f"{language}_{question.id}",
                question_text=question.question,
                template_code="",  # Would need language-specific templates
                config=config
            )
            language_results.update(results)

        multilingual_results[language] = language_results

    # Analyze cross-language performance
    for language, results in multilingual_results.items():
        accuracy = sum(r.is_correct for r in results.values()) / len(results)
        print(f"{language.title()}: {accuracy:.2%} accuracy")

    return multilingual_results
```

This comprehensive set of examples should give you a solid foundation for using Karenina in various scenarios, from basic question extraction to complex multi-model benchmarking workflows.
