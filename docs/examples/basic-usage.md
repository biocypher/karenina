# Basic Usage Examples

This section provides practical examples for getting started with Karenina's core functionality.

## Simple Question Extraction

### Extract from Excel File

```python
from karenina.questions.extractor import extract_and_generate_questions

# Basic extraction
extract_and_generate_questions(
    file_path="data/questions.xlsx",
    output_path="my_questions.py",
    question_column="Question",
    answer_column="Answer"
)

print("Questions extracted to my_questions.py")
```

### Preview File Before Extraction

```python
from karenina.questions.extractor import get_file_preview

# Check file structure first
preview = get_file_preview("data/questions.xlsx", max_rows=5)

if preview["success"]:
    print("Available columns:", preview["columns"])
    print("Sample data:")
    for i, row in enumerate(preview["data"][:3]):
        print(f"  Row {i+1}: {row}")
else:
    print("Error:", preview["error"])
```

### Extract from CSV with Custom Columns

```python
# CSV with different column names
extract_and_generate_questions(
    file_path="data/qa_pairs.csv",
    output_path="csv_questions.py", 
    question_column="query",
    answer_column="response"
)
```

## Simple Answer Template Generation

### Generate Templates for All Questions

```python
from karenina.answers.generator import generate_answer_templates_from_questions_file

# Generate templates using default model (Gemini 2.0 Flash)
templates = generate_answer_templates_from_questions_file("my_questions.py")

print(f"Generated {len(templates)} answer templates")

# Inspect a template
question_id = list(templates.keys())[0]
AnswerClass = templates[question_id]
print(f"Template fields: {list(AnswerClass.model_fields.keys())}")
```

### Use Different LLM for Template Generation

```python
# Use GPT-4 for template generation
templates = generate_answer_templates_from_questions_file(
    "my_questions.py",
    model="gpt-4",
    model_provider="openai"
)

# Use Claude for template generation
templates = generate_answer_templates_from_questions_file(
    "my_questions.py",
    model="claude-3-sonnet", 
    model_provider="anthropic"
)
```

## Simple Benchmark Execution

### Basic Benchmark Run

```python
from karenina.benchmark.runner import run_benchmark

# Prepare test data
questions = {
    "q1": "What is the capital of France?",
    "q2": "What is 2 + 2?"
}

# Simulated model responses
responses = {
    "q1": "The capital of France is Paris.",
    "q2": "2 + 2 equals 4."
}

# Use previously generated templates
# (In practice, question IDs would be MD5 hashes)
templates = {
    "q1": templates["actual_q1_hash"],
    "q2": templates["actual_q2_hash"]
}

# Run benchmark
results = run_benchmark(questions, responses, templates)

# View results
for q_id, result in results.items():
    print(f"Question {q_id}:")
    print(f"  Result: {result}")
    print(f"  Type: {type(result)}")
```

## Complete End-to-End Example

### Full Pipeline with Real Data

```python
from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.runner import run_benchmark
from karenina.llm.interface import call_model
from karenina.questions.reader import read_questions_from_file

def complete_benchmark_example():
    """Complete example from file to results."""
    
    # Step 1: Extract questions from Excel
    print("1. Extracting questions...")
    extract_and_generate_questions(
        file_path="data/sample_questions.xlsx",
        output_path="extracted_questions.py"
    )
    
    # Step 2: Generate answer templates
    print("2. Generating answer templates...")
    templates = generate_answer_templates_from_questions_file("extracted_questions.py")
    print(f"   Generated {len(templates)} templates")
    
    # Step 3: Load questions and collect responses
    print("3. Collecting model responses...")
    questions = read_questions_from_file("extracted_questions.py")
    
    questions_dict = {}
    responses_dict = {}
    
    for question in questions[:3]:  # Test with first 3 questions
        questions_dict[question.id] = question.question
        
        # Get response from GPT-3.5
        try:
            response = call_model(
                model="gpt-3.5-turbo",
                provider="openai",
                message=question.question,
                temperature=0.3
            )
            responses_dict[question.id] = response.message
        except Exception as e:
            print(f"   Error getting response: {e}")
            responses_dict[question.id] = "No response available"
    
    # Step 4: Run benchmark
    print("4. Running benchmark...")
    results = run_benchmark(questions_dict, responses_dict, templates)
    
    # Step 5: Display results
    print("5. Results:")
    for q_id, result in results.items():
        print(f"   Question ID: {q_id}")
        print(f"   Question: {questions_dict[q_id][:50]}...")
        print(f"   Response: {responses_dict[q_id][:50]}...")
        print(f"   Structured Result: {result}")
        print("   ---")
    
    return results

# Run the complete example
if __name__ == "__main__":
    results = complete_benchmark_example()
```

## Working with LLM Providers

### Basic LLM Calls

```python
from karenina.llm.interface import call_model

# Simple call to OpenAI
response = call_model(
    model="gpt-3.5-turbo",
    provider="openai",
    message="What is machine learning?"
)
print("OpenAI says:", response.message)

# Simple call to Google AI
response = call_model(
    model="gemini-2.0-flash",
    provider="google_genai", 
    message="What is machine learning?"
)
print("Google AI says:", response.message)

# Simple call to Anthropic
response = call_model(
    model="claude-3-haiku",
    provider="anthropic",
    message="What is machine learning?"
)
print("Anthropic says:", response.message)
```

### Conversational Examples

```python
from karenina.llm.interface import call_model

def simple_conversation():
    """Simple conversation with session management."""
    
    session_id = None
    
    messages = [
        "Hello, what's your name?",
        "Can you help me with Python?",
        "What's a good way to learn it?"
    ]
    
    for message in messages:
        response = call_model(
            model="gpt-3.5-turbo",
            provider="openai",
            message=message,
            session_id=session_id
        )
        
        session_id = response.session_id
        
        print(f"User: {message}")
        print(f"AI: {response.message}")
        print()

simple_conversation()
```

## Error Handling Examples

### Basic Error Handling

```python
from karenina.llm.interface import call_model, LLMError, LLMNotAvailableError
from karenina.questions.extractor import extract_questions_from_file

def safe_extraction():
    """Extract questions with error handling."""
    
    try:
        questions = extract_questions_from_file(
            "data/questions.xlsx",
            "Question", 
            "Answer"
        )
        print(f"Successfully extracted {len(questions)} questions")
        return questions
        
    except FileNotFoundError:
        print("Error: File not found")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return []

def safe_llm_call():
    """Make LLM call with error handling."""
    
    try:
        response = call_model(
            model="gpt-3.5-turbo",
            provider="openai",
            message="Hello!"
        )
        return response.message
        
    except LLMNotAvailableError:
        print("Error: LangChain not available")
    except LLMError as e:
        print(f"LLM Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

# Usage
questions = safe_extraction()
if questions:
    response = safe_llm_call()
    if response:
        print("Success:", response)
```

## File Format Examples

### Working with Different File Types

```python
from karenina.questions.extractor import extract_questions_from_file

# Excel file
excel_questions = extract_questions_from_file(
    "data/questions.xlsx",
    "Question",
    "Answer", 
    sheet_name="Easy"  # Specific sheet
)

# CSV file
csv_questions = extract_questions_from_file(
    "data/questions.csv",
    "question_text",
    "answer_text"
)

# TSV file
tsv_questions = extract_questions_from_file(
    "data/questions.tsv", 
    "Q",
    "A"
)

print(f"Excel: {len(excel_questions)} questions")
print(f"CSV: {len(csv_questions)} questions") 
print(f"TSV: {len(tsv_questions)} questions")
```

### JSON Output for Web Applications

```python
from karenina.questions.extractor import extract_and_generate_questions

# Get JSON format instead of Python file
questions_json = extract_and_generate_questions(
    file_path="data/questions.xlsx",
    output_path="",  # Not used when return_json=True
    return_json=True
)

print("JSON format questions:")
for q_id, q_data in list(questions_json.items())[:2]:
    print(f"  {q_id}: {q_data}")

# JSON structure:
# {
#   "question_hash": {
#     "question": "Question text",
#     "raw_answer": "Answer text"
#   }
# }
```

## Template Inspection Examples

### Examining Generated Templates

```python
from karenina.answers.generator import generate_answer_templates_from_questions_file

# Generate templates with code blocks
templates, code_blocks = generate_answer_templates_from_questions_file(
    "questions.py",
    return_blocks=True
)

# Inspect first template
question_id = list(templates.keys())[0]
AnswerClass = templates[question_id]
code_block = code_blocks[question_id]

print(f"Template for {question_id}:")
print("Generated code:")
print(code_block)
print()

print("Template fields:")
for field_name, field_info in AnswerClass.model_fields.items():
    print(f"  {field_name}: {field_info.annotation}")
    print(f"    Description: {field_info.description}")
    print(f"    Required: {field_info.is_required()}")
print()

# Test the template
try:
    test_instance = AnswerClass()
    print("Template can be instantiated with no args")
except Exception as e:
    print(f"Template requires arguments: {e}")
```

### Template Validation

```python
def validate_template(AnswerClass, test_data: dict):
    """Test template validation."""
    
    try:
        validated = AnswerClass(**test_data)
        print("✓ Validation successful")
        print("  Validated data:", validated.model_dump())
        return True
        
    except Exception as e:
        print("✗ Validation failed:", e)
        return False

# Test with sample data
test_data = {
    "answer": "Paris",
    "confidence": 0.95,
    "source": "knowledge"
}

success = validate_template(AnswerClass, test_data)
```

## Integration Examples

### Using with Custom Workflows

```python
def custom_benchmark_workflow(input_file: str, target_model: str):
    """Custom workflow with specific requirements."""
    
    # Custom question filtering
    print("Extracting and filtering questions...")
    all_questions = extract_questions_from_file(input_file, "Question", "Answer")
    
    # Filter questions (example: only questions with "what" or "how")
    filtered_questions = [
        q for q in all_questions 
        if any(word in q.question.lower() for word in ['what', 'how'])
    ]
    
    print(f"Filtered to {len(filtered_questions)} questions")
    
    # Generate templates with specific model
    print("Generating templates with Claude...")
    templates = {}
    for question in filtered_questions[:5]:  # Limit for demo
        template_code = generate_answer_template(
            question.question,
            question.model_dump_json(),
            model="claude-3-sonnet",
            model_provider="anthropic"
        )
        
        # Execute template
        code_blocks = extract_and_combine_codeblocks(template_code)
        local_ns = {}
        exec(code_blocks, globals(), local_ns)
        templates[question.id] = local_ns["Answer"]
    
    # Collect responses from target model
    print(f"Collecting responses from {target_model}...")
    questions_dict = {q.id: q.question for q in filtered_questions[:5]}
    responses_dict = {}
    
    for q_id, question in questions_dict.items():
        response = call_model(
            model=target_model,
            provider="openai",  # Assuming OpenAI model
            message=question
        )
        responses_dict[q_id] = response.message
    
    # Run benchmark
    print("Running benchmark...")
    results = run_benchmark(questions_dict, responses_dict, templates)
    
    # Custom analysis
    print("Custom analysis:")
    for q_id, result in results.items():
        result_dict = result.model_dump()
        confidence = result_dict.get('confidence', 'N/A')
        print(f"  Q: {questions_dict[q_id][:30]}...")
        print(f"  Confidence: {confidence}")
    
    return results

# Usage
custom_results = custom_benchmark_workflow("data/questions.xlsx", "gpt-4")
```

### Batch Processing Example

```python
def process_multiple_files(file_list: list):
    """Process multiple question files in batch."""
    
    all_results = {}
    
    for file_path in file_list:
        print(f"Processing {file_path}...")
        
        try:
            # Extract questions
            base_name = file_path.split('/')[-1].split('.')[0]
            questions_file = f"{base_name}_questions.py"
            
            extract_and_generate_questions(file_path, questions_file)
            
            # Generate templates
            templates = generate_answer_templates_from_questions_file(questions_file)
            
            # Store results
            all_results[base_name] = {
                'questions_file': questions_file,
                'template_count': len(templates),
                'templates': templates
            }
            
            print(f"  ✓ Generated {len(templates)} templates")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path}: {e}")
            all_results[base_name] = {'error': str(e)}
    
    return all_results

# Usage
files_to_process = [
    "data/math_questions.xlsx",
    "data/science_questions.csv", 
    "data/history_questions.tsv"
]

batch_results = process_multiple_files(files_to_process)

print("\nBatch processing summary:")
for name, result in batch_results.items():
    if 'error' in result:
        print(f"  {name}: Failed - {result['error']}")
    else:
        print(f"  {name}: Success - {result['template_count']} templates")
```