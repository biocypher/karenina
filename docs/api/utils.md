# Utils Module

The `karenina.utils` module provides utility functions for code processing and parsing operations.

## Code Parser

### extract_and_combine_codeblocks

Extract and combine Python code blocks from LLM-generated text.

::: karenina.utils.code_parser.extract_and_combine_codeblocks

**Parameters:**
- `text` (str): Input text containing code blocks (typically LLM response)

**Returns:**
- `str`: Combined Python code from all code blocks

**Usage Examples:**

```python
from karenina.utils.code_parser import extract_and_combine_codeblocks

# LLM response with code blocks
llm_response = '''
Here's the answer template:

```python
from pydantic import BaseModel, Field
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    city_name: str = Field(description="Capital city name")
    confidence: float = Field(ge=0.0, le=1.0)
```

And here's an additional helper:

```python
def validate_city(city: str) -> bool:
    return len(city) > 0
```
'''

# Extract combined code
combined_code = extract_and_combine_codeblocks(llm_response)
print(combined_code)
# Output:
# from pydantic import BaseModel, Field
# from karenina.schemas.answer_class import BaseAnswer
# 
# class Answer(BaseAnswer):
#     city_name: str = Field(description="Capital city name") 
#     confidence: float = Field(ge=0.0, le=1.0)
# 
# def validate_city(city: str) -> bool:
#     return len(city) > 0
```

**Code Block Detection:**

The function identifies Python code blocks using various patterns:

```python
# Markdown code blocks
text_with_markdown = '''
```python
print("Hello World")
```
'''

# Generic code blocks  
text_with_generic = '''
```
x = 42
y = x * 2
```
'''

# Both will be extracted
code1 = extract_and_combine_codeblocks(text_with_markdown)
code2 = extract_and_combine_codeblocks(text_with_generic)

print(code1)  # print("Hello World")
print(code2)  # x = 42\ny = x * 2
```

**Integration with Answer Generation:**

```python
from karenina.answers.generator import generate_answer_template
from karenina.utils.code_parser import extract_and_combine_codeblocks

# Generate template
question = "What is the capital of France?"
question_json = '{"id": "hash", "question": "What is the capital of France?"}'

llm_response = generate_answer_template(question, question_json)

# Extract executable code
executable_code = extract_and_combine_codeblocks(llm_response)

# Execute safely
local_ns = {}
exec(executable_code, globals(), local_ns)
AnswerClass = local_ns["Answer"]

# Use generated class
answer = AnswerClass(city_name="Paris", confidence=0.95)
```

## Advanced Usage

### Code Validation

```python
import ast

def validate_extracted_code(code: str) -> bool:
    """Validate that extracted code is syntactically correct."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# Usage
llm_response = get_llm_response()
code = extract_and_combine_codeblocks(llm_response)

if validate_extracted_code(code):
    exec(code, globals(), local_ns)
else:
    print("Generated code has syntax errors")
```

### Safe Code Execution

```python
import sys
from io import StringIO

def safe_execute_code(code: str) -> tuple[dict, str]:
    """Safely execute code and capture output."""
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    # Execute in isolated namespace
    local_ns = {}
    error_msg = ""
    
    try:
        exec(code, {"__builtins__": {}}, local_ns)
    except Exception as e:
        error_msg = str(e)
    finally:
        sys.stdout = old_stdout
    
    output = captured_output.getvalue()
    return local_ns, output, error_msg

# Usage
code = extract_and_combine_codeblocks(llm_response)
namespace, output, error = safe_execute_code(code)

if not error:
    AnswerClass = namespace.get("Answer")
    if AnswerClass:
        print("Successfully extracted Answer class")
else:
    print(f"Execution error: {error}")
```

### Code Quality Analysis

```python
def analyze_extracted_code(code: str) -> dict:
    """Analyze extracted code for quality metrics."""
    
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    metrics = {
        'total_lines': len(lines),
        'code_lines': len(non_empty_lines),
        'has_imports': any('import' in line for line in lines),
        'has_class_definition': any('class ' in line for line in lines),
        'has_function_definition': any('def ' in line for line in lines),
        'has_pydantic_fields': any('Field(' in line for line in lines),
        'class_names': []
    }
    
    # Extract class names
    for line in lines:
        if 'class ' in line and ':' in line:
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            metrics['class_names'].append(class_name)
    
    return metrics

# Usage
code = extract_and_combine_codeblocks(llm_response)
analysis = analyze_extracted_code(code)

print(f"Code analysis: {analysis}")
# Output: {'total_lines': 15, 'code_lines': 12, 'has_imports': True, ...}
```

### Multiple Code Block Processing

```python
def extract_code_blocks_separately(text: str) -> list[str]:
    """Extract code blocks as separate strings instead of combining."""
    
    import re
    
    # Pattern for code blocks
    pattern = r'```(?:python)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    return [match.strip() for match in matches]

# Usage
llm_response_with_multiple_blocks = '''
First, the main class:
```python
class Answer(BaseAnswer):
    result: str
```

Then, a helper function:
```python  
def helper():
    return "help"
```
'''

separate_blocks = extract_code_blocks_separately(llm_response_with_multiple_blocks)
print(f"Found {len(separate_blocks)} code blocks")

for i, block in enumerate(separate_blocks):
    print(f"Block {i+1}:")
    print(block)
    print("---")
```

## Error Handling

### Common Issues

```python
# Handle empty or no code blocks
text_without_code = "This is just text without any code blocks."
code = extract_and_combine_codeblocks(text_without_code)
print(repr(code))  # '' (empty string)

# Handle malformed code blocks
malformed_text = '''
```python
class Answer(BaseAnswer:  # Missing closing parenthesis
    field: str
```
'''

code = extract_and_combine_codeblocks(malformed_text)
try:
    exec(code)
except SyntaxError as e:
    print(f"Syntax error in extracted code: {e}")
```

### Robust Processing Pipeline

```python
def robust_code_extraction(llm_response: str) -> tuple[str, bool, str]:
    """Robust code extraction with error handling."""
    
    try:
        # Extract code
        code = extract_and_combine_codeblocks(llm_response)
        
        if not code.strip():
            return "", False, "No code blocks found"
        
        # Validate syntax
        ast.parse(code)
        
        # Check for required elements
        if 'class Answer' not in code:
            return code, False, "No Answer class found in code"
        
        if 'BaseAnswer' not in code:
            return code, False, "Answer class doesn't inherit from BaseAnswer"
        
        return code, True, "Code extraction successful"
        
    except SyntaxError as e:
        return code if 'code' in locals() else "", False, f"Syntax error: {e}"
    except Exception as e:
        return "", False, f"Unexpected error: {e}"

# Usage in answer generation pipeline
def safe_answer_template_generation(question: str, question_json: str):
    """Generate answer template with robust error handling."""
    
    llm_response = generate_answer_template(question, question_json)
    code, success, message = robust_code_extraction(llm_response)
    
    if success:
        local_ns = {}
        exec(code, globals(), local_ns)
        return local_ns.get("Answer"), code
    else:
        print(f"Code extraction failed: {message}")
        return None, code
```

## Integration Examples

### With Logging

```python
import logging

logger = logging.getLogger(__name__)

def logged_code_extraction(text: str) -> str:
    """Code extraction with detailed logging."""
    
    logger.info("Starting code extraction")
    
    code = extract_and_combine_codeblocks(text)
    
    if code:
        lines = len(code.split('\n'))
        logger.info(f"Extracted {lines} lines of code")
        logger.debug(f"Extracted code:\n{code}")
    else:
        logger.warning("No code blocks found in text")
    
    return code
```

### With Caching

```python
import hashlib
import pickle
from pathlib import Path

class CodeExtractionCache:
    """Cache extracted code to avoid re-processing."""
    
    def __init__(self, cache_dir: str = "code_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_cached_code(self, text: str) -> str | None:
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_code(self, text: str, code: str):
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(code, f)
    
    def extract_with_cache(self, text: str) -> str:
        # Check cache first
        cached_code = self.get_cached_code(text)
        if cached_code is not None:
            return cached_code
        
        # Extract and cache
        code = extract_and_combine_codeblocks(text)
        self.cache_code(text, code)
        return code

# Usage
cache = CodeExtractionCache()
code = cache.extract_with_cache(llm_response)
```