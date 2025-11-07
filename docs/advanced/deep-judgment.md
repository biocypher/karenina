# Deep-Judgment Parsing

Deep-judgment is an advanced parsing mode that provides enhanced transparency and accountability by extracting verbatim evidence from LLM responses before drawing conclusions. This guide explains what it is, when to use it, and how to configure it.

## What is Deep-Judgment?

**Deep-judgment parsing** is a multi-stage evaluation process that goes beyond standard template-based verification. Instead of directly extracting structured data from LLM responses, deep-judgment performs a three-stage analysis:

1. **Excerpt Extraction**: Identifies verbatim quotes that support each answer attribute
2. **Reasoning Generation**: Explains how the excerpts map to attribute values
3. **Parameter Extraction**: Extracts final structured values with full context

This approach creates an **audit trail** showing exactly what evidence the LLM provided and how it was interpreted.

### Standard Parsing vs Deep-Judgment

**Standard Parsing** (default):
```
LLM Response → Parse Attributes → Verify Correctness
```

**Deep-Judgment Parsing**:
```
LLM Response → Extract Excerpts → Generate Reasoning → Parse Attributes → Verify Correctness
```

---

## Why Use Deep-Judgment?

Deep-judgment provides several key benefits:

### 1. Transparency

Every extracted attribute is backed by explicit evidence from the LLM response. You can see exactly which parts of the answer support each claim.

**Example**:
```
Question: "What is the approved drug target of Venetoclax?"

LLM Response: "Venetoclax targets the BCL-2 protein, which is an anti-apoptotic
protein. By inhibiting BCL-2, venetoclax promotes apoptosis in cancer cells."

Standard Parsing:
  drug_target: "BCL-2" ✓

Deep-Judgment Parsing:
  Excerpt: "targets the BCL-2 protein"
  Reasoning: "The response explicitly states BCL-2 as the target protein"
  drug_target: "BCL-2" ✓
```

### 2. Accountability

When an attribute cannot be extracted, deep-judgment provides an explanation of what's missing rather than silently failing.

**Example**:
```
Question: "What is the mechanism of action of Venetoclax?"

LLM Response: "Venetoclax is a BCL-2 inhibitor."

Standard Parsing:
  mechanism: null ✗ (Failed - reason unclear)

Deep-Judgment Parsing:
  Excerpt: [none]
  Reasoning: "Response identifies drug class but does not explain mechanism"
  mechanism: null ✗ (Failed - missing mechanism explanation)
```

### 3. Debugging Support

When verification fails, deep-judgment helps you understand why by showing what evidence was (or wasn't) found in the response.

### 4. Quality Assurance

Ensures that LLM responses contain sufficient evidence for their claims, not just plausible-sounding answers.

---

## How Deep-Judgment Works

Deep-judgment uses a **three-stage autoregressive process** where each stage builds on the previous one:

### Stage 1: Excerpt Extraction

The parsing model identifies **verbatim quotes** from the LLM response that support each template attribute.

**For each attribute**:

- Extract 0-3 excerpts (configurable)
- Assign confidence level: low/medium/high
- Validate excerpts actually exist in the response (fuzzy matching)
- If no excerpts found, request explanation from LLM

**Example**:
```python
Question: "What is the approved drug target of Venetoclax?"

Response: "Venetoclax targets BCL-2, a key anti-apoptotic protein"

Excerpts:
{
  "drug_target": [
    {
      "text": "targets BCL-2",
      "confidence": "high",
      "similarity_score": 0.95
    }
  ]
}
```

### Stage 2: Reasoning Generation

The parsing model explains how the excerpts from Stage 1 inform each attribute value.

**Example**:
```python
Reasoning:
{
  "drug_target": "The excerpt 'targets BCL-2' explicitly states BCL-2
                  as the protein target. This directly answers the question."
}
```

### Stage 3: Parameter Extraction

Using the reasoning context from Stage 2, the parsing model extracts structured attribute values using standard template parsing.

**Example**:
```python
Parsed Answer:
{
  "drug_target": "BCL-2"
}
```

### Validation and Auto-Fail

If any attribute has **missing excerpts** (no verbatim evidence found), verification **automatically fails** even if the final parsed answer seems correct. This ensures all claims are backed by explicit evidence.

**Exception**: If abstention is detected (LLM refused to answer), auto-fail is skipped since abstention takes priority.

---

## Search-Enhanced Deep-Judgment

**Search-enhanced deep-judgment** extends the standard three-stage process with an additional validation layer that checks extracted excerpts against external evidence sources. This helps detect potential hallucinations by verifying that the information in excerpts can be corroborated by external search results.

### How It Works

When search enhancement is enabled, the deep-judgment process adds two additional steps between Stage 1 (Excerpt Extraction) and Stage 2 (Reasoning Generation):

**Standard Deep-Judgment**:
```
Stage 1: Extract Excerpts → Stage 2: Generate Reasoning → Stage 3: Parse Attributes
```

**Search-Enhanced Deep-Judgment**:
```
Stage 1: Extract Excerpts
  → Search Validation (query external sources)
  → Stage 1.5: Hallucination Risk Assessment (per excerpt)
  → Stage 2: Generate Reasoning (with risk context)
  → Stage 3: Parse Attributes
```

### Search Validation Process

After excerpts are extracted in Stage 1, the system:

1. **Collects all non-empty excerpts** from the extraction stage
2. **Performs batch search** using a search tool (default: Tavily) to find external evidence for each excerpt
3. **Stores search results** with each excerpt for later analysis
4. **Assesses hallucination risk** (Stage 1.5) by comparing each excerpt against its search results:

    - `none`: Search strongly supports the excerpt with multiple corroborating sources
    - `low`: Search generally supports the excerpt with minor discrepancies
    - `medium`: Search provides mixed evidence or contradictions
    - `high`: Search contradicts the excerpt or provides no supporting evidence

5. **Uses risk scores** in Stage 2 reasoning to inform confidence in each attribute
6. **Calculates attribute-level risk** as the maximum risk across all excerpts for that attribute


### Benefits

**Enhanced Transparency**:

- See not just what the LLM said, but whether external sources support it
- Understand which claims have strong vs. weak external validation

**Hallucination Detection**:

- Automatically flag excerpts that cannot be verified externally
- Identify potential model confabulations or outdated information

**Audit Trail**:

- Full record of search results used to validate each excerpt
- Per-excerpt and per-attribute risk assessments

### Example

```python
Question: "What is the approved drug target of Venetoclax?"

LLM Response: "Venetoclax targets the BCL-2 protein, which is an anti-apoptotic protein."

Standard Deep-Judgment:
  Excerpt: "targets the BCL-2 protein"
  Confidence: high
  Reasoning: "The excerpt explicitly states BCL-2 as the target protein"
  drug_target: "BCL-2" ✓

Search-Enhanced Deep-Judgment:
  Excerpt: "targets the BCL-2 protein"
  Confidence: high
  Search Results: [
    {"title": "Venetoclax - Wikipedia", "content": "Venetoclax is a BCL-2 inhibitor..."},
    {"title": "FDA Drug Label", "content": "Mechanism: Inhibits BCL-2 protein..."}
  ]
  Hallucination Risk: none (strong external corroboration)
  Reasoning: "The excerpt is strongly supported by multiple authoritative sources"
  drug_target: "BCL-2" ✓
```

### Enabling Search Enhancement

Enable search-enhanced deep-judgment in your verification configuration:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig

# Configure models
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

# Enable search-enhanced deep-judgment
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True,              # Enable deep-judgment
    deep_judgment_search_enabled=True,       # Enable search validation
    deep_judgment_search_tool="tavily"       # Use Tavily search (default)
)

results = benchmark.run_verification(config)
```

### Accessing Search Results

When search enhancement is enabled, verification results include additional metadata:

```python
for question_id, result in results.items():
    if result.deep_judgment_search_enabled:
        # Access hallucination risk scores per attribute
        if result.hallucination_risk_assessment:
            for attr, risk in result.hallucination_risk_assessment.items():
                print(f"{attr}: {risk} hallucination risk")

        # Access excerpts with search results
        if result.extracted_excerpts:
            for attr, excerpts in result.extracted_excerpts.items():
                print(f"\nAttribute: {attr}")
                for exc in excerpts:
                    print(f"  Excerpt: {exc['text']}")
                    print(f"  Risk: {exc.get('hallucination_risk', 'unknown')}")

                    # Access search results for this excerpt
                    if 'search_results' in exc:
                        print(f"  Search Results:")
                        for sr in exc['search_results']:
                            print(f"    - {sr.get('title', 'No title')}: {sr['content'][:100]}...")
                            if sr.get('url'):
                                print(f"      URL: {sr['url']}")
```

### Swapping Search Tools

Karenina supports multiple search tool configurations:

#### Built-in Tools

Currently, Tavily is the only built-in search tool:

```python
config = VerificationConfig(
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool="tavily"  # Built-in Tavily search
)
```

**Requirements**: Install Tavily dependencies:
```bash
pip install langchain-community tavily-python
```

Set your Tavily API key:
```bash
export TAVILY_API_KEY="your-api-key"
```

#### Custom Langchain Tools

You can provide any Langchain tool instance that implements the search interface:

```python
from langchain.tools import Tool

def my_custom_search(query: str) -> str:
    """Custom search function that returns JSON array of results."""
    # Your custom search logic here
    results = [
        {
            "title": "Result Title",
            "content": "Result content snippet...",
            "url": "https://example.com/source"
        }
    ]
    return json.dumps(results)

# Create Langchain tool
custom_tool = Tool(
    name="my_search",
    func=my_custom_search,
    description="Custom search tool"
)

# Use in verification config
config = VerificationConfig(
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool=custom_tool  # Pass custom tool instance
)
```

**Tool Interface Requirements**:

- Input: `str` (single query) or `list[str]` (batch queries)
- Output: JSON string or list of dicts with structure:
  ```python
  [
      {
          "title": str | None,      # Optional: Result title
          "content": str,            # Required: Result content/snippet
          "url": str | None          # Optional: Source URL
      }
  ]
  ```

#### MCP Tools (via Langchain Adapters)

You can use MCP (Model Context Protocol) search tools through langchain adapters:

```python
from langchain_mcp_adapters import create_langchain_tool

# Initialize MCP search tool
mcp_search_tool = create_langchain_tool(
    server="your-mcp-server",
    tool_name="search"
)

# Use in verification config
config = VerificationConfig(
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool=mcp_search_tool
)
```

#### Direct Callable Functions

For simple cases, you can pass a callable function directly:

```python
def simple_search(query: str | list[str]) -> list[dict] | list[list[dict]]:
    """Simple search function."""
    if isinstance(query, list):
        # Handle batch queries
        return [[{"title": None, "content": f"Results for {q}", "url": None}] for q in query]
    else:
        # Handle single query
        return [{"title": None, "content": f"Results for {query}", "url": None}]

config = VerificationConfig(
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool=simple_search  # Pass function directly
)
```

### Performance Considerations

Search enhancement adds significant overhead:

- **Standard deep-judgment**: 3-5 LLM calls per question
- **Search-enhanced deep-judgment**: 3-5 LLM calls + N search queries (where N = total number of excerpts)

**Example overhead**:

- Template with 5 attributes
- 3 excerpts per attribute = 15 total excerpts
- **15 search queries** + hallucination assessment call

**Cost implications**:

- Search API costs (e.g., Tavily: $1-5 per 1000 queries depending on plan)
- Additional LLM call for hallucination assessment
- Increased latency (search queries are batched but still add ~1-3 seconds)

**Recommendation**: Use search enhancement selectively for:

- High-stakes verification where hallucination detection is critical
- Domains with rapidly changing information (e.g., current events, medical research)
- Questions about factual claims that can be externally verified

Avoid for:

- High-volume benchmarks where cost/latency are concerns
- Subjective or opinion-based questions
- Information that cannot be found via web search

---

## Enabling Deep-Judgment

Deep-judgment is disabled by default. Enable it in your verification configuration:

### Basic Configuration

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig
from pathlib import Path

# Load benchmark
benchmark = Benchmark.load(Path("genomics_benchmark.jsonld"))

# Configure models
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    interface="langchain"
)

# Enable deep-judgment
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True  # Enable deep-judgment parsing
)

# Run verification
results = benchmark.run_verification(config)
```

### Advanced Configuration

You can tune deep-judgment behavior with additional parameters:

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],

    # Deep-judgment configuration
    deep_judgment_enabled=True,                        # Enable feature
    deep_judgment_max_excerpts_per_attribute=3,        # Excerpts per attribute (1-5)
    deep_judgment_fuzzy_match_threshold=0.80,          # Similarity threshold (0.0-1.0)
    deep_judgment_excerpt_retry_attempts=2,            # Retry attempts (0-5)
)
```

**Configuration Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deep_judgment_enabled` | `bool` | `False` | Enable/disable deep-judgment parsing |
| `deep_judgment_max_excerpts_per_attribute` | `int` | `3` | Maximum excerpts to extract per attribute (1-5) |
| `deep_judgment_fuzzy_match_threshold` | `float` | `0.80` | Similarity threshold for excerpt validation (0.0-1.0). Higher = stricter. |
| `deep_judgment_excerpt_retry_attempts` | `int` | `2` | Retry attempts when excerpt validation fails (0-5) |

---

## Understanding Results

When deep-judgment is enabled, verification results include additional metadata about the extraction process.

### Result Fields

```python
# Access deep-judgment results
for question_id, result in results.items():
    if result.deep_judgment_performed:
        # Metadata
        print(f"Question: {result.question_text}")
        print(f"Stages Completed: {result.deep_judgment_stages_completed}")
        print(f"Model Calls: {result.deep_judgment_model_calls}")
        print(f"Excerpt Retries: {result.deep_judgment_excerpt_retry_count}")

        # Extracted excerpts
        if result.extracted_excerpts:
            for attr, excerpts in result.extracted_excerpts.items():
                print(f"\n  Attribute: {attr}")
                for exc in excerpts:
                    if exc.get("explanation"):
                        # Missing excerpt with explanation
                        print(f"    [Missing] {exc['explanation']}")
                    else:
                        # Found excerpt
                        print(f"    \"{exc['text']}\"")
                        print(f"    Confidence: {exc['confidence']}")
                        print(f"    Similarity: {exc['similarity_score']:.2f}")

        # Reasoning traces
        if result.attribute_reasoning:
            print(f"\nReasoning:")
            for attr, reasoning in result.attribute_reasoning.items():
                print(f"  {attr}: {reasoning}")

        # Missing excerpts (triggers auto-fail)
        if result.attributes_without_excerpts:
            print(f"\n⚠️  Auto-Fail: Missing excerpts for {', '.join(result.attributes_without_excerpts)}")
```

### Excerpt Structure

Each excerpt includes:

```python
{
    "text": str,              # Verbatim quote from response (empty if missing)
    "confidence": str,        # "low" | "medium" | "high" | "none"
    "similarity_score": float, # 0.0-1.0 (fuzzy match validation score)
    "explanation": str        # Optional: why excerpt couldn't be found
}
```

**Example with excerpts**:
```python
{
    "drug_target": [
        {
            "text": "targets BCL-2 protein",
            "confidence": "high",
            "similarity_score": 0.95
        },
        {
            "text": "inhibits BCL-2",
            "confidence": "medium",
            "similarity_score": 0.87
        }
    ]
}
```

**Example without excerpts**:
```python
{
    "mechanism": [
        {
            "text": "",
            "confidence": "none",
            "similarity_score": 0.0,
            "explanation": "Response does not mention the mechanism of action"
        }
    ]
}
```

---

## Complete Example

Here's a complete workflow using deep-judgment with a genomics benchmark:

```python
from karenina import Benchmark
from karenina.schemas import VerificationConfig, ModelConfig
from pathlib import Path

# 1. Create benchmark
benchmark = Benchmark.create(
    name="Drug Mechanisms Benchmark",
    description="Testing LLM knowledge of drug targets and mechanisms",
    version="1.0.0"
)

# 2. Add questions
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"}
)

benchmark.add_question(
    question="What is the mechanism of action of Venetoclax?",
    raw_answer="Inhibits BCL-2 protein to promote apoptosis",
    author={"name": "Pharma Curator"}
)

# 3. Generate templates
model_config = ModelConfig(
    id="gpt-4.1-mini",
    model_provider="openai",
    model_name="gpt-4.1-mini",
    temperature=0.1,
    interface="langchain"
)

benchmark.generate_all_templates(model_config=model_config)

# 4. Run verification WITH deep-judgment
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True,
    deep_judgment_max_excerpts_per_attribute=3,
    deep_judgment_fuzzy_match_threshold=0.80,
    deep_judgment_excerpt_retry_attempts=2
)

results = benchmark.run_verification(config)

# 5. Analyze deep-judgment results
print("\n=== Deep-Judgment Results ===\n")

for question_id, result in results.items():
    question = benchmark.get_question(question_id)

    print(f"Question: {question.question}")
    print(f"Verification: {'✓ PASS' if result.verify_result else '✗ FAIL'}")

    if result.deep_judgment_performed:
        print(f"Stages: {', '.join(result.deep_judgment_stages_completed)}")
        print(f"Model Calls: {result.deep_judgment_model_calls}")

        # Show excerpts
        if result.extracted_excerpts:
            print("\nExcerpts:")
            for attr, excerpts in result.extracted_excerpts.items():
                print(f"  {attr}:")
                for exc in excerpts:
                    if exc.get("explanation"):
                        print(f"    [Missing] {exc['explanation']}")
                    else:
                        print(f"    \"{exc['text']}\" ({exc['confidence']} confidence)")

        # Show reasoning
        if result.attribute_reasoning:
            print("\nReasoning:")
            for attr, reasoning in result.attribute_reasoning.items():
                print(f"  {attr}: {reasoning[:100]}...")

        # Show auto-fail status
        if result.attributes_without_excerpts:
            print(f"\n⚠️  AUTO-FAIL: Missing excerpts for {', '.join(result.attributes_without_excerpts)}")

    print("\n" + "-" * 60 + "\n")

# 6. Save results
benchmark.save(Path("drug_mechanisms_with_deep_judgment.jsonld"))
benchmark.export_verification_results_to_file(
    file_path=Path("deep_judgment_results.csv"),
    format="csv"
)

print("✓ Results saved with deep-judgment metadata")
```

---

## Use Cases

### When to Use Deep-Judgment

✅ **High-stakes evaluation** where evidence transparency is critical:

- Medical diagnosis benchmarks
- Legal document analysis
- Scientific fact-checking
- Regulatory compliance

✅ **Debugging parsing failures**:

- Understanding why verification fails
- Identifying gaps in LLM responses
- Refining question or template design

✅ **Quality assurance**:

- Ensuring responses contain sufficient evidence
- Validating that answers aren't just plausible-sounding
- Auditing LLM reasoning processes

✅ **Research applications**:

- Studying how LLMs construct answers
- Analyzing excerpt quality patterns
- Comparing evidence provision across models

### When NOT to Use Deep-Judgment

❌ **High-volume verification** where speed is critical:

- Deep-judgment is 3-5x slower than standard parsing
- Uses 3-5 LLM calls per question vs. 1 call for standard

❌ **Low-stakes evaluation** where audit trails aren't needed:

- Quick prototyping
- Informal testing
- Cost-sensitive applications

❌ **Questions with templates that don't need evidence**:

- Multiple choice questions
- True/false questions
- Questions where the answer is self-evident

---

## Performance Considerations

### Execution Time

Deep-judgment significantly increases verification time:

- **Standard parsing**: 1 LLM call per question (~500-2000ms)
- **Deep-judgment parsing**: 3-5 LLM calls per question (~1500-10000ms)

  - Stage 1 (excerpts): 1 call + retries
  - Stage 2 (reasoning): 1 call
  - Stage 3 (parameters): 1 call

**Impact**: 3-5x slower than standard verification

### Cost Impact

Deep-judgment uses 3-5 LLM calls per question compared to 1 call for standard parsing:

- **Standard**: 1 call × token count
- **Deep-judgment**: 3-5 calls × token count

**Impact**: 3-5x higher API costs

### Recommendation

Use deep-judgment **selectively** for questions where transparency is critical, and use standard parsing for the rest. You can enable deep-judgment for specific verification runs without modifying your benchmark.

---

## Configuration Tips

### Fuzzy Match Threshold

Controls how strictly excerpts must match the original response:

```python
# Lenient matching (accepts paraphrases)
deep_judgment_fuzzy_match_threshold=0.70

# Default matching (balanced)
deep_judgment_fuzzy_match_threshold=0.80

# Strict matching (only very close matches)
deep_judgment_fuzzy_match_threshold=0.90
```

**Trade-offs**:

- **Lower threshold (0.60-0.75)**: More lenient, may accept paraphrased excerpts
- **Higher threshold (0.85-0.95)**: Stricter, only accepts near-exact matches

### Excerpt Retry Attempts

Controls how many times to retry excerpt extraction if validation fails:

```python
# No retries (fast but may miss excerpts)
deep_judgment_excerpt_retry_attempts=0

# Default retries (balanced)
deep_judgment_excerpt_retry_attempts=2

# More retries (thorough but slower)
deep_judgment_excerpt_retry_attempts=5
```

**Trade-offs**:

- **Fewer retries**: Faster but may miss valid excerpts
- **More retries**: More thorough but slower and more expensive

### Max Excerpts Per Attribute

Controls how many supporting excerpts to extract:

```python
# Minimal (one excerpt per attribute)
deep_judgment_max_excerpts_per_attribute=1

# Default (multiple perspectives)
deep_judgment_max_excerpts_per_attribute=3

# Comprehensive (all evidence)
deep_judgment_max_excerpts_per_attribute=5
```

**Trade-offs**:

- **Fewer excerpts**: Faster, simpler results
- **More excerpts**: More comprehensive evidence, but slower

---

## Best Practices

### 1. Start with Standard Parsing

Begin with standard parsing for your entire benchmark. Only enable deep-judgment when you need to:

- Debug specific parsing failures
- Audit high-stakes results
- Understand model behavior

### 2. Use Clear, Evidence-Based Templates

Design templates that expect explicit evidence in responses:

```python
# Good: Expects specific factual attributes
class DrugAnswer(BaseAnswer):
    drug_target: str = Field(description="The protein target")
    mechanism: str = Field(description="The mechanism of action")

# Less ideal for deep-judgment: Open-ended or subjective
class OpinionAnswer(BaseAnswer):
    is_good: bool = Field(description="Whether the approach is good")
    reasoning: str = Field(description="Why it's good or bad")
```

### 3. Review Missing Excerpt Explanations

When excerpts are missing, read the LLM explanations to understand if:

- The question needs refinement
- The template expects information not in the response
- The answering model's response is incomplete

### 4. Combine with Rubrics

Use deep-judgment for factual correctness (templates) and rubrics for qualitative assessment:

```python
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    deep_judgment_enabled=True,  # Evidence-based verification
    rubric_enabled=True           # Quality assessment
)
```

### 5. Export Results for Analysis

Export deep-judgment metadata to analyze patterns:

```python
# CSV export includes all deep-judgment fields
benchmark.export_verification_results_to_file(
    file_path=Path("results_with_excerpts.csv"),
    format="csv"
)

# Analyze excerpt patterns across results
import pandas as pd

df = pd.read_csv("results_with_excerpts.csv")
print(df["attributes_without_excerpts"].value_counts())
```

---

## Interpreting Auto-Fail Results

When deep-judgment detects missing excerpts, verification automatically fails. Here's how to interpret these results:

### Auto-Fail Conditions

Verification fails when:

1. Deep-judgment is enabled
2. One or more attributes have no supporting excerpts
3. Abstention is NOT detected (abstention takes priority)

### Example Scenario

```
Question: "What is the mechanism of Venetoclax?"
Response: "Venetoclax is a BCL-2 inhibitor."

Template expects:
- drug_target: ✓ (excerpt found: "BCL-2 inhibitor")
- mechanism: ✗ (no excerpt found - missing explanation)

Result:
- verify_result: False (AUTO-FAIL)
- attributes_without_excerpts: ["mechanism"]
- Explanation: "Response identifies drug class but does not explain mechanism"
```

### What to Do

1. **Review the LLM response**: Does it actually contain the missing information?
2. **Check the explanation**: Why couldn't deep-judgment find an excerpt?
3. **Refine your approach**:

   - Adjust the question to be more specific
   - Modify the template to match what's actually answerable
   - Update answering model's system prompt to provide more detail

---

## Related Features

Deep-judgment works alongside other advanced features:

- **[Abstention Detection](abstention-detection.md)**: Detects when LLMs refuse to answer. Takes priority over deep-judgment auto-fail.
- **[Rubrics](../using-karenina/rubrics.md)**: Assess qualitative aspects. Use together with deep-judgment for comprehensive evaluation.
- **[Verification](../using-karenina/verification.md)**: Core verification system. Deep-judgment enhances standard verification with evidence extraction.

---

## Next Steps

- Learn about [Abstention Detection](abstention-detection.md) for handling model refusals
- Explore [Few-Shot Prompting](few-shot.md) for guiding LLM responses
- Review [Verification guide](../using-karenina/verification.md) for core verification concepts
- Check out [Rubrics](../using-karenina/rubrics.md) for qualitative assessment

---

## Related Documentation

- [Verification](../using-karenina/verification.md) - Core verification system
- [Rubrics](../using-karenina/rubrics.md) - Qualitative assessment
- [Abstention Detection](abstention-detection.md) - Handling model refusals
- [Quick Start](../quickstart.md) - Getting started with Karenina
