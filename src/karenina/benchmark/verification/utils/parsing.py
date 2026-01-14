"""Shared utility functions for parsing operations and template utilities.

Note: Template-specific parsing functions are also available in
evaluators/template_parsing.py which is used by TemplateEvaluator.
For new code, prefer using TemplateEvaluator for template parsing operations.
"""

import json
import logging
import re
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ....schemas.domain import BaseAnswer
from ....schemas.shared import SearchResultItem

# Import and re-export _invoke_llm_with_retry with usage tracking support
from ..verification_utils import _invoke_llm_with_retry

# Import shared utilities (consolidated from duplicated code)
from .shared import extract_json_from_text as _extract_json_from_text
from .shared import strip_markdown_fences as _strip_markdown_fences

__all__ = [
    "_invoke_llm_with_retry",
    "invoke_with_structured_output_for_template",
    "parse_template_response",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def invoke_with_structured_output_for_template(
    llm: Any,
    messages: list[Any],  # BaseMessage or dict
    answer_class: type[T],
) -> tuple[T | None, dict[str, Any], bool]:
    """
    Invoke LLM with structured output for template parsing.

    Uses a two-strategy approach:
    1. json_schema method (native structured output) - for providers that support it
    2. Returns None if structured output fails (caller should use fallback parsing)

    Unlike rubric parsing which always returns a result, this returns None on failure
    to allow the caller to fall back to enhanced PydanticOutputParser flow with retries.

    Args:
        llm: LangChain chat model
        messages: List of messages to send
        answer_class: Pydantic model (user-defined Answer class) for structured output

    Returns:
        Tuple of (parsed_result_or_none, usage_metadata, used_structured_output)
        - parsed_result_or_none: Parsed Answer instance, or None if structured output failed
        - usage_metadata: Token usage from the LLM call
        - used_structured_output: True if native structured output succeeded
    """
    from langchain_core.callbacks import get_usage_metadata_callback

    usage_metadata: dict[str, Any] = {}

    # Try json_schema method (native structured output)
    try:
        structured_llm = llm.with_structured_output(answer_class, method="json_schema")

        with get_usage_metadata_callback() as cb:
            result = structured_llm.invoke(messages)

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

        if isinstance(result, answer_class):
            logger.debug(f"json_schema method succeeded for template {answer_class.__name__}")
            return result, usage_metadata, True

    except Exception as e:
        logger.debug(f"json_schema method failed for template: {e}")

    # Return None to signal caller should use fallback parsing
    return None, usage_metadata, False


def parse_template_response(response: str | Any, answer_class: type[T]) -> T:
    """
    Parse raw LLM response into Answer class with multiple fallback strategies.

    Strategy order:
    1. Already a model instance (pass through)
    2. Direct JSON parsing
    3. JSON extraction from mixed text (markdown fences, etc.)
    4. JSON repair with jsonrepair

    Args:
        response: Raw string response from LLM (or already parsed object)
        answer_class: Pydantic model (user-defined Answer class) to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If all parsing strategies fail
    """
    # Already a model instance
    if isinstance(response, answer_class):
        return response

    # Convert to string if needed
    if not isinstance(response, str):
        response = str(response)

    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response.strip())
        return answer_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"Direct JSON parse failed for template: {e}")

    # Strategy 2: Extract JSON from mixed text (handles markdown fences)
    cleaned = _strip_markdown_fences(response)
    if cleaned and cleaned != response:
        # Fences were stripped, try parsing the cleaned text
        try:
            data = json.loads(cleaned)
            return answer_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Cleaned text parse failed for template: {e}")

    # Try extracting JSON object from text
    json_str = _extract_json_from_text(response)
    if json_str:
        try:
            data = json.loads(json_str)
            return answer_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Extracted JSON parse failed for template: {e}")

    # Strategy 3: JSON repair for malformed JSON
    try:
        from json_repair import repair_json

        repaired = repair_json(response)
        data = json.loads(repaired)
        logger.debug(f"JSON repair succeeded for template {answer_class.__name__}")
        return answer_class.model_validate(data)
    except ImportError:
        logger.warning("json-repair not installed, skipping repair strategy")
    except Exception as e:
        logger.debug(f"JSON repair failed for template: {e}")

    # Strategy 4: Try repair on cleaned text
    if cleaned and cleaned != response:
        try:
            from json_repair import repair_json

            repaired = repair_json(cleaned)
            data = json.loads(repaired)
            logger.debug(f"JSON repair on cleaned text succeeded for template {answer_class.__name__}")
            return answer_class.model_validate(data)
        except Exception as e:
            logger.debug(f"JSON repair on cleaned text failed for template: {e}")

    # All strategies failed
    preview = response[:200] if len(response) > 200 else response
    raise ValueError(f"Could not parse response into {answer_class.__name__}: {preview}")


def _extract_attribute_names_from_class(answer_class: type[BaseAnswer]) -> list[str]:
    """Extract attribute names from Answer class, excluding configuration fields.

    This function extracts all field names from a Pydantic BaseAnswer subclass,
    excluding special configuration fields that are not part of the actual answer content.

    Args:
        answer_class: A Pydantic BaseAnswer subclass with model_fields

    Returns:
        List of attribute names (strings) excluding: 'id', 'correct', 'regex'

    Excluded fields:
        - 'id': Question ID (metadata, not an answer field)
        - 'correct': Ground truth indicator (not extracted from LLM response)
        - 'regex': Regex validation configuration (not an answer field)

    Example:
        >>> class MyAnswer(BaseAnswer):
        ...     id: str
        ...     correct: str
        ...     drug_target: str
        ...     mechanism: str
        ...     confidence: str
        >>> _extract_attribute_names_from_class(MyAnswer)
        ['drug_target', 'mechanism', 'confidence']
    """
    # Get model fields from Pydantic v2
    if hasattr(answer_class, "model_fields"):
        model_fields = answer_class.model_fields
        field_names = list(model_fields.keys())
    else:
        # Fallback for older Pydantic versions
        fields = answer_class.__fields__
        field_names = list(fields.keys())  # type: ignore[attr-defined]

    # Exclude configuration fields that aren't part of the answer content
    # 'id': Question ID (metadata)
    # 'correct': Ground truth field (not extracted from LLM response)
    # 'regex': Regex validation configuration (not an answer field)
    return [name for name in field_names if name not in ("id", "correct", "regex")]


def _extract_attribute_descriptions(json_schema: str, attribute_names: list[str]) -> dict[str, str]:
    """Extract attribute descriptions from JSON schema format instructions.

    Parses the JSON schema string to extract description fields for each attribute.
    This is used to provide guidance to the LLM about what evidence to look for
    without requiring it to follow the full schema format.

    Args:
        json_schema: JSON schema string from PydanticOutputParser.get_format_instructions()
        attribute_names: List of attribute names to extract descriptions for

    Returns:
        Dictionary mapping attribute names to their descriptions

    Example:
        >>> schema = '{"properties": {"drug_target": {"description": "The target protein", ...}}}'
        >>> _extract_attribute_descriptions(schema, ["drug_target"])
        {"drug_target": "The target protein"}
    """
    attribute_descriptions = {}
    for attr in attribute_names:
        # Extract description from JSON schema using regex
        # Pattern matches: "attr_name": {..., "description": "text", ...}
        pattern = rf'"{attr}":\s*{{[^}}]*"description":\s*"([^"]*)"'
        match = re.search(pattern, json_schema)
        if match:
            attribute_descriptions[attr] = match.group(1)
        else:
            # Fallback if description not found
            attribute_descriptions[attr] = f"Evidence for {attr}"
    return attribute_descriptions


def format_excerpts_for_reasoning(excerpts: dict[str, list[dict[str, Any]]]) -> str:
    """Format excerpts in a human-readable way for the reasoning stage.

    Formats excerpts with clear hierarchical structure, including search results when present.
    This improves LLM comprehension compared to raw JSON dumps.

    Args:
        excerpts: Dictionary mapping attribute names to lists of excerpt objects.
                  Each excerpt object may contain:
                  - text: The excerpt text (displayed in full)
                  - confidence: Confidence level (none/low/medium/high)
                  - similarity_score: Fuzzy match score (0.0-1.0)
                  - search_results: Optional external search validation (displayed in full)
                  - explanation: Optional explanation when no excerpt found

    Returns:
        Human-readable formatted string suitable for LLM reasoning stage
    """
    if not excerpts:
        return "(No excerpts extracted)"

    lines = []
    for attr, excerpt_list in excerpts.items():
        lines.append(f"{attr}:")

        if not excerpt_list:
            lines.append("  (No excerpts found for this attribute)")
            continue

        for i, excerpt_obj in enumerate(excerpt_list, 1):
            lines.append(f"  Excerpt {i}:")

            # Handle missing excerpt with explanation
            if not excerpt_obj.get("text") and excerpt_obj.get("explanation"):
                lines.append(f"    Explanation: {excerpt_obj['explanation']}")
                continue

            # Regular excerpt with text
            text = excerpt_obj.get("text", "")
            if text:
                lines.append(f'    Text: "{text}"')

            # Add confidence and similarity
            confidence = excerpt_obj.get("confidence", "unknown")
            similarity = excerpt_obj.get("similarity_score", 0.0)
            lines.append(f"    Confidence: {confidence}")
            lines.append(f"    Similarity: {similarity:.3f}")

            # Add hallucination risk if present (from Stage 1.5)
            if "hallucination_risk" in excerpt_obj:
                hallucination_risk = excerpt_obj["hallucination_risk"]
                lines.append(f"    Hallucination Risk: {hallucination_risk.upper()}")
                justification = excerpt_obj.get("hallucination_justification", "")
                if justification:
                    lines.append(f"    Risk Justification: {justification}")

            # Add search results if present
            if "search_results" in excerpt_obj:
                search_results = excerpt_obj["search_results"]
                lines.append("    Search Results:")

                # Handle both string and list formats (list is new structured format)
                if isinstance(search_results, list):
                    # Use the new formatting function for structured results
                    formatted = _format_search_results_for_llm(search_results)
                    search_lines = formatted.split("\n")
                elif isinstance(search_results, str):
                    # Legacy string format
                    search_lines = search_results.split("\n")
                else:
                    # Fallback for unexpected types
                    search_lines = [str(search_results)]

                for search_line in search_lines:
                    lines.append(f"      {search_line}")

        lines.append("")  # Blank line between attributes

    return "\n".join(lines)


def format_reasoning_for_parsing(reasoning: dict[str, str]) -> str:
    """Format reasoning traces in a human-readable way for the parsing stage.

    Formats reasoning traces with clear structure, making it easier for the LLM
    to understand how each attribute should be valued based on the reasoning.

    Args:
        reasoning: Dictionary mapping attribute names to reasoning text strings.
                   Each reasoning text explains how excerpts inform the attribute value.

    Returns:
        Human-readable formatted string suitable for LLM parsing stage
    """
    if not reasoning:
        return "(No reasoning traces generated)"

    lines = []
    for attr, reasoning_text in reasoning.items():
        lines.append(f"{attr}:")
        # Indent the reasoning text for readability
        reasoning_lines = reasoning_text.split("\n")
        for reasoning_line in reasoning_lines:
            lines.append(f"  {reasoning_line}")
        lines.append("")  # Blank line between attributes

    return "\n".join(lines)


def _format_search_results_for_llm(search_results: list[SearchResultItem] | list[dict[str, Any]]) -> str:
    """Format structured search results into human-readable string for LLM prompts.

    This function converts structured search results (SearchResultItem objects or dicts)
    into a formatted string suitable for inclusion in LLM prompts. The formatting is
    designed to be clear and easy for LLMs to understand.

    Handles optional fields gracefully:
    - If title is missing, uses truncated content (first 50 chars)
    - If URL is missing, displays "Source: Not available"

    Args:
        search_results: List of SearchResultItem objects or list of dicts with
                       title/content/url keys. Can be empty.

    Returns:
        Formatted string with numbered search results, one result per block.
        Returns "No search results found." if the list is empty.
    """
    if not search_results:
        return "No search results found."

    formatted_parts = []
    for i, result in enumerate(search_results, 1):
        # Handle both SearchResultItem objects and dicts
        if isinstance(result, SearchResultItem):
            title = result.title
            content = result.content
            url = result.url
        elif isinstance(result, dict):
            title = result.get("title")
            content = result.get("content", "No content")
            url = result.get("url")
        else:
            logger.warning(f"Unexpected search result type in formatting: {type(result)}")
            continue

        # Format with optional fields (use truncated content if no title)
        title_display = title if title else content[:50] + "..."
        url_display = f"Source: {url}" if url else "Source: Not available"

        formatted_parts.append(f"[{i}] {title_display}\n    {content}\n    {url_display}")

    return "\n\n".join(formatted_parts)


def _extract_text_from_search_results(search_results: list[SearchResultItem] | list[dict[str, Any]]) -> str:
    """Extract text content from search results for LLM consumption.

    Args:
        search_results: List of SearchResultItem objects or dicts

    Returns:
        Concatenated text content from all search results
    """
    texts = []
    for result in search_results:
        if isinstance(result, SearchResultItem):
            content = result.content
        elif isinstance(result, dict):
            content = result.get("content", "")
        else:
            continue
        if content:
            texts.append(content)
    return "\n\n".join(texts)


# Template utilities


def create_test_instance_from_answer_class(Answer: type) -> tuple[Any, dict[str, Any] | None]:
    """
    Create a test instance of an Answer class to extract ground truth and validate structure.

    This function provides dummy values for all required fields and then instantiates
    the Answer class to trigger model_post_init and extract the ground truth from
    the 'correct' field.

    Args:
        Answer: The Answer class to instantiate

    Returns:
        Tuple of (test_instance, ground_truth_dict)
        - test_instance: The instantiated Answer object
        - ground_truth_dict: The contents of the 'correct' field (None if not set)

    Raises:
        Exception: If the Answer class cannot be instantiated
    """
    # Get required fields to create a valid test instance
    required_fields: dict[str, Any] = {}
    if hasattr(Answer, "__annotations__"):
        for field_name, field_type in Answer.__annotations__.items():
            if field_name not in ("id", "correct"):  # Skip id and correct fields
                # Provide dummy values for required fields
                if field_type is int or str(field_type) == "int":
                    required_fields[field_name] = 0
                elif field_type is str or str(field_type) == "str":
                    required_fields[field_name] = ""
                elif field_type is float or str(field_type) == "float":
                    required_fields[field_name] = 0.0
                elif field_type is bool or str(field_type) == "bool":
                    required_fields[field_name] = False
                elif field_type is list or str(field_type) == "list":
                    required_fields[field_name] = []
                else:
                    # Handle Literal and other complex types
                    origin = get_origin(field_type)
                    if origin is not None:
                        # Handle Literal types
                        if str(origin) == "typing.Literal":
                            # Get the first literal value
                            args = get_args(field_type)
                            if args:
                                required_fields[field_name] = args[0]
                            else:
                                required_fields[field_name] = ""
                        # Handle List types
                        elif str(origin) == "<class 'list'>" or origin is list:
                            required_fields[field_name] = []
                        else:
                            # Default to empty string for unknown types
                            required_fields[field_name] = ""
                    else:
                        # Default to empty string for unknown types
                        required_fields[field_name] = ""

    # Create test instance to extract ground truth
    test_instance = Answer(**required_fields)

    # Extract ground truth if it exists
    ground_truth = None
    if hasattr(test_instance, "correct"):
        ground_truth = test_instance.correct

    return test_instance, ground_truth


def extract_ground_truth_from_template_code(template_code: str) -> dict[str, Any] | None:
    """
    Extract ground truth from Answer template code by creating a test instance.

    Args:
        template_code: The template code defining an Answer class

    Returns:
        The ground truth dictionary from the 'correct' field, or None if not available

    Raises:
        Exception: If the template code cannot be executed or Answer class cannot be instantiated
    """
    # Execute the template code to get the Answer class
    # Create a namespace with necessary imports
    global_ns = {
        "__builtins__": __builtins__,
        "BaseAnswer": BaseAnswer,
    }

    # Import commonly used pydantic and typing components
    try:
        from pydantic import Field

        global_ns["Field"] = Field
    except ImportError:
        pass

    try:
        from typing import Any, Literal, Optional, Union

        global_ns.update(
            {
                "List": list,
                "Dict": dict,
                "Optional": Optional,
                "Union": Union,
                "Any": Any,
                "Literal": Literal,
            }
        )
    except ImportError:
        pass

    local_ns: dict[str, Any] = {}

    # Execute the template code
    exec(template_code, global_ns, local_ns)

    # Check if Answer class was defined
    if "Answer" not in local_ns:
        raise ValueError("No 'Answer' class found in template code")

    Answer = local_ns["Answer"]

    # Store the template code for exec-created classes
    Answer._source_code = template_code

    # Create test instance and extract ground truth
    _, ground_truth = create_test_instance_from_answer_class(Answer)

    return ground_truth


def extract_rubric_traits_from_template(answer_template: str) -> list[Any]:
    """Extract rubric traits from answer template code.

    Args:
        answer_template: The answer template code string

    Returns:
        List of trait objects found in the template (LLM, regex, callable, or metric)
    """
    try:
        # Prepare minimal execution environment similar to template validation
        from ....schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric

        global_ns = {
            "__builtins__": __builtins__,
            "BaseAnswer": BaseAnswer,
            "Rubric": Rubric,
            "LLMRubricTrait": LLMRubricTrait,
            "RegexTrait": RegexTrait,
            "CallableTrait": CallableTrait,
            "MetricRubricTrait": MetricRubricTrait,
        }
        try:
            from pydantic import Field

            global_ns["Field"] = Field
        except Exception:
            pass
        try:
            from typing import Any, ClassVar, Literal, Optional, Union

            global_ns.update(
                {
                    "List": list,
                    "Dict": dict,
                    "Optional": Optional,
                    "Union": Union,
                    "Any": Any,
                    "Literal": Literal,
                    "ClassVar": ClassVar,
                }
            )
        except Exception:
            pass

        local_ns: dict[str, Any] = {}
        exec(answer_template, global_ns, local_ns)

        # Store the template code for exec-created classes
        if "Answer" in local_ns:
            Answer = local_ns["Answer"]
            Answer._source_code = answer_template

        # Heuristics: check for rubric on Answer class or top-level var
        extracted_traits: list[Any] = []

        def _coerce_traits(obj: Any) -> list[Any]:
            traits_list: list[Any] = []
            if not obj:
                return traits_list
            # If wrapped in Rubric
            if isinstance(obj, Rubric):
                # Collect all trait types
                for llm_trait in obj.llm_traits:
                    if isinstance(llm_trait, LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait):
                        traits_list.append(llm_trait)
                for regex_trait in obj.regex_traits:
                    if isinstance(regex_trait, RegexTrait):
                        traits_list.append(regex_trait)
                for callable_trait in obj.callable_traits:
                    if isinstance(callable_trait, CallableTrait):
                        traits_list.append(callable_trait)
                for metric_trait in obj.metric_traits:
                    if isinstance(metric_trait, MetricRubricTrait):
                        traits_list.append(metric_trait)
                return traits_list
            # If already list of trait objects
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, LLMRubricTrait | RegexTrait | CallableTrait | MetricRubricTrait):
                        traits_list.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        # Try to infer trait type and construct it
                        try:
                            if "kind" in item:
                                # LLMRubricTrait
                                traits_list.append(LLMRubricTrait(**item))
                            elif "pattern" in item:
                                # RegexTrait
                                traits_list.append(RegexTrait(**item))
                            elif "callable_code" in item:
                                # CallableTrait
                                traits_list.append(CallableTrait(**item))
                            elif "metrics" in item:
                                # MetricRubricTrait
                                traits_list.append(MetricRubricTrait(**item))
                        except Exception:
                            continue
            return traits_list

        AnswerCls = local_ns.get("Answer")
        if AnswerCls is not None:
            # Common attribute names that might store rubric traits
            for attr in ("question_rubric", "rubric_traits", "rubric"):
                if hasattr(AnswerCls, attr):
                    extracted_traits = _coerce_traits(getattr(AnswerCls, attr))
                    if extracted_traits:
                        break

        # Also allow a top-level constant like QUESTION_RUBRIC
        if not extracted_traits and "QUESTION_RUBRIC" in local_ns:
            extracted_traits = _coerce_traits(local_ns.get("QUESTION_RUBRIC"))

        return extracted_traits
    except Exception:
        # Silently ignore rubric extraction errors to keep TaskEval lightweight
        return []
