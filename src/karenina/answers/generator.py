"""
Structured answer template generator for Karenina benchmarks.

This module implements a two-phase structured generation approach:
1. Ground truth extraction from question/answer pairs
2. Field description generation for judge prompts
3. Pydantic class code generation

This replaces the previous example-based prompting system with a more
reliable and standardized approach.
"""

import json
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm

from karenina.llm.interface import init_chat_model_unified
from karenina.questions.reader import read_questions_from_file
from karenina.schemas.answer_class import BaseAnswer  # noqa: F401

if TYPE_CHECKING:
    from karenina.benchmark.models import ModelConfig


class GroundTruthField(BaseModel):
    """Schema describing a single ground-truth attribute."""

    name: str = Field(..., description="Attribute identifier suitable for a Pydantic field name.")
    type: str = Field(..., description="Python/Pydantic type annotation such as 'bool', 'str', 'List[str]'.")
    ground_truth: Any = Field(
        ..., description="The expected correct value for this attribute based on the reference answer."
    )


class GroundTruthSpec(BaseModel):
    """Schema for the ground-truth dictionary definition."""

    attributes: list[GroundTruthField] = Field(
        ..., description="Ordered collection of attributes required for judgement."
    )

    @field_validator("attributes")
    @classmethod
    def validate_no_string_attributes(cls, v: list[GroundTruthField]) -> list[GroundTruthField]:
        """Ensure no free string attributes are allowed."""
        forbidden_types = ["str", "List[str]", "Dict[str, str]"]
        errors = []

        for attr in v:
            if attr.type in forbidden_types:
                errors.append(
                    f"Attribute '{attr.name}' uses forbidden type '{attr.type}'. Use boolean attributes instead."
                )
            elif "str" in attr.type and not attr.type.startswith("Literal"):
                errors.append(
                    f"Attribute '{attr.name}' uses type '{attr.type}' which contains strings. Use boolean attributes or Literal types instead."
                )

        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        return v


class AttributeDescriptions(BaseModel):
    """Schema capturing instructions for each attribute."""

    field_descriptions: dict[str, str] = Field(
        ..., description="Mapping from attribute name to guidance text for judge prompts."
    )


class JSONOnlyOutputParser(BaseOutputParser[Any]):
    """Parser ensuring output is valid JSON before delegating to Pydantic parser."""

    def __init__(self, inner: PydanticOutputParser[Any]):
        self._inner = inner

    def parse(self, text: str) -> Any:
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Model response is not valid JSON") from exc
        return self._inner.parse(text)

    @property
    def _type(self) -> str:
        """Return the type of parser for LangChain compatibility."""
        return "json_only_output_parser"


# System prompts adapted from structured_generator.py
GROUND_TRUTH_SYSTEM_PROMPT = """
You are an expert evaluation designer extracting ground-truth attributes from a question and its ideal answer. Build a Pydantic-friendly schema capturing what a judge model should read from candidate responses. Apply the following rules when specifying attributes:

- Prefer concise snake_case names that are stable and unambiguous.
- FORBIDDEN: Never use `str`, `List[str]`, or `Dict[str, str]` types. All text-based evaluations must be converted to boolean checks.
- Use `bool` whenever the judge needs to confirm whether a concept, entity, or pattern is present. This is the primary type for text-based evaluation.
- Use numeric types (int, float) only when measurable quantities are required.
- When the answer implies a categorical classification or grading scheme, use `Literal` types to enumerate all reasonable values in that domain.
- For lists of items (e.g., multiple drugs, genes, etc.), create separate boolean attributes for each expected item rather than using List[str].
- When the reference answer contains compound terms or phrases, treat them as single semantic units and create one boolean attribute for the complete concept, not separate attributes for individual words.
- Avoid redundant attributes; ensure each serves a unique decision-making purpose.
- Frame every attribute as something the judge can extract by reading the candidate response (e.g., `number_of_interacting_genes` to count genes mentioned, `mentions_control_group` to flag a concept).
- For each attribute, derive the `ground_truth` value from the reference answer that represents the expected correct response.
- Ensure the final response is valid JSON without trailing commentary.

Example JSON output:
{{
  "attributes": [
    {{
      "name": "count_of_items",
      "type": "int",
      "ground_truth": 3
    }},
    {{
      "name": "mentions_first_concept",
      "type": "bool",
      "ground_truth": true
    }},
    {{
      "name": "mentions_second_concept",
      "type": "bool",
      "ground_truth": false
    }},
    {{
      "name": "classification_level",
      "type": "Literal['high', 'medium', 'low']",
      "ground_truth": "high"
    }}
  ]
}}
""".strip()

GROUND_TRUTH_USER_PROMPT_TEMPLATE = """
You receive an evaluation sample consisting of a question and its reference answer:

Question:
{question}

Reference Answer:
{answer}

Identify the minimal set of structured attributes that a judge must extract from a candidate response to verify correctness. Construct a JSON object with a single key `attributes` containing a list of attribute definitions. Each definition must include `name`, `type`, and `ground_truth` fields, where `ground_truth` contains the expected correct value derived from the reference answer.

When selecting types, follow these guidelines:
- FORBIDDEN: Never use `str`, `List[str]`, or `Dict[str, str]` types.
- Use `bool` to capture presence or absence of concepts, entities, or patterns. This is the primary evaluation mechanism.
- When the reference answer suggests a categorical classification or scale, use `Literal` types with all reasonable values in that domain.
- For multiple items in the reference answer, create separate boolean attributes for each item instead of using lists.
- When the reference answer contains compound terms or phrases, treat them as single semantic units and create one boolean attribute for the complete concept, not separate attributes for individual words.

Return only valid JSON.
""".strip()

FIELD_DESCRIPTION_SYSTEM_PROMPT = """
You craft instructional text for judge models who must parse a candidate response to a question. For every attribute in the provided ground-truth specification, produce a short, direct description that explains exactly what the judge should read for in the response and how to answer. Highlight boolean expectations clearly.

Guidelines:
- Reference attribute names verbatim.
- Mention the expected type implicitly via phrasing (e.g., "Answer with true or false if ..." for booleans, "Provide the count of ..." for numeric fields).
- For boolean attributes checking concept presence, allow semantic equivalence and related terms rather than requiring exact string matches. Focus on whether the underlying concept is conveyed.
- When relevant, reference concrete response-focused examples such as "Number of interacting genes mentioned in the response" or "Does the response cite a control group?" to reinforce that extraction happens from the candidate answer.
- Stay concise (<= 2 sentences per attribute).
- Return only valid JSON.

Example mapping:
{{
  "field_descriptions": {{
    "count_of_items": "Provide an integer equal to the number of items mentioned in the response.",
    "mentions_first_concept": "Answer with true if the response refers to the first concept or semantically related terms; otherwise answer false.",
    "mentions_second_concept": "Answer with true if the response refers to the second concept or semantically related terms; otherwise answer false.",
    "classification_level": "Select the classification level mentioned in the response from the available options."
  }}
}}
""".strip()

FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE = """
Question:
{question}

Reference Answer:
{answer}

Ground-truth attribute specification:
{spec_json}

Produce JSON with key `field_descriptions` mapping attribute names to their instructional descriptions for judge prompts.

Ensure descriptions communicate type expectations, especially emphasizing when boolean values should be used to flag presence of concepts. For boolean concept checks, focus on semantic meaning rather than exact word matching. Make it explicit that the judge is reading the candidate response to this question.

Return only valid JSON.
""".strip()


def _build_chain(stage: str, config: "ModelConfig") -> Any:
    """Build generation chain for a specific stage."""
    if stage == "ground_truth":
        parser = JSONOnlyOutputParser(inner=PydanticOutputParser(pydantic_object=GroundTruthSpec))
        system_prompt = GROUND_TRUTH_SYSTEM_PROMPT
        user_template = GROUND_TRUTH_USER_PROMPT_TEMPLATE
    elif stage == "field_descriptions":
        parser = JSONOnlyOutputParser(inner=PydanticOutputParser(pydantic_object=AttributeDescriptions))
        system_prompt = FIELD_DESCRIPTION_SYSTEM_PROMPT
        user_template = FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    # No custom instructions in new structured approach

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_template),
        ]
    )

    # Use karenina's unified model interface
    model = init_chat_model_unified(
        model=config.model_name,
        provider=config.model_provider,
        interface=config.interface,
        temperature=config.temperature,
    )

    return prompt | model | parser


def _generate_with_retry(
    stage: str,
    inputs: dict[str, Any],
    config: "ModelConfig",
    max_retries: int = 2,
) -> Any:
    """Generate output with retry logic on validation failures."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Build/rebuild chain for each attempt (for error injection)
            chain = _build_chain(stage, config)

            # Add error context on retry
            if attempt > 0 and last_error:
                error_context = f"\n\nPREVIOUS ATTEMPT FAILED with error: {last_error}\nPlease fix the validation issues and try again."

                if stage == "ground_truth":
                    system_prompt = GROUND_TRUTH_SYSTEM_PROMPT + error_context
                    user_template = GROUND_TRUTH_USER_PROMPT_TEMPLATE
                    parser = JSONOnlyOutputParser(inner=PydanticOutputParser(pydantic_object=GroundTruthSpec))
                else:
                    system_prompt = FIELD_DESCRIPTION_SYSTEM_PROMPT + error_context
                    user_template = FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE
                    parser = JSONOnlyOutputParser(inner=PydanticOutputParser(pydantic_object=AttributeDescriptions))

                # No custom instructions in new structured approach

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("user", user_template),
                    ]
                )

                model = init_chat_model_unified(
                    model=config.model_name,
                    provider=config.model_provider,
                    interface=config.interface,
                    temperature=config.temperature,
                )

                chain = prompt | model | parser

            result = chain.invoke(inputs)
            return result

        except (ValidationError, ValueError) as e:
            last_error = str(e)
            if attempt == max_retries:
                raise ValueError(f"Failed after {max_retries + 1} attempts. Last error: {last_error}") from None
            continue

    raise ValueError("Unexpected error in retry logic")


def _generate_structured_outputs(
    question: str,
    answer: str,
    config: "ModelConfig",
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate structured outputs (ground truth + field descriptions) using two-phase approach."""
    # Phase 1: Generate ground truth specification
    gt_result = _generate_with_retry("ground_truth", {"question": question, "answer": answer}, config, max_retries)

    gt_json = gt_result.model_dump() if isinstance(gt_result, BaseModel) else gt_result

    # Phase 2: Generate field descriptions
    # Remove ground_truth values from spec for field description generation
    spec_for_descriptions = {
        "attributes": [{k: v for k, v in attr.items() if k != "ground_truth"} for attr in gt_json["attributes"]]
    }

    fd_result = _generate_with_retry(
        "field_descriptions",
        {
            "question": question,
            "answer": answer,
            "spec_json": json.dumps(spec_for_descriptions, ensure_ascii=False),
        },
        config,
        max_retries,
    )

    return {
        "attributes": gt_json["attributes"],
        "field_descriptions": fd_result.field_descriptions,
    }


def _python_type_to_annotation(type_str: str) -> str:
    """Convert a Python type string to proper type annotation."""
    # Handle Literal types
    if type_str.startswith("Literal"):
        return type_str

    # Handle basic types
    type_mapping = {"bool": "bool", "int": "int", "float": "float", "str": "str"}

    # Handle List types
    if type_str.startswith("List["):
        inner_type = type_str[5:-1]  # Extract inner type from List[...]
        return f"List[{_python_type_to_annotation(inner_type)}]"

    # Handle Dict types
    if type_str.startswith("Dict["):
        # Extract key, value types from Dict[key, value]
        inner = type_str[5:-1]
        key_type, value_type = inner.split(", ")
        return f"Dict[{_python_type_to_annotation(key_type)}, {_python_type_to_annotation(value_type)}]"

    return type_mapping.get(type_str, type_str)


def _generate_verification_logic(attr_name: str, attr_type: str, tolerance: float = 0.001) -> str:
    """Generate verification logic for a specific attribute type."""
    if attr_type == "float":
        return f'abs(self.{attr_name} - self.correct["{attr_name}"]) <= {tolerance}'
    elif attr_type in ["bool", "int", "str"] or attr_type.startswith("Literal"):
        return f'self.{attr_name} == self.correct["{attr_name}"]'
    elif attr_type.startswith("List["):
        # For lists, compare as sets to ignore order or do exact comparison
        return f'set(self.{attr_name}) == set(self.correct["{attr_name}"])'
    elif attr_type.startswith("Dict["):
        return f'self.{attr_name} == self.correct["{attr_name}"]'
    else:
        # Default to equality check
        return f'self.{attr_name} == self.correct["{attr_name}"]'


def _format_ground_truth_value(value: Any) -> str:
    """Format ground truth value for Python code."""
    if isinstance(value, str):
        return repr(value)  # Handles quotes and escaping
    elif isinstance(value, bool | int | float):
        return str(value)
    elif isinstance(value, list | dict):
        return repr(value)
    else:
        return repr(value)


def _generate_pydantic_class(
    spec_dict: dict[str, Any], class_name: str = "Answer", float_tolerance: float = 0.001
) -> str:
    """Generate a Pydantic class from structured generator output."""
    attributes = spec_dict["attributes"]
    field_descriptions = spec_dict["field_descriptions"]

    # Start building the class
    lines = []

    # Class definition (no imports - they'll be added by existing system)
    lines.append(f"class {class_name}(BaseAnswer):")

    # Field definitions
    for attr in attributes:
        attr_name = attr["name"]
        attr_type = attr["type"]
        description = field_descriptions[attr_name]

        # Convert type to proper annotation
        type_annotation = _python_type_to_annotation(attr_type)

        # Add field definition
        field_def = f'    {attr_name}: {type_annotation} = Field(description="{description}")'
        lines.append(field_def)

    lines.append("")

    # model_post_init method
    lines.append("    def model_post_init(self, __context):")

    # Build correct dictionary
    correct_dict_items = []
    for attr in attributes:
        attr_name = attr["name"]
        ground_truth_value = _format_ground_truth_value(attr["ground_truth"])
        correct_dict_items.append(f'"{attr_name}": {ground_truth_value}')

    correct_dict = "{" + ", ".join(correct_dict_items) + "}"
    lines.append(f"        self.correct = {correct_dict}")
    lines.append("")

    # verify method
    lines.append("    def verify(self) -> bool:")
    if len(attributes) == 1:
        # Single attribute - simple check
        attr = attributes[0]
        verification_logic = _generate_verification_logic(attr["name"], attr["type"], float_tolerance)
        lines.append(f"        return {verification_logic}")
    else:
        # Multiple attributes - all must pass
        lines.append("        return (")
        verification_conditions = []
        for attr in attributes:
            verification_logic = _generate_verification_logic(attr["name"], attr["type"], float_tolerance)
            verification_conditions.append(f"            {verification_logic}")

        lines.append(" and\n".join(verification_conditions))
        lines.append("        )")

    lines.append("")

    # verify_granular method (only if multiple attributes)
    if len(attributes) > 1:
        lines.append("    def verify_granular(self) -> float:")
        lines.append("        correct_count = 0")
        lines.append("        total_count = " + str(len(attributes)))
        lines.append("")

        for attr in attributes:
            verification_logic = _generate_verification_logic(attr["name"], attr["type"], float_tolerance)
            lines.append(f"        if {verification_logic}:")
            lines.append("            correct_count += 1")

        lines.append("")
        lines.append("        return correct_count / total_count")

    # Generate the class code
    return "\n".join(lines)


def inject_question_id_into_answer_class(answer_class: type, question_id: str) -> type:
    """
    Programmatically inject the question ID into an Answer class.

    This creates a new class that automatically sets the question ID when instantiated,
    replacing the need for LLM-generated ID assignment in model_post_init.

    Args:
        answer_class: The Answer class generated by LLM
        question_id: The question ID to inject

    Returns:
        A new Answer class with programmatic ID assignment
    """

    class AnswerWithID(answer_class):  # type: ignore[misc]
        def model_post_init(self, __context: Any) -> None:
            # Call the original model_post_init if it exists
            if hasattr(super(), "model_post_init"):
                super().model_post_init(__context)
            # Set the question ID programmatically
            self.id = question_id

    # Preserve the original class name and metadata
    AnswerWithID.__name__ = answer_class.__name__
    AnswerWithID.__qualname__ = answer_class.__qualname__

    # Preserve the original source code
    if hasattr(answer_class, "_source_code"):
        AnswerWithID._source_code = answer_class._source_code

    return AnswerWithID


def generate_answer_template(
    question: str,
    raw_answer: str,
    model: str | None = None,
    model_provider: str | None = None,
    temperature: float | None = None,
    interface: str | None = None,
    config: "ModelConfig | None" = None,
) -> str:
    """
    Generate an answer template using structured two-phase approach.

    Args:
        question: The question to generate an answer template for.
        raw_answer: The raw answer to the question.
        model: The model to use (when not using config).
        model_provider: The provider of the model (when not using config).
        temperature: The temperature of the model (when not using config).
        interface: The interface to use (when not using config).
        config: ModelConfig object (takes precedence over individual params).

    Returns:
        The python code for the answer template.
    """
    if config is not None:
        # Use provided ModelConfig
        model_config = config
    else:
        # Import ModelConfig dynamically to avoid circular imports
        from karenina.benchmark.models import ModelConfig

        # Create ModelConfig from individual parameters
        model_config = ModelConfig(
            id="template-generator",
            model_name=model or "gemini-2.0-flash",
            model_provider=model_provider or "google_genai",
            temperature=temperature if temperature is not None else 0.0,
            interface=interface or "langchain",  # type: ignore[arg-type]
            system_prompt="",  # Not used in structured generation
        )

    # Phase 1 & 2: Generate structured outputs (ground truth + field descriptions)
    max_retries = getattr(model_config, "max_retries", 2)  # Default to 2 if not set
    spec = _generate_structured_outputs(question, raw_answer, model_config, max_retries=max_retries)

    # Phase 3: Generate Pydantic class code
    template_code = _generate_pydantic_class(spec)

    return template_code


def generate_answer_templates_from_questions_file(
    questions_py_path: str,
    model: str = "gemini-2.0-flash",
    model_provider: str = "google_genai",
    interface: str = "langchain",
    return_blocks: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, str]]:
    """
    Given a path to a questions.py file, dynamically import all_questions from it,
    generate answer templates for each question using the structured approach,
    and return a dictionary mapping question.id to the generated Answer class.
    """
    from karenina.utils.code_parser import extract_and_combine_codeblocks

    # Use the reader function to get all questions
    all_questions = read_questions_from_file(questions_py_path)

    answer_templates = {}
    all_code_blocks = {}
    for _, question in tqdm(enumerate(all_questions)):
        answer_template = generate_answer_template(
            question.question,
            question.raw_answer or "",
            model=model,
            model_provider=model_provider,
            interface=interface,
        )
        # Try to extract code blocks (for old markdown-wrapped responses)
        # If no blocks found, use the answer_template directly (new plain Python responses)
        code_blocks = extract_and_combine_codeblocks(answer_template)
        if not code_blocks:
            code_blocks = answer_template

        # define the class in a local namespace
        local_ns: dict[str, Any] = {}
        exec(code_blocks, globals(), local_ns)
        Answer = local_ns["Answer"]

        # Store the template code for exec-created classes
        Answer._source_code = code_blocks

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, question.id)
        answer_templates[question.id] = AnswerWithID

        if return_blocks:
            all_code_blocks[question.id] = code_blocks

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates


def load_answer_templates_from_json(
    json_file_path: str, return_blocks: bool = False
) -> dict[str, type] | tuple[dict[str, type], dict[str, str]]:
    """
    Load answer templates from a JSON file containing code blocks.

    Args:
        json_file_path: Path to the JSON file containing code blocks
        return_blocks: Whether to also return the code blocks dictionary

    Returns:
        If return_blocks is False: Dictionary mapping question IDs to Answer classes
        If return_blocks is True: Tuple of (answer_templates, code_blocks)
    """
    # Read the JSON file
    with open(json_file_path) as f:
        all_code_blocks = json.load(f)

    answer_templates = {}
    for question_id, code_blocks in all_code_blocks.items():
        # Define the class in a local namespace
        local_ns: dict[str, Any] = {}
        exec(code_blocks, globals(), local_ns)
        Answer = local_ns["Answer"]

        # Store the template code for exec-created classes
        Answer._source_code = code_blocks

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, question_id)
        answer_templates[question_id] = AnswerWithID

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates
