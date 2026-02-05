"""
ADeLe trait conversion and registry.

Converts parsed ADeLe rubrics into LLMRubricTrait objects with kind="literal".
"""

from __future__ import annotations

import functools
from importlib import resources

from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

from .parser import AdeleRubric, parse_adele_file

# Mapping from ADeLe rubric codes to snake_case trait names
ADELE_CODE_TO_NAME: dict[str, str] = {
    "AS": "attention_and_scan",
    "AT": "atypicality",
    "CEc": "comprehension_complexity",
    "CEe": "comprehension_evaluation",
    "CL": "conceptualization_and_learning",
    "KNa": "knowledge_applied_sciences",
    "KNc": "knowledge_cultural",
    "KNf": "knowledge_formal_sciences",
    "KNn": "knowledge_natural_sciences",
    "KNs": "knowledge_social_sciences",
    "MCr": "metacognition_relevance",
    "MCt": "metacognition_task_planning",
    "MCu": "metacognition_uncertainty",
    "MS": "mind_modelling",
    "QLl": "logical_reasoning_logic",
    "QLq": "logical_reasoning_quantitative",
    "SNs": "spatial_physical_understanding",
    "VO": "volume",
}

# Reverse mapping for lookup by trait name
ADELE_NAME_TO_CODE: dict[str, str] = {v: k for k, v in ADELE_CODE_TO_NAME.items()}

# All available ADeLe trait names
ADELE_TRAIT_NAMES: list[str] = list(ADELE_CODE_TO_NAME.values())

# All available ADeLe codes
ADELE_CODES: list[str] = list(ADELE_CODE_TO_NAME.keys())

# Standard level labels (lowercase snake_case for class names)
LEVEL_LABELS: dict[str, str] = {
    "None": "none",
    "Very Low": "very_low",
    "Very low": "very_low",
    "Low": "low",
    "Intermediate": "intermediate",
    "High": "high",
    "Very High": "very_high",
    "Very high": "very_high",
}


def _load_bundled_rubric(code: str) -> str:
    """Load a bundled rubric text file by code.

    Args:
        code: ADeLe rubric code (e.g., "AS", "AT")

    Returns:
        Raw text content of the rubric file

    Raises:
        FileNotFoundError: If the rubric file doesn't exist
    """
    ref = resources.files("karenina.integrations.adele.data").joinpath(f"{code}.txt")
    return ref.read_text(encoding="utf-8")


def _normalize_label(label: str) -> str:
    """Normalize level label to snake_case class name.

    Args:
        label: Original label (e.g., "Very Low", "Intermediate")

    Returns:
        Normalized snake_case label (e.g., "very_low", "intermediate")
    """
    return LEVEL_LABELS.get(label, label.lower().replace(" ", "_"))


def _adele_rubric_to_trait(rubric: AdeleRubric) -> LLMRubricTrait:
    """Convert a parsed ADeLe rubric to an LLMRubricTrait.

    Args:
        rubric: Parsed ADeLe rubric

    Returns:
        LLMRubricTrait with kind="literal" and 6 classes
    """
    trait_name = ADELE_CODE_TO_NAME.get(rubric.code)
    if trait_name is None:
        raise ValueError(f"Unknown ADeLe rubric code: {rubric.code}")

    # Build classes dict (ordered by level index)
    classes: dict[str, str] = {}
    for level in rubric.levels:
        class_name = _normalize_label(level.label)
        class_description = level.to_class_description()
        classes[class_name] = class_description

    return LLMRubricTrait(
        name=trait_name,
        description=rubric.header,
        kind="literal",
        classes=classes,
        min_score=None,  # Auto-derived by model validator from classes
        max_score=None,  # Auto-derived by model validator from classes
        deep_judgment_enabled=False,
        deep_judgment_excerpt_enabled=True,
        deep_judgment_max_excerpts=None,
        deep_judgment_fuzzy_match_threshold=None,
        deep_judgment_excerpt_retry_attempts=None,
        deep_judgment_search_enabled=False,
        higher_is_better=True,  # All ADeLe rubrics use ascending quality (Level 5 is best)
    )


@functools.lru_cache(maxsize=32)
def _load_and_parse_rubric(code: str) -> AdeleRubric:
    """Load and parse a rubric file (cached).

    Args:
        code: ADeLe rubric code

    Returns:
        Parsed AdeleRubric
    """
    content = _load_bundled_rubric(code)
    return parse_adele_file(content, code)


def get_adele_trait(name: str) -> LLMRubricTrait:
    """Get a single ADeLe trait by snake_case name.

    Args:
        name: Snake_case trait name (e.g., "attention_and_scan", "mind_modelling")

    Returns:
        LLMRubricTrait with kind="literal"

    Raises:
        ValueError: If the trait name is not recognized

    Example:
        >>> trait = get_adele_trait("attention_and_scan")
        >>> trait.name
        'attention_and_scan'
        >>> trait.kind
        'literal'
        >>> len(trait.classes)
        6
    """
    code = ADELE_NAME_TO_CODE.get(name)
    if code is None:
        raise ValueError(f"Unknown ADeLe trait name: {name}. Available traits: {', '.join(ADELE_TRAIT_NAMES)}")

    rubric = _load_and_parse_rubric(code)
    return _adele_rubric_to_trait(rubric)


def get_adele_trait_by_code(code: str) -> LLMRubricTrait:
    """Get a single ADeLe trait by its original code.

    Args:
        code: Original ADeLe code (e.g., "AS", "AT", "CEc")

    Returns:
        LLMRubricTrait with kind="literal"

    Raises:
        ValueError: If the code is not recognized

    Example:
        >>> trait = get_adele_trait_by_code("AS")
        >>> trait.name
        'attention_and_scan'
    """
    if code not in ADELE_CODE_TO_NAME:
        raise ValueError(f"Unknown ADeLe code: {code}. Available codes: {', '.join(ADELE_CODES)}")

    rubric = _load_and_parse_rubric(code)
    return _adele_rubric_to_trait(rubric)


def get_all_adele_traits() -> list[LLMRubricTrait]:
    """Get all 18 ADeLe traits.

    Returns:
        List of 18 LLMRubricTrait objects with kind="literal"

    Example:
        >>> traits = get_all_adele_traits()
        >>> len(traits)
        18
        >>> all(t.kind == "literal" for t in traits)
        True
    """
    return [get_adele_trait(name) for name in ADELE_TRAIT_NAMES]


def create_adele_rubric(trait_names: list[str] | None = None) -> Rubric:
    """Create a Rubric with specified ADeLe traits (or all if None).

    Args:
        trait_names: List of snake_case trait names to include.
                    If None, includes all 18 ADeLe traits.

    Returns:
        Rubric containing the specified ADeLe traits as llm_traits

    Raises:
        ValueError: If any trait name is not recognized

    Example:
        >>> # All traits
        >>> rubric = create_adele_rubric()
        >>> len(rubric.llm_traits)
        18

        >>> # Selected traits
        >>> rubric = create_adele_rubric(["attention_and_scan", "mind_modelling"])
        >>> len(rubric.llm_traits)
        2
    """
    traits = get_all_adele_traits() if trait_names is None else [get_adele_trait(name) for name in trait_names]

    return Rubric(llm_traits=traits)
