"""
ADeLe (Assessment Dimensions for Language Evaluation) integration for Karenina.

This module provides 18 pre-defined LLMRubricTrait objects based on the ADeLe
evaluation framework. Each trait uses `kind="literal"` with 6 ordinal classes
(levels 0-5) for evaluating various cognitive and processing dimensions.

Available Traits (by snake_case name):
    - attention_and_scan (AS): Attention and scanning requirements
    - atypicality (AT): Novelty/uniqueness of the task
    - comprehension_complexity (CEc): Comprehension difficulty
    - comprehension_evaluation (CEe): Comprehension evaluation difficulty
    - conceptualization_and_learning (CL): Learning requirements
    - knowledge_applied_sciences (KNa): Applied sciences knowledge
    - knowledge_cultural (KNc): Cultural knowledge
    - knowledge_formal_sciences (KNf): Formal sciences knowledge
    - knowledge_natural_sciences (KNn): Natural sciences knowledge
    - knowledge_social_sciences (KNs): Social sciences knowledge
    - metacognition_relevance (MCr): Metacognitive relevance recognition
    - metacognition_task_planning (MCt): Metacognitive task planning
    - metacognition_uncertainty (MCu): Metacognitive uncertainty handling
    - mind_modelling (MS): Social cognition/mind modelling
    - logical_reasoning_logic (QLl): Logical reasoning
    - logical_reasoning_quantitative (QLq): Quantitative reasoning
    - spatial_physical_understanding (SNs): Spatial/physical understanding
    - volume (VO): Time/effort required

Usage:
    # Get a single trait
    from karenina.integrations.adele import get_adele_trait
    trait = get_adele_trait("attention_and_scan")

    # Get all traits
    from karenina.integrations.adele import get_all_adele_traits
    traits = get_all_adele_traits()

    # Create a Rubric with ADeLe traits
    from karenina.integrations.adele import create_adele_rubric
    rubric = create_adele_rubric()  # All 18 traits
    rubric = create_adele_rubric(["attention_and_scan", "mind_modelling"])  # Selected

    # List available trait names
    from karenina.integrations.adele import ADELE_TRAIT_NAMES
    print(ADELE_TRAIT_NAMES)

    # Classify questions using ADeLe dimensions
    from karenina.integrations.adele import QuestionClassifier
    classifier = QuestionClassifier()
    result = classifier.classify_single("What is the capital of France?")
    print(result.scores)  # {"attention_and_scan": 0, "volume": 1, ...}
"""

from .classifier import QuestionClassifier
from .parser import AdeleLevel, AdeleRubric, parse_adele_file
from .schemas import AdeleTraitInfo, QuestionClassificationResult
from .traits import (
    ADELE_CODE_TO_NAME,
    ADELE_CODES,
    ADELE_NAME_TO_CODE,
    ADELE_TRAIT_NAMES,
    create_adele_rubric,
    get_adele_trait,
    get_adele_trait_by_code,
    get_all_adele_traits,
)

__all__ = [
    # Question classification
    "QuestionClassifier",
    "QuestionClassificationResult",
    "AdeleTraitInfo",
    # Trait API functions
    "get_adele_trait",
    "get_adele_trait_by_code",
    "get_all_adele_traits",
    "create_adele_rubric",
    # Constants
    "ADELE_TRAIT_NAMES",
    "ADELE_CODES",
    "ADELE_CODE_TO_NAME",
    "ADELE_NAME_TO_CODE",
    # Parser types (for advanced usage)
    "AdeleLevel",
    "AdeleRubric",
    "parse_adele_file",
]
