"""Deep judgment configuration helpers.

Utility functions for resolving and applying deep judgment configuration to rubric traits.
These are extracted from rubric_evaluation.py to reduce clutter in the stage file.
"""

import logging
from typing import Any

from karenina.schemas.entities import LLMRubricTrait
from karenina.schemas.verification import DeepJudgmentTraitConfig

# Set up logger
logger = logging.getLogger(__name__)


def resolve_deep_judgment_config_for_trait(
    trait: LLMRubricTrait,
    question_id: str | None,
    config: Any,  # VerificationConfig
) -> DeepJudgmentTraitConfig:
    """
    Resolve deep judgment configuration for a single trait based on mode hierarchy.

    Resolution priority (first match wins):
    1. disabled mode: Deep judgment OFF
    2. enable_all mode: Deep judgment ON for all traits (respects global excerpt toggle)
    3. use_checkpoint mode: Use settings from trait object (loaded from checkpoint)
    4. custom mode: Look up trait in config dict (question-specific → global → disabled)

    Args:
        trait: The rubric trait to resolve configuration for
        question_id: Optional question ID for question-specific config lookup
        config: VerificationConfig with deep judgment mode and settings

    Returns:
        DeepJudgmentTraitConfig with resolved settings
    """
    mode = getattr(config, "deep_judgment_rubric_mode", "disabled")
    logger.debug(f"Resolving deep judgment config for trait '{trait.name}', mode='{mode}'")

    if mode == "disabled":
        # Explicit: Deep judgment OFF
        return DeepJudgmentTraitConfig(enabled=False)

    elif mode == "enable_all":
        # Apply to all traits with global excerpt toggle
        return DeepJudgmentTraitConfig(
            enabled=True,
            excerpt_enabled=getattr(config, "deep_judgment_rubric_global_excerpts", True),
            max_excerpts=getattr(config, "deep_judgment_rubric_max_excerpts_default", 7),
            fuzzy_match_threshold=getattr(config, "deep_judgment_rubric_fuzzy_match_threshold_default", 0.80),
            excerpt_retry_attempts=getattr(config, "deep_judgment_rubric_excerpt_retry_attempts_default", 2),
            search_enabled=getattr(config, "deep_judgment_rubric_search_enabled", False),
        )

    elif mode == "use_checkpoint":
        # Use settings from trait (loaded from checkpoint)
        return DeepJudgmentTraitConfig(
            enabled=trait.deep_judgment_enabled,
            excerpt_enabled=trait.deep_judgment_excerpt_enabled,
            max_excerpts=trait.deep_judgment_max_excerpts,
            fuzzy_match_threshold=trait.deep_judgment_fuzzy_match_threshold,
            excerpt_retry_attempts=trait.deep_judgment_excerpt_retry_attempts,
            search_enabled=trait.deep_judgment_search_enabled,
        )

    elif mode == "custom":
        # Navigate nested config structure
        config_dict = getattr(config, "deep_judgment_rubric_config", None) or {}

        # Try question-specific first
        if (
            question_id
            and "question_specific" in config_dict
            and question_id in config_dict["question_specific"]
            and trait.name in config_dict["question_specific"][question_id]
        ):
            trait_config = config_dict["question_specific"][question_id][trait.name]
            # Validate dict against model
            return DeepJudgmentTraitConfig(**trait_config)

        # Fall back to global
        if "global" in config_dict and trait.name in config_dict["global"]:
            trait_config = config_dict["global"][trait.name]
            # Validate dict against model
            return DeepJudgmentTraitConfig(**trait_config)

        # No config found, disabled
        return DeepJudgmentTraitConfig(enabled=False)

    else:
        # Unknown mode, default to disabled
        logger.warning(f"Unknown deep_judgment_rubric_mode: {mode}, defaulting to disabled")
        return DeepJudgmentTraitConfig(enabled=False)


def apply_deep_judgment_config_to_traits(
    traits: list[LLMRubricTrait],
    question_id: str | None,
    config: Any,  # VerificationConfig
) -> list[LLMRubricTrait]:
    """
    Apply resolved deep judgment configuration to a list of traits.

    Creates shallow copies of traits with resolved deep judgment settings.
    Uses Pydantic's model_copy() for efficient copying since we only modify
    scalar config fields (bool, int, float).

    Args:
        traits: List of traits to configure
        question_id: Optional question ID for question-specific config
        config: VerificationConfig with deep judgment settings

    Returns:
        List of traits with resolved deep judgment configuration applied
    """
    configured_traits = []

    for trait in traits:
        # Resolve configuration for this trait (before copy to avoid unnecessary work)
        dj_config = resolve_deep_judgment_config_for_trait(trait, question_id, config)

        # Use Pydantic's model_copy with update dict for efficient shallow copy
        # This avoids expensive deepcopy for scalar field updates
        trait_copy = trait.model_copy(
            update={
                "deep_judgment_enabled": dj_config.enabled,
                "deep_judgment_excerpt_enabled": dj_config.excerpt_enabled,
                "deep_judgment_max_excerpts": dj_config.max_excerpts,
                "deep_judgment_fuzzy_match_threshold": dj_config.fuzzy_match_threshold,
                "deep_judgment_excerpt_retry_attempts": dj_config.excerpt_retry_attempts,
                "deep_judgment_search_enabled": dj_config.search_enabled,
            }
        )

        configured_traits.append(trait_copy)

    return configured_traits
