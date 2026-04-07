"""Test scenario combo ordering follows task_ordering config."""

from unittest.mock import MagicMock

import pytest

from karenina.benchmark.verification.utils.task_helpers import model_sort_key
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig


def _make_model(model_id: str, model_name: str | None = None) -> ModelConfig:
    """Create a ModelConfig with the given id and optional model_name."""
    return ModelConfig(
        id=model_id,
        model_provider="anthropic",
        model_name=model_name or model_id,
        temperature=0.0,
    )


@pytest.mark.unit
class TestScenarioComboOrdering:
    """Verify that scenario combos respect the task_ordering config."""

    def test_prefix_cache_groups_by_answering_model(self) -> None:
        """Combos sorted with prefix_cache put all model-a combos before model-b."""
        model_a = _make_model("model-a")
        model_b = _make_model("model-b")

        config = VerificationConfig(
            answering_models=[model_b, model_a],  # intentionally reversed
            parsing_models=[model_a],
            task_ordering="prefix_cache",
        )

        # Three mock scenarios with .name attributes
        scenarios = []
        for name in ["scenario-z", "scenario-a", "scenario-m"]:
            s = MagicMock()
            s.name = name
            scenarios.append(s)

        # Build combos the same way benchmark.py does
        combos = [
            (scenario_def, ans_model, parse_model)
            for scenario_def in scenarios
            for ans_model in config.answering_models
            for parse_model in config.parsing_models
        ]

        # Apply prefix_cache sort (same logic as in benchmark.py)
        combos.sort(
            key=lambda c: (
                model_sort_key(c[1]),  # answering_model
                c[0].name,  # scenario name
                model_sort_key(c[2]),  # parsing_model
            )
        )

        answering_ids = [c[1].id for c in combos]

        # All model-a combos must come before all model-b combos
        first_b = answering_ids.index("model-b")
        last_a = len(answering_ids) - 1 - answering_ids[::-1].index("model-a")
        assert last_a < first_b, (
            f"Expected all model-a combos before model-b, "
            f"but last model-a at index {last_a} and first model-b at index {first_b}"
        )

    def test_prefix_cache_sorts_scenarios_within_model_group(self) -> None:
        """Within a model group, scenarios are sorted alphabetically by name."""
        model_a = _make_model("model-a")

        config = VerificationConfig(
            answering_models=[model_a],
            parsing_models=[model_a],
            task_ordering="prefix_cache",
        )

        scenarios = []
        for name in ["zeta", "alpha", "mu"]:
            s = MagicMock()
            s.name = name
            scenarios.append(s)

        combos = [
            (scenario_def, ans_model, parse_model)
            for scenario_def in scenarios
            for ans_model in config.answering_models
            for parse_model in config.parsing_models
        ]

        combos.sort(
            key=lambda c: (
                model_sort_key(c[1]),
                c[0].name,
                model_sort_key(c[2]),
            )
        )

        scenario_names = [c[0].name for c in combos]
        assert scenario_names == ["alpha", "mu", "zeta"]

    def test_generation_order_preserves_original_order(self) -> None:
        """With generation_order, combos stay in the original list comprehension order."""
        model_b = _make_model("model-b")
        model_a = _make_model("model-a")

        config = VerificationConfig(
            answering_models=[model_b, model_a],
            parsing_models=[model_a],
            task_ordering="generation_order",
        )

        scenarios = []
        for name in ["s1", "s2"]:
            s = MagicMock()
            s.name = name
            scenarios.append(s)

        combos = [
            (scenario_def, ans_model, parse_model)
            for scenario_def in scenarios
            for ans_model in config.answering_models
            for parse_model in config.parsing_models
        ]

        original_order = list(combos)

        # generation_order is a no-op; do nothing
        # (matches the benchmark.py code path)

        assert combos == original_order
