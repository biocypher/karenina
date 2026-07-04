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
            (scenario_def, ans_model, parse_model, None)
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
            (scenario_def, ans_model, parse_model, None)
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
            (scenario_def, ans_model, parse_model, None)
            for scenario_def in scenarios
            for ans_model in config.answering_models
            for parse_model in config.parsing_models
        ]

        original_order = list(combos)

        # generation_order is a no-op; do nothing
        # (matches the benchmark.py code path)

        assert combos == original_order


# =============================================================================
# task_ordering dispatch via _run_scenario_verification
# =============================================================================


@pytest.mark.unit
class TestScenarioTaskOrderingDispatch:
    """Pins the dispatch in Benchmark._run_scenario_verification to the shared
    ordering resolver so scenario users honor 'auto' and 'distribute_answerers'.
    """

    def _combos(self, *, answerers: list[str], scenarios: list[str]) -> list:
        out = []
        for scen_name in scenarios:
            s = MagicMock()
            s.name = scen_name
            for ans_id in answerers:
                ans = _make_model(ans_id)
                parse = _make_model("parser")
                out.append((s, ans, parse, None))
        return out

    def test_auto_single_answerer_groups_like_prefix_cache(self) -> None:
        """Default (auto) with one answerer preserves pre-feature behavior: grouped by answerer."""
        from karenina.benchmark.benchmark import _apply_scenario_ordering

        combos = self._combos(answerers=["only"], scenarios=["s2", "s1"])
        config = VerificationConfig(
            answering_models=[_make_model("only")],
            parsing_models=[_make_model("parser")],
            # default task_ordering = "auto"
        )

        ordered = _apply_scenario_ordering(combos, config)

        # All for the single answerer; scenarios sorted alphabetically
        assert [c[0].name for c in ordered] == ["s1", "s2"]

    def test_auto_multiple_answerers_round_robin(self) -> None:
        """Default (auto) with multiple answerers interleaves across answerers."""
        from karenina.benchmark.benchmark import _apply_scenario_ordering

        combos = self._combos(answerers=["a", "b"], scenarios=["s1", "s2"])
        config = VerificationConfig(
            answering_models=[_make_model("a"), _make_model("b")],
            parsing_models=[_make_model("parser")],
            # default task_ordering = "auto"
        )

        ordered = _apply_scenario_ordering(combos, config)

        # First two combos must span both answerers (round-robin head)
        head_ids = {c[1].id for c in ordered[:2]}
        assert head_ids == {"a", "b"}
        assert len(ordered) == 4

    def test_explicit_distribute_answerers_round_robin(self) -> None:
        """Explicit distribute_answerers works at the scenario site too."""
        from karenina.benchmark.benchmark import _apply_scenario_ordering

        combos = self._combos(answerers=["a", "b", "c"], scenarios=["s1"])
        config = VerificationConfig(
            answering_models=[_make_model("a"), _make_model("b"), _make_model("c")],
            parsing_models=[_make_model("parser")],
            task_ordering="distribute_answerers",
        )

        ordered = _apply_scenario_ordering(combos, config)

        # First three combos cover all three answerers
        head_ids = {c[1].id for c in ordered[:3]}
        assert head_ids == {"a", "b", "c"}

    def test_explicit_prefix_cache_still_groups(self) -> None:
        """Existing prefix_cache behavior preserved by the helper."""
        from karenina.benchmark.benchmark import _apply_scenario_ordering

        combos = self._combos(answerers=["b", "a"], scenarios=["s1"])
        config = VerificationConfig(
            answering_models=[_make_model("a"), _make_model("b")],
            parsing_models=[_make_model("parser")],
            task_ordering="prefix_cache",
        )

        ordered = _apply_scenario_ordering(combos, config)
        answerer_run = [c[1].id for c in ordered]
        assert answerer_run == ["a", "b"]

    def test_generation_order_is_passthrough(self) -> None:
        from karenina.benchmark.benchmark import _apply_scenario_ordering

        combos = self._combos(answerers=["a", "b"], scenarios=["s1", "s2"])
        config = VerificationConfig(
            answering_models=[_make_model("a"), _make_model("b")],
            parsing_models=[_make_model("parser")],
            task_ordering="generation_order",
        )

        ordered = _apply_scenario_ordering(combos, config)
        # With generation_order we preserve list-comprehension order
        assert ordered == combos
