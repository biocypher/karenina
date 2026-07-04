"""Tests for R2: scenario replicate_count executor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from karenina.scenario.manager import ScenarioManager


@pytest.mark.unit
class TestScenarioManagerReplicateThreading:
    """Verify ScenarioManager.run threads the replicate value correctly."""

    def test_execution_result_carries_replicate(self, monkeypatch):
        """ScenarioManager.run(..., replicate=2) returns a result with replicate=2."""
        from karenina.schemas.scenario.state import ScenarioExecutionResult

        captured: dict[str, object] = {}

        def fake_run(self, scenario, config, base_answering_model, base_parsing_model, **kwargs):
            captured["replicate"] = kwargs.get("replicate")
            final_state = MagicMock()
            return ScenarioExecutionResult(
                scenario_id=scenario.name,
                status="completed",
                path=[],
                turn_count=0,
                history=[],
                turn_results=[],
                final_state=final_state,
                outcome_results={},
                replicate=kwargs.get("replicate"),
            )

        monkeypatch.setattr(ScenarioManager, "run", fake_run, raising=True)
        manager = ScenarioManager()
        scenario = MagicMock(name="scenario_def")
        scenario.name = "foo"
        result = manager.run(
            scenario=scenario,
            config=MagicMock(),
            base_answering_model=MagicMock(),
            base_parsing_model=MagicMock(),
            replicate=2,
        )
        assert result.replicate == 2
        assert captured["replicate"] == 2


@pytest.mark.unit
class TestScenarioExecutorForwardsReplicate:
    """Verify ScenarioExecutor forwards replicate from the 4-tuple to ScenarioManager.run."""

    def test_sequential_forwards_replicate_kwarg(self, monkeypatch):
        from karenina.benchmark.verification import scenario_executor as se
        from karenina.schemas.scenario.state import ScenarioExecutionResult

        captured: list[int | None] = []

        def fake_run(self, *args, **kwargs):
            captured.append(kwargs.get("replicate"))
            return ScenarioExecutionResult(
                scenario_id=kwargs["scenario"].name,
                status="completed",
                path=[],
                turn_count=0,
                history=[],
                turn_results=[],
                final_state=MagicMock(),
                outcome_results={},
                replicate=kwargs.get("replicate"),
            )

        monkeypatch.setattr(se.ScenarioManager, "run", fake_run, raising=True)

        scenario = MagicMock()
        scenario.name = "foo"
        ans = MagicMock()
        ans.model_name = "a"
        parse = MagicMock()
        parse.model_name = "p"

        combos = [
            (scenario, ans, parse, None),
            (scenario, ans, parse, 1),
            (scenario, ans, parse, 2),
        ]
        executor = se.ScenarioExecutor(parallel=False, config=se.ScenarioExecutorConfig(enable_cache=False))
        results, errors = executor.run_batch(combos=combos, config=MagicMock())
        assert not errors
        assert captured == [None, 1, 2]
        assert [r.replicate for r in results] == [None, 1, 2]

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_parallel_forwards_replicate_kwarg(self, mock_manager_cls: MagicMock) -> None:
        """The parallel executor branch forwards replicate into ScenarioManager.run.

        Patches the module-level ScenarioManager reference in scenario_executor
        so the fake applies across worker threads (monkeypatch on the class has
        the same global effect, but @patch matches the convention already used
        in test_scenario_executor_parallel.py).
        """
        import threading

        from karenina.benchmark.verification import scenario_executor as se
        from karenina.schemas.scenario.state import ScenarioExecutionResult

        captured: list[int | None] = []
        capture_lock = threading.Lock()

        def fake_run(**kwargs):
            replicate = kwargs.get("replicate")
            with capture_lock:
                captured.append(replicate)
            return ScenarioExecutionResult(
                scenario_id=kwargs["scenario"].name,
                status="completed",
                path=[],
                turn_count=0,
                history=[],
                turn_results=[],
                final_state=MagicMock(),
                outcome_results={},
                replicate=replicate,
            )

        mock_manager_cls.return_value.run.side_effect = fake_run

        scenario = MagicMock()
        scenario.name = "foo"
        ans = MagicMock()
        ans.model_name = "a"
        ans.id = "a-id"
        parse = MagicMock()
        parse.model_name = "p"
        parse.id = "p-id"

        combos = [
            (scenario, ans, parse, None),
            (scenario, ans, parse, 1),
            (scenario, ans, parse, 2),
        ]
        executor = se.ScenarioExecutor(
            parallel=True,
            config=se.ScenarioExecutorConfig(max_workers=3, enable_cache=False),
        )
        results, errors = executor.run_batch(combos=combos, config=MagicMock())

        assert not errors
        # Parallel completion may reorder; compare as sets so the test is
        # insensitive to scheduling order.
        assert set(captured) == {None, 1, 2}
        assert {r.replicate for r in results} == {None, 1, 2}


@pytest.mark.unit
class TestEndToEndReplicatePropagation:
    """End-to-end checks: replicate reaches VerificationContext and ScenarioExecutionResult."""

    def _build_two_node_scenario(self, name: str = "two_turn_scenario"):
        """Build a minimal two-node scenario used by several tests in this class.

        The scenario walks n1 then n2 then END; paired with a fake _run_turn
        (or orchestrator stub) this yields exactly two turns per ``run``.
        """
        from karenina.schemas.entities import Question
        from karenina.schemas.scenario.definition import ScenarioDefinition
        from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode

        q1 = Question(question="first?", raw_answer="y", answer_template="class Answer: pass")
        q2 = Question(question="second?", raw_answer="y", answer_template="class Answer: pass")
        return ScenarioDefinition(
            name=name,
            nodes={
                "n1": ScenarioNode(node_id="n1", question=q1),
                "n2": ScenarioNode(node_id="n2", question=q2),
            },
            edges=[
                ScenarioEdge(source="n1", target="n2"),
                ScenarioEdge(source="n2", target=END),
            ],
            entry_node="n1",
        )

    def _make_config(self):
        """Minimal VerificationConfig double shared across this class.

        ``custom_error_patterns`` must be an iterable (not ``None``) because
        ``_run_turn`` unconditionally passes it through
        ``_build_error_registry``. The outer ``_run_turn``-stubbing test never
        reaches that call site, but the orchestrator-stubbing test does.
        """
        config = MagicMock()
        config.replay_store = None
        config.replicate_count = 3
        config.request_timeout = None
        config.evaluation_mode = "template_only"
        config.scenario_turn_limit = 5
        config.custom_error_patterns = []
        config.use_full_trace_for_template = False
        config.use_full_trace_for_rubric = False
        return config

    def test_turn_contexts_share_run_replicate(self, monkeypatch):
        """Every turn in one scenario execution sees the same replicate.

        Drives ScenarioManager.run through two turns (a two-node scenario) and
        intercepts _run_turn to capture the replicate kwarg per turn. Asserts
        that every captured value matches the run-level replicate and that the
        returned ScenarioExecutionResult.replicate mirrors it.
        """
        from karenina.scenario import manager as mgr_mod

        observed_replicates: list[int | None] = []

        def fake_run_turn(self, **kwargs):
            observed_replicates.append(kwargs.get("replicate"))
            vr = MagicMock()
            vr.metadata.failure = None
            vr.metadata.replicate = kwargs.get("replicate")
            vr.metadata.result_id = f"rid_{len(observed_replicates)}"
            vr.template.verify_result = True
            vr.rubric = None
            return (vr, [], None, None)

        monkeypatch.setattr(mgr_mod.ScenarioManager, "_run_turn", fake_run_turn, raising=True)

        scenario = self._build_two_node_scenario()
        manager = mgr_mod.ScenarioManager()
        config = self._make_config()

        result = manager.run(
            scenario=scenario,
            config=config,
            base_answering_model=MagicMock(
                id="m",
                model_name="m",
                system_prompt="",
                request_timeout=None,
            ),
            base_parsing_model=MagicMock(
                id="p",
                model_name="p",
                system_prompt="",
                request_timeout=None,
            ),
            replicate=2,
        )

        assert result.replicate == 2
        assert len(observed_replicates) == 2  # two turns in this scenario
        assert all(r == 2 for r in observed_replicates)

    def test_run_turn_threads_replicate_into_verification_context(self, monkeypatch):
        """_run_turn builds VerificationContext with replicate set from the run-level arg.

        Unlike ``test_turn_contexts_share_run_replicate``, this test does NOT
        stub ``_run_turn``. Instead, it stubs one level deeper
        (``StageOrchestrator.execute``) so the assertion chain exercises the
        real ``VerificationContext(replicate=replicate, ...)`` construction in
        ``ScenarioManager._run_turn``. A future refactor that drops the
        ``replicate=replicate`` keyword from the context construction would be
        caught here; the outer-level ``_run_turn``-stubbing test would not see
        it.
        """
        from karenina.benchmark.verification.stages.core import orchestrator as orch_mod
        from karenina.scenario import manager as mgr_mod
        from karenina.schemas.config import ModelConfig

        observed_contexts: list[object] = []

        def fake_execute(self, context):
            observed_contexts.append(context)
            vr = MagicMock()
            vr.metadata.failure = None
            vr.metadata.replicate = context.replicate
            vr.metadata.result_id = f"rid_{len(observed_contexts)}"
            vr.template.verify_result = True
            vr.rubric = None
            return vr

        monkeypatch.setattr(orch_mod.StageOrchestrator, "execute", fake_execute, raising=True)

        scenario = self._build_two_node_scenario(name="ctx_capture_scenario")
        manager = mgr_mod.ScenarioManager()
        config = self._make_config()

        # Real ModelConfig instances so ModelIdentity.from_model_config passes
        # pydantic validation for interface/model_name. MagicMock model doubles
        # do not satisfy the string type constraint inside _run_turn.
        answering_model = ModelConfig(
            id="m",
            model_name="m",
            model_provider="test",
            interface="langchain",
            system_prompt="",
        )
        parsing_model = ModelConfig(
            id="p",
            model_name="p",
            model_provider="test",
            interface="langchain",
            system_prompt="",
        )

        result = manager.run(
            scenario=scenario,
            config=config,
            base_answering_model=answering_model,
            base_parsing_model=parsing_model,
            replicate=2,
        )

        assert result.replicate == 2
        assert len(observed_contexts) >= 1
        assert all(c.replicate == 2 for c in observed_contexts)

    def test_cache_key_replicate_isolation(self):
        """Two replicates of the same scenario node produce distinct cache keys."""
        from karenina.scenario.manager import build_scenario_cache_key

        k_rep1 = build_scenario_cache_key("s", "n", "m", ["hi"], replicate=1)
        k_rep2 = build_scenario_cache_key("s", "n", "m", ["hi"], replicate=2)
        assert k_rep1 != k_rep2


@pytest.mark.unit
class TestScenarioWorkspaceReplicateIsolation:
    """Verify per-replicate workspace directories do not collide.

    R2 I-1 regression: without a replicate segment in the scenario workspace
    path, two replicates of the same scenario collide under
    ``{workspace_root}/{scenario}/{model}/turn_{N}`` and either interleave
    writes (parallel) or overwrite each other (sequential). The fix inserts
    ``rep_{N}`` between the model segment and the turn segment when
    ``replicate is not None`` and preserves the pre-R2 path shape exactly
    when ``replicate is None``.
    """

    def _build_two_turn_scenario(self):
        """Build a minimal two-turn scenario; mirrors TestEndToEndReplicatePropagation."""
        from karenina.schemas.entities import Question
        from karenina.schemas.scenario.definition import ScenarioDefinition
        from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode

        q1 = Question(question="first?", raw_answer="y", answer_template="class Answer: pass")
        q2 = Question(question="second?", raw_answer="y", answer_template="class Answer: pass")
        return ScenarioDefinition(
            name="ws_iso_scenario",
            nodes={
                "n1": ScenarioNode(node_id="n1", question=q1),
                "n2": ScenarioNode(node_id="n2", question=q2),
            },
            edges=[
                ScenarioEdge(source="n1", target="n2"),
                ScenarioEdge(source="n2", target=END),
            ],
            entry_node="n1",
        )

    def _make_config(self):
        config = MagicMock()
        config.replay_store = None
        config.replicate_count = 3
        config.request_timeout = None
        config.evaluation_mode = "template_only"
        config.scenario_turn_limit = 5
        config.custom_error_patterns = None
        config.use_full_trace_for_template = False
        config.use_full_trace_for_rubric = False
        return config

    def _run_and_capture(self, tmp_path, replicate, monkeypatch):
        from karenina.scenario import manager as mgr_mod

        captured: list = []

        def fake_run_turn(self, **kwargs):
            captured.append(kwargs.get("turn_workspace_path"))
            vr = MagicMock()
            vr.metadata.failure = None
            vr.metadata.replicate = kwargs.get("replicate")
            vr.metadata.result_id = f"rid_{len(captured)}"
            vr.template.verify_result = True
            vr.rubric = None
            return (vr, [], None, None)

        monkeypatch.setattr(mgr_mod.ScenarioManager, "_run_turn", fake_run_turn, raising=True)

        manager = mgr_mod.ScenarioManager()
        manager.run(
            scenario=self._build_two_turn_scenario(),
            config=self._make_config(),
            base_answering_model=MagicMock(id="m", model_name="m", system_prompt="", request_timeout=None),
            base_parsing_model=MagicMock(id="p", model_name="p", system_prompt="", request_timeout=None),
            workspace_root=tmp_path,
            replicate=replicate,
        )
        return captured

    def test_replicates_get_distinct_workspace_paths(self, tmp_path, monkeypatch):
        """replicate=1 and replicate=2 must emit turn workspaces under rep_1/ and rep_2/."""
        paths_rep1 = self._run_and_capture(tmp_path, replicate=1, monkeypatch=monkeypatch)
        paths_rep2 = self._run_and_capture(tmp_path, replicate=2, monkeypatch=monkeypatch)

        # Each run produces two turns.
        assert len(paths_rep1) == 2
        assert len(paths_rep2) == 2

        # Replicate isolation: the full captured paths must differ pairwise.
        assert set(paths_rep1).isdisjoint(paths_rep2), (
            f"Replicate 1 and 2 share workspace paths: {paths_rep1} vs {paths_rep2}"
        )

        # Each path carries the rep_{N} segment.
        for p in paths_rep1:
            assert p is not None
            assert "rep_1" in p.parts, f"Expected 'rep_1' in {p.parts}"
        for p in paths_rep2:
            assert p is not None
            assert "rep_2" in p.parts, f"Expected 'rep_2' in {p.parts}"

    def test_replicate_none_preserves_pre_r2_path_shape(self, tmp_path, monkeypatch):
        """replicate=None must produce paths WITHOUT any rep_ segment (byte-for-byte)."""
        paths_none = self._run_and_capture(tmp_path, replicate=None, monkeypatch=monkeypatch)

        assert len(paths_none) == 2
        for p in paths_none:
            assert p is not None
            # No rep_N segment should appear anywhere in the path.
            assert not any(part.startswith("rep_") for part in p.parts), f"Expected no 'rep_*' segment in {p.parts}"
            # Pre-R2 shape: {workspace_root}/{scenario}/{model}/turn_{M}
            assert p.parts[-2] == "ws_iso_scenario" or "ws_iso_scenario" in p.parts
            assert p.parts[-1].startswith("turn_"), f"Expected final segment to be 'turn_*', got {p.parts[-1]}"


@pytest.mark.unit
class TestFacadeComboExpansion:
    """Verify Benchmark._run_scenario_verification expands the replicate axis."""

    def _capture_combos(
        self,
        monkeypatch,
        replicate_count: int,
        task_ordering: str = "generation_order",
        scenario_names: list[str] | None = None,
    ):
        """Helper: run _run_scenario_verification and return the combos list passed to ScenarioExecutor.run_batch.

        Args:
            scenario_names: Names of scenarios to register on the Benchmark
                double. Defaults to ``["foo"]`` to match the single-scenario
                tests in this class.
        """
        from karenina.benchmark import benchmark as bm
        from karenina.benchmark.verification import scenario_executor as se

        names = scenario_names if scenario_names is not None else ["foo"]

        captured: dict[str, object] = {}

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def run_batch(self, combos, config, **kwargs):  # noqa: ARG002
                captured["combos"] = combos
                return [], []

        # Patch on the source module; the facade does a local import at call time.
        monkeypatch.setattr(se, "ScenarioExecutor", FakeExecutor, raising=True)

        # Minimal Benchmark double with scenarios + config
        benchmark_obj = MagicMock()
        scenarios: dict[str, MagicMock] = {}
        for name in names:
            scen = MagicMock(name=f"scenario_def_{name}", nodes={})
            scen.name = name
            scenarios[name] = scen
        benchmark_obj._scenarios = scenarios
        benchmark_obj._rubric_manager.get_global_rubric.return_value = None
        benchmark_obj._workspace_root = None

        config = MagicMock()
        config.answering_models = [MagicMock(model_name="a", request_timeout=None, retry_policy=None)]
        config.parsing_models = [MagicMock(model_name="p", request_timeout=None, retry_policy=None)]
        config.replicate_count = replicate_count
        config.task_ordering = task_ordering
        config.request_timeout = None
        config.retry_policy = None
        config.async_max_workers = 1
        config.max_concurrent_requests = None

        bm.Benchmark._run_scenario_verification(benchmark_obj, config=config)
        return captured["combos"]

    def test_replicate_count_one_emits_single_none_combo(self, monkeypatch):
        combos = self._capture_combos(monkeypatch, replicate_count=1)
        assert len(combos) == 1
        assert combos[0][3] is None

    def test_replicate_count_three_produces_three_combos(self, monkeypatch):
        combos = self._capture_combos(monkeypatch, replicate_count=3)
        assert [c[3] for c in combos] == [1, 2, 3]

    def test_prefix_cache_sort_is_none_safe(self, monkeypatch):
        # replicate_count=1 -> single combo with replicate=None.
        # The prefix_cache sort must not raise TypeError when comparing None with int.
        combos = self._capture_combos(monkeypatch, replicate_count=1, task_ordering="prefix_cache")
        assert combos[0][3] is None

    def test_prefix_cache_groups_replicates_adjacent(self, monkeypatch):
        """Under prefix_cache ordering, replicates of the same (ans, scenario, parse) are adjacent and ordered 1..N.

        Guards the replicate tiebreaker at the tail of the prefix_cache sort
        key: two scenarios times three replicates must produce six combos in
        which each scenario's three replicates stay contiguous and ordered
        ``1, 2, 3``. The two scenario groups themselves are contiguous (no
        interleaving) because the outer sort key is
        ``(answering_model, scenario_name, parsing_model, replicate)`` and
        this test uses one answering model and one parsing model, leaving
        scenario name as the dominant grouping dimension.
        """
        from itertools import groupby

        combos = self._capture_combos(
            monkeypatch,
            replicate_count=3,
            task_ordering="prefix_cache",
            scenario_names=["foo", "bar"],
        )

        # 2 scenarios x 1 ans x 1 parse x 3 replicates = 6 combos
        assert len(combos) == 6

        # Split into per-scenario runs while preserving order. Each group must
        # have length 3 and its replicates must be [1, 2, 3]. There must be
        # exactly two groups, so replicates never interleave across scenarios.
        groups = [list(g) for _, g in groupby(combos, key=lambda c: c[0].name)]
        assert len(groups) == 2, f"Expected two contiguous scenario groups, got {len(groups)}"
        for group in groups:
            assert [c[3] for c in group] == [1, 2, 3]
