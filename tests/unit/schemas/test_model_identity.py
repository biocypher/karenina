"""Unit tests for ModelIdentity and its integration with VerificationResultMetadata.

Tests cover:
- ModelIdentity construction and field defaults
- from_model_config factory with various interfaces and roles
- display_string format (with and without tools)
- canonical_key determinism and sort stability
- Equality semantics
- compute_result_id differentiation by interface and tools
- Backward-compatible answering_model/parsing_model properties on metadata
"""

import pytest

from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification import (
    VerificationResultMetadata,
)
from karenina.schemas.verification.model_identity import ModelIdentity

# =============================================================================
# ModelIdentity Construction
# =============================================================================


@pytest.mark.unit
class TestModelIdentityConstruction:
    """Tests for direct ModelIdentity construction."""

    def test_basic_construction(self) -> None:
        identity = ModelIdentity(interface="langchain", model_name="gpt-4")
        assert identity.interface == "langchain"
        assert identity.model_name == "gpt-4"
        assert identity.tools == []

    def test_construction_with_tools(self) -> None:
        identity = ModelIdentity(
            interface="claude_agent_sdk",
            model_name="claude-sonnet-4-20250514",
            tools=["brave", "filesystem"],
        )
        assert identity.interface == "claude_agent_sdk"
        assert identity.model_name == "claude-sonnet-4-20250514"
        assert identity.tools == ["brave", "filesystem"]

    def test_equality_same_fields(self) -> None:
        """Two instances with same fields are equal."""
        a = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave"])
        b = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave"])
        assert a == b

    def test_inequality_different_interface(self) -> None:
        a = ModelIdentity(interface="langchain", model_name="gpt-4")
        b = ModelIdentity(interface="openrouter", model_name="gpt-4")
        assert a != b

    def test_inequality_different_tools(self) -> None:
        a = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave"])
        b = ModelIdentity(interface="langchain", model_name="gpt-4", tools=[])
        assert a != b


# =============================================================================
# from_model_config Factory
# =============================================================================


@pytest.mark.unit
class TestFromModelConfig:
    """Tests for ModelIdentity.from_model_config factory."""

    def test_langchain_interface(self) -> None:
        config = ModelConfig(
            id="claude-sonnet-4-20250514",
            interface="langchain",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
        )
        identity = ModelIdentity.from_model_config(config)
        assert identity.interface == "langchain"
        assert identity.model_name == "claude-sonnet-4-20250514"
        assert identity.tools == []

    def test_claude_tool_interface(self) -> None:
        config = ModelConfig(
            id="claude-sonnet-4-20250514",
            interface="claude_tool",
            model_name="claude-sonnet-4-20250514",
        )
        identity = ModelIdentity.from_model_config(config)
        assert identity.interface == "claude_tool"
        assert identity.model_name == "claude-sonnet-4-20250514"
        assert identity.tools == []

    def test_openrouter_interface(self) -> None:
        config = ModelConfig(
            id="anthropic/claude-sonnet-4-20250514",
            interface="openrouter",
            model_name="anthropic/claude-sonnet-4-20250514",
        )
        identity = ModelIdentity.from_model_config(config)
        assert identity.interface == "openrouter"
        assert identity.model_name == "anthropic/claude-sonnet-4-20250514"

    def test_with_mcp_urls_dict_answering_role(self) -> None:
        """Answering models with mcp_urls_dict should have sorted tools."""
        config = ModelConfig(
            id="claude-sonnet-4-20250514",
            interface="langchain",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            mcp_urls_dict={"brave": "http://localhost:8001", "fs": "http://localhost:8002"},
        )
        identity = ModelIdentity.from_model_config(config, role="answering")
        assert identity.tools == ["brave", "fs"]

    def test_with_mcp_urls_dict_parsing_role_always_empty_tools(self) -> None:
        """Parsing models always produce tools=[] regardless of mcp_urls_dict."""
        config = ModelConfig(
            id="claude-sonnet-4-20250514",
            interface="langchain",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            mcp_urls_dict={"brave": "http://localhost:8001", "fs": "http://localhost:8002"},
        )
        identity = ModelIdentity.from_model_config(config, role="parsing")
        assert identity.tools == []

    def test_missing_model_name_defaults_to_unknown(self) -> None:
        """When model_name is None, from_model_config defaults to 'unknown'.

        ModelConfig validators prevent model_name=None for most interfaces,
        but from_model_config handles it defensively. We test this by mocking
        the config to bypass validation.
        """
        from unittest.mock import MagicMock

        config = MagicMock(spec=ModelConfig)
        config.interface = "langchain"
        config.model_name = None
        config.id = None
        config.mcp_urls_dict = None

        identity = ModelIdentity.from_model_config(config)
        assert identity.model_name == "unknown"

    def test_tools_are_sorted(self) -> None:
        """Tools are sorted alphabetically regardless of input order."""
        config = ModelConfig(
            id="gpt-4",
            interface="langchain",
            model_name="gpt-4",
            model_provider="openai",
            mcp_urls_dict={
                "z_server": "http://z",
                "a_server": "http://a",
                "m_server": "http://m",
            },
        )
        identity = ModelIdentity.from_model_config(config, role="answering")
        assert identity.tools == ["a_server", "m_server", "z_server"]


# =============================================================================
# display_string Property
# =============================================================================


@pytest.mark.unit
class TestDisplayString:
    """Tests for ModelIdentity.display_string formatting."""

    def test_without_tools(self) -> None:
        identity = ModelIdentity(interface="langchain", model_name="claude-sonnet-4-20250514")
        assert identity.display_string == "langchain:claude-sonnet-4-20250514"

    def test_with_tools(self) -> None:
        identity = ModelIdentity(
            interface="langchain",
            model_name="claude-sonnet-4-20250514",
            tools=["brave", "fs"],
        )
        assert identity.display_string == "langchain:claude-sonnet-4-20250514 +[brave, fs]"

    def test_single_tool(self) -> None:
        identity = ModelIdentity(
            interface="claude_agent_sdk",
            model_name="claude-sonnet-4-20250514",
            tools=["brave"],
        )
        assert identity.display_string == "claude_agent_sdk:claude-sonnet-4-20250514 +[brave]"


# =============================================================================
# canonical_key Property
# =============================================================================


@pytest.mark.unit
class TestCanonicalKey:
    """Tests for ModelIdentity.canonical_key determinism and format."""

    def test_without_tools(self) -> None:
        identity = ModelIdentity(interface="langchain", model_name="gpt-4")
        assert identity.canonical_key == "langchain:gpt-4:"

    def test_with_tools(self) -> None:
        identity = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave", "fs"])
        assert identity.canonical_key == "langchain:gpt-4:brave|fs"

    def test_determinism(self) -> None:
        """Same inputs always produce the same canonical_key."""
        a = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["x", "y"])
        b = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["x", "y"])
        assert a.canonical_key == b.canonical_key

    def test_sort_stability(self) -> None:
        """Tools in canonical_key are always sorted, regardless of input order."""
        a = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["z_server", "a_server"])
        assert a.canonical_key == "langchain:gpt-4:a_server|z_server"

    def test_different_interface_different_key(self) -> None:
        """Same model_name but different interface => different canonical_key."""
        a = ModelIdentity(interface="langchain", model_name="gpt-4")
        b = ModelIdentity(interface="openrouter", model_name="gpt-4")
        assert a.canonical_key != b.canonical_key

    def test_different_tools_different_key(self) -> None:
        a = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave"])
        b = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["fs"])
        assert a.canonical_key != b.canonical_key


# =============================================================================
# compute_result_id with ModelIdentity
# =============================================================================


@pytest.mark.unit
class TestComputeResultId:
    """Tests for compute_result_id using ModelIdentity objects."""

    def test_deterministic(self) -> None:
        """Same inputs produce same result_id."""
        answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
        id1 = VerificationResultMetadata.compute_result_id("q1", answering, parsing, "2025-01-01T00:00:00Z")
        id2 = VerificationResultMetadata.compute_result_id("q1", answering, parsing, "2025-01-01T00:00:00Z")
        assert id1 == id2
        assert len(id1) == 16

    def test_different_interface_different_hash(self) -> None:
        """Same model_name but different interface produces different result_id."""
        a_answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        b_answering = ModelIdentity(interface="openrouter", model_name="gpt-4")
        parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
        ts = "2025-01-01T00:00:00Z"

        id_a = VerificationResultMetadata.compute_result_id("q1", a_answering, parsing, ts)
        id_b = VerificationResultMetadata.compute_result_id("q1", b_answering, parsing, ts)
        assert id_a != id_b

    def test_different_tools_different_hash(self) -> None:
        """Same model but different tools produces different result_id."""
        a = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave"])
        b = ModelIdentity(interface="langchain", model_name="gpt-4", tools=["fs"])
        parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
        ts = "2025-01-01T00:00:00Z"

        id_a = VerificationResultMetadata.compute_result_id("q1", a, parsing, ts)
        id_b = VerificationResultMetadata.compute_result_id("q1", b, parsing, ts)
        assert id_a != id_b

    def test_with_replicate(self) -> None:
        """Replicate parameter changes the result_id."""
        answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
        ts = "2025-01-01T00:00:00Z"

        id_no_rep = VerificationResultMetadata.compute_result_id("q1", answering, parsing, ts)
        id_rep1 = VerificationResultMetadata.compute_result_id("q1", answering, parsing, ts, replicate=1)
        id_rep2 = VerificationResultMetadata.compute_result_id("q1", answering, parsing, ts, replicate=2)
        assert id_no_rep != id_rep1
        assert id_rep1 != id_rep2

    def test_scenario_context_changes_hash(self) -> None:
        """Scenario turns with shared node questions must not collide."""
        answering = ModelIdentity(interface="langchain", model_name="gpt-4")
        parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
        ts = "2025-01-01T00:00:00Z"

        first = VerificationResultMetadata.compute_result_id(
            "shared-guardrail-question",
            answering,
            parsing,
            ts,
            scenario_id="scenario-a",
            scenario_node="guardrail_check",
            scenario_turn=2,
        )
        second = VerificationResultMetadata.compute_result_id(
            "shared-guardrail-question",
            answering,
            parsing,
            ts,
            scenario_id="scenario-b",
            scenario_node="guardrail_check",
            scenario_turn=2,
        )

        assert first != second


# =============================================================================
# Backward-Compatible Properties on VerificationResultMetadata
# =============================================================================


@pytest.mark.unit
class TestBackwardCompatProperties:
    """Tests for answering_model/parsing_model backward-compat properties."""

    def _make_metadata(
        self,
        answering: ModelIdentity | None = None,
        parsing: ModelIdentity | None = None,
    ) -> VerificationResultMetadata:
        return VerificationResultMetadata(
            question_id="q1",
            template_id="t1",
            failure=None,
            caveats=[],
            question_text="test",
            answering=answering or ModelIdentity(interface="langchain", model_name="gpt-4"),
            parsing=parsing or ModelIdentity(interface="langchain", model_name="claude-haiku-4-5"),
            execution_time=1.0,
            timestamp="2025-01-01T00:00:00Z",
            result_id="abc123",
        )

    def test_answering_model_returns_display_string(self) -> None:
        meta = self._make_metadata(answering=ModelIdentity(interface="langchain", model_name="gpt-4", tools=["brave"]))
        assert meta.answering_model == "langchain:gpt-4 +[brave]"

    def test_parsing_model_returns_display_string(self) -> None:
        meta = self._make_metadata(parsing=ModelIdentity(interface="claude_tool", model_name="claude-haiku-4-5"))
        assert meta.parsing_model == "claude_tool:claude-haiku-4-5"

    def test_no_tools_display(self) -> None:
        meta = self._make_metadata()
        assert meta.answering_model == "langchain:gpt-4"
        assert meta.parsing_model == "langchain:claude-haiku-4-5"


# =============================================================================
# config_id Field
# =============================================================================


@pytest.mark.unit
class TestModelIdentityConfigId:
    """Tests for ModelIdentity.config_id field and its effects on display/key."""

    def test_config_id_set_when_differs_from_model_name(self) -> None:
        """from_model_config sets config_id when config.id != model_name."""
        config = ModelConfig(
            id="high-temp",
            interface="langchain",
            model_name="gpt-4",
            model_provider="openai",
        )
        identity = ModelIdentity.from_model_config(config)
        assert identity.config_id == "high-temp"

    def test_config_id_none_when_matches_model_name(self) -> None:
        """from_model_config leaves config_id None when config.id == model_name."""
        config = ModelConfig(
            id="gpt-4",
            interface="langchain",
            model_name="gpt-4",
            model_provider="openai",
        )
        identity = ModelIdentity.from_model_config(config)
        assert identity.config_id is None

    def test_display_string_includes_config_id(self) -> None:
        """display_string includes config_id in parentheses."""
        identity = ModelIdentity(
            interface="langchain",
            model_name="gpt-4",
            config_id="high-temp",
        )
        assert identity.display_string == "langchain:gpt-4 (high-temp)"

    def test_display_string_without_config_id_unchanged(self) -> None:
        """display_string is unchanged when config_id is None."""
        identity = ModelIdentity(interface="langchain", model_name="gpt-4")
        assert identity.display_string == "langchain:gpt-4"

    def test_display_string_with_config_id_and_tools(self) -> None:
        """display_string places config_id before tools."""
        identity = ModelIdentity(
            interface="claude_agent_sdk",
            model_name="claude-sonnet-4-20250514",
            config_id="my-config",
            tools=["brave"],
        )
        assert identity.display_string == "claude_agent_sdk:claude-sonnet-4-20250514 (my-config) +[brave]"

    def test_canonical_key_includes_config_id(self) -> None:
        """canonical_key has 4 segments when config_id is present."""
        identity = ModelIdentity(
            interface="langchain",
            model_name="gpt-4",
            config_id="high-temp",
        )
        assert identity.canonical_key == "langchain:gpt-4:high-temp:"

    def test_canonical_key_without_config_id_backward_compatible(self) -> None:
        """canonical_key has 3 segments when config_id is None (backward compatible)."""
        identity = ModelIdentity(interface="langchain", model_name="gpt-4")
        assert identity.canonical_key == "langchain:gpt-4:"

    def test_different_config_ids_produce_different_identities(self) -> None:
        """Two identities with same model but different config_ids are not equal."""
        a = ModelIdentity(interface="langchain", model_name="gpt-4", config_id="high-temp")
        b = ModelIdentity(interface="langchain", model_name="gpt-4", config_id="low-temp")
        assert a != b
        assert a.canonical_key != b.canonical_key


# =============================================================================
# CSV Round-Trip with config_id
# =============================================================================


@pytest.mark.unit
class TestModelIdentityCSVRoundTrip:
    """Tests for ModelIdentity CSV round-trip via display_string and _parse_model_identity."""

    def test_round_trip_with_config_id(self) -> None:
        """display_string -> _parse_model_identity preserves config_id."""
        from karenina.benchmark.core.results_io import ResultsIOManager

        original = ModelIdentity(
            interface="langchain",
            model_name="gpt-4",
            config_id="high-temp",
        )
        reconstructed = ResultsIOManager._parse_model_identity(original.display_string)
        assert reconstructed.interface == original.interface
        assert reconstructed.model_name == original.model_name
        assert reconstructed.config_id == original.config_id
        assert reconstructed.tools == original.tools

    def test_round_trip_without_config_id(self) -> None:
        """display_string -> _parse_model_identity works without config_id."""
        from karenina.benchmark.core.results_io import ResultsIOManager

        original = ModelIdentity(
            interface="langchain",
            model_name="gpt-4",
        )
        reconstructed = ResultsIOManager._parse_model_identity(original.display_string)
        assert reconstructed.interface == original.interface
        assert reconstructed.model_name == original.model_name
        assert reconstructed.config_id is None
        assert reconstructed.tools == original.tools

    def test_round_trip_with_config_id_and_tools(self) -> None:
        """display_string -> _parse_model_identity preserves config_id and tools."""
        from karenina.benchmark.core.results_io import ResultsIOManager

        original = ModelIdentity(
            interface="claude_agent_sdk",
            model_name="claude-sonnet-4-20250514",
            config_id="my-config",
            tools=["brave"],
        )
        reconstructed = ResultsIOManager._parse_model_identity(original.display_string)
        assert reconstructed.interface == original.interface
        assert reconstructed.model_name == original.model_name
        assert reconstructed.config_id == original.config_id
        assert reconstructed.tools == original.tools
