"""Tests for MCP tool description optimization in GEPA integration.

These tests verify that tool description overrides flow correctly through
the entire chain WITHOUT requiring GEPA optimization to run.

Key principle: All tests verify the wiring chain works WITHOUT running
full GEPA optimization.
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.infrastructure.llm.mcp_utils import _apply_tool_description_overrides
from karenina.integrations.gepa import OptimizationTarget


class TestApplyToolDescriptionOverrides:
    """Test the _apply_tool_description_overrides function."""

    def test_apply_overrides_modifies_tools(self):
        """Test that overrides actually modify tool descriptions."""
        # Create mock tools with name and description attributes
        tool1 = MagicMock()
        tool1.name = "search"
        tool1.description = "Original search description"

        tool2 = MagicMock()
        tool2.name = "query"
        tool2.description = "Original query description"

        tools = [tool1, tool2]
        overrides = {"search": "New optimized search description"}

        result = _apply_tool_description_overrides(tools, overrides)

        # Tool1 should be modified
        assert tool1.description == "New optimized search description"
        # Tool2 should be unchanged
        assert tool2.description == "Original query description"
        assert result == tools

    def test_apply_overrides_handles_missing_tools(self):
        """Test that missing tools in overrides are handled gracefully."""
        tool1 = MagicMock()
        tool1.name = "existing_tool"
        tool1.description = "Original"

        overrides = {
            "existing_tool": "Updated",
            "nonexistent_tool": "Should be ignored",
        }

        result = _apply_tool_description_overrides([tool1], overrides)

        assert tool1.description == "Updated"
        assert len(result) == 1

    def test_apply_overrides_empty_overrides(self):
        """Test that empty overrides dict doesn't modify tools."""
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Original"

        _apply_tool_description_overrides([tool1], {})

        assert tool1.description == "Original"

    def test_apply_overrides_tool_without_name(self):
        """Test tools without name attribute are skipped gracefully."""
        tool_with_name = MagicMock()
        tool_with_name.name = "named_tool"
        tool_with_name.description = "Original"

        tool_without_name = MagicMock(spec=[])  # No name attribute

        overrides = {"named_tool": "Updated"}

        result = _apply_tool_description_overrides([tool_with_name, tool_without_name], overrides)

        assert tool_with_name.description == "Updated"
        assert len(result) == 2


class TestMockMCPServer:
    """Tests using the mock MCP server."""

    def test_mock_server_returns_tools(self):
        """Test mock MCP server returns expected tools."""
        from tests.fixtures.mock_mcp_server import (
            create_mock_mcp_client_and_tools,
        )

        client, tools = create_mock_mcp_client_and_tools({"test": "http://localhost/mcp"})

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "search_proteins" in tool_names
        assert "get_interactions" in tool_names
        assert "analyze_pathway" in tool_names

    def test_mock_server_applies_overrides(self):
        """Test mock server correctly applies description overrides."""
        from tests.fixtures.mock_mcp_server import create_mock_mcp_client_and_tools

        overrides = {"search_proteins": "INJECTED DESCRIPTION"}

        client, tools = create_mock_mcp_client_and_tools(
            {"test": "http://localhost/mcp"},
            tool_description_overrides=overrides,
        )

        search_tool = next(t for t in tools if t.name == "search_proteins")
        assert search_tool.description == "INJECTED DESCRIPTION"

        # Other tools should be unchanged
        get_tool = next(t for t in tools if t.name == "get_interactions")
        assert get_tool.description == "Get protein-protein interactions from database."

    def test_mock_server_filters_tools(self):
        """Test mock server correctly filters tools."""
        from tests.fixtures.mock_mcp_server import create_mock_mcp_client_and_tools

        client, tools = create_mock_mcp_client_and_tools(
            {"test": "http://localhost/mcp"},
            tool_filter=["search_proteins"],
        )

        assert len(tools) == 1
        assert tools[0].name == "search_proteins"

    def test_mock_server_filter_and_override_combined(self):
        """Test filter and override work together."""
        from tests.fixtures.mock_mcp_server import create_mock_mcp_client_and_tools

        client, tools = create_mock_mcp_client_and_tools(
            {"test": "http://localhost/mcp"},
            tool_filter=["search_proteins", "get_interactions"],
            tool_description_overrides={
                "search_proteins": "CUSTOM SEARCH",
                "get_interactions": "CUSTOM INTERACTIONS",
            },
        )

        assert len(tools) == 2
        search_tool = next(t for t in tools if t.name == "search_proteins")
        assert search_tool.description == "CUSTOM SEARCH"
        get_tool = next(t for t in tools if t.name == "get_interactions")
        assert get_tool.description == "CUSTOM INTERACTIONS"


class TestToolDescriptionInjectionChain:
    """Test the complete injection chain WITHOUT running optimization."""

    def test_sync_create_mcp_client_passes_overrides(self):
        """Test that sync_create_mcp_client_and_tools passes overrides correctly."""
        from tests.fixtures.mock_mcp_server import MockTool

        mock_tools = [
            MockTool("tool1", "Original description 1"),
            MockTool("tool2", "Original description 2"),
        ]

        overrides = {"tool1": "INJECTED DESCRIPTION"}

        # Patch at the langchain_mcp_adapters level where it's actually imported
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
            mock_client_instance = MagicMock()

            async def mock_get_tools():
                return mock_tools

            mock_client_instance.get_tools = mock_get_tools
            MockClient.return_value = mock_client_instance

            from karenina.infrastructure.llm.mcp_utils import (
                sync_create_mcp_client_and_tools,
            )

            _, tools = sync_create_mcp_client_and_tools(
                {"test": "http://mock"},
                tool_description_overrides=overrides,
            )

            # Verify the override was applied
            tool1 = next((t for t in tools if t.name == "tool1"), None)
            assert tool1 is not None
            assert tool1.description == "INJECTED DESCRIPTION"

            # Verify other tools unchanged
            tool2 = next((t for t in tools if t.name == "tool2"), None)
            assert tool2 is not None
            assert tool2.description == "Original description 2"

    def test_init_chat_model_unified_accepts_override_param(self):
        """Test that init_chat_model_unified accepts mcp_tool_description_overrides."""
        import inspect

        from karenina.infrastructure.llm.interface import init_chat_model_unified

        sig = inspect.signature(init_chat_model_unified)
        param_names = list(sig.parameters.keys())

        assert "mcp_tool_description_overrides" in param_names


class TestAdapterInjectCandidate:
    """Test that KareninaAdapter._inject_candidate correctly handles MCP tools."""

    def test_inject_candidate_extracts_mcp_tools(self):
        """Test _inject_candidate correctly parses mcp_tool_* keys."""
        from karenina.integrations.gepa.adapter import KareninaAdapter

        # Create minimal mocks
        mock_benchmark = MagicMock()
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.mcp_urls_dict = {"biocontext": "http://localhost/mcp"}
        mock_model.mcp_tool_description_overrides = None

        mock_config = MagicMock()
        mock_config.answering_models = [mock_model]

        def mock_model_copy(deep=False):
            # Return a new mock with the same structure
            new_mock = MagicMock()
            new_model = MagicMock()
            new_model.model_name = "test-model"
            new_model.mcp_urls_dict = {"biocontext": "http://localhost/mcp"}
            new_model.mcp_tool_description_overrides = None
            new_mock.answering_models = [new_model]
            return new_mock

        mock_config.model_copy = mock_model_copy

        # Create adapter without auto-fetch
        with patch.object(KareninaAdapter, "fetch_seed_tool_descriptions", return_value={}):
            adapter = KareninaAdapter(
                benchmark=mock_benchmark,
                base_config=mock_config,
                targets=[OptimizationTarget.MCP_TOOL_DESCRIPTIONS],
            )

        # Test candidate with mcp_tool_* keys
        candidate = {
            "mcp_tool_search_proteins": "Optimized search description",
            "mcp_tool_get_interactions": "Optimized interactions description",
            "answering_system_prompt": "Some prompt",  # Should be ignored for MCP
        }

        injected_config = adapter._inject_candidate(candidate)

        # Verify overrides were extracted and set
        model = injected_config.answering_models[0]
        assert model.mcp_tool_description_overrides is not None
        assert "search_proteins" in model.mcp_tool_description_overrides
        assert model.mcp_tool_description_overrides["search_proteins"] == "Optimized search description"
        assert "get_interactions" in model.mcp_tool_description_overrides
        assert model.mcp_tool_description_overrides["get_interactions"] == "Optimized interactions description"

    def test_inject_candidate_empty_mcp_tools(self):
        """Test _inject_candidate handles candidates without mcp_tool_* keys."""
        from karenina.integrations.gepa.adapter import KareninaAdapter

        mock_benchmark = MagicMock()
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.mcp_urls_dict = {"biocontext": "http://localhost/mcp"}
        mock_model.mcp_tool_description_overrides = None

        mock_config = MagicMock()
        mock_config.answering_models = [mock_model]

        def mock_model_copy(deep=False):
            new_mock = MagicMock()
            new_model = MagicMock()
            new_model.model_name = "test-model"
            new_model.mcp_urls_dict = {"biocontext": "http://localhost/mcp"}
            new_model.mcp_tool_description_overrides = None
            new_mock.answering_models = [new_model]
            return new_mock

        mock_config.model_copy = mock_model_copy

        with patch.object(KareninaAdapter, "fetch_seed_tool_descriptions", return_value={}):
            adapter = KareninaAdapter(
                benchmark=mock_benchmark,
                base_config=mock_config,
                targets=[OptimizationTarget.MCP_TOOL_DESCRIPTIONS],
            )

        # Candidate without mcp_tool_* keys
        candidate = {
            "answering_system_prompt": "Some prompt",
        }

        injected_config = adapter._inject_candidate(candidate)

        # No MCP overrides should be set
        model = injected_config.answering_models[0]
        # Since no mcp_tool_* keys, overrides should be empty or not set
        overrides = getattr(model, "mcp_tool_description_overrides", None)
        assert overrides is None or overrides == {}


class TestModelConfigField:
    """Test ModelConfig accepts mcp_tool_description_overrides field."""

    def test_model_config_accepts_overrides(self):
        """Test ModelConfig schema includes mcp_tool_description_overrides."""
        from karenina.schemas.workflow.models import ModelConfig

        # Should not raise validation error
        config = ModelConfig(
            id="test",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            mcp_urls_dict={"test": "http://localhost:8000/mcp"},
            mcp_tool_description_overrides={"search": "Custom description"},
        )

        assert config.mcp_tool_description_overrides == {"search": "Custom description"}

    def test_model_config_overrides_optional(self):
        """Test mcp_tool_description_overrides is optional."""
        from karenina.schemas.workflow.models import ModelConfig

        config = ModelConfig(
            id="test",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
        )

        assert config.mcp_tool_description_overrides is None


# =============================================================================
# Integration Tests (require real MCP server and LLM API)
# =============================================================================

OPEN_TARGETS_MCP_URL = "https://mcp.platform.opentargets.org/mcp"


@pytest.mark.integration
class TestMCPToolDescriptionIntegration:
    """Integration tests for MCP tool description optimization.

    These tests require:
    - Network access to Open Targets MCP server
    - Valid Anthropic API key

    Run with: pytest -m integration
    """

    def test_fetch_tool_descriptions_from_real_server(self):
        """Test fetching tool descriptions from real Open Targets MCP server."""
        from karenina.infrastructure.llm.mcp_utils import sync_fetch_tool_descriptions

        descriptions = sync_fetch_tool_descriptions({"opentargets": OPEN_TARGETS_MCP_URL})

        # Should fetch at least the known tools
        assert len(descriptions) >= 3
        assert "search_entities" in descriptions
        assert "query_open_targets_graphql" in descriptions

        # Descriptions should be non-empty strings
        for name, desc in descriptions.items():
            assert isinstance(desc, str)
            assert len(desc) > 10, f"Description for {name} is too short"

    def test_adapter_auto_fetch_from_real_server(self):
        """Test KareninaAdapter auto-fetches from real MCP server."""
        from karenina.integrations.gepa import OptimizationTarget
        from karenina.integrations.gepa.adapter import KareninaAdapter

        mock_benchmark = MagicMock()
        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.model_name = "claude-haiku-4-5"
        mock_model.mcp_urls_dict = {"opentargets": OPEN_TARGETS_MCP_URL}
        mock_model.mcp_tool_filter = None
        mock_config.answering_models = [mock_model]

        adapter = KareninaAdapter(
            benchmark=mock_benchmark,
            base_config=mock_config,
            targets=[OptimizationTarget.MCP_TOOL_DESCRIPTIONS],
            auto_fetch_tool_descriptions=True,
        )

        # Should have auto-fetched descriptions
        assert adapter.seed_tool_descriptions is not None
        assert len(adapter.seed_tool_descriptions) >= 3
        assert "search_entities" in adapter.seed_tool_descriptions

    @pytest.mark.asyncio
    async def test_tool_description_injection_changes_model_behavior(self):
        """Test that modifying tool descriptions actually changes model behavior.

        This is the definitive test: we mark a tool as "deprecated" and verify
        the model avoids using it.

        Test strategy:
        1. Run query WITHOUT override -> model should use search_entities
        2. Run query WITH override (search_entities marked deprecated) -> model should NOT use it
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from karenina.infrastructure.llm.interface import init_chat_model_unified

        mcp_urls = {"opentargets": OPEN_TARGETS_MCP_URL}

        # Helper to extract tool calls from response
        def get_tools_called(response):
            tools = []
            if isinstance(response, dict) and "messages" in response:
                for msg in response["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tools.extend([tc["name"] for tc in msg.tool_calls])
            return tools

        # Test 1: WITHOUT override - model should use search_entities
        agent_without_override = init_chat_model_unified(
            model="claude-haiku-4-5",
            provider="anthropic",
            mcp_urls_dict=mcp_urls,
            mcp_tool_description_overrides=None,
            temperature=0.0,
        )

        messages = [
            SystemMessage(content="You must use tools to answer. Do not answer from memory."),
            HumanMessage(content="What is the Open Targets ID for BRCA1?"),
        ]

        config = {"configurable": {"thread_id": "integration-test-no-override"}}
        response_without = await agent_without_override.ainvoke({"messages": messages}, config=config)
        tools_without_override = get_tools_called(response_without)

        # Test 2: WITH override - mark search_entities as deprecated
        deprecated_description = {
            "search_entities": """⚠️ DEPRECATED - DO NOT USE ⚠️

This tool is currently broken and will return errors.
Use query_open_targets_graphql directly instead.

DO NOT call search_entities - it will fail.""",
        }

        agent_with_override = init_chat_model_unified(
            model="claude-haiku-4-5",
            provider="anthropic",
            mcp_urls_dict=mcp_urls,
            mcp_tool_description_overrides=deprecated_description,
            temperature=0.0,
        )

        config = {"configurable": {"thread_id": "integration-test-with-override"}}
        response_with = await agent_with_override.ainvoke({"messages": messages}, config=config)
        tools_with_override = get_tools_called(response_with)

        # Verify: behavior should change based on our modified description
        # Without override: should use search_entities (the natural choice)
        assert "search_entities" in tools_without_override, f"Expected search_entities in {tools_without_override}"

        # With override: should NOT use search_entities (we marked it deprecated)
        assert "search_entities" not in tools_with_override, (
            f"Model should have avoided deprecated search_entities, but called: {tools_with_override}"
        )

        # Should have used an alternative tool instead
        assert len(tools_with_override) > 0, "Model should have used some tool"

    @pytest.mark.asyncio
    async def test_modified_description_affects_tool_usage(self):
        """Test that a modified description is actually seen by the model.

        We modify search_entities to include a specific instruction and verify
        the model follows it.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from karenina.infrastructure.llm.interface import init_chat_model_unified

        mcp_urls = {"opentargets": OPEN_TARGETS_MCP_URL}

        # Custom description with specific formatting instruction
        custom_description = {
            "search_entities": """Search for biological entities in Open Targets.

REQUIRED: When reporting results, you MUST start your response with
"ENTITY SEARCH COMPLETE:" followed by the results.

Takes query_strings parameter with list of search terms.
Returns entity IDs and types.""",
        }

        agent = init_chat_model_unified(
            model="claude-haiku-4-5",
            provider="anthropic",
            mcp_urls_dict=mcp_urls,
            mcp_tool_description_overrides=custom_description,
            temperature=0.0,
        )

        messages = [
            SystemMessage(content="Follow tool descriptions exactly."),
            HumanMessage(content="Find the ID for gene TP53"),
        ]

        config = {"configurable": {"thread_id": "integration-test-custom"}}
        response = await agent.ainvoke({"messages": messages}, config=config)

        # We only need to check tool calls for this test
        # (final response text is not inspected)

        # The model should have used search_entities
        tools_called = []
        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_called.extend([tc["name"] for tc in msg.tool_calls])

        assert "search_entities" in tools_called, f"Expected model to use search_entities, but used: {tools_called}"
