"""Tests for system prompt injection into tagged_messages by ScenarioManager."""

from __future__ import annotations

import pytest

from karenina.ports.messages import Message
from karenina.scenario.handover import TaggedMessage, format_transcript


@pytest.mark.unit
class TestSystemPromptInjectionViaTranscript:
    """Verify system prompt injection by checking format_transcript output
    on hand-built tagged_messages lists that simulate ScenarioManager behavior."""

    def test_first_node_injects_system_prompt(self) -> None:
        """First node should have system prompt in tagged_messages."""
        tagged: list[TaggedMessage] = []
        previous_agent_id = None
        agent_id = "qwen"
        system_prompt = "Be helpful."

        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        if agent_changed and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Q1"), agent_id="__user__"))
        tagged.append(TaggedMessage(Message.assistant("A1"), agent_id="qwen"))

        result = format_transcript(tagged)
        assert "[qwen:system:text] Be helpful." in result

    def test_same_agent_does_not_reinject(self) -> None:
        """Second node with same agent should not have a second system prompt."""
        tagged: list[TaggedMessage] = []
        previous_agent_id = None
        previous_system_prompt = None

        # Node A
        agent_id = "qwen"
        system_prompt = "Be helpful."
        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        prompt_changed = system_prompt != previous_system_prompt
        if (agent_changed or prompt_changed) and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Q1"), agent_id="__user__"))
        tagged.append(TaggedMessage(Message.assistant("A1"), agent_id="qwen"))
        previous_agent_id = agent_id
        previous_system_prompt = system_prompt

        # Node B (same agent)
        agent_id = "qwen"
        system_prompt = "Be helpful."
        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        prompt_changed = system_prompt != previous_system_prompt
        if (agent_changed or prompt_changed) and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Q2"), agent_id="__user__"))
        tagged.append(TaggedMessage(Message.assistant("A2"), agent_id="qwen"))

        result = format_transcript(tagged)
        assert result.count("[qwen:system:text]") == 1

    def test_agent_change_injects_new_system_prompt(self) -> None:
        """Different agent should get a new system prompt injection."""
        tagged: list[TaggedMessage] = []
        previous_agent_id = None
        previous_system_prompt = None

        # Node A (qwen)
        agent_id = "qwen"
        system_prompt = "Be a biomedical expert."
        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        prompt_changed = system_prompt != previous_system_prompt
        if (agent_changed or prompt_changed) and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Q1"), agent_id="__user__"))
        tagged.append(TaggedMessage(Message.assistant("A1"), agent_id="qwen"))
        previous_agent_id = agent_id
        previous_system_prompt = system_prompt

        # Node C (guardrail, different agent)
        agent_id = "guardrail"
        system_prompt = "Be a guardrail judge."
        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        prompt_changed = system_prompt != previous_system_prompt
        if (agent_changed or prompt_changed) and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Evaluate"), agent_id="__user__"))

        result = format_transcript(tagged)
        assert "[qwen:system:text] Be a biomedical expert." in result
        assert "[guardrail:system:text] Be a guardrail judge." in result

    def test_model_override_changes_system_prompt_same_agent(self) -> None:
        """model_override changing system_prompt (but same agent_identity) should inject."""
        tagged: list[TaggedMessage] = []
        previous_agent_id = None
        previous_system_prompt = None

        # Node A
        agent_id = "qwen"
        system_prompt = "Be a biomedical expert."
        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        prompt_changed = system_prompt != previous_system_prompt
        if (agent_changed or prompt_changed) and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Q1"), agent_id="__user__"))
        tagged.append(TaggedMessage(Message.assistant("A1"), agent_id="qwen"))
        previous_agent_id = agent_id
        previous_system_prompt = system_prompt

        # Node B (same agent_id, but model_override changed system_prompt)
        agent_id = "qwen"
        system_prompt = "Be a chemistry expert."
        agent_changed = previous_agent_id is None or previous_agent_id != agent_id
        prompt_changed = system_prompt != previous_system_prompt
        if (agent_changed or prompt_changed) and system_prompt:
            tagged.append(TaggedMessage(Message.system(system_prompt), agent_id=agent_id))
        tagged.append(TaggedMessage(Message.user("Q2"), agent_id="__user__"))
        tagged.append(TaggedMessage(Message.assistant("A2"), agent_id="qwen"))

        result = format_transcript(tagged)
        assert "[qwen:system:text] Be a biomedical expert." in result
        assert "[qwen:system:text] Be a chemistry expert." in result
        assert result.count("[qwen:system:text]") == 2
