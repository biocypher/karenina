"""Tests for ManualLLMAdapter protocol shape (T9).

ManualLLMAdapter.astream must match the LLMPort protocol shape: calling
it returns an async context manager yielding StreamingLLMResponse, and
the ManualInterfaceError fires when the context is entered, not at call
time.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

import pytest

from karenina.adapters.manual import ManualInterfaceError, ManualLLMAdapter
from karenina.ports import Message

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


@pytest.mark.unit
class TestManualLLMAdapterAstreamShape:
    def test_astream_call_returns_async_context_manager(self) -> None:
        adapter = ManualLLMAdapter(model_config=None)
        ctx = adapter.astream([Message.user("Hi")])
        assert isinstance(ctx, AbstractAsyncContextManager)

    async def test_astream_raises_on_enter(self) -> None:
        adapter = ManualLLMAdapter(model_config=None)
        with pytest.raises(ManualInterfaceError):
            async with adapter.astream([Message.user("Hi")]):
                pass
