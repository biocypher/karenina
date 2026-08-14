"""Tests for Maraviroc evidence inspection."""

from __future__ import annotations

import json

import pytest

from paper.otp_response_characterization.analyses.maraviroc import trace_support_flags


@pytest.mark.unit
class TestMaravirocEvidence:
    """Validate structured approval evidence detection."""

    def test_successful_tool_payload_exposes_both_signals(self) -> None:
        content = json.dumps(
            {
                "status": "success",
                "result": {
                    "clinicalStage": "APPROVAL",
                    "firstApprovalYear": 2007,
                },
            }
        )
        assert trace_support_flags([{"role": "tool", "content": content}]) == (
            True,
            True,
        )

    def test_failed_or_non_tool_messages_are_ignored(self) -> None:
        payload = json.dumps(
            {
                "status": "error",
                "result": {"stage": "APPROVAL", "approvalDate": "2007-08-06"},
            }
        )
        assert trace_support_flags(
            [
                {"role": "assistant", "content": payload},
                {"role": "tool", "content": payload},
            ]
        ) == (False, False)
