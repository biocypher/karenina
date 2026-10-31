"""Availability checks for the Oh My Pi adapter."""

from __future__ import annotations

import importlib.util
import shutil

from karenina.adapters.registry import AdapterAvailability


def check_omp_available() -> AdapterAvailability:
    """Return whether both the OMP CLI and ACP Python client are installed."""
    if shutil.which("omp") is None:
        return AdapterAvailability(
            available=False,
            reason="Oh My Pi CLI not found. Install it from https://omp.sh/.",
            fallback_interface="langchain",
        )
    if importlib.util.find_spec("acp") is None:
        return AdapterAvailability(
            available=False,
            reason="ACP Python client not installed. Install karenina with the 'omp' extra.",
            fallback_interface="langchain",
        )
    return AdapterAvailability(
        available=True,
        reason="Oh My Pi CLI and ACP Python client are installed",
    )
