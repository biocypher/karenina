"""Tests for the response-characterization entry point."""

from pathlib import Path

import pytest

from paper.config import RESPONSE_OUTPUT_DIR
from paper.otp_response_characterization.run import parse_args


@pytest.mark.unit
def test_no_argument_command_selects_default_full_run() -> None:
    """The documented no-argument command must parse as the full fresh run."""
    args = parse_args([])

    assert args.analyses == []
    assert args.output_dir == RESPONSE_OUTPUT_DIR
    assert args.reuse_stored_judgments is False


@pytest.mark.unit
def test_regeneration_can_use_an_isolated_output_directory(tmp_path: Path) -> None:
    """A distinct output directory avoids overwriting prior analysis files."""
    args = parse_args(["--output-dir", str(tmp_path), "--reuse-stored-judgments"])

    assert args.output_dir == tmp_path
    assert args.reuse_stored_judgments is True
