"""Unit tests for karenina.utils.file_ops.atomic_write().

Tests cover:
- Successful file write with correct content
- Atomic rename behavior (target file not corrupted on failure)
- Partial file cleanup on write failure
- Overwriting existing files
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from karenina.utils.file_ops import atomic_write


@pytest.mark.unit
class TestAtomicWrite:
    """Tests for atomic_write() function."""

    def test_writes_content_correctly(self, tmp_path: Path) -> None:
        """File content matches what was written."""
        filepath = tmp_path / "output.json"
        content = '{"key": "value"}'

        atomic_write(filepath, content)

        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == content

    def test_partial_file_removed_after_success(self, tmp_path: Path) -> None:
        """The .partial temp file is not left behind after successful write."""
        filepath = tmp_path / "output.json"
        partial_path = filepath.with_suffix(".json.partial")

        atomic_write(filepath, "data")

        assert not partial_path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Existing file is replaced with new content."""
        filepath = tmp_path / "output.json"
        filepath.write_text("old content", encoding="utf-8")

        atomic_write(filepath, "new content")

        assert filepath.read_text(encoding="utf-8") == "new content"

    def test_original_preserved_on_write_failure(self, tmp_path: Path) -> None:
        """If write fails, original file content is preserved."""
        filepath = tmp_path / "output.json"
        filepath.write_text("original", encoding="utf-8")

        # Simulate write failure by making the partial file write raise
        with (
            patch("karenina.utils.file_ops.open", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            atomic_write(filepath, "should not appear")

        assert filepath.read_text(encoding="utf-8") == "original"

    def test_partial_file_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        """Partial file is removed after a failure during rename."""
        filepath = tmp_path / "output.json"
        partial_path = filepath.with_suffix(".json.partial")

        # Write the partial file successfully but fail on replace
        with (
            patch.object(Path, "replace", side_effect=OSError("rename failed")),
            pytest.raises(OSError, match="rename failed"),
        ):
            atomic_write(filepath, "data")

        assert not partial_path.exists()

    def test_creates_file_in_existing_directory(self, tmp_path: Path) -> None:
        """Can create a new file in a nested existing directory."""
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        filepath = nested / "result.json"

        atomic_write(filepath, "nested data")

        assert filepath.read_text(encoding="utf-8") == "nested data"

    def test_handles_empty_content(self, tmp_path: Path) -> None:
        """Writing empty string creates an empty file."""
        filepath = tmp_path / "empty.json"

        atomic_write(filepath, "")

        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == ""

    def test_handles_unicode_content(self, tmp_path: Path) -> None:
        """Unicode content is written correctly."""
        filepath = tmp_path / "unicode.json"
        content = '{"emoji": "🧪", "kanji": "漢字"}'

        atomic_write(filepath, content)

        assert filepath.read_text(encoding="utf-8") == content


@pytest.mark.unit
class TestAtomicWriter:
    """Tests for atomic_writer() context manager."""

    def test_writes_streamed_content_correctly(self, tmp_path: Path) -> None:
        """Content written chunk by chunk ends up in the target file."""
        from karenina.utils.file_ops import atomic_writer

        filepath = tmp_path / "output.json"
        with atomic_writer(filepath) as f:
            f.write('{"a":')
            f.write("1")
            f.write("}")

        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == '{"a":1}'

    def test_partial_removed_after_success(self, tmp_path: Path) -> None:
        """The .partial sidecar is not left behind after a clean exit."""
        from karenina.utils.file_ops import atomic_writer

        filepath = tmp_path / "output.json"
        partial_path = filepath.with_suffix(".json.partial")
        with atomic_writer(filepath) as f:
            f.write("data")

        assert not partial_path.exists()

    def test_exception_inside_with_block_removes_partial(self, tmp_path: Path) -> None:
        """If the caller raises inside the with block, partial is cleaned up and target never appears."""
        from karenina.utils.file_ops import atomic_writer

        filepath = tmp_path / "output.json"
        partial_path = filepath.with_suffix(".json.partial")

        with pytest.raises(RuntimeError, match="boom"), atomic_writer(filepath) as f:
            f.write("some-data")
            raise RuntimeError("boom")

        assert not filepath.exists()
        assert not partial_path.exists()

    def test_write_error_cleans_up_partial(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If an OSError is raised during write, the .partial file is still removed."""
        from karenina.utils.file_ops import atomic_writer

        filepath = tmp_path / "output.json"
        partial_path = filepath.with_suffix(".json.partial")

        real_open = open

        class FailingFile:
            def __init__(self, path: str, mode: str, **kwargs: object) -> None:
                self._path = path
                self._mode = mode
                self._real = real_open(path, mode, **kwargs)

            def write(self, data: str) -> int:  # noqa: ARG002
                raise OSError("disk full")

            def fileno(self) -> int:
                return self._real.fileno()

            def flush(self) -> None:
                self._real.flush()

            @property
            def closed(self) -> bool:
                return self._real.closed

            def close(self) -> None:
                if not self._real.closed:
                    self._real.close()

            def __enter__(self) -> "FailingFile":
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        def fake_open(path: str, mode: str = "r", **kwargs: object) -> object:
            if str(path).endswith(".partial"):
                return FailingFile(path, mode, **kwargs)
            return real_open(path, mode, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        with pytest.raises(OSError, match="disk full"), atomic_writer(filepath) as f:
            f.write("data")

        assert not filepath.exists()
        assert not partial_path.exists()
