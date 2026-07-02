from __future__ import annotations

from dataclasses import dataclass

from karenina.adapters.langchain_deep_agents.read_only_backend import ReadOnlyBackend


class _StringReadDelegate:
    def read(self, _file_path: str, _offset: int = 0, _limit: int = 2000) -> str:
        return "x" * 200


@dataclass
class _ReadResult:
    file_data: dict[str, str]
    error: str | None = None


class _StructuredReadDelegate:
    def read(self, _file_path: str, _offset: int = 0, _limit: int = 2000) -> _ReadResult:
        return _ReadResult(file_data={"content": "x" * 200, "encoding": "utf-8"})


def test_read_only_backend_truncates_string_reads_when_enabled() -> None:
    backend = ReadOnlyBackend(_StringReadDelegate(), read_max_bytes=160)

    result = backend.read("/wide.csv")

    assert isinstance(result, str)
    assert len(result.encode("utf-8")) <= 160
    assert "File read truncated at 160 bytes" in result


def test_read_only_backend_truncates_structured_reads_when_enabled() -> None:
    backend = ReadOnlyBackend(_StructuredReadDelegate(), read_max_bytes=160)

    result = backend.read("/wide.csv")

    content = result.file_data["content"]
    assert len(content.encode("utf-8")) <= 160
    assert "File read truncated at 160 bytes" in content


def test_read_only_backend_leaves_reads_uncapped_by_default() -> None:
    backend = ReadOnlyBackend(_StringReadDelegate())

    result = backend.read("/wide.csv")

    assert result == "x" * 200
