"""Read-only wrapper for DeepAgents backends."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from deepagents.backends.protocol import (
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)


class ReadOnlyBackend:
    """Backend wrapper that delegates reads but blocks mutation and execution.

    This intentionally does not expose an ``execute`` method, so DeepAgents
    should not construct a shell execution tool for this backend.
    """

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    def ls_info(self, path: str) -> list[FileInfo]:
        return cast(list[FileInfo], self.delegate.ls_info(path))

    async def als_info(self, path: str) -> list[FileInfo]:
        return await asyncio.to_thread(self.ls_info, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return cast(str, self.delegate.read(file_path, offset, limit))

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return cast(list[GrepMatch] | str, self.delegate.grep_raw(pattern, path, glob))

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return cast(list[FileInfo], self.delegate.glob_info(pattern, path))

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.glob_info, pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:  # noqa: ARG002
        return WriteResult(error=f"Read-only backend: writing '{file_path}' is not allowed.")

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,  # noqa: ARG002
        new_string: str,  # noqa: ARG002
        replace_all: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> EditResult:
        return EditResult(error=f"Read-only backend: editing '{file_path}' is not allowed.")

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _content in files]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return cast(list[FileDownloadResponse], self.delegate.download_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)

    def __getattr__(self, name: str) -> Any:
        if name in {"execute", "aexecute"}:
            raise AttributeError(name)
        return getattr(self.delegate, name)
