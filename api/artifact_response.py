"""FileResponse transport semantics over an already-authorized open file."""

from __future__ import annotations

from secrets import token_hex

import anyio
from starlette.datastructures import MutableHeaders
from starlette.responses import FileResponse


class OpenedArtifactResponse(FileResponse):
    """Keep Range/If-Range/stat headers without reopening an untrusted path."""

    def __init__(self, path, handle, stat_result, *, media_type):
        super().__init__(path, filename=path.name, stat_result=stat_result, media_type=media_type)
        self._handle = handle

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Also close for malformed/unsatisfiable Range, HEAD or disconnected
            # clients; FileResponse may return before a background task runs.
            self._handle.close()

    async def _body(self, send, start, end):
        stream = anyio.wrap_file(self._handle)
        await stream.seek(start)
        while start < end:
            chunk = await stream.read(min(self.chunk_size, end - start))
            if not chunk:
                raise OSError("Artifact changed during download")
            start += len(chunk)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

    async def _handle_simple(self, send, send_header_only, send_pathsend):
        # Ignore pathsend: handing the path to the server would reopen it after
        # the owner's manifest and opened-identity checks.
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        if not send_header_only:
            await self._body(send, 0, self.stat_result.st_size)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _handle_single_range(self, send, start, end, file_size, send_header_only):
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if not send_header_only:
            await self._body(send, start, end)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _handle_multiple_ranges(self, send, ranges, file_size, send_header_only):
        boundary = token_hex(13)

        def header(start, end):
            return (
                f"--{boundary}\r\nContent-Type: {self.headers['content-type']}\r\n"
                f"Content-Range: bytes {start}-{end - 1}/{file_size}\r\n\r\n"
            ).encode("latin-1")

        length = sum(len(header(start, end)) + end - start + 2 for start, end in ranges) + len(boundary) + 4
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(length)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if not send_header_only:
            for start, end in ranges:
                await send({"type": "http.response.body", "body": header(start, end), "more_body": True})
                await self._body(send, start, end)
                await send({"type": "http.response.body", "body": b"\r\n", "more_body": True})
            await send({"type": "http.response.body", "body": f"--{boundary}--".encode(), "more_body": False})
        else:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
