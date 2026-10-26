from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class StreamStallError(TimeoutError):
    """Raised when an LLM stream stops producing chunks."""


@contextmanager
def force_stream_assemble(*, stall_seconds: float, connect_seconds: float = 30.0) -> Iterator[None]:
    """Route DSPy LiteLLM completions through streaming with inter-chunk timeouts.

    Non-streaming calls can hang until the full response timeout even when the
    upstream is wedged. Streaming lets httpx treat ``read`` as the gap between
    chunks, so a silent provider fails fast and DSPy retries.
    """
    import dspy.clients.lm as lm_mod
    import httpx
    import litellm

    stall = max(5.0, float(stall_seconds))
    connect = max(5.0, float(connect_seconds))
    original_sync = lm_mod.litellm_completion
    original_async = lm_mod.alitellm_completion

    def _timeout() -> httpx.Timeout:
        # read timeout applies between streamed chunks, not to the whole reply.
        return httpx.Timeout(connect=connect, read=stall, write=connect, pool=connect)

    def _assemble_sync(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None) -> Any:
        ui_stream = lm_mod._get_stream_completion_fn(request, cache, sync=True)
        if ui_stream is not None:
            return ui_stream()
        req = dict(request)
        req["stream"] = True
        req.setdefault("stream_options", {"include_usage": True})
        req["timeout"] = _timeout()
        started = time.monotonic()
        last_chunk_at = started
        chunks: list[Any] = []
        stream = litellm.completion(**req, num_retries=num_retries, cache=cache)
        try:
            for chunk in stream:
                now = time.monotonic()
                gap = now - last_chunk_at
                if gap > stall:
                    raise StreamStallError(
                        f"LLM stream stalled for {gap:.1f}s (limit {stall:.1f}s) after {len(chunks)} chunk(s)"
                    )
                last_chunk_at = now
                chunks.append(chunk)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass
        if not chunks:
            raise StreamStallError("LLM stream returned no chunks")
        built = litellm.stream_chunk_builder(chunks)
        logger.debug(
            "stream_assemble chunks=%s elapsed=%.2fs stall_limit=%.1fs",
            len(chunks),
            time.monotonic() - started,
            stall,
        )
        return built

    async def _assemble_async(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None) -> Any:
        ui_stream = lm_mod._get_stream_completion_fn(request, cache, sync=False)
        if ui_stream is not None:
            return await ui_stream()
        req = dict(request)
        req["stream"] = True
        req.setdefault("stream_options", {"include_usage": True})
        req["timeout"] = _timeout()
        started = time.monotonic()
        last_chunk_at = started
        chunks: list[Any] = []
        stream = await litellm.acompletion(**req, num_retries=num_retries, cache=cache)
        try:
            async for chunk in stream:
                now = time.monotonic()
                gap = now - last_chunk_at
                if gap > stall:
                    raise StreamStallError(
                        f"LLM stream stalled for {gap:.1f}s (limit {stall:.1f}s) after {len(chunks)} chunk(s)"
                    )
                last_chunk_at = now
                chunks.append(chunk)
        finally:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # pragma: no cover
                    pass
        if not chunks:
            raise StreamStallError("LLM stream returned no chunks")
        built = litellm.stream_chunk_builder(chunks)
        logger.debug(
            "stream_assemble_async chunks=%s elapsed=%.2fs stall_limit=%.1fs",
            len(chunks),
            time.monotonic() - started,
            stall,
        )
        return built

    lm_mod.litellm_completion = _assemble_sync  # type: ignore[assignment]
    lm_mod.alitellm_completion = _assemble_async  # type: ignore[assignment]
    try:
        yield
    finally:
        lm_mod.litellm_completion = original_sync
        lm_mod.alitellm_completion = original_async
