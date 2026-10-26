from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from .streaming import StreamStallError


logger = logging.getLogger(__name__)

T = TypeVar("T")

_OVERLOAD_MARKERS = (
    "overloaded",
    "currently overloaded",
    "midstreamfallback",
    "mid_stream_fallback",
    "rate limit",
    "ratelimit",
    "too many requests",
    "server is busy",
    "service unavailable",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "read timeout",
    "apiconnectionerror",
    "timeout",
    "timed out",
    "stream stalled",
)


def is_overload_error(exc: BaseException) -> bool:
    """Return True for provider overload / transient transport failures worth retrying."""
    if isinstance(exc, StreamStallError):
        return True
    name = type(exc).__name__.casefold()
    if any(token in name for token in ("midstream", "ratelimit", "timeout", "apiconnection", "serviceunavailable")):
        return True
    text = str(exc).casefold()
    if any(marker in text for marker in _OVERLOAD_MARKERS):
        return True
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException) and cause is not exc:
        return is_overload_error(cause)
    context = getattr(exc, "__context__", None)
    if isinstance(context, BaseException) and context is not exc:
        return is_overload_error(context)
    return False


def call_with_overload_retry(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    backoff_seconds: float = 5.0,
    backoff_max_seconds: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "provider_call",
) -> T:
    """Run ``fn`` and retry on overload/transient provider errors with exponential backoff."""
    attempts = max(1, int(retries) + 1)
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classify then re-raise or retry
            last_error = exc
            if attempt >= attempts or not is_overload_error(exc):
                raise
            delay = min(backoff_max_seconds, backoff_seconds * (2 ** (attempt - 1)))
            delay = delay + random.uniform(0.0, min(1.0, delay * 0.1))
            logger.warning(
                "%s overloaded/transient failure attempt=%s/%s sleep=%.1fs error=%s",
                label,
                attempt,
                attempts,
                delay,
                exc,
            )
            sleep(delay)
    assert last_error is not None
    raise last_error
