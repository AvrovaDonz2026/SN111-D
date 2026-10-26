from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator


_DSPY_CONFIGURE_LOCK = threading.Lock()
_AGENT_SLOT_LOCK = threading.Lock()
_AGENT_SLOTS: threading.Semaphore | None = None


def agent_max_concurrent() -> int:
    """Cap simultaneous agent_v1 model runs (batch papers share this pool)."""
    raw = os.getenv("SUBNET_CLAIMS_AGENT_MAX_CONCURRENT", "").strip()
    if raw:
        return max(1, int(raw))
    # Fall back to batch worker setting so one env knob can drive both.
    batch = os.getenv("CLAIMS_MINER_BATCH_MAX_WORKERS", "4").strip() or "4"
    return max(1, int(batch))


def _agent_slots() -> threading.Semaphore:
    global _AGENT_SLOTS
    with _AGENT_SLOT_LOCK:
        if _AGENT_SLOTS is None:
            _AGENT_SLOTS = threading.Semaphore(agent_max_concurrent())
        return _AGENT_SLOTS


@contextmanager
def acquire_agent_slot() -> Iterator[None]:
    """Bound concurrent LLM extractions so provider rate limits don't cascade."""
    slots = _agent_slots()
    slots.acquire()
    try:
        yield
    finally:
        slots.release()


@contextmanager
def dspy_configure_lock() -> Iterator[None]:
    """Serialize legacy ``dspy.configure`` when ``dspy.context`` is unavailable."""
    with _DSPY_CONFIGURE_LOCK:
        yield


def reset_agent_slots_for_tests() -> None:
    """Test helper: rebuild the semaphore after env changes."""
    global _AGENT_SLOTS
    with _AGENT_SLOT_LOCK:
        _AGENT_SLOTS = None
