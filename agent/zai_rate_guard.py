"""Serialize outgoing Z.AI requests to stay under Coding Plan concurrency caps.

Z.AI's GLM Coding Plan has an undocumented ~1-concurrent cap per key (see
opencode #8618, letta-code #1394). Hermes normally fans out main + LCM +
session_search + flush_memories + title_generation in parallel on the same
key, blowing the cap and surfacing as 429 code 1302 / 1305. This module
gates `.chat.completions.create()` entry points with a bounded semaphore
so at most N zai requests start at once.

Gates .create() only, not the full stream lifecycle — full serialization
would require intrusive changes to the streaming loop. In practice this
still flattens the main + aux fanout that is the dominant collision case
(aux calls are short, so the semaphore turns over quickly).

Env var:
    ZAI_MAX_CONCURRENT (default 2) — max in-flight .create() calls.
"""
import contextlib
import os
import threading

_MAX_CONCURRENT = max(1, int(os.getenv("ZAI_MAX_CONCURRENT", "2")))
_sem = threading.BoundedSemaphore(_MAX_CONCURRENT)


@contextlib.contextmanager
def zai_guard(provider):
    """Serialize zai-provider .create() calls. No-op for other providers."""
    if (provider or "").lower() != "zai":
        yield
        return
    _sem.acquire()
    try:
        yield
    finally:
        _sem.release()
