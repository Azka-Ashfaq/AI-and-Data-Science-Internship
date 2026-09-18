"""
resilience.py — System hardening utilities (Day 5, Task 1).

Provides a single, consistent way for every graph node to call an external
tool (pandas lookups, the joblib models, langchain @tool.invoke calls):

  * Bounded execution time  -> no single slow tool call can hang a request.
  * Uniform error shape     -> every failure becomes the same
                                {"status": "error", "error": "..."} dict
                                that nodes_validation.py already knows how
                                to route to the fallback node.
  * One place to log        -> every tool call is timed and logged, which
                                doubles as the raw data for the monitoring
                                dashboard described in Task 4.

Nothing here changes *what* a tool returns on success — only how failures
and slow calls are handled.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict

logger = logging.getLogger("afl_assistant")

# A small, shared pool. Tool calls are I/O/CPU bound pandas/sklearn lookups,
# not network calls, so a handful of workers is plenty for a single-process
# API deployment; increase POOL_WORKERS if you later run this under high
# concurrency.
POOL_WORKERS = 8
_executor = ThreadPoolExecutor(max_workers=POOL_WORKERS, thread_name_prefix="afl-tool")

# Per-tool-type timeouts. Retrieval is a couple of pandas filters; prediction
# additionally loads/runs a scikit-learn pipeline, so it gets more headroom.
DEFAULT_TIMEOUT_SECONDS = 4.0
TIMEOUTS = {
    "retrieval": 6.0,
    "prediction": 8.0,
}


def safe_call(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    tool_name: str = "unknown_tool",
    request_id: str | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run `fn(*args, **kwargs)` with a wall-clock timeout and a catch-all
    exception handler. Always returns a dict:

      {"status": "success", "result": <whatever fn returned>, "latency_ms": float}
      {"status": "error", "error": "<message>", "error_type": "timeout"|"exception",
       "latency_ms": float}

    This is the single choke point every node routes tool calls through, so
    Task 1's "consistent error handling across every node/tool" and "timeouts
    for slow tool calls" are enforced in one place instead of copy-pasted
    into each node.
    """
    request_id = request_id or uuid.uuid4().hex[:8]
    start = time.monotonic()
    future = _executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "tool_call_ok", extra={
                "request_id": request_id, "tool": tool_name,
                "latency_ms": latency_ms,
            },
        )
        return {"status": "success", "result": result, "latency_ms": latency_ms}
    except FutureTimeoutError:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        future.cancel()
        logger.warning(
            "tool_call_timeout", extra={
                "request_id": request_id, "tool": tool_name,
                "timeout_s": timeout, "latency_ms": latency_ms,
            },
        )
        return {
            "status": "error",
            "error": f"'{tool_name}' took longer than {timeout:.0f}s and was aborted.",
            "error_type": "timeout",
            "latency_ms": latency_ms,
        }
    except Exception as e:  # noqa: BLE001 — this is the intentional catch-all boundary
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.error(
            "tool_call_exception", extra={
                "request_id": request_id, "tool": tool_name,
                "error": str(e), "latency_ms": latency_ms,
            },
            exc_info=True,
        )
        return {
            "status": "error",
            "error": f"'{tool_name}' failed: {e}",
            "error_type": "exception",
            "latency_ms": latency_ms,
        }


def safe_langchain_tool(tool_obj, tool_input: dict, **kw) -> Dict[str, Any]:
    """Convenience wrapper for langchain `@tool` objects (called via .invoke)."""
    return safe_call(tool_obj.invoke, tool_input, tool_name=tool_obj.name, **kw)