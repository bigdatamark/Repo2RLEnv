# [MODULE DOC] Small bounded retry for observational (read-only) operations on transient connection
# errors; never to be used for model calls or job dispatch.
"""Small retry bound for observational operations; never use for model or job dispatch."""

from __future__ import annotations

import time
from collections.abc import Callable


def transient_connection_error(error: Exception) -> bool:
    # [FUNCTION DOC] Returns True for TimeoutError/ConnectionError, errors with status_code 408, 429
    # or 5xx gateway codes, or known Daytona/httpx timeout and connection error class names.
    return (
        isinstance(error, (TimeoutError, ConnectionError))
        or getattr(error, "status_code", None) in {408, 429, 500, 502, 503, 504}
        or type(error).__name__
        in {
            "DaytonaConnectionError",
            "DaytonaConnectionTimeoutError",
            "DaytonaTimeoutError",
            "DaytonaRateLimitError",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectError",
            "ConnectionError",
        }
    )


def retry_read[Result](operation: Callable[[], Result], *, attempts: int = 3) -> Result:
    # [FUNCTION DOC] Calls operation up to attempts times (1-3, default 3; ValueError otherwise),
    # sleeping 0.5s * 2^attempt between tries, retrying only transient connection errors. Returns
    # the result or re-raises the last/non-transient error.
    if not 1 <= attempts <= 3:
        raise ValueError("Read retries allow one to three attempts")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if attempt + 1 == attempts or not transient_connection_error(error):
                raise
            time.sleep(0.5 * 2**attempt)
    raise AssertionError("Unreachable: the last attempt must return or raise")
