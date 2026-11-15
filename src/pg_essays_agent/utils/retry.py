
import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def is_transient(exc: Exception) -> bool:
    """Determine whether an exception should be retried."""

    if isinstance(exc, (KeyError, TypeError, ValueError, AssertionError)):
        return False

    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
    )

    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500

    # Preserve the original implementation's fallback behavior.
    return True


def with_retries(
    fn: Callable[[], T],
    what: str,
    retries: int = 5,
    base: float = 1.0,
) -> T:
    """
    Execute a function with exponential backoff and jitter.

    Retries only transient failures. Permanent failures are
    immediately propagated to the caller.
    """

    if retries < 1:
        raise ValueError("retries must be at least 1")

    if base < 0:
        raise ValueError("base cannot be negative")

    for attempt in range(1, retries + 1):
        try:
            return fn()

        except Exception as exc:
            if attempt == retries or not is_transient(exc):
                raise

            wait = (
                base
                * 2 ** (attempt - 1)
                * (0.5 + random.random())
            )

            logger.warning(
                "%s failed (%s); retry %d/%d in %.1fs",
                what,
                exc,
                attempt,
                retries - 1,
                wait,
            )

            time.sleep(wait)

    raise RuntimeError("Unexpected retry loop termination")
