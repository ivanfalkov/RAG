"""Groq rate-limit classification shared by API callers."""

from __future__ import annotations

import re

from beartype import beartype


class DailyRateLimitExceeded(Exception):
    """Groq daily quota (RPD/TPD) is exhausted; retries will not help today."""


@beartype
def is_daily_rate_limit(exc: BaseException) -> bool:
    """Return True when the error is a daily quota, not a per-minute throttle."""
    msg = str(exc).lower()
    if "rate limit" not in msg and "429" not in msg:
        return False
    if any(token in msg for token in ("per day", "per-day", "rpd", "tpd")):
        return True
    if any(token in msg for token in ("per minute", "per-minute", "rpm", "tpm")):
        return False
    return re.search(r"\d+\s*h", msg) is not None
