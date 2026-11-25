"""Skip rules: decide, before any cache lookup, whether a request must not use the cache (design §5.3).
A bypassed request is still forwarded to the upstream; it is just never looked up or stored."""

from collections.abc import Mapping

from gateway.bypass_patterns import IDENTIFIERS, TIME_SENSITIVE
from gateway.keys import last_user_text


def bypass_reason(body: dict, headers: Mapping[str, str], max_temperature: float) -> str | None:
    """Return why this request must skip the cache, or None if it is cacheable.
    The first matching rule wins. Cheap field checks come first and the regexes last,
    so most bypassed requests never pay for the text scan."""
    if body.get("stream"):
        return "stream"
    if "no-cache" in headers.get("cache-control", "").lower():
        return "no-cache-header"
    # Missing means OpenAI's default of 1.0. A non-number bypasses instead of crashing.
    temperature = body.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or temperature > max_temperature:
        return "temperature"
    n = body.get("n", 1)
    if not isinstance(n, int) or n > 1:
        return "n"
    if body.get("tools") or body.get("functions"):
        return "tools"
    text = last_user_text(body)
    if text is None:
        return "no-user-text"
    if any(pattern.search(text) for pattern in TIME_SENSITIVE):
        return "time-sensitive"
    if any(pattern.search(text) for pattern in IDENTIFIERS):
        return "identifier"
    return None
