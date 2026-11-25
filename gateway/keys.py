"""Cache keys: the one matching rule (design §5.2).
Two requests may share an answer only if everything except the wording of the last
user message is identical, so the context key hashes the request without that wording."""

import copy
import hashlib
import json


def last_user_text(body: dict) -> str | None:
    """Return the text of the final message if it is a non-empty user string, else None.
    None means there is nothing safe to cache on, so the caller should bypass the cache."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    content = last.get("content")
    # A list of parts (multimodal) is not supported, so it counts as unusable.
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace and drop trailing ?.! so trivial variants match exactly."""
    return " ".join(text.lower().split()).rstrip("?.! ")


def context_key(body: dict) -> str:
    """sha256 of the request with the last user wording and "stream" removed.
    Equal keys mean same model, settings, system prompt and earlier turns."""
    if last_user_text(body) is None:
        raise ValueError("request has no usable last user message")
    body = copy.deepcopy(body)  # the caller still forwards the original body
    body.pop("stream", None)
    del body["messages"][-1]["content"]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def exact_key(ctx: str, text: str) -> str:
    """sha256 of the context key plus the normalized last user text: the exact-match cache key."""
    return hashlib.sha256(f"{ctx}\n{normalize(text)}".encode()).hexdigest()
