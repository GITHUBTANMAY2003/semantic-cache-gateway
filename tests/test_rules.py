"""Tests for the skip rules: which requests must never be answered from the cache (design §5.3)."""

import pytest

from gateway.rules import bypass_reason


def req(text, temperature=0, **extra):
    return {"model": "m", "temperature": temperature, "messages": [{"role": "user", "content": text}], **extra}


@pytest.mark.parametrize(
    "text",
    [
        "How do I cancel my order?",
        "my account settings help",
        "reset my password",
        "How do I change my current password?",
        "What is your refund policy?",
        "I want to track my order number",
        "Is 2026 a leap year?",
    ],
)
def test_generic_wording_is_cacheable(text):
    assert bypass_reason(req(text), {}, 0.3) is None


@pytest.mark.parametrize(
    "text",
    [
        "cancel order #48213",
        "refund for ORD-99812",
        "email me at a@b.com",
        "call +91 98765 43210",
        "where is {{Order Number}}",
        "my ticket 482 is stuck",
        "tracking number 123456789",
    ],
)
def test_identifiers_bypass(text):
    assert bypass_reason(req(text), {}, 0.3) == "identifier"


@pytest.mark.parametrize(
    "text",
    [
        "what's the latest version",
        "news today",
        "price on 2026-12-01",
        "what happened yesterday",
        "is the store open now",
        "deals this week",
    ],
)
def test_time_sensitive_bypass(text):
    assert bypass_reason(req(text), {}, 0.3) == "time-sensitive"


def test_temperature_missing_bypasses():
    body = req("reset my password")
    del body["temperature"]  # OpenAI's default is 1.0
    assert bypass_reason(body, {}, 0.3) == "temperature"


def test_temperature_above_max_bypasses():
    assert bypass_reason(req("reset my password", temperature=0.5), {}, 0.3) == "temperature"


def test_temperature_equal_to_max_is_cacheable():
    assert bypass_reason(req("reset my password", temperature=0.3), {}, 0.3) is None


def test_temperature_not_a_number_bypasses():
    assert bypass_reason(req("reset my password", temperature="0.2"), {}, 0.3) == "temperature"


def test_n_not_a_number_bypasses():
    assert bypass_reason(req("reset my password", n="2"), {}, 0.3) == "n"


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({"stream": True}, "stream"),
        ({"n": 2}, "n"),
        ({"n": 1}, None),
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
        ({"functions": [{"name": "f"}]}, "tools"),
        ({"tools": []}, None),
    ],
)
def test_body_fields(extra, expected):
    assert bypass_reason(req("reset my password", **extra), {}, 0.3) == expected


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"cache-control": "no-cache"}, "no-cache-header"),
        ({"cache-control": "max-age=0, no-cache"}, "no-cache-header"),
        ({"cache-control": "max-age=60"}, None),
    ],
)
def test_cache_control_header(headers, expected):
    assert bypass_reason(req("reset my password"), headers, 0.3) == expected


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "You are helpful."}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}],
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    ],
)
def test_no_user_text_bypass(messages):
    body = {"model": "m", "temperature": 0, "messages": messages}
    assert bypass_reason(body, {}, 0.3) == "no-user-text"


def test_first_rule_wins():
    body = req("reset my password", stream=True)
    del body["temperature"]
    assert bypass_reason(body, {}, 0.3) == "stream"
