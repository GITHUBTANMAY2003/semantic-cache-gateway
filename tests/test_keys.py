"""Tests for the matching rule: two requests may share an answer only if everything
except the wording of the last user message is identical (design §5.2)."""

import copy

import pytest

from gateway.keys import context_key, exact_key, last_user_text, normalize


def req(text, system=None, model="m", temperature=0, history=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages += history or []
    messages.append({"role": "user", "content": text})
    return {"model": model, "temperature": temperature, "messages": messages}


def test_context_key_ignores_last_user_wording():
    assert context_key(req("reset password?")) == context_key(req("password reset steps"))


def test_context_key_differs_on_system_prompt():
    assert context_key(req("x", system="A")) != context_key(req("x", system="B"))


def test_context_key_differs_on_model_and_temperature():
    assert context_key(req("x", model="a")) != context_key(req("x", model="b"))
    assert context_key(req("x", temperature=0)) != context_key(req("x", temperature=0.2))


def test_context_key_differs_on_earlier_turns():
    history_a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    history_b = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey there."}]
    assert context_key(req("x", history=history_a)) != context_key(req("x", history=history_b))


def test_context_key_differs_on_max_tokens():
    assert context_key({**req("x"), "max_tokens": 50}) != context_key({**req("x"), "max_tokens": 100})


def test_context_key_ignores_stream():
    assert context_key({**req("x"), "stream": True}) == context_key(req("x"))


def test_context_key_does_not_mutate_input():
    body = {**req("x", system="s"), "stream": True}
    before = copy.deepcopy(body)
    context_key(body)
    assert body == before


def test_context_key_raises_without_last_user_text():
    with pytest.raises(ValueError):
        context_key({"model": "m", "messages": [{"role": "system", "content": "s"}]})


def test_context_key_is_sha256_hex():
    key = context_key(req("x"))
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_exact_key_normalizes_text():
    expected = exact_key("c", "reset password")
    assert exact_key("c", "Reset  Password?") == expected
    assert exact_key("c", "  RESET password ?! ") == expected


def test_exact_key_differs_when_context_differs():
    assert exact_key("c1", "x") != exact_key("c2", "x")


def test_normalize():
    assert normalize("Hello   World?") == "hello world"
    assert normalize("a.b?") == "a.b"  # only trailing punctuation goes
    assert normalize("  ?  ") == ""


def test_last_user_text_none_cases():
    user = {"role": "user", "content": "hi"}
    assert last_user_text({"messages": [{"role": "system", "content": "s"}]}) is None
    assert last_user_text({"messages": [{"role": "user", "content": "   "}]}) is None
    assert last_user_text({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}) is None
    assert last_user_text({"messages": []}) is None
    assert last_user_text({"model": "m"}) is None
    assert last_user_text({"messages": [user, {"role": "assistant", "content": "hello"}]}) is None


def test_last_user_text_returns_content_unchanged():
    assert last_user_text(req("  Reset Password?  ")) == "  Reset Password?  "
