from collections.abc import Iterator

import pytest

from ccgram.handlers import callback_tokens


@pytest.fixture(autouse=True)
def clear_tokens() -> Iterator[None]:
    callback_tokens._tokens.clear()
    yield
    callback_tokens._tokens.clear()


def test_token_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(callback_tokens, "_MAX_CALLBACK_TOKENS", 2)

    callbacks = [
        callback_tokens.compact_callback_data(
            "st:ss:", "x" * 65 + str(index), f"@{index}"
        )
        for index in range(3)
    ]

    assert len(callback_tokens._tokens) == 2
    assert (
        callback_tokens.resolve_callback_data(callbacks[0], 1, lambda *_: True) is None
    )
    assert (
        callback_tokens.resolve_callback_data(callbacks[-1], 1, lambda *_: True)
        == "x" * 65 + "2"
    )


def test_expired_tokens_are_pruned_before_new_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(callback_tokens.time, "monotonic", lambda: now)
    expired = callback_tokens.compact_callback_data("st:ss:", "x" * 65, "@expired")
    monkeypatch.setattr(
        callback_tokens.time,
        "monotonic",
        lambda: now + callback_tokens._TOKEN_TTL_SECONDS + 1,
    )

    callback_tokens.compact_callback_data("st:ss:", "y" * 65, "@fresh")

    assert callback_tokens.resolve_callback_data(expired, 1, lambda *_: True) is None
    assert len(callback_tokens._tokens) == 1


def test_revoke_window_tokens_removes_only_that_window() -> None:
    revoked = callback_tokens.compact_callback_data("st:ss:", "x" * 65, "@revoked")
    retained = callback_tokens.compact_callback_data("st:ss:", "y" * 65, "@retained")

    callback_tokens.revoke_window_tokens("@revoked")

    assert callback_tokens.resolve_callback_data(revoked, 1, lambda *_: True) is None
    assert (
        callback_tokens.resolve_callback_data(retained, 1, lambda *_: True) == "y" * 65
    )
