"""Desk-mode digest: counting, rendering, send/edit cadence, close-out."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.config import config
from ccgram.expandable_quote import EXPANDABLE_QUOTE_START
from ccgram.handlers.presence import digest

USER, CHAT, THREAD, WINDOW = 42, -100777, 9, "hc-t5"


def _last(mock: AsyncMock):
    call = mock.await_args
    assert call is not None
    return call


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(digest, "_clock", fake)
    return fake


@pytest.fixture
def badge(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[str, str] | None]:
    holder: dict[str, tuple[str, str] | None] = {"value": ("active", "🟡")}
    monkeypatch.setattr(digest, "topic_state_badge", lambda _c, _t: holder["value"])
    return holder


@pytest.fixture
def send(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    counter = {"id": 500}

    async def _send(*_args, **_kwargs):
        counter["id"] += 1
        return MagicMock(message_id=counter["id"])

    mock = AsyncMock(side_effect=_send)
    monkeypatch.setattr(digest, "rate_limit_send_message", mock)
    return mock


@pytest.fixture
def edit(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr(digest, "edit_with_fallback", mock)
    return mock


@pytest.fixture(autouse=True)
def _setup(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, badge: dict, send, edit
) -> Iterator[None]:
    monkeypatch.setattr(config, "desk_digest_minutes", 10.0)
    monkeypatch.setattr(digest, "_wall_time", lambda: "14:32")
    digest._reset_digest_for_testing()
    yield
    digest._reset_digest_for_testing()


def _reply(text: str = "All tests pass.") -> None:
    digest.record_content(
        USER, CHAT, THREAD, WINDOW, role="assistant", content_type="text", text=text
    )


def _tool() -> None:
    digest.record_content(
        USER,
        CHAT,
        THREAD,
        WINDOW,
        role="assistant",
        content_type="tool_use",
        text="Bash",
    )


def test_only_replies_and_tool_calls_count() -> None:
    digest.record_content(
        USER, CHAT, THREAD, WINDOW, role="user", content_type="text", text="hi"
    )
    digest.record_content(
        USER, CHAT, THREAD, WINDOW, role="assistant", content_type="thinking", text="x"
    )
    assert not digest.has_digest(USER, CHAT, THREAD)
    _reply()
    _tool()
    entry = digest._digests[(USER, CHAT, THREAD)]
    assert (entry.messages, entry.tool_calls, entry.dirty) == (1, 1, True)


def test_render_format() -> None:
    _reply("Done: 3 files changed.")
    _tool()
    _tool()
    text = digest.render(digest._digests[(USER, CHAT, THREAD)], CHAT, THREAD)
    lines = text.split("\n")
    assert lines[0] == "🟡 working · 1 message, 2 tool calls since last update"
    assert lines[1] == "Latest reply:"
    assert lines[2].startswith(EXPANDABLE_QUOTE_START)
    assert "Done: 3 files changed." in lines[2]
    assert lines[-1] == "updated 14:32 · desk mode"


def test_render_waiting_state_and_long_reply(badge: dict) -> None:
    badge["value"] = ("waiting", "🔴")
    _reply("x" * 5000)
    digest.record_error(USER, CHAT, THREAD, WINDOW, "API error — overloaded")
    text = digest.render(digest._digests[(USER, CHAT, THREAD)], CHAT, THREAD)
    assert text.startswith("🔴 waiting for your choice · 1 message, 0 tool calls")
    assert "/last for the full reply" in text
    assert "⚠ API error — overloaded" in text
    assert len(text) < 4096


async def test_first_send_waits_one_interval_and_is_silent(
    clock: _Clock, send: AsyncMock
) -> None:
    _reply()
    clock.now += 599
    await digest.flush_due(MagicMock())
    send.assert_not_awaited()
    clock.now += 1
    await digest.flush_due(MagicMock())
    send.assert_awaited_once()
    kwargs = _last(send).kwargs
    assert kwargs["disable_notification"] is True
    assert kwargs["message_thread_id"] == THREAD
    entry = digest._digests[(USER, CHAT, THREAD)]
    assert (entry.messages, entry.tool_calls, entry.dirty) == (0, 0, False)


async def test_later_refreshes_edit_the_same_message(
    clock: _Clock, send: AsyncMock, edit: AsyncMock
) -> None:
    _reply()
    clock.now += 600
    await digest.flush_due(MagicMock())
    message_id = digest._digests[(USER, CHAT, THREAD)].message_id
    _tool()
    clock.now += 300
    await digest.flush_due(MagicMock())
    edit.assert_not_awaited()  # not due yet
    clock.now += 300
    await digest.flush_due(MagicMock())
    edit.assert_awaited_once()
    assert _last(edit).args[2] == message_id
    assert send.await_count == 1


async def test_nothing_new_means_no_edit(clock: _Clock, edit: AsyncMock) -> None:
    _reply()
    clock.now += 600
    await digest.flush_due(MagicMock())
    clock.now += 600
    await digest.flush_due(MagicMock())
    edit.assert_not_awaited()


async def test_state_change_alone_refreshes(
    clock: _Clock, badge: dict, edit: AsyncMock
) -> None:
    _reply()
    clock.now += 600
    await digest.flush_due(MagicMock())
    badge["value"] = ("idle", "🟢")
    clock.now += 600
    await digest.flush_due(MagicMock())
    edit.assert_awaited_once()
    assert "🟢 idle" in _last(edit).args[3]


async def test_failed_edit_sends_a_new_message(
    clock: _Clock, send: AsyncMock, edit: AsyncMock
) -> None:
    _reply()
    clock.now += 600
    await digest.flush_due(MagicMock())
    edit.return_value = False  # e.g. the user deleted the digest
    _tool()
    clock.now += 600
    await digest.flush_due(MagicMock())
    assert send.await_count == 2
    assert digest._digests[(USER, CHAT, THREAD)].message_id == 502


async def test_close_out_from_telegram_updates_once_and_forgets(
    send: AsyncMock,
) -> None:
    _reply()
    await digest.close_out(MagicMock(), USER, reason="telegram")
    send.assert_awaited_once()
    assert _last(send).kwargs["disable_notification"] is True
    assert not digest.has_digest(USER, CHAT, THREAD)


async def test_close_out_without_news_sends_nothing(
    clock: _Clock, send: AsyncMock, edit: AsyncMock
) -> None:
    _reply()
    clock.now += 600
    await digest.flush_due(MagicMock())
    await digest.close_out(MagicMock(), USER, reason="telegram")
    edit.assert_not_awaited()
    assert send.await_count == 1


async def test_idle_close_out_notifies_for_a_settled_topic(
    clock: _Clock, badge: dict, send: AsyncMock, edit: AsyncMock
) -> None:
    _reply()
    clock.now += 600
    await digest.flush_due(MagicMock())
    badge["value"] = ("idle", "🟢")
    await digest.close_out(MagicMock(), USER, reason="idle")
    edit.assert_not_awaited()
    assert send.await_count == 2
    assert _last(send).kwargs["disable_notification"] is False


async def test_idle_close_out_while_working_stays_silent(
    send: AsyncMock,
) -> None:
    _reply()
    await digest.close_out(MagicMock(), USER, reason="idle")
    assert _last(send).kwargs["disable_notification"] is True


def test_topic_cleanup_drops_the_digest() -> None:
    _reply()
    digest.clear_topic_digest(CHAT, THREAD)
    assert not digest.has_digest(USER, CHAT, THREAD)
