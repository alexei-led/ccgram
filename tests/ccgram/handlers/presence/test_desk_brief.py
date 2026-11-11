"""Desk brief: one notifying direct message for every session (CCGRAM_DESK_BRIEF=dm)."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import RetryAfter

from ccgram.config import config
from ccgram.entity_formatting import convert_to_entities
from ccgram.handlers.presence import desk_brief, digest

USER = 42
CHAT = -1004477516122
PRIVATE = 42
A, B, C = 9, 10, 11


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
def badges(monkeypatch: pytest.MonkeyPatch) -> dict[int, tuple[str, str]]:
    holder: dict[int, tuple[str, str]] = {}
    monkeypatch.setattr(
        digest,
        "topic_state_badge",
        lambda _chat, thread: holder.get(thread, ("active", "🟡")),
    )
    return holder


@pytest.fixture
def topic_send(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    counter = {"id": 500}

    async def _send(*_args, **_kwargs):
        counter["id"] += 1
        return MagicMock(message_id=counter["id"])

    mock = AsyncMock(side_effect=_send)
    monkeypatch.setattr(digest, "rate_limit_send_message", mock)
    return mock


@pytest.fixture
def dm_send(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=MagicMock(message_id=900))
    monkeypatch.setattr(desk_brief, "rate_limit_send_message", mock)
    return mock


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int | None, int, str]]:
    topics: list[tuple[int, int | None, int, str]] = [
        (USER, CHAT, A, "hc-a"),
        (USER, CHAT, B, "hc-b"),
        (USER, CHAT, C, "hc-c"),
        (7, CHAT, 99, "hc-other-user"),
    ]
    monkeypatch.setattr(digest, "iter_bound_topics", lambda: list(topics))
    return topics


@pytest.fixture(autouse=True)
def _setup(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    badges: dict,
    topic_send: AsyncMock,
    dm_send: AsyncMock,
    bound: list,
) -> Iterator[None]:
    monkeypatch.setattr(config, "desk_digest_minutes", 30.0)
    monkeypatch.setattr(config, "desk_brief", "dm")
    monkeypatch.setattr(digest, "_wall_time", lambda: "14:32")
    monkeypatch.setattr(
        digest,
        "view_window",
        lambda window_id: SimpleNamespace(
            window_name=f"Claude ▸ Sessions ▸ {window_id.removeprefix('hc-')}"
        ),
    )
    monkeypatch.setattr(digest, "edit_with_fallback", AsyncMock(return_value=True))
    digest._reset_digest_for_testing()
    yield
    digest._reset_digest_for_testing()


def _reply(thread: int, text: str = "All tests pass.") -> None:
    digest.record_content(
        USER,
        CHAT,
        thread,
        f"hc-{'abc'[thread - A]}",
        role="assistant",
        content_type="text",
        text=text,
    )


def _last(mock: AsyncMock):
    call = mock.await_args
    assert call is not None
    return call


def _dm_text(mock: AsyncMock) -> str:
    call = _last(mock)
    assert call.args[1] == USER
    return call.args[2]


# --- rendering ---------------------------------------------------------------


def test_topic_link_only_for_supergroups() -> None:
    assert desk_brief.topic_link(CHAT, 123) == "https://t.me/c/4477516122/123"
    assert desk_brief.topic_link(PRIVATE, 123) is None


def test_excerpt_flattens_and_cuts() -> None:
    assert desk_brief.excerpt("line one\n\n  line two") == "line one line two"
    assert (
        desk_brief.excerpt("## Plan\n**Done**: ran `make test`, see __init__")
        == "Plan Done: ran make test, see init"
    )
    long = desk_brief.excerpt("x" * 400)
    assert long == "x" * desk_brief.EXCERPT_CHARS + "…"


def test_sections_sort_by_urgency() -> None:
    def section(title: str, state: str, errors: list[str] | None = None):
        return desk_brief.BriefSection(
            title=title, state=state, emoji="", label=state, errors=errors or []
        )

    ordered = desk_brief.sort_sections(
        [
            section("running", "active"),
            section("finished", "done"),
            section("broken", "active", ["API error"]),
            section("asking", "waiting"),
        ]
    )
    assert [s.title for s in ordered] == ["asking", "broken", "finished", "running"]


def test_render_escapes_titles_and_counts_quiet_sessions() -> None:
    text = desk_brief.render_brief(
        [
            desk_brief.BriefSection(
                title="my_project ▸ *wip*",
                state="done",
                emoji="✅",
                label="done",
                messages=2,
                tool_calls=1,
                latest_reply="Finished the refactor.",
                link="https://t.me/c/1/2",
            )
        ],
        quiet_count=3,
        wall_time="14:32",
    )
    plain, _entities = convert_to_entities(text)
    assert plain.startswith("🗂 Desk brief · 1 session with news")
    assert "✅ my_project ▸ *wip* · done" in plain
    assert "2 messages, 1 tool call" in plain
    assert "Finished the refactor." in plain
    assert "Open topic" in plain
    assert "3 other sessions quiet" in plain
    assert plain.rstrip().endswith("updated 14:32 · desk mode")


# --- periodic cadence --------------------------------------------------------


async def test_one_notifying_brief_for_all_topics(
    clock: _Clock, dm_send: AsyncMock, topic_send: AsyncMock, badges: dict
) -> None:
    _reply(A, "Deployed.")
    clock.now += 600
    _reply(B, "Need a decision.")
    badges[B] = ("waiting", "🔴")

    clock.now += 29 * 60 - 600
    await digest.flush_due(MagicMock())
    dm_send.assert_not_awaited()

    clock.now += 61
    await digest.flush_due(MagicMock())
    dm_send.assert_awaited_once()
    assert "disable_notification" not in _last(dm_send).kwargs
    text = _dm_text(dm_send)
    assert "2 sessions with news" in text
    assert text.index("Sessions ▸ b") < text.index("Sessions ▸ a")
    assert "1 other session quiet" in text
    assert "https://t.me/c/4477516122/9" in text

    # Topic digests are refreshed at the same moment, silently.
    assert topic_send.await_count == 2
    for call in topic_send.await_args_list:
        assert call.kwargs["disable_notification"] is True


async def test_no_news_means_no_brief(clock: _Clock, dm_send: AsyncMock) -> None:
    _reply(A)
    clock.now += 30 * 60
    await digest.flush_due(MagicMock())
    dm_send.reset_mock()

    clock.now += 2 * 60 * 60
    await digest.flush_due(MagicMock())
    dm_send.assert_not_awaited()


async def test_next_period_starts_at_the_next_news(
    clock: _Clock, dm_send: AsyncMock
) -> None:
    _reply(A)
    clock.now += 30 * 60
    await digest.flush_due(MagicMock())
    dm_send.reset_mock()

    clock.now += 2 * 60 * 60
    _reply(A, "More work.")
    await digest.flush_due(MagicMock())
    dm_send.assert_not_awaited()
    clock.now += 30 * 60
    await digest.flush_due(MagicMock())
    dm_send.assert_awaited_once()


async def test_rate_limited_brief_retries_without_touching_topics(
    clock: _Clock, dm_send: AsyncMock, topic_send: AsyncMock
) -> None:
    _reply(A)
    clock.now += 30 * 60
    dm_send.side_effect = RetryAfter(5)
    await digest.flush_due(MagicMock())
    topic_send.assert_not_awaited()

    dm_send.side_effect = None
    await digest.flush_due(MagicMock())
    assert dm_send.await_count == 2
    topic_send.assert_awaited_once()


# --- going away --------------------------------------------------------------


async def test_idle_away_sends_one_brief_and_topics_stay_silent(
    clock: _Clock, dm_send: AsyncMock, topic_send: AsyncMock, badges: dict
) -> None:
    _reply(A, "Done with it.")
    _reply(B, "Which option?")
    badges[A] = ("done", "✅")
    badges[B] = ("waiting", "🔴")
    clock.now += 5 * 60

    await digest.close_out(MagicMock(), USER, reason="idle")

    dm_send.assert_awaited_once()
    assert _dm_text(dm_send).startswith("🚶 You stepped away · 2 sessions")
    assert topic_send.await_count == 2
    for call in topic_send.await_args_list:
        assert call.kwargs["disable_notification"] is True
    assert not digest.has_digest(USER, CHAT, A)


async def test_failed_brief_falls_back_to_notifying_topics(
    dm_send: AsyncMock, topic_send: AsyncMock, badges: dict
) -> None:
    _reply(A)
    badges[A] = ("done", "✅")
    dm_send.return_value = None  # e.g. the user never started the bot privately

    await digest.close_out(MagicMock(), USER, reason="idle")

    topic_send.assert_awaited_once()
    assert _last(topic_send).kwargs["disable_notification"] is False


async def test_telegram_away_sends_no_brief(
    dm_send: AsyncMock, topic_send: AsyncMock
) -> None:
    _reply(A)
    await digest.close_out(MagicMock(), USER, reason="telegram")
    dm_send.assert_not_awaited()
    topic_send.assert_awaited_once()
    assert _last(topic_send).kwargs["disable_notification"] is True


async def test_away_clears_the_brief_period(clock: _Clock, dm_send: AsyncMock) -> None:
    _reply(A)
    await digest.close_out(MagicMock(), USER, reason="telegram")
    clock.now += 60 * 60
    await digest.flush_due(MagicMock())
    dm_send.assert_not_awaited()


async def test_off_mode_never_sends_a_brief(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, dm_send: AsyncMock
) -> None:
    monkeypatch.setattr(config, "desk_brief", "off")
    _reply(A)
    clock.now += 30 * 60
    await digest.flush_due(MagicMock())
    await digest.close_out(MagicMock(), USER, reason="idle")
    dm_send.assert_not_awaited()
