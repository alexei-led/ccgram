"""Desk/away presence transitions."""

from collections.abc import Iterator

import pytest

from ccgram.config import config
from ccgram.handlers.presence import presence


class _Idle:
    def __init__(self, value: float | None = 5.0) -> None:
        self.value = value
        self.calls = 0

    async def __call__(self) -> float | None:
        self.calls += 1
        return self.value


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


USER = 42


@pytest.fixture
def idle() -> _Idle:
    return _Idle()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(presence, "_clock", fake)
    return fake


@pytest.fixture(autouse=True)
def _desk_mode(
    monkeypatch: pytest.MonkeyPatch, idle: _Idle, clock: _Clock
) -> Iterator[None]:
    monkeypatch.setattr(config, "desk_mode", True)
    monkeypatch.setattr(config, "away_idle_minutes", 10.0)
    presence._reset_presence_for_testing(idle)
    yield
    presence._reset_presence_for_testing()


def test_default_is_away_and_not_quiet() -> None:
    assert presence.mode(USER) == presence.AWAY
    assert not presence.is_quiet(USER)


async def test_terminal_prompt_switches_to_desk() -> None:
    assert await presence.note_desk_prompt(USER, "fix the tests")
    assert presence.is_quiet(USER)
    # Already at the desk: no second transition.
    assert not await presence.note_desk_prompt(USER, "and the docs")


async def test_disabled_feature_never_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "desk_mode", False)
    assert not await presence.note_desk_prompt(USER, "fix the tests")
    assert not presence.is_quiet(USER)


async def test_text_sent_from_telegram_is_not_desk_input(clock: _Clock) -> None:
    presence.remember_telegram_text(USER, "  run the build\r\n")
    clock.now += 300  # a busy agent wrote it to the transcript late
    assert not await presence.note_desk_prompt(USER, "run the build")
    assert not presence.is_quiet(USER)


async def test_telegram_text_memory_expires(clock: _Clock) -> None:
    presence.remember_telegram_text(USER, "run the build")
    clock.now += 601
    assert await presence.note_desk_prompt(USER, "run the build")


async def test_prompt_while_computer_idle_is_ignored(idle: _Idle) -> None:
    idle.value = 600.0
    assert not await presence.note_desk_prompt(USER, "fix the tests")
    assert not presence.is_quiet(USER)


async def test_unknown_idle_time_still_allows_desk(idle: _Idle) -> None:
    idle.value = None
    assert await presence.note_desk_prompt(USER, "fix the tests")


async def test_interrupt_notice_is_not_a_prompt() -> None:
    assert not await presence.note_desk_prompt(USER, "[Request interrupted by user]")


async def test_telegram_activity_switches_back_to_away() -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    assert presence.note_telegram_activity(USER)
    assert not presence.is_quiet(USER)
    assert not presence.note_telegram_activity(USER)


async def test_idle_computer_switches_desk_users_to_away(idle: _Idle) -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    idle.value = 599.0
    presence._reset_idle_cache_for_testing()
    assert await presence.refresh_idle() == []
    idle.value = 600.0
    presence._reset_idle_cache_for_testing()
    assert await presence.refresh_idle() == [USER]
    assert not presence.is_quiet(USER)


async def test_idle_rule_can_be_disabled(
    idle: _Idle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "away_idle_minutes", 0.0)
    await presence.note_desk_prompt(USER, "fix the tests")
    idle.value = 10_000.0
    presence._reset_idle_cache_for_testing()
    assert await presence.refresh_idle() == []
    assert presence.is_quiet(USER)


async def test_idle_read_is_cached(idle: _Idle, clock: _Clock) -> None:
    assert await presence.idle_seconds() == 5.0
    clock.now += 10
    assert await presence.idle_seconds() == 15.0  # extrapolated, no new read
    assert idle.calls == 1
    clock.now += 6
    await presence.idle_seconds()
    assert idle.calls == 2
