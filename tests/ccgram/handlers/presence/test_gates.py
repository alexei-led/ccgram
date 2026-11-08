"""Desk-mode gates outside message routing, and the Telegram/periodic wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.config import config
from ccgram.handlers.interactive import interactive_ui
from ccgram.handlers.messaging_pipeline import message_queue
from ccgram.handlers.presence import desk_mode, presence

USER = 100


def _last(mock: AsyncMock):
    call = mock.await_args
    assert call is not None
    return call


@pytest.fixture
def desk(monkeypatch: pytest.MonkeyPatch):
    async def _recent_input() -> float:
        return 1.0

    monkeypatch.setattr(config, "desk_mode", True)
    monkeypatch.setattr(config, "allowed_users", {USER})
    presence._reset_presence_for_testing(_recent_input)
    yield
    presence._reset_presence_for_testing()


async def test_status_updates_are_dropped_at_the_desk(desk) -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    with patch.object(message_queue, "get_or_create_queue") as queue:
        await message_queue.enqueue_status_update(
            MagicMock(), USER, "@5", "Working…", thread_id=42
        )
        queue.assert_not_called()


async def test_status_clears_still_pass_at_the_desk(desk) -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    with (
        patch.object(message_queue, "get_or_create_queue") as queue,
        patch.object(message_queue, "is_window_live", return_value=True),
    ):
        queue.return_value = MagicMock()
        await message_queue.enqueue_status_update(
            MagicMock(), USER, "@5", None, thread_id=42
        )
        queue.assert_called_once()
    message_queue._pending_status_clears.clear()


async def test_interactive_ui_is_not_sent_at_the_desk(desk) -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    with patch.object(
        interactive_ui, "_capture_interactive_content", new=AsyncMock()
    ) as capture:
        assert not await interactive_ui.handle_interactive_ui(
            MagicMock(), USER, "@5", 42
        )
        capture.assert_not_awaited()


def _update(user_id: int | None) -> MagicMock:
    update = MagicMock()
    update.effective_user = None if user_id is None else MagicMock(id=user_id)
    return update


async def test_telegram_update_switches_to_away_and_closes_digests(desk) -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    with patch.object(desk_mode.digest, "close_out", new=AsyncMock()) as close_out:
        await desk_mode.telegram_activity_handler(_update(USER), MagicMock())
        for task in list(desk_mode._background):
            await task
    assert not presence.is_quiet(USER)
    close_out.assert_awaited_once()
    assert _last(close_out).kwargs == {"reason": "telegram"}


async def test_updates_from_others_are_ignored(desk) -> None:
    await presence.note_desk_prompt(USER, "fix the tests")
    await desk_mode.telegram_activity_handler(_update(999), MagicMock())
    await desk_mode.telegram_activity_handler(_update(None), MagicMock())
    assert presence.is_quiet(USER)


async def test_tick_closes_idle_users_then_flushes(desk) -> None:
    with (
        patch.object(
            desk_mode.presence, "refresh_idle", new=AsyncMock(return_value=[USER])
        ),
        patch.object(desk_mode.digest, "close_out", new=AsyncMock()) as close_out,
        patch.object(desk_mode.digest, "flush_due", new=AsyncMock()) as flush,
    ):
        await desk_mode.run_desk_mode_tick(MagicMock())
    close_out.assert_awaited_once()
    assert _last(close_out).kwargs == {"reason": "idle"}
    flush.assert_awaited_once()


async def test_tick_is_a_no_op_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "desk_mode", False)
    with patch.object(desk_mode.digest, "flush_due", new=AsyncMock()) as flush:
        await desk_mode.run_desk_mode_tick(MagicMock())
    flush.assert_not_awaited()


async def test_topic_renames_are_held_at_the_desk(desk) -> None:
    from ccgram.handlers.status import topic_emoji

    topic_emoji.reset_all_state()
    client = MagicMock()
    client.edit_forum_topic = AsyncMock()
    await presence.note_desk_prompt(USER, "fix the tests")
    with patch.object(topic_emoji, "_resolve_topic_name", return_value=("s", True)):
        await topic_emoji.update_topic_emoji(
            client, -1, 42, "active", "s", user_id=USER
        )
        client.edit_forum_topic.assert_not_awaited()
        # The digest header still sees the state the topic should show.
        assert topic_emoji.topic_state_badge(-1, 42) == (
            "active",
            topic_emoji._state_emoji_map()["active"],
        )

        # Back to away: the held state has outlasted its debounce and is
        # applied on the next poll without waiting again.
        presence.note_telegram_activity(USER)
        first_seen = topic_emoji._pending_transitions[(-1, 42)][1]
        topic_emoji._pending_transitions[(-1, 42)] = ("active", first_seen - 60)
        with (
            patch.object(topic_emoji, "_resolve_approval_mode", return_value="normal"),
            patch.object(topic_emoji, "_resolve_rc_mode", return_value=False),
        ):
            await topic_emoji.update_topic_emoji(
                client, -1, 42, "active", "s", user_id=USER
            )
    client.edit_forum_topic.assert_awaited_once()
    topic_emoji.reset_all_state()
