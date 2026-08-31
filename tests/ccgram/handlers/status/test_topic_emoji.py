import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, RetryAfter, TelegramError

from _helpers import make_mock_provider

from ccgram.handlers.status.topic_emoji import (
    DEBOUNCE_TERMINAL_SECONDS,
    DEBOUNCE_TO_ACTIVE_SECONDS,
    DEBOUNCE_TO_IDLE_SECONDS,
    EMOJI_ACTIVE,
    EMOJI_DEAD,
    EMOJI_DONE,
    EMOJI_GREEN_CIRCLE,
    EMOJI_IDLE,
    EMOJI_RC,
    EMOJI_YELLOW_CIRCLE,
    EMOJI_YOLO,
    clear_topic_emoji_state,
    format_topic_name_for_mode,
    mark_awaiting_first_paint,
    reset_all_state,
    sync_topic_name,
    strip_emoji_prefix,
    update_topic_emoji,
)

_DEBOUNCE_FOR: dict[str, float] = {
    "active": DEBOUNCE_TO_ACTIVE_SECONDS,
    "idle": DEBOUNCE_TO_IDLE_SECONDS,
    "done": DEBOUNCE_TERMINAL_SECONDS,
    "dead": DEBOUNCE_TERMINAL_SECONDS,
}


def _debounce_for(state: str) -> float:
    return _DEBOUNCE_FOR[state]


@pytest.fixture(autouse=True)
def _reset():
    from ccgram.handlers.polling.polling_state import terminal_poll_state

    reset_all_state()
    terminal_poll_state.reset_all_seen_status()
    yield
    reset_all_state()
    terminal_poll_state.reset_all_seen_status()


class TestStripEmojiPrefix:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (f"{EMOJI_ACTIVE} myproject", "myproject"),
            (f"{EMOJI_IDLE} myproject", "myproject"),
            (f"{EMOJI_DONE} myproject", "myproject"),
            (f"{EMOJI_DEAD} myproject", "myproject"),
            (f"{EMOJI_GREEN_CIRCLE} myproject", "myproject"),
            (f"{EMOJI_YELLOW_CIRCLE} myproject", "myproject"),
            (f"{EMOJI_YOLO} myproject", "myproject"),
            (f"{EMOJI_RC} myproject", "myproject"),
            (f"{EMOJI_ACTIVE} {EMOJI_YOLO} myproject", "myproject"),
            (f"{EMOJI_ACTIVE} {EMOJI_RC} myproject", "myproject"),
            (f"{EMOJI_ACTIVE} {EMOJI_RC} {EMOJI_YOLO} myproject", "myproject"),
            ("myproject", "myproject"),
            # Only one state prefix is stripped — a name that legitimately
            # starts with an emoji keeps it.
            (f"{EMOJI_ACTIVE} {EMOJI_IDLE} myproject", f"{EMOJI_IDLE} myproject"),
        ],
    )
    def test_strips_status_prefixes(self, name: str, expected: str) -> None:
        assert strip_emoji_prefix(name) == expected


_PATCH_MONOTONIC = "ccgram.handlers.status.topic_emoji.time.monotonic"


def _assert_emoji_call(
    mock_emoji: MagicMock,
    bot: AsyncMock,
    chat_id: int,
    thread_id: int,
    state: str,
    display: str,
) -> None:
    """Assert update_topic_emoji was called once with PTBTelegramClient(bot)."""
    from ccgram.telegram_client import PTBTelegramClient

    mock_emoji.assert_called_once()
    args = mock_emoji.call_args.args
    assert isinstance(args[0], PTBTelegramClient)
    assert args[0].bot is bot
    assert args[1:] == (chat_id, thread_id, state, display)


async def _debounced_update(
    bot: AsyncMock,
    chat_id: int,
    thread_id: int,
    state: str,
    display_name: str,
) -> None:
    with patch(_PATCH_MONOTONIC) as mock_monotonic:
        mock_monotonic.return_value = 0.0
        await update_topic_emoji(bot, chat_id, thread_id, state, display_name)
        mock_monotonic.return_value = _debounce_for(state) + 0.1
        await update_topic_emoji(bot, chat_id, thread_id, state, display_name)


class TestInheritedTopicsRepaintImmediately:
    async def test_seeded_topic_paints_without_waiting(self) -> None:
        bot = AsyncMock()
        mark_awaiting_first_paint(-100, 42)

        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")

        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} myproject",
        )

    async def test_only_the_first_sighting_skips_the_debounce(self) -> None:
        bot = AsyncMock()
        mark_awaiting_first_paint(-100, 42)
        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")
        bot.edit_forum_topic.reset_mock()

        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 42, "active", "myproject")

        bot.edit_forum_topic.assert_not_called()

    async def test_unseeded_topic_still_debounces(self) -> None:
        bot = AsyncMock()
        mark_awaiting_first_paint(-100, 42)

        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 99, "idle", "other")

        bot.edit_forum_topic.assert_not_called()


_STATE_EMOJI = [
    ("active", EMOJI_ACTIVE),
    ("idle", EMOJI_IDLE),
    ("done", EMOJI_DONE),
    ("dead", EMOJI_DEAD),
]


class TestUpdateTopicEmoji:
    async def test_first_call_starts_debounce(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_not_called()

    @pytest.mark.parametrize("state,emoji", _STATE_EMOJI)
    async def test_sets_emoji_after_debounce(self, state: str, emoji: str) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, state, "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{emoji} myproject",
        )

    async def test_skips_same_state(self) -> None:
        """Same state and same name → no Telegram call (herdr tab not renamed)."""
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_updates_on_state_change(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        await _debounced_update(bot, -100, 42, "idle", "myproject")
        bot.edit_forum_topic.assert_called_once()

    async def test_updates_name_immediately_when_state_is_unchanged(self) -> None:
        """A rename must repaint at once — the debounce only gates state flips.

        This is how a herdr tab rename reaches Telegram: ``sync_display_names``
        updates the display name and the next poll passes it in unchanged-state.
        """
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "idle", "fish")
        bot.edit_forum_topic.reset_mock()

        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 42, "idle", "bun")

        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} bun",
        )

    async def test_strips_existing_prefix(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "idle", f"{EMOJI_ACTIVE} myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} myproject",
        )

    async def test_rapid_toggling_suppressed(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            for i in range(10):
                mock_monotonic.return_value = float(i)
                state = "active" if i % 2 == 0 else "idle"
                await update_topic_emoji(bot, -100, 42, state, "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_stable_state_after_flickering(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            for i in range(4):
                mock_monotonic.return_value = float(i)
                state = "active" if i % 2 == 0 else "idle"
                await update_topic_emoji(bot, -100, 42, state, "myproject")
            bot.edit_forum_topic.assert_not_called()

            mock_monotonic.return_value = 4.0
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            mock_monotonic.return_value = 4.0 + _debounce_for("active") + 0.1
            await update_topic_emoji(bot, -100, 42, "active", "myproject")

        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} myproject",
        )

    async def test_permission_error_disables_chat(self) -> None:
        bot = AsyncMock()
        bot.edit_forum_topic.side_effect = BadRequest("Not enough rights")
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        await _debounced_update(bot, -100, 42, "idle", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_topic_not_modified_still_tracks(self) -> None:
        bot = AsyncMock()
        bot.edit_forum_topic.side_effect = BadRequest("TOPIC_NOT_MODIFIED")
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_other_telegram_error_ignored(self) -> None:
        bot = AsyncMock()
        bot.edit_forum_topic.side_effect = TelegramError("Network error")
        await _debounced_update(bot, -100, 42, "active", "myproject")
        assert bot.edit_forum_topic.called

    async def test_invalid_state_ignored(self) -> None:
        bot = AsyncMock()
        await update_topic_emoji(bot, -100, 42, "unknown", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_debounce_not_reached(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            mock_monotonic.return_value = _debounce_for("active") - 0.1
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_active_fires_faster_than_idle(self) -> None:
        bot = AsyncMock()
        midpoint = DEBOUNCE_TO_ACTIVE_SECONDS + 0.1
        assert midpoint < DEBOUNCE_TO_IDLE_SECONDS

        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            mock_monotonic.return_value = midpoint
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} myproject",
        )

    async def test_idle_does_not_fire_at_active_debounce_time(self) -> None:
        bot = AsyncMock()
        midpoint = DEBOUNCE_TO_ACTIVE_SECONDS + 0.1
        assert midpoint < DEBOUNCE_TO_IDLE_SECONDS

        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")
            mock_monotonic.return_value = midpoint
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_brief_pause_during_work_stays_green(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            mock_monotonic.return_value = DEBOUNCE_TO_ACTIVE_SECONDS + 0.1
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        assert bot.edit_forum_topic.call_count == 1
        bot.edit_forum_topic.reset_mock()

        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 10.0
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")
            mock_monotonic.return_value = 20.0
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_not_called()

    async def test_yolo_mode_adds_rocket_badge(self) -> None:
        bot = AsyncMock()
        with patch(
            "ccgram.handlers.status.topic_emoji._resolve_approval_mode",
            return_value="yolo",
        ):
            await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} {EMOJI_YOLO} myproject",
        )


class TestFormatTopicNameForMode:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("yolo", f"{EMOJI_YOLO} myproject"), ("normal", "myproject")],
    )
    def test_badges_the_name_for_the_mode(self, mode: str, expected: str) -> None:
        assert format_topic_name_for_mode("myproject", mode) == expected


class TestTopicNamePreservation:
    async def test_updates_stored_name_when_display_name_changes(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        await _debounced_update(bot, -100, 42, "idle", "renamed")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} renamed",
        )

    async def test_emoji_prefix_does_not_trigger_name_change(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        await _debounced_update(bot, -100, 42, "idle", f"{EMOJI_ACTIVE} myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} myproject",
        )

    async def test_clear_resets_stored_name(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        clear_topic_emoji_state(-100, 42)
        bot.edit_forum_topic.reset_mock()
        await _debounced_update(bot, -100, 42, "active", "renamed")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} renamed",
        )


class TestSyncTopicName:
    async def test_preserves_cached_state_while_refreshing_clean_name(self) -> None:
        from ccgram.handlers.status.topic_emoji import _topic_states

        bot = AsyncMock()
        _topic_states[(-100, 42)] = ("idle", "normal", False)
        with (
            patch(
                "ccgram.handlers.status.topic_emoji._resolve_approval_mode",
                return_value="normal",
            ),
            patch(
                "ccgram.handlers.status.topic_emoji._resolve_rc_mode",
                return_value=False,
            ),
        ):
            await sync_topic_name(bot, -100, 42, "ccgram-codex")

        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} ccgram-codex",
        )


class TestClearTopicEmojiState:
    async def test_clear_resets_pending_transition(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC, return_value=0.0):
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        clear_topic_emoji_state(-100, 42)
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 100.0
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            bot.edit_forum_topic.assert_not_called()
            mock_monotonic.return_value = 100.0 + _debounce_for("active") + 0.1
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once()


_APPLY = "ccgram.handlers.polling.window_tick.apply"


@contextmanager
def _status_poll_env(*, has_status: bool, pane_command: str = "node"):
    """Patch the window-tick collaborators and yield ``(bot, mock_emoji)``."""
    with (
        patch(f"{_APPLY}.tmux_manager") as mock_tm,
        patch(f"{_APPLY}.window_query"),
        patch(f"{_APPLY}.thread_router") as mock_tr,
        patch(f"{_APPLY}.update_topic_emoji") as mock_emoji,
        patch(f"{_APPLY}.enqueue_status_update"),
        patch(f"{_APPLY}.get_interactive_window", return_value=None),
        patch(
            f"{_APPLY}.get_provider_for_window",
            return_value=make_mock_provider(has_status=has_status),
        ),
    ):
        window = MagicMock()
        window.pane_current_command = pane_command
        mock_tm.find_window_by_id = AsyncMock(return_value=window)
        mock_tm.capture_pane = AsyncMock(return_value="some output")
        mock_tm.get_pane_title = AsyncMock(return_value="")
        mock_tr.resolve_chat_id.return_value = -100
        mock_tr.get_display_name.return_value = "myproject"
        yield AsyncMock(), mock_emoji, mock_tr


class TestStatusPollingIntegration:
    """The 1s poll resolves a window state and hands it to update_topic_emoji."""

    async def test_active_window_with_status_updates_emoji(self) -> None:
        from ccgram.handlers.polling.window_tick import _update_status

        with _status_poll_env(has_status=True) as (bot, mock_emoji, _tr):
            await _update_status(bot, 1, "@0", thread_id=42)

        _assert_emoji_call(mock_emoji, bot, -100, 42, "active", "myproject")

    async def test_idle_window_without_status_updates_emoji(self) -> None:
        from ccgram.handlers.polling.polling_state import terminal_poll_state
        from ccgram.handlers.polling.window_tick import _update_status

        terminal_poll_state.get_state("@0").has_seen_status = True

        with _status_poll_env(has_status=False) as (bot, mock_emoji, _tr):
            await _update_status(bot, 1, "@0", thread_id=42)

        _assert_emoji_call(mock_emoji, bot, -100, 42, "idle", "myproject")

    async def test_startup_window_shows_active_not_idle(self) -> None:
        from ccgram.handlers.polling.polling_state import terminal_poll_state
        from ccgram.handlers.polling.window_tick import _update_status

        # Never polled before: no status seen yet must not read as idle.
        terminal_poll_state._states.pop("@99", None)

        with _status_poll_env(has_status=False) as (bot, mock_emoji, mock_tr):
            mock_tr.get_display_name.return_value = "newproject"
            await _update_status(bot, 1, "@99", thread_id=99)

        _assert_emoji_call(mock_emoji, bot, -100, 99, "active", "newproject")

    async def test_done_when_shell_prompt(self) -> None:
        from ccgram.handlers.polling.window_tick import _update_status

        with _status_poll_env(has_status=False, pane_command="zsh") as (
            bot,
            mock_emoji,
            _tr,
        ):
            await _update_status(bot, 1, "@0", thread_id=42)

        _assert_emoji_call(mock_emoji, bot, -100, 42, "done", "myproject")

    async def test_no_thread_id_skips_emoji(self) -> None:
        from ccgram.handlers.polling.window_tick import _update_status

        with _status_poll_env(has_status=True) as (bot, mock_emoji, _tr):
            await _update_status(bot, 1, "@0", thread_id=None)

        mock_emoji.assert_not_called()


class TestUpdateStoredTopicName:
    @pytest.mark.parametrize("cached", ["old-name", None], ids=["cached", "fresh"])
    def test_sets_cached_name(self, cached: str | None) -> None:
        from ccgram.handlers.status.topic_emoji import (
            _topic_names,
            update_stored_topic_name,
        )

        if cached is not None:
            _topic_names[(-100, 42)] = cached
        update_stored_topic_name(-100, 42, "new-name")
        assert _topic_names[(-100, 42)] == "new-name"


class TestRemoteControlBadge:
    async def test_rc_active_adds_badge(self) -> None:
        bot = AsyncMock()
        with (
            patch(
                "ccgram.handlers.status.topic_emoji._resolve_approval_mode",
                return_value="normal",
            ),
            patch(
                "ccgram.handlers.status.topic_emoji._resolve_rc_mode", return_value=True
            ),
        ):
            await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} {EMOJI_RC} myproject",
        )

    async def test_rc_and_yolo_badges(self) -> None:
        bot = AsyncMock()
        with (
            patch(
                "ccgram.handlers.status.topic_emoji._resolve_approval_mode",
                return_value="yolo",
            ),
            patch(
                "ccgram.handlers.status.topic_emoji._resolve_rc_mode", return_value=True
            ),
        ):
            await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} {EMOJI_RC} {EMOJI_YOLO} myproject",
        )


class TestStatusMode:
    @pytest.fixture
    def _user_mode(self, monkeypatch):
        from ccgram.config import config

        monkeypatch.setattr(config, "status_mode", "user")
        yield

    def test_default_uses_system_mode(self) -> None:
        from ccgram.handlers.status.topic_emoji import _state_emoji_map

        # Without any monkeypatch, default config.status_mode is "system".
        table = _state_emoji_map()
        assert table["active"] == EMOJI_GREEN_CIRCLE
        assert table["idle"] == EMOJI_YELLOW_CIRCLE

    def test_user_mode_swaps_active_idle_colors(self, _user_mode) -> None:
        from ccgram.handlers.status.topic_emoji import _state_emoji_map

        table = _state_emoji_map()
        assert table["active"] == EMOJI_YELLOW_CIRCLE
        assert table["idle"] == EMOJI_GREEN_CIRCLE
        # done/dead are unchanged across modes.
        assert table["done"] == EMOJI_DONE
        assert table["dead"] == EMOJI_DEAD

    async def test_user_mode_emits_yellow_for_active(self, _user_mode) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_YELLOW_CIRCLE} myproject",
        )

    async def test_user_mode_emits_green_for_idle(self, _user_mode) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "idle", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_GREEN_CIRCLE} myproject",
        )


MOD = "ccgram.handlers.status.topic_emoji"


class TestFloodControlCooldown:
    """Upstream #199: a RetryAfter on a topic rename must pause the chat's
    renames instead of silently re-arming the debounce forever."""

    async def test_retry_after_pauses_renames_for_the_chat(self) -> None:
        from ccgram.handlers.status.topic_emoji import FLOOD_COOLDOWN_SECONDS

        bot = AsyncMock()
        bot.edit_forum_topic.side_effect = RetryAfter(3)
        mark_awaiting_first_paint(-100, 42)

        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            # First paint hits flood control: cooldown starts, no crash.
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")

            bot.edit_forum_topic.reset_mock()
            bot.edit_forum_topic.side_effect = None
            # While paused: no rename attempt for this chat, by any path.
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            await sync_topic_name(bot, -100, 42, "myproject")
            bot.edit_forum_topic.assert_not_called()

            # After the cooldown lapses, debounced renames apply again.
            mock_monotonic.return_value = FLOOD_COOLDOWN_SECONDS + 0.1
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
            mock_monotonic.return_value = (
                FLOOD_COOLDOWN_SECONDS + 0.1 + _debounce_for("active") + 0.1
            )
            await update_topic_emoji(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once()


class TestFirstPaintPacing:
    """Upstream #199: the inherited-topic repaint burst at startup must be
    spaced across poll cycles, one topic per chat at a time."""

    async def test_burst_of_inherited_topics_is_paced(self) -> None:
        from ccgram.handlers.status.topic_emoji import CHAT_EDIT_MIN_INTERVAL

        bot = AsyncMock()
        mark_awaiting_first_paint(-100, 1)
        mark_awaiting_first_paint(-100, 2)

        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            await update_topic_emoji(bot, -100, 1, "idle", "alpha")
            await update_topic_emoji(bot, -100, 2, "idle", "beta")
            # Only the first rename of the burst goes out immediately.
            assert bot.edit_forum_topic.await_count == 1

            mock_monotonic.return_value = CHAT_EDIT_MIN_INTERVAL + 0.1
            await update_topic_emoji(bot, -100, 2, "idle", "beta")
        assert bot.edit_forum_topic.await_count == 2

    async def test_same_topic_updates_are_not_paced(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            mark_awaiting_first_paint(-100, 1)
            await update_topic_emoji(bot, -100, 1, "idle", "alpha")
            # Same key, immediate follow-up (name change): never deferred.
            mock_monotonic.return_value = 0.2
            await update_topic_emoji(bot, -100, 1, "idle", "alpha2")
        assert bot.edit_forum_topic.await_count == 2

    async def test_paced_name_change_survives_deferral(self) -> None:
        """Regression (comment-compliance review 2026-08-29): a name change
        arriving inside another topic's spacing window must defer, not be
        dropped. The deferral rolls back the write-through name cache so
        the next cycle re-detects the rename instead of consuming it."""
        from ccgram.handlers.status.topic_emoji import (
            CHAT_EDIT_MIN_INTERVAL,
            _topic_states,
        )

        bot = AsyncMock()
        _topic_states[(-100, 2)] = ("idle", "normal", False)
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            mark_awaiting_first_paint(-100, 1)
            await update_topic_emoji(bot, -100, 1, "idle", "alpha")
            assert bot.edit_forum_topic.await_count == 1

            # Topic 2's rename (token unchanged, name changed) lands inside
            # topic 1's spacing window: deferred this cycle...
            mock_monotonic.return_value = 0.1
            await update_topic_emoji(bot, -100, 2, "idle", "beta-renamed")
            assert bot.edit_forum_topic.await_count == 1

            # ...and actually sent on the next one.
            mock_monotonic.return_value = CHAT_EDIT_MIN_INTERVAL + 0.2
            await update_topic_emoji(bot, -100, 2, "idle", "beta-renamed")
        assert bot.edit_forum_topic.await_count == 2

    async def test_sync_sleeps_out_chat_spacing(self) -> None:
        """The /sync path cannot defer to a later poll cycle; it sleeps
        out the per-chat spacing instead, so a sync rename cannot land
        inside the interval opened by another topic's rename (#199)."""
        from ccgram.handlers.status.topic_emoji import CHAT_EDIT_MIN_INTERVAL

        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            mark_awaiting_first_paint(-100, 1)
            await update_topic_emoji(bot, -100, 1, "idle", "alpha")
            assert bot.edit_forum_topic.await_count == 1

            clock = {"now": 0.2}
            mock_monotonic.side_effect = lambda: clock["now"]

            async def advance(seconds: float) -> None:
                clock["now"] += seconds

            with patch(
                f"{MOD}.asyncio.sleep", new=AsyncMock(side_effect=advance)
            ) as sleep_mock:
                await sync_topic_name(bot, -100, 2, "beta")
        sleep_mock.assert_awaited_once_with(CHAT_EDIT_MIN_INTERVAL - 0.2)
        assert bot.edit_forum_topic.await_count == 2

    async def test_sync_lock_cleanup_never_replaces_held_chat_lock(self) -> None:
        from ccgram.handlers.status.topic_emoji import (
            _MAX_DISABLED_CHATS,
            _sync_rename_locks,
        )

        held_lock = asyncio.Lock()
        await held_lock.acquire()
        keepalive = [held_lock]
        _sync_rename_locks[-100] = held_lock
        for chat_id in range(10_000, 10_000 + _MAX_DISABLED_CHATS + 1):
            lock = asyncio.Lock()
            keepalive.append(lock)
            _sync_rename_locks[chat_id] = lock

        bot = AsyncMock()
        task = asyncio.create_task(sync_topic_name(bot, -100, 2, "beta"))
        await asyncio.sleep(0)

        try:
            bot.edit_forum_topic.assert_not_awaited()
        finally:
            held_lock.release()
            await task

        bot.edit_forum_topic.assert_awaited_once()
        assert keepalive

    async def test_sync_rechecks_stamp_after_sleeping(self) -> None:
        """Greptile #206 P1: while /sync sleeps out the spacing, the poll
        task may rename another topic and move the stamp; sync must
        re-check and sleep again rather than send inside the interval."""
        from ccgram.handlers.status.topic_emoji import (
            CHAT_EDIT_MIN_INTERVAL,
            _last_chat_edit,
        )

        bot = AsyncMock()
        _last_chat_edit[-100] = (0.0, (-100, 1))
        clock = {"now": 0.2}

        async def poll_renames_during_sleep(_seconds: float) -> None:
            # The poll task (no sync lock) renames another topic just as
            # sync wakes, then time advances by the spacing interval.
            clock["now"] += CHAT_EDIT_MIN_INTERVAL
            _last_chat_edit[-100] = (clock["now"], (-100, 3))

        with (
            patch(_PATCH_MONOTONIC, side_effect=lambda: clock["now"]),
            patch(
                f"{MOD}.asyncio.sleep",
                new=AsyncMock(side_effect=poll_renames_during_sleep),
            ) as sleep_mock,
        ):
            await sync_topic_name(bot, -100, 2, "beta")

        # Every sleep is undercut by a fresh different-topic stamp, so the
        # bounded loop sleeps three times and then sends (degrading to the
        # unspaced send rather than starving the command).
        assert sleep_mock.await_count == 3
        assert bot.edit_forum_topic.await_count == 1

    def test_topic_cleanup_keeps_chat_pacing_stamp(self) -> None:
        """Greptile #206: the pacing stamp is chat-scoped and must survive
        a single topic's teardown (only the permission set is cleared)."""
        from ccgram.handlers.status.topic_emoji import (
            _disabled_chats,
            _last_chat_edit,
            clear_disabled_chat,
        )

        _disabled_chats.add(-100)
        _last_chat_edit[-100] = (0.0, (-100, 1))

        clear_disabled_chat(-100, 42)

        assert -100 not in _disabled_chats
        assert _last_chat_edit[-100] == (0.0, (-100, 1))
