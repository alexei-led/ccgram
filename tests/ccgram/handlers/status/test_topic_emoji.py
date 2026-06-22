from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, TelegramError

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
    EMOJI_YELLOW_CIRCLE,
    EMOJI_YOLO,
    clear_topic_emoji_state,
    format_topic_name_for_mode,
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
        "emoji", [EMOJI_ACTIVE, EMOJI_IDLE, EMOJI_DONE, EMOJI_DEAD]
    )
    def test_strips_known_emoji(self, emoji: str) -> None:
        assert strip_emoji_prefix(f"{emoji} myproject") == "myproject"

    def test_no_prefix(self) -> None:
        assert strip_emoji_prefix("myproject") == "myproject"

    def test_double_prefix_strips_once(self) -> None:
        result = strip_emoji_prefix(f"{EMOJI_ACTIVE} {EMOJI_IDLE} myproject")
        assert result == f"{EMOJI_IDLE} myproject"

    def test_strips_yolo_prefix(self) -> None:
        assert strip_emoji_prefix(f"{EMOJI_YOLO} myproject") == "myproject"

    def test_strips_state_and_yolo_prefix(self) -> None:
        assert (
            strip_emoji_prefix(f"{EMOJI_ACTIVE} {EMOJI_YOLO} myproject") == "myproject"
        )


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

    async def test_idle_fires_after_full_debounce(self) -> None:
        bot = AsyncMock()
        with patch(_PATCH_MONOTONIC) as mock_monotonic:
            mock_monotonic.return_value = 0.0
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")
            mock_monotonic.return_value = DEBOUNCE_TO_IDLE_SECONDS + 0.1
            await update_topic_emoji(bot, -100, 42, "idle", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} myproject",
        )

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
    def test_formats_yolo_name(self) -> None:
        assert (
            format_topic_name_for_mode("myproject", "yolo") == f"{EMOJI_YOLO} myproject"
        )

    def test_formats_normal_name(self) -> None:
        assert format_topic_name_for_mode("myproject", "normal") == "myproject"


class TestTopicNamePreservation:
    async def test_stores_name_on_first_update(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_ACTIVE} myproject",
        )

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


class TestHerdrToTelegramRename:
    """herdr tab rename propagates to Telegram topic via display-name sync."""

    async def test_display_name_change_triggers_topic_relabel(self) -> None:
        """When herdr renames a tab, the next update_topic_emoji call relabels the topic.

        Flow: sync_display_names updates display_name → update_topic_emoji receives
        new display_name → _resolve_topic_name detects change → edit_forum_topic called.
        """
        bot = AsyncMock()
        # Prime the cache with the old label (simulates a prior poll cycle)
        await _debounced_update(bot, -100, 42, "idle", "workspace ▸ old-agent")
        bot.edit_forum_topic.reset_mock()

        # herdr renames the tab: next poll delivers the new display_name.
        # Same state (idle), same debounce; name change bypasses debounce.
        with patch(_PATCH_MONOTONIC, return_value=1000.0):
            await update_topic_emoji(bot, -100, 42, "idle", "workspace ▸ new-agent")

        bot.edit_forum_topic.assert_called_once_with(
            chat_id=-100,
            message_thread_id=42,
            name=f"{EMOJI_IDLE} workspace ▸ new-agent",
        )

    async def test_same_display_name_after_sync_does_not_relabel(self) -> None:
        """No Telegram call when the tab name did not change."""
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "idle", "workspace ▸ agent")
        bot.edit_forum_topic.reset_mock()

        # Same display name — no relabel expected (state already set, name unchanged)
        with patch(_PATCH_MONOTONIC, return_value=1000.0):
            await update_topic_emoji(bot, -100, 42, "idle", "workspace ▸ agent")

        bot.edit_forum_topic.assert_not_called()


class TestClearTopicEmojiState:
    async def test_clear_allows_re_update(self) -> None:
        bot = AsyncMock()
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.reset_mock()
        clear_topic_emoji_state(-100, 42)
        await _debounced_update(bot, -100, 42, "active", "myproject")
        bot.edit_forum_topic.assert_called_once()


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


class TestStatusPollingIntegration:
    async def test_active_window_with_status_updates_emoji(self) -> None:
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji"
            ) as mock_emoji,
            patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ),
        ):
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "myproject"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=42)

            _assert_emoji_call(mock_emoji, bot, -100, 42, "active", "myproject")

    async def test_idle_window_without_status_updates_emoji(self) -> None:
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji"
            ) as mock_emoji,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
                return_value=make_mock_provider(has_status=False),
            ),
        ):
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )
            from ccgram.handlers.polling.polling_state import terminal_poll_state

            terminal_poll_state.get_state("@0").has_seen_status = True

            mock_window = MagicMock()
            mock_window.pane_current_command = "node"
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "myproject"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=42)

            _assert_emoji_call(mock_emoji, bot, -100, 42, "idle", "myproject")

    async def test_startup_window_shows_active_not_idle(self) -> None:
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji"
            ) as mock_emoji,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
                return_value=make_mock_provider(has_status=False),
            ),
        ):
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )
            from ccgram.handlers.polling.polling_state import terminal_poll_state

            terminal_poll_state._states.pop("@99", None)

            mock_window = MagicMock()
            mock_window.pane_current_command = "node"
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "newproject"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@99", thread_id=99)

            _assert_emoji_call(mock_emoji, bot, -100, 99, "active", "newproject")

    async def test_done_when_shell_prompt(self) -> None:
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji"
            ) as mock_emoji,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
                return_value=make_mock_provider(has_status=False),
            ),
        ):
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            mock_window = MagicMock()
            mock_window.pane_current_command = "zsh"
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "myproject"

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=42)

            _assert_emoji_call(mock_emoji, bot, -100, 42, "done", "myproject")

    async def test_no_thread_id_skips_emoji(self) -> None:
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji"
            ) as mock_emoji,
            patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
                return_value=make_mock_provider(has_status=True),
            ),
        ):
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            mock_tm.find_window_by_id = AsyncMock(return_value=MagicMock())
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tm.get_pane_title = AsyncMock(return_value="")

            bot = AsyncMock()
            await update_status_message(bot, 1, "@0", thread_id=None)

            mock_emoji.assert_not_called()


class TestUpdateStoredTopicName:
    def test_overwrites_cached_name(self) -> None:
        from ccgram.handlers.status.topic_emoji import (
            _topic_names,
            update_stored_topic_name,
        )

        _topic_names[(-100, 42)] = "old-name"
        update_stored_topic_name(-100, 42, "new-name")
        assert _topic_names[(-100, 42)] == "new-name"

    def test_sets_name_when_not_cached(self) -> None:
        from ccgram.handlers.status.topic_emoji import (
            _topic_names,
            update_stored_topic_name,
        )

        update_stored_topic_name(-100, 99, "fresh-name")
        assert _topic_names[(-100, 99)] == "fresh-name"


class TestRemoteControlBadge:
    def test_strip_rc_prefix(self) -> None:
        from ccgram.handlers.status.topic_emoji import EMOJI_RC

        assert strip_emoji_prefix(f"{EMOJI_RC} myproject") == "myproject"

    def test_strip_state_and_rc_prefix(self) -> None:
        from ccgram.handlers.status.topic_emoji import EMOJI_RC

        assert strip_emoji_prefix(f"{EMOJI_ACTIVE} {EMOJI_RC} myproject") == "myproject"

    def test_strip_state_rc_yolo_prefix(self) -> None:
        from ccgram.handlers.status.topic_emoji import EMOJI_RC

        assert (
            strip_emoji_prefix(f"{EMOJI_ACTIVE} {EMOJI_RC} {EMOJI_YOLO} myproject")
            == "myproject"
        )

    async def test_rc_active_adds_badge(self) -> None:
        from ccgram.handlers.status.topic_emoji import EMOJI_RC

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
        from ccgram.handlers.status.topic_emoji import EMOJI_RC

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

    def test_strip_handles_both_modes(self) -> None:
        # Whichever mode wrote the prefix, both colors are always strippable.
        assert strip_emoji_prefix(f"{EMOJI_GREEN_CIRCLE} x") == "x"
        assert strip_emoji_prefix(f"{EMOJI_YELLOW_CIRCLE} x") == "x"
