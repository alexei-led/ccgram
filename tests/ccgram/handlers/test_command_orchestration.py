"""Tests for command_orchestration module public API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.command_orchestration import (
    forward_command_handler,
    get_global_provider_menu,
    set_global_provider_menu,
    sync_scoped_provider_menu,
)


def _make_update(
    *,
    user_id: int = 100,
    thread_id: int = 42,
    text: str = "/clear",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    msg = AsyncMock()
    msg.text = text
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.id = -100999
    msg.chat.is_forum = True
    msg.is_topic_message = True
    msg.get_bot = MagicMock(return_value=MagicMock(send_chat_action=AsyncMock()))
    update.message = msg
    update.callback_query = None
    return update


class TestForwardKnownCommand:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mock_tr = MagicMock()
        self.mock_tr.resolve_window_for_thread.return_value = "@1"
        self.mock_tr.get_display_name.return_value = "project"
        self.mock_tr.set_group_chat_id = MagicMock()

        self.mock_sm = MagicMock()
        self.mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
        self.mock_sm.get_window_state.return_value = SimpleNamespace(
            transcript_path="", session_id="s1", cwd="/w"
        )

        self.mock_tm = MagicMock()
        self.mock_tm.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@1")
        )
        self.mock_tm.capture_pane = AsyncMock(return_value="")

        provider = SimpleNamespace(
            capabilities=SimpleNamespace(
                name="claude",
                supports_incremental_read=True,
                supports_status_snapshot=False,
            )
        )

        with (
            patch(
                "ccgram.bot.is_user_allowed",
                return_value=True,
            ),
            patch("ccgram.handlers.command_orchestration.thread_router", self.mock_tr),
            patch(
                "ccgram.handlers.command_orchestration.session_manager", self.mock_sm
            ),
            patch("ccgram.handlers.command_orchestration.tmux_manager", self.mock_tm),
            patch(
                "ccgram.handlers.command_orchestration.get_provider_for_window",
                return_value=provider,
            ),
            patch(
                "ccgram.handlers.command_orchestration._get_provider_command_metadata",
                return_value=({"clear": "clear"}, {"/clear"}),
            ),
            patch(
                "ccgram.handlers.command_orchestration._command_known_in_other_provider",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.command_orchestration._capture_command_probe_context",
                new_callable=AsyncMock,
                return_value=(None, None, None),
            ),
            patch(
                "ccgram.handlers.command_orchestration._spawn_command_failure_probe",
                MagicMock(),
            ),
            patch(
                "ccgram.handlers.command_orchestration.sync_scoped_provider_menu",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_forward_known_command(self) -> None:
        update = _make_update(text="/clear")
        await forward_command_handler(update, MagicMock())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/clear")

    async def test_forward_unknown_command_warns(self) -> None:
        with patch(
            "ccgram.handlers.command_orchestration._command_known_in_other_provider",
            return_value=True,
        ):
            update = _make_update(text="/cost")
            await forward_command_handler(update, MagicMock())

        self.mock_sm.send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "not supported" in reply_text


class TestMenuCacheInvalidation:
    async def test_menu_cache_invalidated_on_provider_change(self) -> None:
        from ccgram.handlers.command_orchestration import (
            _scoped_provider_menu,
            _chat_scoped_provider_menu,
        )

        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        set_global_provider_menu("old")
        try:
            message = AsyncMock()
            message.chat.id = -100
            message.get_bot.return_value = object()
            codex = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
            claude = SimpleNamespace(capabilities=SimpleNamespace(name="claude"))

            with patch(
                "ccgram.handlers.command_orchestration.register_commands",
                new_callable=AsyncMock,
            ) as mock_reg:
                await sync_scoped_provider_menu(message, 1, codex)  # type: ignore[arg-type]
                await sync_scoped_provider_menu(message, 1, claude)  # type: ignore[arg-type]

            assert mock_reg.call_count == 2
            assert _scoped_provider_menu[(-100, 1)] == "claude"
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            set_global_provider_menu("claude")


class TestGlobalProviderMenu:
    def test_get_set_global_provider_menu(self) -> None:
        old = get_global_provider_menu()
        try:
            set_global_provider_menu("test-provider")
            assert get_global_provider_menu() == "test-provider"
        finally:
            if old is not None:
                set_global_provider_menu(old)
