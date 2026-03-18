"""Tests for provider selection UI in directory browser flow."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_DIR_CANCEL,
    CB_MODE_SELECT,
    CB_PROV_SELECT,
)
from ccgram.handlers.directory_browser import build_mode_picker, build_provider_picker
from ccgram.handlers.directory_callbacks import (
    _handle_confirm,
    _handle_mode_select,
    _handle_provider_select,
)
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT


class TestBuildProviderPicker:
    def test_returns_text_and_keyboard(self) -> None:
        text, keyboard = build_provider_picker("/home/user/project")
        assert "Select Provider" in text
        assert isinstance(keyboard, InlineKeyboardMarkup)

    def test_shows_all_three_providers(self) -> None:
        text, keyboard = build_provider_picker("/tmp/test")
        buttons = keyboard.inline_keyboard
        labels = [btn.text for row in buttons for btn in row]
        assert any("Claude" in label for label in labels)
        assert any("Codex" in label for label in labels)
        assert any("Gemini" in label for label in labels)

    def test_claude_marked_as_default(self) -> None:
        _text, keyboard = build_provider_picker("/tmp/test")
        buttons = keyboard.inline_keyboard
        claude_labels = [
            btn.text for row in buttons for btn in row if "Claude" in btn.text
        ]
        assert any("default" in label for label in claude_labels)

    def test_callback_data_uses_prov_prefix(self) -> None:
        _text, keyboard = build_provider_picker("/tmp/test")
        buttons = keyboard.inline_keyboard
        provider_callbacks = [
            btn.callback_data
            for row in buttons
            for btn in row
            if btn.callback_data and btn.callback_data.startswith(CB_PROV_SELECT)
        ]
        assert f"{CB_PROV_SELECT}claude" in provider_callbacks
        assert f"{CB_PROV_SELECT}codex" in provider_callbacks
        assert f"{CB_PROV_SELECT}gemini" in provider_callbacks

    def test_has_cancel_button(self) -> None:
        _text, keyboard = build_provider_picker("/tmp/test")
        buttons = keyboard.inline_keyboard
        cancel_callbacks = [btn.callback_data for row in buttons for btn in row]
        assert CB_DIR_CANCEL in cancel_callbacks

    def test_displays_directory_path(self) -> None:
        text, _keyboard = build_provider_picker("/home/user/my-project")
        assert "my-project" in text

    def test_tilde_substitution(self) -> None:
        home = str(Path.home())
        text, _keyboard = build_provider_picker(f"{home}/project")
        assert "~/project" in text


class TestBuildModePicker:
    def test_returns_text_and_keyboard(self) -> None:
        text, keyboard = build_mode_picker("/home/user/project", "claude")
        assert "Select Session Mode" in text
        assert isinstance(keyboard, InlineKeyboardMarkup)

    def test_mode_callbacks(self) -> None:
        _text, keyboard = build_mode_picker("/tmp/test", "codex")
        callbacks = [
            btn.callback_data for row in keyboard.inline_keyboard for btn in row
        ]
        assert f"{CB_MODE_SELECT}codex:normal" in callbacks
        assert f"{CB_MODE_SELECT}codex:yolo" in callbacks
        assert CB_DIR_CANCEL in callbacks


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    return ctx


def _make_query(data: str = "") -> AsyncMock:
    query = AsyncMock()
    query.data = data
    query.answer = AsyncMock()
    query.message = MagicMock()
    return query


def _make_update(thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 100
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.message_thread_id = thread_id
    return update


class TestHandleConfirmShowsProviderPicker:
    @patch("ccgram.handlers.directory_callbacks.safe_edit", new_callable=AsyncMock)
    @patch("ccgram.handlers.directory_callbacks.session_manager")
    async def test_confirm_shows_provider_picker(
        self, mock_sm: MagicMock, mock_edit: AsyncMock
    ) -> None:
        mock_sm.get_window_for_thread.return_value = None
        user_data = {
            "browse_path": "/tmp/test",
            PENDING_THREAD_ID: 42,
        }
        query = _make_query()
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_confirm(query, 100, update, context)

        mock_edit.assert_called_once()
        call_args = mock_edit.call_args
        text = call_args[0][1]
        assert "Select Provider" in text
        keyboard = call_args.kwargs.get("reply_markup") or call_args[0][2]
        assert isinstance(keyboard, InlineKeyboardMarkup)

    @patch("ccgram.handlers.directory_callbacks.safe_edit", new_callable=AsyncMock)
    @patch("ccgram.handlers.directory_callbacks.session_manager")
    async def test_confirm_clears_browse_state(
        self, mock_sm: MagicMock, mock_edit: AsyncMock
    ) -> None:
        mock_sm.get_window_for_thread.return_value = None
        user_data = {
            "browse_path": "/tmp/test",
            "browse_page": 2,
            "browse_dirs": ["a", "b"],
            "state": "browsing_directory",
            PENDING_THREAD_ID: 42,
        }
        query = _make_query()
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_confirm(query, 100, update, context)

        assert "browse_path" in user_data
        assert "state" in user_data


class TestHandleProviderSelect:
    @patch("ccgram.handlers.directory_callbacks.safe_edit", new_callable=AsyncMock)
    @patch("ccgram.handlers.directory_callbacks.tmux_manager")
    @patch("ccgram.handlers.directory_callbacks.provider_registry")
    @patch("ccgram.handlers.directory_callbacks.session_manager")
    async def test_shows_mode_picker(
        self,
        mock_sm: MagicMock,
        mock_registry: MagicMock,
        mock_tmux: MagicMock,
        mock_edit: AsyncMock,
    ) -> None:
        mock_registry.is_valid.return_value = True
        mock_sm.get_window_for_thread.return_value = None
        mock_tmux.create_window = AsyncMock()

        user_data = {"browse_path": "/tmp/test", PENDING_THREAD_ID: 42}
        query = _make_query(data=f"{CB_PROV_SELECT}codex")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_provider_select(
            query, 100, f"{CB_PROV_SELECT}codex", update, context
        )

        mock_tmux.create_window.assert_not_called()
        mock_edit.assert_called_once()
        text = mock_edit.call_args[0][1]
        assert "Select Session Mode" in text

    @patch("ccgram.handlers.directory_callbacks.provider_registry")
    async def test_rejects_unknown_provider(self, mock_registry: MagicMock) -> None:
        mock_registry.is_valid.return_value = False
        query = _make_query(data=f"{CB_PROV_SELECT}unknown")
        update = _make_update()
        context = _make_context()

        await _handle_provider_select(
            query, 100, f"{CB_PROV_SELECT}unknown", update, context
        )
        query.answer.assert_any_call("Unknown provider", show_alert=True)


class TestHandleModeSelect:
    @patch("ccgram.providers.resolve_launch_command")
    @patch("ccgram.handlers.directory_callbacks.safe_edit", new_callable=AsyncMock)
    @patch("ccgram.handlers.directory_callbacks.session_manager")
    @patch("ccgram.handlers.directory_callbacks.tmux_manager")
    @patch("ccgram.handlers.directory_callbacks.provider_registry")
    async def test_creates_window_with_yolo_mode(
        self,
        mock_registry: MagicMock,
        mock_tmux: MagicMock,
        mock_sm: MagicMock,
        mock_edit: AsyncMock,
        mock_resolve_launch: MagicMock,
    ) -> None:
        mock_registry.is_valid.return_value = True
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_registry.get.return_value = mock_provider

        mock_resolve_launch.return_value = (
            "codex --dangerously-bypass-approvals-and-sandbox"
        )
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'proj'", "proj", "@5")
        )
        mock_sm.get_window_for_thread.return_value = None
        mock_sm.resolve_chat_id.return_value = 123
        mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
        mock_sm.get_window_state.return_value = MagicMock()

        user_data = {"browse_path": "/tmp/proj", PENDING_THREAD_ID: 42}
        query = _make_query(data=f"{CB_MODE_SELECT}codex:yolo")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}codex:yolo", update, context
        )

        mock_resolve_launch.assert_called_once_with("codex", approval_mode="yolo")
        mock_tmux.create_window.assert_called_once_with(
            "/tmp/proj",
            launch_command="codex --dangerously-bypass-approvals-and-sandbox",
        )
        mock_sm.set_window_provider.assert_called_once_with("@5", "codex")
        mock_sm.set_window_approval_mode.assert_called_once_with("@5", "yolo")

    @patch("ccgram.handlers.directory_callbacks.provider_registry")
    async def test_rejects_unknown_mode(self, mock_registry: MagicMock) -> None:
        mock_registry.is_valid.return_value = True
        query = _make_query(data=f"{CB_MODE_SELECT}codex:unknown")
        update = _make_update()
        context = _make_context()

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}codex:unknown", update, context
        )
        query.answer.assert_any_call("Unknown mode", show_alert=True)

    @patch("ccgram.providers.resolve_launch_command")
    @patch("ccgram.handlers.directory_callbacks.safe_edit", new_callable=AsyncMock)
    @patch("ccgram.handlers.directory_callbacks.session_manager")
    @patch("ccgram.handlers.directory_callbacks.tmux_manager")
    @patch("ccgram.handlers.directory_callbacks.provider_registry")
    async def test_forwards_pending_text(
        self,
        mock_registry: MagicMock,
        mock_tmux: MagicMock,
        mock_sm: MagicMock,
        mock_edit: AsyncMock,
        mock_resolve_launch: MagicMock,
    ) -> None:
        mock_registry.is_valid.return_value = True
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_registry.get.return_value = mock_provider

        mock_resolve_launch.return_value = "claude"
        mock_tmux.create_window = AsyncMock(
            return_value=(True, "Created window 'proj'", "proj", "@1")
        )
        mock_sm.get_window_for_thread.return_value = None
        mock_sm.resolve_chat_id.return_value = 123
        mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
        mock_sm.get_window_state.return_value = MagicMock()

        user_data = {
            "browse_path": "/tmp/proj",
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "hello world",
        }
        query = _make_query(data=f"{CB_MODE_SELECT}claude:normal")
        update = _make_update(thread_id=42)
        context = _make_context(user_data)

        await _handle_mode_select(
            query, 100, f"{CB_MODE_SELECT}claude:normal", update, context
        )

        mock_sm.send_to_window.assert_called_once_with("@1", "hello world")
        assert PENDING_THREAD_TEXT not in user_data
