"""Tests for /bind and for the directory browser's project-root listing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from ccgram.handlers.topics.bind_command import bind_command
from ccgram.handlers.topics.directory_browser import (
    STATE_KEY,
    STATE_SELECTING_WINDOW,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    build_window_picker,
)
from ccgram.handlers.user_state import PENDING_THREAD_ID


def _buttons(keyboard) -> list:
    return [btn for row in keyboard.inline_keyboard for btn in row]


def _window(window_id: str, name: str, cwd: str) -> MagicMock:
    win = MagicMock()
    win.window_id = window_id
    win.window_name = name
    win.cwd = cwd
    return win


class TestWindowPickerEntries:
    def test_unbound_flow_offers_a_new_session(self) -> None:
        _text, keyboard, ids = build_window_picker([("@1", "proj", "/tmp")])
        callbacks = [btn.callback_data for btn in _buttons(keyboard)]
        assert ids == ["@1"]
        assert CB_WIN_NEW in callbacks
        assert CB_WIN_BIND + "0" in callbacks

    def test_bind_command_offers_no_new_session(self) -> None:
        _text, keyboard, _ids = build_window_picker(
            [("@1", "proj", "/tmp")], include_new=False
        )
        callbacks = [btn.callback_data for btn in _buttons(keyboard)]
        assert CB_WIN_NEW not in callbacks
        assert CB_WIN_CANCEL in callbacks


class TestBindCommand:
    async def test_lists_unbound_windows(self) -> None:
        update = MagicMock()
        update.effective_user.id = 100
        update.message.chat.id = -100999
        update.message.message_thread_id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.user_data = {}

        with (
            patch("ccgram.handlers.topics.bind_command.config") as mock_config,
            patch("ccgram.handlers.topics.bind_command.tmux_manager") as mock_mux,
            patch("ccgram.handlers.topics.bind_command.thread_router") as mock_router,
        ):
            mock_config.is_user_allowed.return_value = True
            mock_mux.list_windows = AsyncMock(
                return_value=[_window("@1", "proj", "/tmp"), _window("@2", "x", "/tmp")]
            )
            mock_router.get_window_for_thread.return_value = None
            mock_router.iter_thread_bindings.return_value = [(100, 7, "@2")]

            await bind_command(update, context)

        text, kwargs = update.message.reply_text.call_args
        assert "Bind to Existing Window" in text[0]
        assert context.user_data[UNBOUND_WINDOWS_KEY] == ["@1"]
        assert context.user_data[STATE_KEY] == STATE_SELECTING_WINDOW
        assert context.user_data[PENDING_THREAD_ID] == 42
        assert kwargs["reply_markup"] is not None

    async def test_existing_session_is_reported_not_replaced(self) -> None:
        update = MagicMock()
        update.effective_user.id = 100
        update.message.chat.id = -100999
        update.message.message_thread_id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.user_data = {}

        with (
            patch("ccgram.handlers.topics.bind_command.config") as mock_config,
            patch("ccgram.handlers.topics.bind_command.thread_router") as mock_router,
        ):
            mock_config.is_user_allowed.return_value = True
            mock_router.get_window_for_thread.return_value = "@1"

            await bind_command(update, context)

        assert "/unbind" in update.message.reply_text.call_args[0][0]

    async def test_no_unbound_windows_explains_the_typed_line(self) -> None:
        update = MagicMock()
        update.effective_user.id = 100
        update.message.chat.id = -100999
        update.message.message_thread_id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.user_data = {}

        with (
            patch("ccgram.handlers.topics.bind_command.config") as mock_config,
            patch("ccgram.handlers.topics.bind_command.tmux_manager") as mock_mux,
            patch("ccgram.handlers.topics.bind_command.thread_router") as mock_router,
        ):
            mock_config.is_user_allowed.return_value = True
            mock_mux.list_windows = AsyncMock(return_value=[])
            mock_router.get_window_for_thread.return_value = None
            mock_router.iter_thread_bindings.return_value = []

            await bind_command(update, context)

        assert "codex" in update.message.reply_text.call_args[0][0]


class TestProjectsRootListing:
    """Under a projects root only git checkouts are listed."""

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from ccgram.config import config

        root = tmp_path / "Workspace"
        root.mkdir()
        (root / "ccgram" / ".git").mkdir(parents=True)
        (root / "notes").mkdir()
        (root / ".hidden" / ".git").mkdir(parents=True)
        monkeypatch.setattr(config, "projects_roots", [root], raising=False)
        monkeypatch.setattr(config, "dir_browser_all_dirs", False, raising=False)
        monkeypatch.setattr(config, "show_hidden_dirs", False, raising=False)
        return root

    def test_only_git_repos_are_listed_at_the_root(self, root: Path) -> None:
        _text, _keyboard, subdirs = build_directory_browser(str(root))
        assert subdirs == ["ccgram"]

    def test_deeper_levels_list_every_subdirectory(self, root: Path) -> None:
        (root / "ccgram" / "src").mkdir()
        _text, _keyboard, subdirs = build_directory_browser(str(root / "ccgram"))
        assert subdirs == ["src"]

    def test_all_directories_mode_lists_plain_dirs(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "dir_browser_all_dirs", True, raising=False)
        _text, _keyboard, subdirs = build_directory_browser(str(root))
        assert subdirs == ["ccgram", "notes"]
