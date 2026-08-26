"""Tests for window picker callback handlers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, CallbackQuery, Update
from telegram.ext import ContextTypes

from ccgram.handlers.callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from ccgram.handlers.topics.directory_browser import UNBOUND_WINDOWS_KEY
from ccgram.handlers.topics.new_command import new_command
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT
from ccgram.handlers.topics.window_callbacks import (
    _forward_pending_text,
    handle_window_callback,
)

_MODULE = "ccgram.handlers.topics.window_callbacks."
_FIND_WINDOW = "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id"


def _make_query_update_context(
    thread_id: int = 42,
    user_data: dict | None = None,
) -> tuple[AsyncMock, MagicMock, MagicMock]:
    query = AsyncMock(spec=CallbackQuery)
    query.answer = AsyncMock()

    msg = MagicMock()
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.id = -100999

    update = MagicMock(spec=Update)
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = msg

    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.user_data = user_data if user_data is not None else {}
    context.bot = AsyncMock()
    return query, update, context


def _unbound_window(name: str, pane_current_command: str | None = None) -> MagicMock:
    window = MagicMock()
    window.window_name = name
    if pane_current_command is not None:
        window.pane_current_command = pane_current_command
    return window


async def _reset_flow_with_new_command(context: MagicMock) -> None:
    """Run /new, which drops the picker state every stale-guard test needs gone."""
    update = MagicMock()
    update.effective_user = MagicMock(id=100)
    update.message = AsyncMock()
    with patch(
        "ccgram.handlers.topics.new_command.config.is_user_allowed", return_value=True
    ):
        await new_command(update, context)


@dataclass
class _BindMocks:
    session: MagicMock
    router: MagicMock
    find_window: AsyncMock
    edit: MagicMock


@contextlib.contextmanager
def _bind_env(window: MagicMock | None = None) -> Iterator[_BindMocks]:
    """Patch the collaborators a CB_WIN_BIND tap reaches, ready for a live window."""
    with (
        patch(f"{_MODULE}session_manager") as session,
        patch(f"{_MODULE}thread_router") as router,
        patch(_FIND_WINDOW, new_callable=AsyncMock, return_value=window) as find_window,
        patch(f"{_MODULE}safe_edit") as edit,
        patch(f"{_MODULE}format_topic_name_for_mode"),
    ):
        router.resolve_chat_id.return_value = -100
        session.get_approval_mode.return_value = "normal"
        yield _BindMocks(
            session=session, router=router, find_window=find_window, edit=edit
        )


class TestBindWindowCallback:
    async def test_bind_existing_window(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        )

        with _bind_env(_unbound_window("my-project")) as m:
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        m.router.bind_thread.assert_called_once_with(
            100, 42, "@5", window_name="my-project", chat_id=-100999
        )
        m.router.set_group_chat_id.assert_called_once_with(100, 42, -100999)
        m.edit.assert_called_once()
        assert "my-project" in m.edit.call_args[0][1]

    async def test_bind_forwards_pending_text(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={
                UNBOUND_WINDOWS_KEY: ["@5"],
                PENDING_THREAD_ID: 42,
                PENDING_THREAD_TEXT: "hello agent",
            }
        )

        with (
            _bind_env(_unbound_window("proj")),
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_send.assert_called_once_with(100, "@5", 42, "hello agent", -100999)
        assert PENDING_THREAD_TEXT not in context.user_data

    @pytest.mark.parametrize(
        ("suffix", "answer_args", "answer_kwargs"),
        [
            ("abc", ("Invalid data",), {}),
            ("5", ("Window list changed, please retry",), {"show_alert": True}),
        ],
        ids=["not_a_number", "out_of_range"],
    )
    async def test_bind_rejects_unusable_index(
        self, suffix: str, answer_args: tuple[str, ...], answer_kwargs: dict
    ) -> None:
        query, update, context = _make_query_update_context(
            user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        )

        await handle_window_callback(
            query, 100, f"{CB_WIN_BIND}{suffix}", update, context
        )

        query.answer.assert_called_once_with(*answer_args, **answer_kwargs)

    async def test_bind_rejected_after_new_command_reset(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        )
        await _reset_flow_with_new_command(context)

        with patch(_FIND_WINDOW, new_callable=AsyncMock) as mock_find:
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_find.assert_not_called()
        query.answer.assert_called_once_with(
            "Window list changed, please retry", show_alert=True
        )


class TestNewWindowCallback:
    async def test_transitions_to_directory_browser(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={PENDING_THREAD_ID: 42}
        )

        with (
            patch(
                f"{_MODULE}build_directory_browser",
                return_value=("Browse:", MagicMock(), ["/a", "/b"]),
            ),
            patch(f"{_MODULE}safe_edit") as mock_edit,
            patch(f"{_MODULE}clear_window_picker_state"),
        ):
            await handle_window_callback(query, 100, CB_WIN_NEW, update, context)

        mock_edit.assert_called_once()
        query.answer.assert_called_once_with()

    async def test_new_rejected_after_new_command_reset(self) -> None:
        from ccgram.handlers.topics.directory_browser import BROWSE_PATH_KEY, STATE_KEY

        query, update, context = _make_query_update_context(
            user_data={PENDING_THREAD_ID: 42}
        )
        await _reset_flow_with_new_command(context)

        with patch(f"{_MODULE}build_directory_browser") as mock_build:
            await handle_window_callback(query, 100, CB_WIN_NEW, update, context)

        mock_build.assert_not_called()
        query.answer.assert_called_once_with(
            "Stale picker (flow reset)", show_alert=True
        )
        assert STATE_KEY not in context.user_data
        assert BROWSE_PATH_KEY not in context.user_data


class TestCancelCallback:
    async def test_cancel_clears_state(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "some text"}
        )

        with (
            patch(f"{_MODULE}safe_edit") as mock_edit,
            patch(f"{_MODULE}clear_window_picker_state"),
        ):
            await handle_window_callback(query, 100, CB_WIN_CANCEL, update, context)

        mock_edit.assert_called_once_with(query, "Cancelled")
        query.answer.assert_called_once_with("Cancelled")
        assert PENDING_THREAD_ID not in context.user_data
        assert PENDING_THREAD_TEXT not in context.user_data


class TestCrossTopicGuard:
    @pytest.mark.parametrize(
        "data",
        [f"{CB_WIN_BIND}0", CB_WIN_NEW, CB_WIN_CANCEL],
        ids=["bind", "new", "cancel"],
    )
    async def test_tap_from_another_topic_is_rejected(self, data: str) -> None:
        query, update, context = _make_query_update_context(
            thread_id=42,
            user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 99},
        )

        await handle_window_callback(query, 100, data, update, context)

        query.answer.assert_called_once_with(
            "Stale picker (topic mismatch)", show_alert=True
        )


class TestBindProviderDetection:
    @pytest.mark.parametrize(
        ("pane_command", "detected", "expects_prompt_setup"),
        [
            ("fish", "shell", True),
            ("", "shell", True),
            ("claude", "claude", False),
        ],
        ids=["shell_window", "bare_herdr_shell", "claude_window"],
    )
    async def test_detected_provider_drives_shell_prompt_setup(
        self, pane_command: str, detected: str, expects_prompt_setup: bool
    ) -> None:
        """herdr leaves pane_current_command empty; detection must still run."""
        query, update, context = _make_query_update_context(
            user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        )

        with (
            _bind_env(_unbound_window("bound-window", pane_command)) as m,
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value=detected,
            ) as mock_detect,
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_detect.assert_awaited_once_with(pane_command, window_id="@5")
        m.session.set_window_provider.assert_called_once_with("@5", detected)
        if expects_prompt_setup:
            mock_ensure.assert_awaited_once()
            assert mock_ensure.call_args[0] == ("@5", "external_bind")
        else:
            mock_ensure.assert_not_awaited()

    async def test_bind_shell_pending_text_routes_through_shell_handler(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={
                UNBOUND_WINDOWS_KEY: ["@5"],
                PENDING_THREAD_ID: 42,
                PENDING_THREAD_TEXT: "ls -la",
            }
        )

        with (
            _bind_env(_unbound_window("my-shell", "bash")),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_MODULE}_forward_pending_text", new_callable=AsyncMock
            ) as mock_forward,
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_forward.assert_awaited_once()
        forward_args = mock_forward.call_args.args
        assert forward_args[1:] == (100, 42, "@5", "ls -la", "shell")
        assert forward_args[0].bot is context.bot
        assert mock_forward.call_args.kwargs == {
            "is_existing_window": True,
            "chat_id": -100999,
        }


class TestForwardPendingText:
    async def test_existing_shell_window_sends_raw(self) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(f"{_MODULE}session_manager"),
            patch(
                "ccgram.handlers.shell.shell_commands.handle_shell_message",
                new_callable=AsyncMock,
            ) as mock_shell,
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
        ):
            await _forward_pending_text(
                bot, 1, 42, "@5", "list files", "shell", is_existing_window=True
            )

        mock_shell.assert_not_awaited()
        mock_send.assert_called_once_with(1, "@5", 42, "list files", None)

    async def test_new_shell_window_routes_through_handler(self) -> None:
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
        ) as mock_shell:
            await _forward_pending_text(
                bot, 1, 42, "@5", "list files", "shell", is_existing_window=False
            )

        mock_shell.assert_awaited_once()
        args = mock_shell.call_args.args
        assert args[1:] == (1, 42, "@5", "list files")
        assert args[0] is bot
