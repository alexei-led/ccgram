"""Integration tests for toolbar callback dispatch through the PTB application.

Exercises the full path from a Telegram CallbackQuery → callback_registry
dispatch → toolbar_callbacks._dispatch → action handler. Mocks tmux_manager
and session_manager but uses a real PTB Application + real callback_registry
to verify the single ``CB_TOOLBAR`` prefix wiring and per-type dispatch.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram import CallbackQuery
from telegram.ext import Application, CallbackQueryHandler

from ccgram.handlers.callback_data import CB_TOOLBAR
from ccgram.handlers.callback_registry import (
    dispatch as callback_dispatch,
    load_handlers,
)
from ccgram.handlers.toolbar.toolbar_keyboard import (
    build_toolbar_keyboard,
    reload_toolbar_config,
)
from ccgram.toolbar_config import (
    BUILTIN_ACTIONS,
    DEFAULT_LAYOUTS,
    ToolbarAction,
    ToolbarConfig,
    ToolbarLayout,
)

pytestmark = pytest.mark.integration

TEST_USER_ID = 12345
TEST_THREAD_ID = 42
TEST_WINDOW_ID = "@5"

_MOD_CB = "ccgram.handlers.toolbar.toolbar_callbacks"
_MOD_KB = "ccgram.handlers.toolbar.toolbar_keyboard"


@pytest.fixture(autouse=True)
def _reset_toolbar_config():
    reload_toolbar_config()
    yield
    reload_toolbar_config()


def _register(application: Application) -> None:
    load_handlers()
    application.add_handler(CallbackQueryHandler(callback_dispatch))


@pytest.fixture
async def app(make_ptb_app) -> Application:
    """Real PTB Application with the callback_registry dispatch installed."""
    return await make_ptb_app(_register)


@contextmanager
def _toolbar_env(cfg: ToolbarConfig | None = None):
    """Patch the toolbar's collaborators and yield (tmux, answer) mocks.

    Every dispatch test needs the same three stand-ins — ownership check,
    multiplexer, and ``CallbackQuery.answer`` — plus, for custom-config tests,
    the same config in both the keyboard builder and the callback module.
    """
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch(f"{_MOD_CB}.user_owns_window", return_value=True))
        mock_tmux = enter(patch(f"{_MOD_CB}.tmux_manager"))
        mock_answer = enter(
            patch.object(CallbackQuery, "answer", new_callable=AsyncMock)
        )
        if cfg is not None:
            enter(patch(f"{_MOD_KB}.get_toolbar_config", return_value=cfg))
            enter(patch(f"{_MOD_CB}.get_toolbar_config", return_value=cfg))
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id=TEST_WINDOW_ID)
        )
        mock_tmux.send_keys = AsyncMock()
        yield mock_tmux, mock_answer


class TestKeyboardBuild:
    @pytest.mark.parametrize(
        "provider,expected_rows",
        [
            ("claude", 4),
            ("codex", 4),
            ("gemini", 4),
            ("pi", 4),
            ("shell", 3),
        ],
    )
    def test_default_keyboard_for_each_provider(
        self, provider: str, expected_rows: int
    ) -> None:
        kb = build_toolbar_keyboard(TEST_WINDOW_ID, provider)
        assert len(kb.inline_keyboard) == expected_rows
        for row in kb.inline_keyboard:
            assert 1 <= len(row) <= 8
            for btn in row:
                cb = btn.callback_data
                assert isinstance(cb, str)
                assert cb.startswith(CB_TOOLBAR)
                assert TEST_WINDOW_ID in cb

    def test_callback_data_under_64_bytes(self) -> None:
        kb = build_toolbar_keyboard("@99999", "claude")
        for row in kb.inline_keyboard:
            for btn in row:
                cb = btn.callback_data
                assert isinstance(cb, str)
                assert len(cb.encode("utf-8")) <= 64


class TestDispatchRoundTrip:
    @pytest.mark.parametrize(
        ("action_name", "expected_key", "expected_literal"),
        [
            ("esc", "Escape", False),
            ("enter", "Enter", False),
            ("tab", "Tab", False),
            ("eof", "C-d", False),
            ("susp", "C-z", False),
            ("ctrlc", "C-c", False),
            ("mode", "\x1b[Z", True),
        ],
    )
    async def test_key_action_dispatched_to_send_keys(
        self,
        app: Application,
        make_callback_update,
        action_name: str,
        expected_key: str,
        expected_literal: bool,
    ) -> None:
        update = make_callback_update(f"tb:{TEST_WINDOW_ID}:{action_name}", bot=app.bot)
        with (
            _toolbar_env() as (mock_tmux, mock_answer),
            patch(
                f"{_MOD_CB}.refresh_button_label",
                new=AsyncMock(return_value="Edit"),
            ),
        ):
            await app.process_update(update)

        mock_tmux.send_keys.assert_awaited_once_with(
            TEST_WINDOW_ID, expected_key, enter=False, literal=expected_literal
        )
        mock_answer.assert_awaited()

    async def test_text_action_sends_payload_to_window(
        self, app: Application, make_callback_update
    ) -> None:
        clear = ToolbarAction(
            name="clear",
            emoji="\U0001f9f9",
            text="Clear",
            action_type="text",
            payload="/clear",
        )
        cfg = ToolbarConfig(
            layouts=dict(DEFAULT_LAYOUTS),
            actions={**BUILTIN_ACTIONS, "clear": clear},
        )
        update = make_callback_update(f"tb:{TEST_WINDOW_ID}:clear", bot=app.bot)
        with (
            _toolbar_env(cfg),
            patch(
                f"{_MOD_CB}.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            await app.process_update(update)

        mock_send.assert_awaited_once_with(
            TEST_USER_ID, TEST_WINDOW_ID, TEST_THREAD_ID, "/clear", ANY
        )

    async def test_dismiss_deletes_message(
        self, app: Application, make_callback_update
    ) -> None:
        update = make_callback_update(f"tb:{TEST_WINDOW_ID}:close", bot=app.bot)
        with (
            _toolbar_env(),
            patch.object(
                CallbackQuery, "delete_message", new_callable=AsyncMock
            ) as mock_delete,
        ):
            await app.process_update(update)

        mock_delete.assert_awaited_once()

    async def test_unknown_action_alerts_user(
        self, app: Application, make_callback_update
    ) -> None:
        update = make_callback_update(f"tb:{TEST_WINDOW_ID}:nothereyet", bot=app.bot)
        with _toolbar_env() as (_tmux, mock_answer):
            await app.process_update(update)

        mock_answer.assert_awaited_once()
        args, kwargs = mock_answer.call_args
        assert "nothereyet" in args[0]
        assert kwargs.get("show_alert") is True


class TestCustomConfigDispatch:
    async def test_custom_layout_renders_and_dispatches(
        self, app: Application, make_callback_update
    ) -> None:
        summary = ToolbarAction(
            name="summary",
            emoji="📝",
            text="Sum",
            action_type="text",
            payload="/summary",
        )
        custom_layout = ToolbarLayout(
            style="text",
            buttons=(("screen", "summary"), ("close",)),
        )
        cfg = ToolbarConfig(
            layouts={**DEFAULT_LAYOUTS, "claude": custom_layout},
            actions={**BUILTIN_ACTIONS, "summary": summary},
        )

        with patch(f"{_MOD_KB}.get_toolbar_config", return_value=cfg):
            kb = build_toolbar_keyboard(TEST_WINDOW_ID, "claude")
        assert len(kb.inline_keyboard) == 2
        assert kb.inline_keyboard[0][0].text == "Screen"
        assert kb.inline_keyboard[0][1].text == "Sum"
        assert kb.inline_keyboard[1][0].text == "Close"

        update = make_callback_update(f"tb:{TEST_WINDOW_ID}:summary", bot=app.bot)
        with (
            _toolbar_env(cfg),
            patch(
                f"{_MOD_CB}.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            await app.process_update(update)

        mock_send.assert_awaited_once_with(
            TEST_USER_ID, TEST_WINDOW_ID, TEST_THREAD_ID, "/summary", ANY
        )

    async def test_user_config_overrides_builtin_action(
        self, app: Application, make_callback_update
    ) -> None:
        custom_mode = ToolbarAction(
            name="mode",
            emoji="🆕",
            text="Mode",
            action_type="key",
            payload="C-x",  # different key than the default \x1b[Z
            literal=False,
            read_state=False,
        )
        cfg = ToolbarConfig(
            layouts=dict(DEFAULT_LAYOUTS),
            actions={**BUILTIN_ACTIONS, "mode": custom_mode},
        )
        update = make_callback_update(f"tb:{TEST_WINDOW_ID}:mode", bot=app.bot)
        with _toolbar_env(cfg) as (mock_tmux, _answer):
            await app.process_update(update)

        mock_tmux.send_keys.assert_awaited_once_with(
            TEST_WINDOW_ID, "C-x", enter=False, literal=False
        )
