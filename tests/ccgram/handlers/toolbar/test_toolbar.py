"""Tests for toolbar_callbacks (dispatch) and toolbar_keyboard (keyboard build + labels)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_TOOLBAR
from ccgram.handlers.toolbar.toolbar_callbacks import (
    _parse_callback_data,
    handle_toolbar_callback,
)
from ccgram.handlers.toolbar.toolbar_keyboard import (
    _clear_toolbar_labels,
    _get_action_label,
    _set_action_label,
    build_toolbar_keyboard,
    refresh_button_label,
    reload_toolbar_config,
    seed_button_states,
)
from ccgram.toolbar_config import (
    BUILTIN_ACTIONS,
    DEFAULT_LAYOUTS,
    ToolbarAction,
    ToolbarConfig,
    ToolbarLayout,
)


@pytest.fixture(autouse=True)
def _fresh_toolbar_config():
    reload_toolbar_config()
    yield
    reload_toolbar_config()


# ──────────────────────────────────────────────────────────────────────
# build_toolbar_keyboard
# ──────────────────────────────────────────────────────────────────────


class TestBuildToolbarKeyboard:
    @pytest.mark.parametrize("provider", ["claude", "codex", "gemini", "shell"])
    def test_default_grid_has_valid_shape(self, provider: str) -> None:
        kb = build_toolbar_keyboard("@5", provider)
        assert 3 <= len(kb.inline_keyboard) <= 4
        for row in kb.inline_keyboard:
            assert 1 <= len(row) <= 8

    @pytest.mark.parametrize("provider", ["claude", "codex", "gemini", "shell"])
    def test_callback_data_uses_single_prefix(self, provider: str) -> None:
        kb = build_toolbar_keyboard("@5", provider)
        for row in kb.inline_keyboard:
            for btn in row:
                cb = btn.callback_data
                assert isinstance(cb, str)
                assert cb.startswith(CB_TOOLBAR)
                assert ":@5:" in cb

    def test_unknown_provider_falls_back_to_claude(self) -> None:
        kb_default = build_toolbar_keyboard("@1", "claude")
        kb_unknown = build_toolbar_keyboard("@1", "aider")
        labels_default = [[b.text for b in row] for row in kb_default.inline_keyboard]
        labels_unknown = [[b.text for b in row] for row in kb_unknown.inline_keyboard]
        assert labels_default == labels_unknown

    def test_emoji_text_style_renders_emoji_and_text(self) -> None:
        kb = build_toolbar_keyboard("@1", "claude")
        first = kb.inline_keyboard[0][0]
        assert "Screen" in first.text
        assert "\U0001f4f7" in first.text


class TestBuildToolbarKeyboardCustom:
    def test_text_style_renders_text_only(self) -> None:
        custom_layout = ToolbarLayout(
            style="text",
            buttons=(("ctrlc", "esc"),),
        )
        custom_cfg = ToolbarConfig(
            layouts={"claude": custom_layout},
            actions=dict(BUILTIN_ACTIONS),
        )
        with patch(
            "ccgram.handlers.toolbar.toolbar_keyboard.get_toolbar_config",
            return_value=custom_cfg,
        ):
            kb = build_toolbar_keyboard("@7", "claude")
        assert kb.inline_keyboard[0][0].text == "Ctrl-C"
        assert kb.inline_keyboard[0][1].text == "Esc"

    def test_emoji_style_renders_emoji_only(self) -> None:
        custom_layout = ToolbarLayout(
            style="emoji",
            buttons=(("ctrlc",),),
        )
        custom_cfg = ToolbarConfig(
            layouts={"claude": custom_layout},
            actions=dict(BUILTIN_ACTIONS),
        )
        with patch(
            "ccgram.handlers.toolbar.toolbar_keyboard.get_toolbar_config",
            return_value=custom_cfg,
        ):
            kb = build_toolbar_keyboard("@7", "claude")
        assert kb.inline_keyboard[0][0].text == "\u23f9"


# ──────────────────────────────────────────────────────────────────────
# _parse_callback_data
# ──────────────────────────────────────────────────────────────────────


class TestParseCallbackData:
    def test_simple_window_id(self) -> None:
        assert _parse_callback_data("tb:@5:mode") == ("@5", "mode")

    def test_missing_prefix_returns_none(self) -> None:
        assert _parse_callback_data("foo:bar") is None

    def test_no_colon_returns_none(self) -> None:
        assert _parse_callback_data("tb:") is None


# ──────────────────────────────────────────────────────────────────────
# Dispatch — common test helpers
# ──────────────────────────────────────────────────────────────────────


def _make_query(data: str) -> AsyncMock:
    query = AsyncMock()
    query.data = data
    query.answer = AsyncMock()
    query.delete_message = AsyncMock()
    query.get_bot = MagicMock(return_value=MagicMock())
    return query


def _make_update_with_user(user_id: int = 100) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = MagicMock(id=-100, type="supergroup")
    update.effective_message = MagicMock(message_thread_id=42)
    return update


def _make_context() -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


# ──────────────────────────────────────────────────────────────────────
# Dispatch — key actions
# ──────────────────────────────────────────────────────────────────────


class TestDispatchKey:
    async def test_key_action_sends_tmux_key(self) -> None:
        query = _make_query("tb:@5:esc")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@5")
            )
            mock_tmux.send_keys = AsyncMock()
            await handle_toolbar_callback(query, 100, "tb:@5:esc", update, context)
        mock_tmux.send_keys.assert_awaited_once_with(
            "@5", "Escape", enter=False, literal=False
        )
        query.answer.assert_awaited_once()

    async def test_mode_action_uses_literal_true(self) -> None:
        query = _make_query("tb:@5:mode")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.refresh_button_label",
                new=AsyncMock(return_value="Edit"),
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@5")
            )
            mock_tmux.send_keys = AsyncMock()
            await handle_toolbar_callback(query, 100, "tb:@5:mode", update, context)
        mock_tmux.send_keys.assert_awaited_once_with(
            "@5", "\x1b[Z", enter=False, literal=True
        )
        query.answer.assert_awaited_once_with("\U0001f500 Edit")

    async def test_window_not_found_alerts(self) -> None:
        query = _make_query("tb:@5:esc")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            await handle_toolbar_callback(query, 100, "tb:@5:esc", update, context)
        query.answer.assert_awaited_once_with("Window not found", show_alert=True)


# ──────────────────────────────────────────────────────────────────────
# Dispatch — text actions
# ──────────────────────────────────────────────────────────────────────


class TestDispatchText:
    async def test_text_action_sends_with_enter_literal(self) -> None:
        clear_action = ToolbarAction(
            name="clear",
            emoji="\U0001f9f9",
            text="Clear",
            action_type="text",
            payload="/clear",
        )
        custom_cfg = ToolbarConfig(
            layouts=dict(DEFAULT_LAYOUTS),
            actions={**BUILTIN_ACTIONS, "clear": clear_action},
        )
        query = _make_query("tb:@5:clear")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.get_toolbar_config",
                return_value=custom_cfg,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@5")
            )
            mock_tmux.send_keys = AsyncMock()
            await handle_toolbar_callback(query, 100, "tb:@5:clear", update, context)
        mock_tmux.send_keys.assert_awaited_once_with(
            "@5", "/clear", enter=True, literal=True
        )
        query.answer.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# Dispatch — builtin actions
# ──────────────────────────────────────────────────────────────────────


class TestDispatchBuiltinCtrlc:
    async def test_sends_ctrl_c(self) -> None:
        query = _make_query("tb:@5:ctrlc")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@5")
            )
            mock_tmux.send_keys = AsyncMock()
            await handle_toolbar_callback(query, 100, "tb:@5:ctrlc", update, context)
        mock_tmux.send_keys.assert_awaited_once_with(
            "@5", "C-c", enter=False, literal=False
        )


class TestDispatchBuiltinDismiss:
    async def test_deletes_message(self) -> None:
        query = _make_query("tb:@5:close")
        update = _make_update_with_user()
        context = _make_context()
        with patch(
            "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
            return_value=True,
        ):
            await handle_toolbar_callback(query, 100, "tb:@5:close", update, context)
        query.delete_message.assert_awaited_once()


class TestDispatchBuiltinGetfile:
    async def test_no_cwd_alerts(self) -> None:
        query = _make_query("tb:@5:getfile")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch("ccgram.handlers.toolbar.toolbar_callbacks.view_window") as mock_view,
        ):
            mock_view.return_value = None
            await handle_toolbar_callback(query, 100, "tb:@5:getfile", update, context)
        query.answer.assert_awaited_once_with(
            "Working directory not available", show_alert=True
        )


class TestDispatchBuiltinLast:
    async def test_calls_send_last_reply(self) -> None:
        query = _make_query("tb:@5:last")
        update = _make_update_with_user()
        context = _make_context()
        mock_send_last = AsyncMock()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.thread_router"
            ) as mock_router,
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.telegram_client.PTBTelegramClient",
                return_value=MagicMock(),
            ),
            patch(
                "ccgram.handlers.last_reply.send_last_reply",
                mock_send_last,
            ),
        ):
            mock_router.resolve_chat_id.return_value = -100
            await handle_toolbar_callback(query, 100, "tb:@5:last", update, context)
        mock_send_last.assert_awaited_once()
        args = mock_send_last.call_args.args
        assert args[1] == -100
        assert args[2] == 42
        assert args[3] == "@5"
        query.answer.assert_awaited_once()

    async def test_no_user_context_alerts(self) -> None:
        query = _make_query("tb:@5:last")
        update = MagicMock()
        update.effective_user = None
        update.effective_chat = MagicMock(id=-100, type="supergroup")
        update.effective_message = MagicMock(message_thread_id=42)
        context = _make_context()
        with patch(
            "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
            return_value=True,
        ):
            await handle_toolbar_callback(query, 100, "tb:@5:last", update, context)
        query.answer.assert_awaited_once_with("No user context", show_alert=True)

    async def test_no_topic_alerts(self) -> None:
        query = _make_query("tb:@5:last")
        update = _make_update_with_user()
        context = _make_context()
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.get_thread_id",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.toolbar.toolbar_callbacks.thread_router"
            ) as mock_router,
        ):
            mock_router.resolve_chat_id.return_value = None
            await handle_toolbar_callback(query, 100, "tb:@5:last", update, context)
        query.answer.assert_awaited_once_with("Use in a topic", show_alert=True)


# ──────────────────────────────────────────────────────────────────────
# Dispatch — error paths
# ──────────────────────────────────────────────────────────────────────


class TestDispatchErrorPaths:
    async def test_bad_callback_data_format(self) -> None:
        query = _make_query("notb:foo")
        update = _make_update_with_user()
        context = _make_context()
        await handle_toolbar_callback(query, 100, "notb:foo", update, context)
        query.answer.assert_awaited_once_with("Bad toolbar callback", show_alert=True)

    async def test_ownership_rejection(self) -> None:
        query = _make_query("tb:@5:esc")
        update = _make_update_with_user()
        context = _make_context()
        with patch(
            "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
            return_value=False,
        ):
            await handle_toolbar_callback(query, 100, "tb:@5:esc", update, context)
        query.answer.assert_awaited_once_with("Not your session", show_alert=True)

    async def test_unknown_action_alerts(self) -> None:
        query = _make_query("tb:@5:doesnotexist")
        update = _make_update_with_user()
        context = _make_context()
        with patch(
            "ccgram.handlers.toolbar.toolbar_callbacks.user_owns_window",
            return_value=True,
        ):
            await handle_toolbar_callback(
                query, 100, "tb:@5:doesnotexist", update, context
            )
        call_args = query.answer.call_args
        assert call_args is not None
        assert "doesnotexist" in call_args.args[0]
        assert call_args.kwargs.get("show_alert") is True


# ──────────────────────────────────────────────────────────────────────
# State readback — button label updates via provider.scrape_current_mode
# ──────────────────────────────────────────────────────────────────────


class TestRefreshButtonLabel:
    async def test_mode_click_updates_button_label(self) -> None:
        query = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()
        mode_action = BUILTIN_ACTIONS["mode"]
        mock_provider = AsyncMock()
        mock_provider.scrape_current_mode = AsyncMock(return_value="Edit")
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_keyboard.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.toolbar.toolbar_keyboard.window_query") as mock_sm,
        ):
            mock_sm.view_window.return_value = MagicMock(provider_name="claude")
            result = await refresh_button_label(mode_action, query, "@5", delay=0)
        assert result == "Edit"
        assert _get_action_label("@5", "mode") == "Edit"
        query.edit_message_reply_markup.assert_awaited_once()

    async def test_keyboard_rebuild_shows_stored_label(self) -> None:
        reload_toolbar_config()
        _set_action_label("@9", "mode", "Plan")
        kb = build_toolbar_keyboard("@9", "claude")
        mode_btn = None
        for row in kb.inline_keyboard:
            for btn in row:
                cb = btn.callback_data
                if isinstance(cb, str) and cb.endswith(":mode"):
                    mode_btn = btn
                    break
        assert mode_btn is not None
        assert "Plan" in mode_btn.text
        assert "\U0001f500" in mode_btn.text

    async def test_seed_button_states_populates_mode_label(self) -> None:
        _clear_toolbar_labels("@111")
        mock_provider = AsyncMock()
        mock_provider.scrape_current_mode = AsyncMock(return_value="Plan")
        with patch(
            "ccgram.handlers.toolbar.toolbar_keyboard.get_provider_for_window",
            return_value=mock_provider,
        ):
            await seed_button_states("@111")
        assert _get_action_label("@111", "mode") == "Plan"

    async def test_seed_button_states_no_label_when_none(self) -> None:
        _clear_toolbar_labels("@222")
        mock_provider = AsyncMock()
        mock_provider.scrape_current_mode = AsyncMock(return_value=None)
        with patch(
            "ccgram.handlers.toolbar.toolbar_keyboard.get_provider_for_window",
            return_value=mock_provider,
        ):
            await seed_button_states("@222")
        assert _get_action_label("@222", "mode") is None

    async def test_capture_failure_returns_def_label(self) -> None:
        query = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()
        mock_provider = AsyncMock()
        mock_provider.scrape_current_mode = AsyncMock(return_value=None)
        with (
            patch(
                "ccgram.handlers.toolbar.toolbar_keyboard.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch("ccgram.handlers.toolbar.toolbar_keyboard.window_query") as mock_sm,
        ):
            mock_sm.view_window.return_value = MagicMock(provider_name="claude")
            result = await refresh_button_label(
                BUILTIN_ACTIONS["mode"], query, "@42", delay=0
            )
        assert result == "Def"
        assert _get_action_label("@42", "mode") == "Def"
