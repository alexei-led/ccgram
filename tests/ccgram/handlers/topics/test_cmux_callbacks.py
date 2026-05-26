"""Tests for the /cmux workspace picker (Task 4)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes

from ccgram.handlers.callback_data import CB_CMUX_BIND, CB_CMUX_CANCEL, CB_CMUX_LIST
from ccgram.handlers.topics.cmux_callbacks import (
    CMUX_WORKSPACES_KEY,
    _dispatch,
    build_cmux_picker,
    cmux_command,
)
from ccgram.terminal_backends.base import (
    BACKEND_CMUX,
    TerminalBackendUnavailableError,
    TerminalUnit,
    TerminalUnitRef,
)
from ccgram.terminal_backends.config import TerminalBackendConfig
from ccgram.terminal_backends.router import reset_router_for_testing


def _unit(workspace_id: str, **overrides: Any) -> TerminalUnit:
    return TerminalUnit(
        ref=TerminalUnitRef(backend=BACKEND_CMUX, unit_id=workspace_id),
        title=overrides.pop("title", workspace_id),
        cwd=overrides.pop("cwd", "/repo"),
        provider_name=overrides.pop("provider_name", "claude"),
        supports_capture=overrides.pop("supports_capture", True),
        supports_send_text=overrides.pop("supports_send_text", True),
        supports_send_key=overrides.pop("supports_send_key", True),
    )


def _enabled_config() -> TerminalBackendConfig:
    from pathlib import Path

    return TerminalBackendConfig(
        cmux_enabled=True,
        cmux_sidecar_socket=Path("/tmp/cmux-sidecar.sock"),
    )


def _disabled_config() -> TerminalBackendConfig:
    return TerminalBackendConfig()


@pytest.fixture(autouse=True)
def _isolated_router():
    reset_router_for_testing()
    yield
    reset_router_for_testing()


def _make_query_update_context(
    thread_id: int = 42,
    user_data: dict | None = None,
    chat_id: int = -100123,
) -> tuple[AsyncMock, MagicMock, MagicMock]:
    query = AsyncMock(spec=CallbackQuery)
    query.answer = AsyncMock()
    msg = MagicMock()
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.id = chat_id
    query.message = msg

    update = MagicMock(spec=Update)
    update.message = None
    update.callback_query = query
    update.callback_query.message = msg
    update.effective_user = MagicMock(id=100)

    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.user_data = user_data if user_data is not None else {}
    context.bot = AsyncMock()
    return query, update, context


class TestBuildCmuxPicker:
    def test_renders_units_with_token_callbacks(self) -> None:
        units = [_unit("ws-a"), _unit("ws-b", title="beta", provider_name="codex")]
        text, keyboard = build_cmux_picker(units, picker_id="pick")
        assert "ws-a" in text
        assert "beta" in text
        callbacks = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert f"{CB_CMUX_BIND}pick:0" in callbacks
        assert f"{CB_CMUX_BIND}pick:1" in callbacks
        assert CB_CMUX_LIST in callbacks
        assert CB_CMUX_CANCEL in callbacks

    def test_no_terminal_surface_unit_has_no_bind_button(self) -> None:
        units = [_unit("ws-a", supports_send_text=False)]
        text, keyboard = build_cmux_picker(units, picker_id="pick")
        callbacks = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert "no terminal surface" in text
        assert f"{CB_CMUX_BIND}pick:0" not in callbacks

    def test_empty_units_keeps_refresh_and_cancel_only(self) -> None:
        text, keyboard = build_cmux_picker([])
        assert "No workspaces" in text
        callbacks = [
            btn.callback_data
            for row in keyboard.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
        ]
        assert callbacks == [CB_CMUX_LIST, CB_CMUX_CANCEL]


class TestCmuxCommandDisabledBackend:
    async def test_disabled_config_replies_with_setup_hint(self) -> None:
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.message_thread_id = 42
        update.message.is_topic_message = True
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.user_data = {}

        with (
            patch(
                "ccgram.handlers.topics.cmux_callbacks._load_backend_config",
                return_value=_disabled_config(),
            ),
            patch("ccgram.handlers.topics.cmux_callbacks.config") as mock_cfg,
            patch("ccgram.handlers.topics.cmux_callbacks.safe_reply") as mock_reply,
        ):
            mock_cfg.is_user_allowed.return_value = True
            await cmux_command(update, context)

        mock_reply.assert_awaited_once()
        body = mock_reply.call_args[0][1]
        assert "disabled" in body.lower()


class TestCmuxCommandBackendNotRegistered:
    async def test_router_missing_cmux_returns_hint(self) -> None:
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.message_thread_id = 42
        update.message.is_topic_message = True
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.user_data = {}

        with (
            patch(
                "ccgram.handlers.topics.cmux_callbacks._load_backend_config",
                return_value=_enabled_config(),
            ),
            patch("ccgram.handlers.topics.cmux_callbacks.config") as mock_cfg,
            patch("ccgram.handlers.topics.cmux_callbacks.safe_reply") as mock_reply,
        ):
            mock_cfg.is_user_allowed.return_value = True
            await cmux_command(update, context)

        body = mock_reply.call_args[0][1]
        assert "not wired" in body.lower()


class TestCmuxCommandSidecarUnavailable:
    async def test_backend_raises_unavailable(self) -> None:
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.message_thread_id = 42
        update.message.is_topic_message = True
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.user_data = {}

        fake_backend = MagicMock()
        fake_backend.name = BACKEND_CMUX
        fake_backend.list_units = AsyncMock(
            side_effect=TerminalBackendUnavailableError("socket gone")
        )

        from ccgram.terminal_backends.router import get_router

        get_router().register(fake_backend)

        with (
            patch(
                "ccgram.handlers.topics.cmux_callbacks._load_backend_config",
                return_value=_enabled_config(),
            ),
            patch("ccgram.handlers.topics.cmux_callbacks.config") as mock_cfg,
            patch("ccgram.handlers.topics.cmux_callbacks.safe_reply") as mock_reply,
        ):
            mock_cfg.is_user_allowed.return_value = True
            await cmux_command(update, context)

        body = mock_reply.call_args[0][1]
        assert "cmux sidecar error" in body.lower()


class TestCmuxCommandHappyPath:
    async def test_lists_workspaces_and_stores_state(self) -> None:
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.message_thread_id = 42
        update.message.is_topic_message = True
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.user_data = {}

        fake_backend = MagicMock()
        fake_backend.name = BACKEND_CMUX
        units = [_unit("ws-a"), _unit("ws-b", title="beta")]
        fake_backend.list_units = AsyncMock(return_value=units)

        from ccgram.terminal_backends.router import get_router

        get_router().register(fake_backend)

        with (
            patch(
                "ccgram.handlers.topics.cmux_callbacks._load_backend_config",
                return_value=_enabled_config(),
            ),
            patch("ccgram.handlers.topics.cmux_callbacks.config") as mock_cfg,
            patch("ccgram.handlers.topics.cmux_callbacks.safe_reply") as mock_reply,
        ):
            mock_cfg.is_user_allowed.return_value = True
            await cmux_command(update, context)

        store = context.user_data[CMUX_WORKSPACES_KEY]
        assert len(store) == 1
        entry = next(iter(store.values()))
        assert entry["thread_id"] == 42
        assert entry["units"] == units
        body = mock_reply.call_args[0][1]
        assert "ws-a" in body
        assert "beta" in body

    async def test_outside_topic_replies_without_listing(self) -> None:
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.message_thread_id = None
        update.message.is_topic_message = False
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.user_data = {}

        with (
            patch("ccgram.handlers.topics.cmux_callbacks.config") as mock_cfg,
            patch("ccgram.handlers.topics.cmux_callbacks.safe_reply") as mock_reply,
            patch(
                "ccgram.handlers.topics.cmux_callbacks._list_cmux_workspaces"
            ) as mock_list,
        ):
            mock_cfg.is_user_allowed.return_value = True
            await cmux_command(update, context)

        mock_list.assert_not_called()
        body = mock_reply.call_args[0][1]
        assert "topic" in body.lower()


class TestCmuxBindCallback:
    async def test_bind_persists_identity_and_renames_topic(self) -> None:
        # Trigger SessionManager construction so window_store proxy is wired.
        import ccgram.session  # noqa: F401
        from ccgram.window_state_store import get_window_store

        store = get_window_store()
        store.window_states.clear()

        units = [_unit("ws-a", title="alpha", cwd="/repo/a")]
        user_data = {CMUX_WORKSPACES_KEY: {"pick": {"thread_id": 42, "units": units}}}
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = f"{CB_CMUX_BIND}pick:0"

        with (
            patch("ccgram.handlers.topics.cmux_callbacks.thread_router") as mock_tr,
            patch("ccgram.handlers.topics.cmux_callbacks.safe_edit") as mock_edit,
            patch(
                "ccgram.handlers.topics.cmux_callbacks.PTBTelegramClient"
            ) as mock_client_cls,
        ):
            mock_tr.resolve_chat_id.return_value = -100123
            client = AsyncMock()
            mock_client_cls.return_value = client

            await _dispatch(update, context)

        bound = store.window_states["cmux:ws-a"]
        assert bound.terminal_backend == BACKEND_CMUX
        assert bound.terminal_unit_id == "ws-a"
        assert bound.cwd == "/repo/a"
        assert bound.provider_name == "claude"
        assert bound.window_name == "alpha"

        mock_tr.bind_thread.assert_called_once_with(
            100, 42, "cmux:ws-a", window_name="alpha"
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -100123)
        mock_edit.assert_awaited_once()
        assert "alpha" in mock_edit.call_args[0][1]
        assert CMUX_WORKSPACES_KEY not in context.user_data

    async def test_bind_invalid_index_alerts_user(self) -> None:
        user_data = {CMUX_WORKSPACES_KEY: {"pick": {"thread_id": 42, "units": []}}}
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = f"{CB_CMUX_BIND}pick:0"

        await _dispatch(update, context)
        query.answer.assert_awaited()
        assert "list changed" in query.answer.call_args[0][0].lower()

    async def test_bind_non_integer_index_rejected(self) -> None:
        user_data = {
            CMUX_WORKSPACES_KEY: {"pick": {"thread_id": 42, "units": [_unit("ws-a")]}}
        }
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = f"{CB_CMUX_BIND}pick:abc"

        await _dispatch(update, context)
        query.answer.assert_awaited()
        assert "invalid" in query.answer.call_args[0][0].lower()

    async def test_bind_stale_topic_blocked(self) -> None:
        user_data = {
            CMUX_WORKSPACES_KEY: {"pick": {"thread_id": 999, "units": [_unit("ws-a")]}}
        }
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = f"{CB_CMUX_BIND}pick:0"

        await _dispatch(update, context)
        query.answer.assert_awaited()
        body = query.answer.call_args[0][0].lower()
        assert "stale" in body or "topic mismatch" in body

    async def test_bind_no_terminal_surface_rejected(self) -> None:
        units = [_unit("ws-a", supports_send_text=False)]
        user_data = {CMUX_WORKSPACES_KEY: {"pick": {"thread_id": 42, "units": units}}}
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = f"{CB_CMUX_BIND}pick:0"

        await _dispatch(update, context)

        query.answer.assert_awaited()
        assert "no terminal" in query.answer.call_args[0][0].lower()

    async def test_old_picker_token_keeps_original_workspace_list(self) -> None:
        import ccgram.session  # noqa: F401
        from ccgram.window_state_store import get_window_store

        store = get_window_store()
        store.window_states.clear()
        user_data = {
            CMUX_WORKSPACES_KEY: {
                "old": {"thread_id": 42, "units": [_unit("ws-old", title="old")]},
                "new": {"thread_id": 42, "units": [_unit("ws-new", title="new")]},
            }
        }
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = f"{CB_CMUX_BIND}old:0"

        with (
            patch("ccgram.handlers.topics.cmux_callbacks.thread_router"),
            patch("ccgram.handlers.topics.cmux_callbacks.safe_edit"),
            patch(
                "ccgram.handlers.topics.cmux_callbacks.PTBTelegramClient",
                return_value=AsyncMock(),
            ),
        ):
            await _dispatch(update, context)

        assert "cmux:ws-old" in store.window_states
        assert "cmux:ws-new" not in store.window_states


class TestCmuxBindCallbackPrefixSelectorWiresDispatcher:
    """Smoke check that the @register decorator picks up the prefixes."""

    def test_dispatch_is_wired_for_known_prefixes(self) -> None:
        from ccgram.handlers.callback_registry import _registry

        assert CB_CMUX_LIST in _registry
        assert CB_CMUX_BIND in _registry
        assert CB_CMUX_CANCEL in _registry
        # All three route to the same dispatch func
        assert _registry[CB_CMUX_LIST] is _dispatch
        assert _registry[CB_CMUX_BIND] is _dispatch
        assert _registry[CB_CMUX_CANCEL] is _dispatch


class TestCmuxCancelCallback:
    async def test_cancel_clears_state(self) -> None:
        user_data = {
            CMUX_WORKSPACES_KEY: {"pick": {"thread_id": 42, "units": [_unit("ws-a")]}},
        }
        query, update, context = _make_query_update_context(user_data=user_data)
        query.data = CB_CMUX_CANCEL

        with patch("ccgram.handlers.topics.cmux_callbacks.safe_edit") as mock_edit:
            await _dispatch(update, context)

        mock_edit.assert_awaited_once()
        assert CMUX_WORKSPACES_KEY not in context.user_data
