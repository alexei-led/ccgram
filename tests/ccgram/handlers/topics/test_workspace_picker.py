"""Tests for Task 9: workspace picker step in the /new flow.

Covers:
- herdr backend (native_agent_status=True): workspace picker shown when workspaces exist
- herdr backend: fall-through to provider picker when list_workspaces returns []
- CB_WS_SELECT stores chosen workspace_id via index into cached list
- CB_WS_SKIP clears pending workspace id and goes to provider picker
- workspace_id threaded to create_window
- tmux backend (native_agent_status=False): no picker shown (regression)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.callback_data import CB_WS_SELECT, CB_WS_SKIP
from ccgram.handlers.topics.directory_callbacks import (
    _handle_workspace_callback,
    _show_workspace_picker_or_provider,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_WORKSPACE_ID,
    PENDING_WORKSPACES,
)
from ccgram.multiplexer.base import MultiplexerCapabilities, WorkspaceRef
from ccgram.handlers.topics.directory_browser import BROWSE_PATH_KEY


# ── Fixtures & helpers ─────────────────────────────────────────────────


def _capabilities(native_agent_status: bool) -> MultiplexerCapabilities:
    return MultiplexerCapabilities(
        name="herdr" if native_agent_status else "tmux",
        ids_stable_across_restart=not native_agent_status,
        exposes_pane_tty=not native_agent_status,
        native_agent_status=native_agent_status,
        read_max_lines=1000 if native_agent_status else None,
        self_identify_env="HERDR_PANE_ID" if native_agent_status else "TMUX_PANE",
        supports_event_stream=native_agent_status,
    )


def _make_query() -> AsyncMock:
    query = AsyncMock()
    query.answer = AsyncMock()
    query.message = MagicMock()
    query.message.chat.type = "supergroup"
    query.message.chat.id = -100999
    return query


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot = AsyncMock()
    ctx.bot.edit_forum_topic = AsyncMock()
    return ctx


def _make_update(thread_id: int = 42) -> MagicMock:
    update = MagicMock()
    update.message = None  # force get_thread_id to use callback_query path
    update.callback_query = MagicMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.message_thread_id = thread_id
    return update


_WORKSPACES = [
    WorkspaceRef(workspace_id="ws1", label="my-project", cwd="/home/user/project"),
    WorkspaceRef(workspace_id="ws2", label="other", cwd="/home/user/other"),
]


# ── _show_workspace_picker_or_provider ────────────────────────────────


class TestShowWorkspacePickerOrProvider:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.tmux_manager")
    async def test_herdr_with_workspaces_shows_picker(
        self, mock_mux: MagicMock, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        """On herdr with workspaces, workspace picker is shown."""
        mock_mux.capabilities = _capabilities(native_agent_status=True)
        mock_mux.list_workspaces = AsyncMock(return_value=_WORKSPACES)

        user_data: dict = {}
        context = _make_context(user_data)
        await _show_workspace_picker_or_provider(_make_query(), str(tmp_path), context)

        text = mock_edit.call_args[0][1]
        assert "Select Workspace" in text
        assert user_data[PENDING_WORKSPACES] == [
            ("ws1", "my-project", "/home/user/project"),
            ("ws2", "other", "/home/user/other"),
        ]

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.tmux_manager")
    async def test_herdr_empty_workspaces_shows_provider_picker(
        self, mock_mux: MagicMock, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        """On herdr with no workspaces, fall through to provider picker."""
        mock_mux.capabilities = _capabilities(native_agent_status=True)
        mock_mux.list_workspaces = AsyncMock(return_value=[])

        context = _make_context()
        await _show_workspace_picker_or_provider(_make_query(), str(tmp_path), context)

        text = mock_edit.call_args[0][1]
        assert "Select Provider" in text

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.tmux_manager")
    async def test_tmux_skips_workspace_picker(
        self, mock_mux: MagicMock, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        """On tmux (native_agent_status=False), workspace picker is never shown."""
        mock_mux.capabilities = _capabilities(native_agent_status=False)

        context = _make_context()
        await _show_workspace_picker_or_provider(_make_query(), str(tmp_path), context)

        text = mock_edit.call_args[0][1]
        assert "Select Provider" in text
        # list_workspaces must not be called on tmux path
        mock_mux.list_workspaces.assert_not_called()


# ── _handle_workspace_callback ────────────────────────────────────────


class TestHandleWorkspaceCallback:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_select_stores_workspace_id_and_shows_provider(
        self, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        """CB_WS_SELECT<idx> stores workspace_id and shows provider picker."""
        ws_triples = [
            ("ws1", "my-project", "/home/user/project"),
            ("ws2", "other", "/home/user/other"),
        ]
        user_data = {
            PENDING_THREAD_ID: 42,
            BROWSE_PATH_KEY: str(tmp_path),
            PENDING_WORKSPACES: ws_triples,
        }
        context = _make_context(user_data)

        await _handle_workspace_callback(
            _make_query(), f"{CB_WS_SELECT}1", _make_update(42), context
        )

        assert user_data[PENDING_WORKSPACE_ID] == "ws2"
        text = mock_edit.call_args[0][1]
        assert "Select Provider" in text

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_skip_clears_workspace_id_and_shows_provider(
        self, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        """CB_WS_SKIP clears any stale workspace id and shows provider picker."""
        user_data = {
            PENDING_THREAD_ID: 42,
            BROWSE_PATH_KEY: str(tmp_path),
            PENDING_WORKSPACE_ID: "stale-ws",
        }
        context = _make_context(user_data)

        await _handle_workspace_callback(
            _make_query(), CB_WS_SKIP, _make_update(42), context
        )

        assert PENDING_WORKSPACE_ID not in user_data
        text = mock_edit.call_args[0][1]
        assert "Select Provider" in text

    async def test_stale_flow_reset_fails_closed(self, tmp_path: Path) -> None:
        """No PENDING_THREAD_ID → stale guard, alert shown, no edit."""
        query = _make_query()
        context = _make_context({BROWSE_PATH_KEY: str(tmp_path)})

        await _handle_workspace_callback(query, CB_WS_SKIP, _make_update(42), context)

        query.answer.assert_awaited_once()
        call_kwargs = query.answer.call_args[1]
        assert call_kwargs.get("show_alert") is True

    async def test_topic_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """Thread mismatch → stale guard, alert shown."""
        query = _make_query()
        user_data = {PENDING_THREAD_ID: 99, BROWSE_PATH_KEY: str(tmp_path)}
        context = _make_context(user_data)

        await _handle_workspace_callback(
            query,
            CB_WS_SKIP,
            _make_update(42),
            context,  # thread 42 ≠ pending 99
        )

        query.answer.assert_awaited_once()
        assert query.answer.call_args[1].get("show_alert") is True

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    async def test_out_of_range_index_shows_error(
        self, mock_edit: AsyncMock, tmp_path: Path
    ) -> None:
        """Out-of-range index shows error, does not crash."""
        user_data = {
            PENDING_THREAD_ID: 42,
            BROWSE_PATH_KEY: str(tmp_path),
            PENDING_WORKSPACES: [("ws1", "label", "/cwd")],
        }
        context = _make_context(user_data)

        await _handle_workspace_callback(
            _make_query(), f"{CB_WS_SELECT}5", _make_update(42), context
        )

        text = mock_edit.call_args[0][1]
        assert "❌" in text


# ── workspace_id threaded to create_window ────────────────────────────


class TestWorkspaceIdThreaded:
    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.tmux_manager")
    @patch("ccgram.handlers.topics.directory_callbacks.session_manager")
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    @patch("ccgram.handlers.topics.directory_callbacks.topic_orchestration")
    @patch("ccgram.handlers.topics.directory_callbacks.user_preferences")
    @patch("ccgram.handlers.topics.directory_callbacks.session_map_sync")
    async def test_chosen_workspace_id_passed_to_create_window(
        self,
        mock_sms: MagicMock,
        mock_prefs: MagicMock,
        mock_orch: MagicMock,
        mock_tr: MagicMock,
        mock_sm: MagicMock,
        mock_mux: MagicMock,
        mock_edit: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """PENDING_WORKSPACE_ID is forwarded as workspace_id= to create_window."""
        from ccgram.handlers.topics.directory_callbacks import _create_window_and_bind

        mock_mux.create_window = AsyncMock(return_value=(True, "ok", "my-tab", "w1:t1"))
        mock_mux.stamp_pane_title = AsyncMock()
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.resolve_chat_id.return_value = -100999
        mock_sm.set_window_provider = MagicMock()
        mock_sm.set_window_origin = MagicMock()
        mock_sm.set_window_cwd = MagicMock()
        mock_sm.set_window_approval_mode = MagicMock()
        mock_sms.wait_for_session_map_entry = AsyncMock()

        # Patch provider resolution + capabilities
        with (
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
            patch(
                "ccgram.handlers.topics.directory_callbacks.provider_registry"
            ) as mock_reg,
        ):
            mock_caps = MagicMock()
            mock_caps.chat_first_command_path = False
            mock_caps.has_yolo_confirmation = False
            mock_caps.supports_hook = False
            mock_reg.get.return_value.capabilities = mock_caps

            user_data = {
                PENDING_THREAD_ID: 42,
                PENDING_WORKSPACE_ID: "ws-chosen",
            }
            query = _make_query()
            context = _make_context(user_data)

            await _create_window_and_bind(
                query, 100, str(tmp_path), "claude", "normal", context
            )

        mock_mux.create_window.assert_awaited_once()
        call_kwargs = mock_mux.create_window.call_args[1]
        assert call_kwargs.get("workspace_id") == "ws-chosen"

    @patch(
        "ccgram.handlers.topics.directory_callbacks.safe_edit", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.topics.directory_callbacks.tmux_manager")
    @patch("ccgram.handlers.topics.directory_callbacks.session_manager")
    @patch("ccgram.handlers.topics.directory_callbacks.thread_router")
    @patch("ccgram.handlers.topics.directory_callbacks.topic_orchestration")
    @patch("ccgram.handlers.topics.directory_callbacks.user_preferences")
    @patch("ccgram.handlers.topics.directory_callbacks.session_map_sync")
    async def test_no_workspace_id_passes_none(
        self,
        mock_sms: MagicMock,
        mock_prefs: MagicMock,
        mock_orch: MagicMock,
        mock_tr: MagicMock,
        mock_sm: MagicMock,
        mock_mux: MagicMock,
        mock_edit: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """When no workspace was chosen, workspace_id=None is passed (auto-resolve)."""
        from ccgram.handlers.topics.directory_callbacks import _create_window_and_bind

        mock_mux.create_window = AsyncMock(return_value=(True, "ok", "my-tab", "w1:t1"))
        mock_mux.stamp_pane_title = AsyncMock()
        mock_tr.get_window_for_thread.return_value = None
        mock_tr.resolve_chat_id.return_value = -100999
        mock_sm.set_window_provider = MagicMock()
        mock_sm.set_window_origin = MagicMock()
        mock_sm.set_window_cwd = MagicMock()
        mock_sm.set_window_approval_mode = MagicMock()
        mock_sms.wait_for_session_map_entry = AsyncMock()

        with (
            patch("ccgram.providers.resolve_launch_command", return_value="claude"),
            patch(
                "ccgram.handlers.topics.directory_callbacks.provider_registry"
            ) as mock_reg,
        ):
            mock_caps = MagicMock()
            mock_caps.chat_first_command_path = False
            mock_caps.has_yolo_confirmation = False
            mock_caps.supports_hook = False
            mock_reg.get.return_value.capabilities = mock_caps

            # No PENDING_WORKSPACE_ID in user_data
            user_data = {PENDING_THREAD_ID: 42}
            query = _make_query()
            context = _make_context(user_data)

            await _create_window_and_bind(
                query, 100, str(tmp_path), "claude", "normal", context
            )

        call_kwargs = mock_mux.create_window.call_args[1]
        assert call_kwargs.get("workspace_id") is None
