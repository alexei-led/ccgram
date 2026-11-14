"""Tests for window picker callback handlers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from telegram import Bot, CallbackQuery, Update
from telegram.ext import ContextTypes

from ccgram.handlers.callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from ccgram.handlers.topics.directory_browser import UNBOUND_WINDOWS_KEY
from ccgram.handlers.topics.new_command import new_command
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT
from ccgram.multiplexer.base import ForegroundInfo, WindowRef
from ccgram.handlers.topics.window_callbacks import (
    _forward_pending_text,
    _send_verified_shell_input,
    handle_window_callback,
)
from ccgram.session_protocol.backend import Target
from ccgram.session_protocol.models import Status, utc_now
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.store import ProtocolStore
from ccgram.session_protocol.telegram import TelegramSessionAdapter


@pytest.fixture(autouse=True)
def _isolated_window_state_owner():
    from ccgram.window_state_store import (
        WindowStateStore,
        get_window_store,
        install_window_store,
    )

    previous = get_window_store()
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
    )
    install_window_store(store)
    try:
        yield
    finally:
        install_window_store(previous)


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
    context.bot_data = {}
    return query, update, context


def _unbound_window(name: str, pane_current_command: str | None = None) -> MagicMock:
    window = MagicMock()
    window.window_name = name
    window.window_id = "@5"
    window.topic_markable = True
    window.topic_mark_owner = "owner:@5"
    window.topic_marked = True
    if pane_current_command is not None:
        window.pane_current_command = pane_current_command
    return window


def _protocol_adapter(tmp_path: Path, provider_name: str):
    target = Target(
        id="@5",
        title="my-shell",
        cwd="/workspace",
        provider=provider_name,
        conversation_id="",
        transcript=None,
        pane_id="%5",
        pane_title=provider_name,
        rows=24,
        cols=80,
        process_id=5,
        approval_mode="normal",
        status=Status("idle", "", "screen", utc_now()),
        host="test-host",
        foreground_argv=("bash",) if provider_name == "shell" else ("claude",),
    )
    backend = MagicMock()
    backend.inspect = AsyncMock(return_value=target)
    backend.prompt = AsyncMock(return_value=None)
    backend.send_text = AsyncMock(return_value=True)
    store = ProtocolStore(tmp_path / f"{provider_name}-delivery.sqlite3")
    service = LocalSessionService(backend, store)
    return TelegramSessionAdapter(service), backend, target, store


async def _reset_flow_with_new_command(context: MagicMock) -> None:
    """Run /new, which drops the picker state every stale-guard test needs gone."""
    update = MagicMock()
    update.effective_user = MagicMock(id=100)
    update.message = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.new_command.commands_command",
            new_callable=AsyncMock,
        ),
        patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ),
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
        patch(
            "ccgram.multiplexer.multiplexer",
            new=SimpleNamespace(
                find_window_by_id=AsyncMock(
                    side_effect=lambda wid: SimpleNamespace(
                        window_id=wid,
                        topic_markable=True,
                        topic_mark_owner="owner:" + wid,
                        topic_marked=True,
                    )
                ),
                set_topic_mark=AsyncMock(return_value=True),
            ),
        ),
        patch("ccgram.session.session_manager"),
        patch(f"{_MODULE}thread_router") as router,
        patch(_FIND_WINDOW, new_callable=AsyncMock, return_value=window) as find_window,
        patch(f"{_MODULE}safe_edit") as edit,
        patch(f"{_MODULE}format_topic_name_for_mode"),
        patch(f"{_MODULE}adapter_for", return_value=MagicMock()),
    ):
        router.resolve_chat_id.return_value = -100
        session.get_approval_mode.return_value = "normal"
        yield _BindMocks(
            session=session, router=router, find_window=find_window, edit=edit
        )


@pytest.mark.parametrize("mode", ["auto", "marked"])
@pytest.mark.parametrize("phase", ["before-selection", "during-failed-mark"])
@pytest.mark.parametrize(
    "initial_owner,markable,current_owner",
    [
        ("original-O", True, "original-O"),
        ("original-O", True, "replacement-P"),
        ("original-O", True, None),
        (None, True, None),
        (None, True, "replacement-P"),
        (None, False, None),
    ],
)
async def test_picker_keeps_original_native_owner_across_selection_awaits(
    monkeypatch, mode, phase, initial_owner, markable, current_owner
):
    from ccgram.config import config
    from ccgram.thread_router import ThreadRouter
    from ccgram.window_state_ports import lifecycle_state
    from ccgram.window_state_store import get_window_store

    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    store = get_window_store()
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=store.has_window)
    router.bind_thread(200, 77, "@5", chat_id=-100888)
    lifecycle_state.record_topic_mark_owner("@5", initial_owner)
    initial = WindowRef(
        "@5",
        "selected",
        "/p",
        pane_current_command="claude",
        topic_marked=True,
        topic_mark_owner=initial_owner,
        topic_markable=markable,
    )
    replacement = replace(initial, topic_mark_owner=current_owner)
    calls = 0
    current = initial

    async def find_window(_wid):
        nonlocal calls, current
        calls += 1
        if calls > 1 and phase == "before-selection":
            current = replacement
            lifecycle_state.observe_topic_mark_owner("@5", current_owner)
        return current

    async def native_mark(_wid, _desired, *, owner):
        nonlocal current
        if phase == "during-failed-mark":
            assert owner == initial_owner
            current = replacement
            lifecycle_state.observe_topic_mark_owner("@5", current_owner)
            return False
        return True

    native = SimpleNamespace(
        find_window_by_id=AsyncMock(side_effect=find_window),
        set_topic_mark=AsyncMock(side_effect=native_mark),
    )
    manager = SimpleNamespace(
        flush_state=lambda: None, set_window_provider=lambda *_args: None
    )
    query, update, context = _make_query_update_context(
        user_data={
            UNBOUND_WINDOWS_KEY: ["@5"],
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "keep this off replacement terminals",
        }
    )
    with (
        patch(f"{_MODULE}thread_router", router),
        patch(f"{_MODULE}session_manager", manager),
        patch("ccgram.session.session_manager", manager),
        patch(f"{_MODULE}tmux_manager", native),
        patch("ccgram.multiplexer.multiplexer", native),
        patch(
            "ccgram.providers.detect_provider_from_pane",
            AsyncMock(return_value="claude"),
        ),
        patch(f"{_MODULE}adapter_for", return_value=MagicMock()),
        patch(f"{_MODULE}safe_edit", AsyncMock()),
        patch(
            f"{_MODULE}send_telegram_to_window", AsyncMock(return_value=(True, ""))
        ) as sender,
    ):
        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

    if initial_owner is None:
        bound = not markable or mode == "auto"
    else:
        bound = current_owner == initial_owner and (
            phase == "before-selection" or mode == "auto"
        )
    assert router.get_window_for_chat_thread(-100999, 42) == ("@5" if bound else None)
    assert router.get_window_for_chat_thread(-100888, 77) == "@5"
    assert sender.await_count == int(bound)
    if not bound:
        assert (
            context.user_data[PENDING_THREAD_TEXT]
            == "keep this off replacement terminals"
        )
    if (
        initial_owner is not None
        and phase == "before-selection"
        and current_owner != initial_owner
    ):
        native.set_topic_mark.assert_not_awaited()
        assert not lifecycle_state.pending_topic_selections()
    if initial_owner is not None and phase == "during-failed-mark":
        assert [op.owner for op in lifecycle_state.pending_topic_selections()] == [
            "original-O"
        ]
        native.set_topic_mark.assert_awaited_once_with("@5", True, owner="original-O")

    if not markable or (initial_owner is None and mode == "marked"):
        native.set_topic_mark.assert_not_awaited()
        assert not lifecycle_state.pending_topic_selections()


@pytest.mark.parametrize("mode", ["auto", "marked"])
async def test_picker_native_selection_failure_gates_only_marked(mode, monkeypatch):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", mode, raising=False)
    query, update, context = _make_query_update_context(
        user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
    )
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@5",
                topic_markable=True,
                topic_mark_owner="owner:" + "@5",
                topic_marked=True,
            )
        ),
        set_topic_mark=AsyncMock(return_value=False),
    )
    with (
        _bind_env(_unbound_window("selected")) as m,
        patch("ccgram.multiplexer.multiplexer", native),
    ):
        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

    assert lifecycle_state.get_pending_topic_mark("@5") is True
    native.set_topic_mark.assert_awaited_once_with("@5", True, owner="owner:" + "@5")
    if mode == "marked":
        m.router.bind_thread.assert_not_called()
        assert context.user_data[UNBOUND_WINDOWS_KEY] == ["@5"]
        query.answer.assert_awaited_once()
        assert "Could not verify" in query.answer.call_args.args[0]
    else:
        m.router.bind_thread.assert_called_once()


async def test_auto_picker_cannot_bypass_unfinished_detach_clear(monkeypatch):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state
    from ccgram.window_state_store import WindowStateStore

    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
    )
    monkeypatch.setattr(lifecycle_state, "window_store", store)
    monkeypatch.setattr(config, "topic_discovery_mode", "auto", raising=False)
    lifecycle_state.set_topic_selection("@5", detached=True, pending=False, owner=None)
    query, update, context = _make_query_update_context(
        user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
    )
    with _bind_env(_unbound_window("selected")) as m:
        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)
    m.router.bind_thread.assert_not_called()
    assert lifecycle_state.is_topic_detach_pending("@5", owner=None)


async def test_auto_bind_never_uses_a_route_that_disappears_during_mark(monkeypatch):
    from ccgram.config import config

    monkeypatch.setattr(config, "topic_discovery_mode", "auto")
    query, update, context = _make_query_update_context(
        user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
    )
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    window_id="@5", topic_markable=True, topic_mark_owner="owner:@5"
                ),
                None,
            ]
        ),
        set_topic_mark=AsyncMock(return_value=False),
    )
    with (
        _bind_env(_unbound_window("selected")) as m,
        patch("ccgram.multiplexer.multiplexer", native),
    ):
        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)
    m.router.bind_thread.assert_not_called()
    assert "Session changed" in query.answer.call_args.args[0]


async def test_existing_provisioning_claim_blocks_bind_before_owner_capture():
    from ccgram.window_state_ports import lifecycle_state

    lifecycle_state.set_topic_selection(
        "@5", detached=False, pending=None, owner="original-owner"
    )
    query, update, context = _make_query_update_context(
        user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
    )
    with _bind_env(_unbound_window("replacement")) as m:
        m.router.has_target_provisioning.return_value = True
        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)
    assert lifecycle_state.get_topic_mark_owner("@5") == "original-owner"
    assert not lifecycle_state.pending_topic_selections()
    m.router.bind_thread.assert_not_called()


@pytest.mark.parametrize(
    "current_owner,binds",
    [
        pytest.param(None, False, id="captured-owner-lost"),
        pytest.param("owner:@5", True, id="stable-owner-mark-write-failed"),
    ],
)
async def test_auto_bind_rechecks_known_owner_with_real_router(
    monkeypatch, current_owner, binds
):
    from ccgram.config import config
    from ccgram.multiplexer.base import WindowRef
    from ccgram.thread_router import ThreadRouter
    from ccgram.window_state_store import get_window_store

    monkeypatch.setattr(config, "topic_discovery_mode", "auto")
    router = ThreadRouter(
        schedule_save=lambda: None,
        has_window_state=get_window_store().has_window,
    )
    query, update, context = _make_query_update_context(
        user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
    )
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            side_effect=[
                WindowRef(
                    "@5",
                    "captured",
                    "/tmp",
                    pane_current_command="claude",
                    topic_mark_owner="owner:@5",
                    topic_marked=False,
                ),
                WindowRef(
                    "@5",
                    "current",
                    "/tmp",
                    pane_current_command="claude",
                    topic_mark_owner=current_owner,
                    topic_marked=False,
                ),
            ]
        ),
        set_topic_mark=AsyncMock(return_value=False),
        foreground=AsyncMock(
            return_value=ForegroundInfo(
                pid=12345, pgid=12345, argv=["claude"], cwd="/tmp"
            )
        ),
    )
    with (
        _bind_env(_unbound_window("selected", "claude")),
        patch("ccgram.multiplexer.multiplexer", native),
        patch(f"{_MODULE}thread_router", router),
    ):
        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)
    assert router.get_window_for_thread(100, 42, -100999) == ("@5" if binds else None)
    native.set_topic_mark.assert_awaited_once_with("@5", True, owner="owner:@5")


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

    async def test_private_topic_binds_and_renames_with_regular_topic_api(self) -> None:
        query, update, context = _make_query_update_context(
            user_data={UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        )
        message = update.callback_query.message
        message.message_thread_id = 42
        message.is_topic_message = True
        message.chat.type = "private"
        message.chat.id = 100

        with _bind_env(_unbound_window("my-project")) as m:
            m.router.resolve_chat_id.return_value = 100
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        m.router.bind_thread.assert_called_once_with(
            100, 42, "@5", window_name="my-project", chat_id=100
        )
        context.bot.edit_forum_topic.assert_awaited_once()

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
                f"{_MODULE}adapter_for", return_value=MagicMock()
            ) as mock_adapter_for,
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        adapter = mock_adapter_for.return_value
        mock_send.assert_called_once_with(
            100, "@5", 42, "hello agent", -100999, adapter=adapter
        )
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
            assert mock_ensure.call_args.kwargs["adapter"] is not None
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
                f"{_MODULE}adapter_for", return_value=MagicMock()
            ) as mock_adapter_for,
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
            "adapter": mock_adapter_for.return_value,
            "is_existing_window": True,
            "shell_exit_detected": False,
            "chat_id": -100999,
        }

    async def test_tracked_agent_returned_to_shell_does_not_receive_pending_text(
        self, tmp_path: Path
    ) -> None:
        query, update, context = _make_query_update_context(
            user_data={
                UNBOUND_WINDOWS_KEY: ["@5"],
                PENDING_THREAD_ID: 42,
                PENDING_THREAD_TEXT: "do not send to a shell",
            }
        )
        adapter, backend, _, store = _protocol_adapter(tmp_path, "shell")
        window = _unbound_window("former-agent", "node")
        mux = MagicMock()
        mux.find_window_by_id = AsyncMock(return_value=window)
        mux.set_topic_mark = AsyncMock(return_value=True)
        mux.foreground = AsyncMock(
            side_effect=[
                ForegroundInfo(
                    pid=1,
                    pgid=101,
                    argv=["node", "/opt/claude"],
                    cwd="/workspace",
                ),
                ForegroundInfo(
                    pid=2,
                    pgid=102,
                    argv=["bash"],
                    cwd="/workspace",
                ),
            ]
        )
        claude_provider = MagicMock()
        claude_provider.capabilities.chat_first_command_path = False
        shell_provider = MagicMock()
        shell_provider.capabilities.chat_first_command_path = True

        def provider_for_window(_window_id: str, provider_name: str):
            return {"claude": claude_provider, "shell": shell_provider}[provider_name]

        with (
            _bind_env(window),
            patch(f"{_MODULE}window_query.get_window_provider", return_value="claude"),
            patch(f"{_MODULE}get_initial_provider_name", return_value="claude"),
            patch(
                "ccgram.window_state_ports.identity_state.get_initial_provider_name",
                return_value="claude",
            ),
            patch("ccgram.multiplexer.multiplexer", mux),
            patch("ccgram.providers.process_detection._pgid_cache", {}),
            patch(
                "ccgram.providers.get_provider_for_window",
                side_effect=provider_for_window,
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ),
            patch(f"{_MODULE}adapter_for", return_value=adapter),
            patch(f"{_MODULE}safe_send", new_callable=AsyncMock) as mock_safe_send,
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mux.foreground.assert_has_awaits([call("@5"), call("@5")])
        assert mux.foreground.await_count == 2
        backend.send_text.assert_not_awaited()
        mock_safe_send.assert_awaited_once()
        assert "recover the session" in mock_safe_send.call_args.args[2]
        query.answer.assert_awaited_with("Bound")
        store.close()

    async def test_verified_shell_binding_submits_pending_text_once(
        self, tmp_path: Path
    ) -> None:
        query, update, context = _make_query_update_context(
            user_data={
                UNBOUND_WINDOWS_KEY: ["@5"],
                PENDING_THREAD_ID: 42,
                PENDING_THREAD_TEXT: "ls -la",
            }
        )
        adapter, backend, target, store = _protocol_adapter(tmp_path, "shell")
        provider = MagicMock()
        provider.capabilities.chat_first_command_path = True

        with (
            _bind_env(_unbound_window("my-shell", "bash")),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ),
            patch(
                f"{_MODULE}get_initial_provider_name",
                return_value="",
            ),
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ),
            patch(f"{_MODULE}adapter_for", return_value=adapter),
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        backend.send_text.assert_awaited_once_with(
            target, "ls -la", enter=True, raw=True
        )
        assert PENDING_THREAD_TEXT not in context.user_data
        store.close()

    async def test_agent_binding_keeps_message_action(self, tmp_path: Path) -> None:
        query, update, context = _make_query_update_context(
            user_data={
                UNBOUND_WINDOWS_KEY: ["@5"],
                PENDING_THREAD_ID: 42,
                PENDING_THREAD_TEXT: "continue the task",
            }
        )
        adapter, backend, target, store = _protocol_adapter(tmp_path, "claude")
        provider = MagicMock()
        provider.capabilities.chat_first_command_path = False

        with (
            _bind_env(_unbound_window("agent-window", "claude")),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="claude",
            ),
            patch(f"{_MODULE}window_query.get_window_provider", return_value="claude"),
            patch(f"{_MODULE}get_initial_provider_name", return_value="claude"),
            patch(
                "ccgram.window_state_ports.identity_state.get_initial_provider_name",
                return_value="claude",
            ),
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
            patch(f"{_MODULE}adapter_for", return_value=adapter),
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        backend.send_text.assert_awaited_once_with(
            target, "continue the task", enter=True
        )
        store.close()

    @pytest.mark.parametrize(
        ("current_provider", "initial_provider", "submits"),
        [
            ("claude", "shell", True),
            ("shell", "", False),
            ("mystery", "mystery", False),
            (None, "", True),
        ],
        ids=[
            "explicit-shell-origin",
            "legacy-shell-state",
            "unknown-provider",
            "untracked-verified-shell",
        ],
    )
    async def test_shell_exit_policy_uses_captured_origin(
        self,
        tmp_path: Path,
        current_provider: str | None,
        initial_provider: str,
        submits: bool,
    ) -> None:
        query, update, context = _make_query_update_context(
            user_data={
                UNBOUND_WINDOWS_KEY: ["@5"],
                PENDING_THREAD_ID: 42,
                PENDING_THREAD_TEXT: "ls -la",
            }
        )
        adapter, backend, target, store = _protocol_adapter(tmp_path, "shell")
        provider = MagicMock()
        provider.capabilities.chat_first_command_path = True

        with (
            _bind_env(_unbound_window("my-shell", "bash")),
            patch(
                f"{_MODULE}window_query.get_window_provider",
                return_value=current_provider,
            ),
            patch(
                f"{_MODULE}get_initial_provider_name",
                return_value=initial_provider,
            ),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ),
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ),
            patch(f"{_MODULE}adapter_for", return_value=adapter),
            patch(f"{_MODULE}safe_send", new_callable=AsyncMock) as mock_safe_send,
        ):
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        if submits:
            backend.send_text.assert_awaited_once_with(
                target, "ls -la", enter=True, raw=True
            )
            mock_safe_send.assert_not_awaited()
        else:
            backend.send_text.assert_not_awaited()
            mock_safe_send.assert_awaited_once()
            assert "recover the session" in mock_safe_send.call_args.args[2]
        store.close()


class TestForwardPendingText:
    async def test_new_shell_window_routes_through_handler(self) -> None:
        bot = AsyncMock(spec=Bot)
        adapter = MagicMock()
        with patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
        ) as mock_shell:
            await _forward_pending_text(
                bot,
                1,
                42,
                "@5",
                "list files",
                "shell",
                adapter=adapter,
                is_existing_window=False,
            )
        mock_shell.assert_awaited_once()
        args = mock_shell.call_args.args
        assert args[1:] == (1, 42, "@5", "list files")
        assert args[0] is bot
        assert mock_shell.call_args.kwargs["adapter"] is adapter


@pytest.mark.parametrize(
    "binding_change",
    [
        {"provider": "claude", "foreground_argv": ("claude",)},
        {"process_id": 99},
        {"pane_id": "%6"},
    ],
    ids=["provider", "process", "active-pane"],
)
async def test_verified_shell_input_rejects_binding_change_after_verification(
    tmp_path: Path, binding_change: dict[str, object]
) -> None:
    adapter, backend, target, store = _protocol_adapter(tmp_path, "shell")

    def inject_then_change_binding(*args):
        backend.inspect = AsyncMock(return_value=replace(target, **binding_change))
        return object()

    with patch(
        "ccgram.handlers.telegram_origin.remember_telegram_injection",
        side_effect=inject_then_change_binding,
    ):
        success, message = await _send_verified_shell_input(
            100, "@5", 42, "ls -la", -100999, adapter
        )

    assert not success
    assert message == "Session binding changed"
    backend.send_text.assert_not_awaited()
    store.close()
