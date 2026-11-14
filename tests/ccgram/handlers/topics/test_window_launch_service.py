"""Tests for window_launch_service.py — pure helpers plus the launch_window flow."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topics.window_launch_service import (
    WindowLaunchRequest,
    _create_topic_window,
    _cwd_within,
    _follow_supersession,
    _persist_worktree_state,
    agent_process_started,
    launch_window,
)
from ccgram.handlers.user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
)


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


_MODULE = "ccgram.handlers.topics.window_launch_service."


# ── agent_process_started ────────────────────────────────────────────────────


class TestAgentProcessStarted:
    async def test_unknown_when_the_window_is_missing(self) -> None:
        adapter = MagicMock()
        adapter.resolve = AsyncMock(side_effect=Exception("missing"))
        from ccgram.session_protocol.models import ProtocolError

        adapter.resolve.side_effect = ProtocolError(404, "not_found", "missing")
        assert await agent_process_started("@7", adapter) is None

    async def test_shell_is_not_started(self) -> None:
        adapter = MagicMock()
        adapter.resolve = AsyncMock(
            return_value=(SimpleNamespace(foreground_argv=("-zsh",)), None)
        )
        assert await agent_process_started("@7", adapter) is False

    async def test_agent_command_is_started(self) -> None:
        adapter = MagicMock()
        adapter.resolve = AsyncMock(
            return_value=(SimpleNamespace(foreground_argv=("agy",)), None)
        )
        assert await agent_process_started("@7", adapter) is True


# ── _cwd_within ──────────────────────────────────────────────────────────────


class TestCwdWithin:
    def test_exact_match_returns_true(self, tmp_path):
        assert _cwd_within(str(tmp_path), str(tmp_path)) is True

    def test_subdir_returns_true(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        assert _cwd_within(str(sub), str(tmp_path)) is True

    def test_sibling_returns_false(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert _cwd_within(str(a), str(b)) is False

    def test_parent_returns_false(self, tmp_path):
        sub = tmp_path / "child"
        sub.mkdir()
        assert _cwd_within(str(tmp_path), str(sub)) is False

    def test_nonexistent_path_returns_false(self):
        # Path.resolve() is non-strict by default — does not raise on nonexistent paths.
        # Both paths resolve to themselves, so a != b → False.
        assert _cwd_within("/nonexistent/a", "/nonexistent/b") is False


# ── _persist_worktree_state ──────────────────────────────────────────────────


class TestPersistWorktreeState:
    @staticmethod
    def _context(user_data: dict) -> MagicMock:
        context = MagicMock()
        context.user_data = user_data
        return context

    def test_writes_worktree_state_and_clears_keys_when_cwd_matches(
        self, tmp_path
    ) -> None:
        wt_path = str(tmp_path)
        user_data = {
            PENDING_WORKTREE_PATH: wt_path,
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }

        with patch(f"{_MODULE}session_manager") as mock_sm:
            _persist_worktree_state("@1", wt_path, self._context(user_data))

        mock_sm.set_window_worktree.assert_called_once_with("@1", wt_path, "ccg/feat")
        assert PENDING_WORKTREE_PATH not in user_data
        assert PENDING_WORKTREE_BRANCH not in user_data

    def test_skips_write_when_cwd_outside_worktree(self, tmp_path) -> None:
        """A stale path from an aborted attempt must not attach to a new window."""
        user_data = {
            PENDING_WORKTREE_PATH: str(tmp_path / "worktrees" / "feat"),
            PENDING_WORKTREE_BRANCH: "ccg/feat",
        }

        with patch(f"{_MODULE}session_manager") as mock_sm:
            _persist_worktree_state(
                "@1", str(tmp_path / "other"), self._context(user_data)
            )

        mock_sm.set_window_worktree.assert_not_called()
        assert PENDING_WORKTREE_PATH not in user_data

    def test_no_op_when_worktree_path_missing(self, tmp_path) -> None:
        with patch(f"{_MODULE}session_manager") as mock_sm:
            _persist_worktree_state("@1", str(tmp_path), self._context({}))
        mock_sm.set_window_worktree.assert_not_called()


# ── _follow_supersession ─────────────────────────────────────────────────────


class TestFollowSupersession:
    def test_stable_id_leaves_the_creation_guard_alone(self) -> None:
        with (
            patch(f"{_MODULE}window_query") as mock_wq,
            patch(f"{_MODULE}topic_orchestration") as mock_orch,
        ):
            mock_wq.resolve_window_alias.return_value = "@5"
            assert _follow_supersession("@5") == "@5"

        mock_orch.register_pending_creation.assert_not_called()
        mock_orch.clear_pending_creation.assert_not_called()

    def test_case_only_supersession_keeps_the_existing_guard(self) -> None:
        with (
            patch(f"{_MODULE}window_query") as mock_wq,
            patch(f"{_MODULE}topic_orchestration") as mock_orch,
        ):
            mock_wq.resolve_window_alias.return_value = "ABC-DEF"
            assert _follow_supersession("abc-def") == "ABC-DEF"

        mock_orch.register_pending_creation.assert_not_called()
        mock_orch.clear_pending_creation.assert_not_called()

    def test_superseded_id_carries_the_creation_guard_forward(self) -> None:
        """Herdr re-keys a target mid-creation; the guard must move with it."""
        with (
            patch(f"{_MODULE}window_query") as mock_wq,
            patch(f"{_MODULE}topic_orchestration") as mock_orch,
        ):
            mock_wq.resolve_window_alias.return_value = "w1:t9"
            assert _follow_supersession("w1:t1") == "w1:t9"

        mock_orch.register_pending_creation.assert_called_once_with("w1:t9")
        mock_orch.clear_pending_creation.assert_called_once_with("w1:t1")


# ── _create_topic_window ─────────────────────────────────────────────────────


class TestCreateTopicWindow:
    @staticmethod
    def _worktree_context() -> MagicMock:
        context = MagicMock()
        context.user_data = {
            PENDING_WORKTREE_REPO: "/repo",
            PENDING_WORKTREE_BRANCH: "ccg/feat",
            PENDING_WORKTREE_PATH: "/repo.worktrees/ccg-feat",
        }
        return context

    async def test_native_worktree_delegates_to_one_call(self) -> None:
        with (
            patch(f"{_MODULE}tmux_manager") as mux,
            patch(f"{_MODULE}adapter_for") as adapter_for,
        ):
            mux.capabilities.native_worktrees = True
            adapter = MagicMock()
            adapter.create = AsyncMock(
                return_value=SimpleNamespace(target_id="w1:t1", title="ccg-feat")
            )
            adapter_for.return_value = adapter
            result = await _create_topic_window(
                "/repo.worktrees/ccg-feat",
                "claude",
                "normal",
                None,
                self._worktree_context(),
            )

        assert result == (
            True,
            "Created topic target 'ccg-feat'",
            "ccg-feat",
            "w1:t1",
        )
        adapter.create.assert_awaited_once_with(
            {
                "provider": "claude",
                "directory": "/repo",
                "worktree": True,
                "branch": "ccg/feat",
            },
            approval_mode="normal",
        )

    async def test_preselected_workspace_blocks_native_worktree_creation(self) -> None:
        """The native worktree API cannot pin a workspace — refuse, don't override."""
        with (
            patch(f"{_MODULE}tmux_manager") as mux,
            patch(f"{_MODULE}adapter_for") as adapter_for,
        ):
            mux.capabilities.native_worktrees = True
            adapter = MagicMock()
            adapter.create = AsyncMock()
            adapter_for.return_value = adapter
            success, message, name, window_id = await _create_topic_window(
                "/repo.worktrees/ccg-feat",
                "claude",
                "normal",
                "ws-chosen",
                self._worktree_context(),
            )

        assert success is False
        assert message == "Selected workspace cannot create a native worktree"
        assert (name, window_id) == ("", "")
        adapter.create.assert_not_awaited()

    async def test_tmux_backend_uses_create_window(self) -> None:
        context = MagicMock()
        context.user_data = {}
        with patch(f"{_MODULE}adapter_for") as adapter_for:
            adapter = MagicMock()
            adapter.create = AsyncMock(
                return_value=SimpleNamespace(target_id="@7", title="proj")
            )
            adapter_for.return_value = adapter
            result = await _create_topic_window(
                "/proj", "claude", "normal", "ws1", context
            )

        assert result == (True, "Created topic target 'proj'", "proj", "@7")
        adapter.create.assert_awaited_once_with(
            {"provider": "claude", "directory": "/proj", "workspaceId": "ws1"},
            approval_mode="normal",
        )

    async def test_guarded_target_failure_is_reported_not_raised(self) -> None:
        context = MagicMock()
        context.user_data = {}
        with patch(f"{_MODULE}adapter_for") as adapter_for:
            from ccgram.session_protocol.models import ProtocolError

            adapter = MagicMock()
            adapter.create = AsyncMock(
                side_effect=ProtocolError(400, "invalid_request", "no socket")
            )
            adapter_for.return_value = adapter
            success, message, name, window_id = await _create_topic_window(
                "/proj", "claude", "normal", None, context
            )

        assert success is False
        assert message == "no socket"
        assert (name, window_id) == ("", "")


# ── launch_window ────────────────────────────────────────────────────────────


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


@dataclass
class _LaunchMocks:
    """Every collaborator ``launch_window`` reaches for, so tests can steer them."""

    mux: MagicMock
    session: MagicMock
    router: MagicMock
    orchestration: MagicMock
    preferences: MagicMock
    session_map: MagicMock
    edit: AsyncMock
    registry: MagicMock
    adapter: MagicMock


@contextlib.contextmanager
def _launch_env(
    *,
    supports_hook: bool = False,
    has_yolo_confirmation: bool = False,
    chat_first_command_path: bool = False,
    launch_command: str = "claude",
) -> Iterator[_LaunchMocks]:
    """Patch launch_window's collaborators for a successful hookless launch.

    Defaults describe a guarded-target backend creating ``@5`` with no hook to
    wait for; each test overrides only the piece it is about.
    """
    with (
        patch(f"{_MODULE}tmux_manager") as mux,
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
        patch(f"{_MODULE}session_manager") as session,
        patch(f"{_MODULE}thread_router") as router,
        patch(f"{_MODULE}topic_orchestration") as orchestration,
        patch(f"{_MODULE}user_preferences") as preferences,
        patch(f"{_MODULE}session_map_sync") as session_map,
        patch(f"{_MODULE}safe_edit", new_callable=AsyncMock) as edit,
        patch(f"{_MODULE}provider_registry") as registry,
        patch(f"{_MODULE}adapter_for") as adapter_for,
        patch("ccgram.providers.resolve_launch_command", return_value=launch_command),
    ):
        adapter = MagicMock()
        adapter.create = AsyncMock(
            return_value=SimpleNamespace(target_id="@5", title="my-win")
        )
        adapter.close = AsyncMock(return_value=(True, "Closed"))
        adapter.resolve = AsyncMock(
            return_value=(
                SimpleNamespace(
                    foreground_argv=(launch_command,),
                    provider={"agy": "antigravity"}.get(launch_command, launch_command),
                    available=True,
                ),
                None,
            )
        )
        adapter_for.return_value = adapter
        mux.stamp_pane_title = AsyncMock()
        mux.capabilities.native_worktrees = False
        mux.capabilities.native_agent_status = True
        mux.capabilities.native_topic_targets = True
        router.get_window_for_thread.return_value = None
        router.resolve_chat_id.return_value = -100999
        router.begin_topic_provisioning.return_value = SimpleNamespace(
            claim_id="claim-1"
        )
        router.commit_topic_provisioning.return_value = True
        session_map.wait_for_session_map_entry = AsyncMock(return_value=True)
        registry.get.return_value.capabilities = MagicMock(
            supports_hook=supports_hook,
            has_yolo_confirmation=has_yolo_confirmation,
            chat_first_command_path=chat_first_command_path,
        )
        yield _LaunchMocks(
            mux=mux,
            session=session,
            router=router,
            orchestration=orchestration,
            preferences=preferences,
            session_map=session_map,
            edit=edit,
            registry=registry,
            adapter=adapter,
        )


def _request(**overrides) -> WindowLaunchRequest:
    fields = {
        "user_id": 100,
        "thread_id": 42,
        "provider_name": "claude",
        "cwd": "/tmp/proj",
        "mode": "normal",
        "pending_text": None,
    }
    fields.update(overrides)
    return WindowLaunchRequest(**fields)


@pytest.mark.parametrize("mode", ["auto", "marked"])
async def test_failed_native_selection_is_retryable_and_gates_only_marked(
    mode, monkeypatch
):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", mode, raising=False)
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
    with _launch_env() as m, patch("ccgram.multiplexer.multiplexer", native):
        result = await launch_window(_make_query(), _make_context(), _request())

    assert result.success is (mode == "auto")
    assert lifecycle_state.get_pending_topic_mark("@5") is True
    native.set_topic_mark.assert_awaited_once_with("@5", True, owner="owner:" + "@5")
    m.adapter.close.assert_not_awaited()
    if mode == "marked":
        m.router.commit_topic_provisioning.assert_not_called()
        m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")
    else:
        m.router.commit_topic_provisioning.assert_called_once()


async def test_cancellation_during_selection_preserves_provisioning(monkeypatch):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", "marked", raising=False)
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@5",
                topic_markable=True,
                topic_mark_owner="owner:" + "@5",
                topic_marked=True,
            )
        ),
        set_topic_mark=AsyncMock(side_effect=asyncio.CancelledError),
    )
    with (
        _launch_env() as m,
        patch("ccgram.multiplexer.multiplexer", native),
        pytest.raises(asyncio.CancelledError),
    ):
        await launch_window(_make_query(), _make_context(), _request())

    assert lifecycle_state.get_pending_topic_mark("@5") is True
    m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")
    m.router.commit_topic_provisioning.assert_not_called()
    m.adapter.close.assert_not_awaited()


async def test_native_selection_precedes_creation_commit(monkeypatch):
    from ccgram.config import config

    monkeypatch.setattr(config, "topic_discovery_mode", "marked", raising=False)
    with _launch_env() as m:

        async def mark(window_id, selected, *, owner):
            assert window_id == "@5" and selected is True
            m.router.commit_topic_provisioning.assert_not_called()
            return True

        native = SimpleNamespace(
            find_window_by_id=AsyncMock(
                return_value=SimpleNamespace(
                    window_id="@5",
                    topic_markable=True,
                    topic_mark_owner="owner:" + "@5",
                    topic_marked=True,
                )
            ),
            set_topic_mark=AsyncMock(side_effect=mark),
        )
        with patch("ccgram.multiplexer.multiplexer", native):
            result = await launch_window(_make_query(), _make_context(), _request())

    assert result.success
    native.set_topic_mark.assert_awaited_once()
    m.router.commit_topic_provisioning.assert_called_once()


async def test_agent_exit_during_native_mark_keeps_creation_quarantined(monkeypatch):
    from ccgram.config import config

    monkeypatch.setattr(config, "topic_discovery_mode", "auto")
    with _launch_env() as m:

        async def mark(_wid, _selected, *, owner):
            m.adapter.resolve.return_value = (
                SimpleNamespace(
                    foreground_argv=("zsh",), provider="shell", available=True
                ),
                None,
            )
            return True

        native = SimpleNamespace(
            find_window_by_id=AsyncMock(
                return_value=SimpleNamespace(
                    window_id="@5",
                    topic_markable=True,
                    topic_marked=True,
                    topic_mark_owner="owner:@5",
                )
            ),
            set_topic_mark=AsyncMock(side_effect=mark),
        )
        with patch("ccgram.multiplexer.multiplexer", native):
            result = await launch_window(_make_query(), _make_context(), _request())
    assert not result.success
    m.router.commit_topic_provisioning.assert_not_called()
    m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")
    m.adapter.close.assert_not_awaited()


async def test_new_terminal_cannot_inherit_old_detached_ttl_at_reused_id(monkeypatch):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", "marked")
    lifecycle_state.set_topic_selection(
        "@5", detached=True, pending=False, owner="old-terminal"
    )
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@5",
                topic_markable=True,
                topic_marked=False,
                topic_mark_owner="new-terminal",
            )
        ),
        set_topic_mark=AsyncMock(return_value=False),
    )
    with _launch_env() as m, patch("ccgram.multiplexer.multiplexer", native):
        result = await launch_window(_make_query(), _make_context(), _request())
    assert not result.success
    assert not lifecycle_state.is_topic_detached("@5")
    assert lifecycle_state.get_topic_mark_owner("@5") == "new-terminal"
    assert {
        op.owner: op.desired for op in lifecycle_state.pending_topic_selections()
    } == {
        "old-terminal": False,
        "new-terminal": True,
    }
    m.router.commit_topic_provisioning.assert_not_called()


@pytest.mark.parametrize(
    "current_owner,commits",
    [
        pytest.param(None, False, id="captured-owner-lost"),
        pytest.param("owner:@5", True, id="stable-owner-mark-write-failed"),
    ],
)
async def test_auto_launch_rechecks_known_owner_before_real_claim_commit(
    monkeypatch, current_owner, commits
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
    )
    with (
        _launch_env() as launch,
        patch("ccgram.multiplexer.multiplexer", native),
        patch(f"{_MODULE}thread_router", router),
    ):
        result = await launch_window(_make_query(), _make_context(), _request())
    assert result.success is commits
    assert router.get_window_for_thread(100, 42, -100999) == ("@5" if commits else None)
    if not commits:
        (claim,) = router.iter_topic_provisionings()
        assert claim.uncertain
    launch.adapter.close.assert_not_awaited()


class TestLaunchWindowSuccess:
    async def test_creates_window_and_binds_thread(self, tmp_path) -> None:
        query = _make_query()
        context = _make_context({PENDING_THREAD_ID: 42})

        with _launch_env() as m:
            await launch_window(query, context, _request(cwd=str(tmp_path)))

        m.adapter.create.assert_awaited_once_with(
            {"provider": "claude", "directory": str(tmp_path)},
            approval_mode="normal",
        )
        m.router.begin_topic_provisioning.assert_called_once_with(
            100, -100999, thread_id=42, kind="target_for_topic"
        )
        m.router.attach_provisioning_target.assert_called_once_with("claim-1", "@5")
        m.router.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="my-win"
        )
        m.orchestration.pending_creation_transaction.assert_called_once_with()
        m.orchestration.register_pending_creation.assert_called_once_with("@5")
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")
        m.edit.assert_awaited_once()
        assert "✅" in m.edit.call_args[0][1]

    @patch(f"{_MODULE}_accept_yolo_confirmation", new_callable=AsyncMock)
    async def test_yolo_creation_guard_outlives_configured_confirmation(
        self, mock_accept: AsyncMock, tmp_path, monkeypatch
    ) -> None:
        mock_accept.return_value = True
        monkeypatch.setattr(
            "ccgram.handlers.topics.window_launch_service.config.yolo_confirmation_timeout",
            45.0,
        )

        with _launch_env(has_yolo_confirmation=True) as m:
            await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42}),
                _request(cwd=str(tmp_path), mode="yolo"),
            )

        m.orchestration.register_pending_creation.assert_called_once_with(
            "@5", ttl_s=55.0
        )

    async def test_no_thread_success_releases_pending_creation_guard(
        self, tmp_path
    ) -> None:
        with _launch_env() as m:
            result = await launch_window(
                _make_query(),
                _make_context(),
                _request(thread_id=None, cwd=str(tmp_path)),
            )

        assert result.success
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")

    async def test_pending_text_is_forwarded_after_binding(self, tmp_path) -> None:
        """A non-chat-first provider's first prompt uses the bound-window send path."""
        user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hello agent"}
        context = _make_context(user_data)

        with (
            _launch_env(launch_command="agy") as m,
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            await launch_window(
                _make_query(),
                context,
                _request(
                    provider_name="antigravity",
                    cwd=str(tmp_path),
                    pending_text="hello agent",
                ),
            )

        m.adapter.create.assert_awaited_once_with(
            {"provider": "antigravity", "directory": str(tmp_path)},
            approval_mode="normal",
        )
        mock_send.assert_awaited_once_with(
            100,
            "@5",
            42,
            "hello agent",
            -100999,
            adapter=m.adapter,
        )
        assert PENDING_THREAD_TEXT not in user_data

    @patch(f"{_MODULE}asyncio.sleep", new_callable=AsyncMock)
    async def test_hookless_launch_waits_for_agent_before_binding(
        self, mock_sleep: AsyncMock, tmp_path
    ) -> None:
        """The pane still runs the launch shell until the agent CLI starts.

        Binding before then lets the first poll mistake the shell for an agent
        exit and kill the window, so the bind must wait for the agent process.
        """
        order: list[str] = []
        panes = iter(["zsh", "zsh", "agy", "agy"])

        async def resolve(_window_id: str) -> tuple[SimpleNamespace, None]:
            command = next(panes)
            order.append(command)
            return SimpleNamespace(
                foreground_argv=(command,),
                provider="shell" if command == "zsh" else "antigravity",
                available=True,
            ), None

        with _launch_env(launch_command="agy") as m:
            m.adapter.resolve = AsyncMock(side_effect=resolve)
            m.router.commit_topic_provisioning.side_effect = lambda *_a, **_k: (
                order.append("bind") or True
            )
            await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42}),
                _request(provider_name="antigravity", cwd=str(tmp_path)),
            )

        assert order == ["zsh", "zsh", "agy", "agy", "bind"]
        assert mock_sleep.await_count == 2

    @patch(f"{_MODULE}asyncio.sleep", new_callable=AsyncMock)
    async def test_hookless_agent_timeout_quarantines_without_binding(
        self, mock_sleep: AsyncMock, tmp_path
    ) -> None:
        """A pane that never leaves its launch shell must not be bound.

        Binding would report success while the next poll reads the shell as an
        exited agent and kills the window.
        """
        user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hi"}
        with (
            _launch_env(launch_command="agy") as m,
            patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            m.adapter.resolve = AsyncMock(
                return_value=(
                    SimpleNamespace(
                        foreground_argv=("zsh",), provider="shell", available=True
                    ),
                    None,
                )
            )
            result = await launch_window(
                _make_query(),
                _make_context(user_data),
                _request(provider_name="antigravity", cwd=str(tmp_path)),
            )

        assert not result.success and "still starting" in (result.error_message or "")
        assert mock_sleep.await_count == 50
        m.router.commit_topic_provisioning.assert_not_called()
        m.router.bind_thread.assert_not_called()
        m.adapter.close.assert_not_awaited()
        assert user_data[PENDING_THREAD_ID] == 42


class TestLaunchWindowFailure:
    async def test_create_failure_aborts_without_binding_and_clears_flow_state(
        self, tmp_path
    ) -> None:
        user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hi"}
        context = _make_context(user_data)

        with _launch_env() as m:
            from ccgram.session_protocol.models import ProtocolError

            m.adapter.create = AsyncMock(
                side_effect=ProtocolError(400, "invalid_request", "tmux error")
            )
            result = await launch_window(
                _make_query(), context, _request(cwd=str(tmp_path))
            )

        assert result.success is False
        m.router.bind_thread.assert_not_called()
        m.edit.assert_awaited_once()
        assert "❌" in m.edit.call_args[0][1]
        assert PENDING_THREAD_ID not in user_data
        assert PENDING_THREAD_TEXT not in user_data

    async def test_create_failure_from_backend_rpc_is_quarantined(
        self, tmp_path
    ) -> None:
        with _launch_env() as m:
            from ccgram.session_protocol.models import ProtocolError

            m.adapter.create = AsyncMock(
                side_effect=ProtocolError(
                    503, "backend_unavailable", "multiplexer socket unavailable"
                )
            )
            with pytest.raises(ProtocolError, match="multiplexer socket unavailable"):
                await launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )

        m.router.abort_topic_provisioning.assert_not_called()
        m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")

    async def test_initial_checkpoint_failure_does_not_start_backend(
        self, tmp_path
    ) -> None:
        with _launch_env() as m:
            m.session.flush_state.side_effect = RuntimeError("state disk unavailable")
            with pytest.raises(RuntimeError, match="state disk unavailable"):
                await launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )

        m.adapter.create.assert_not_awaited()
        m.router.abort_topic_provisioning.assert_called_once_with(
            "claim-1", target_confirmed_absent=True
        )

    async def test_attach_checkpoint_failure_keeps_completed_target_quarantined(
        self, tmp_path
    ) -> None:
        flush_calls = 0

        def flush_state() -> None:
            nonlocal flush_calls
            flush_calls += 1
            if flush_calls == 2:
                raise RuntimeError("attach checkpoint failed")

        with _launch_env() as m:
            m.session.flush_state.side_effect = flush_state
            with pytest.raises(RuntimeError, match="attach checkpoint failed"):
                await launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )

        m.adapter.close.assert_not_awaited()
        m.router.abort_topic_provisioning.assert_not_called()
        m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")

    async def test_cancellation_settles_completed_creation_and_cleans_target(
        self, tmp_path
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def create_target(*_args, **_kwargs):
            started.set()
            await release.wait()
            return SimpleNamespace(target_id="@5", title="my-win")

        with _launch_env() as m:
            m.adapter.create = AsyncMock(side_effect=create_target)
            task = asyncio.create_task(
                launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )
            )
            await started.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        m.adapter.close.assert_awaited_once_with("@5")
        m.router.abort_topic_provisioning.assert_called_once_with(
            "claim-1", target_confirmed_absent=True
        )

    async def test_post_create_stamp_error_closes_target_before_reraising(
        self, tmp_path
    ) -> None:
        with _launch_env() as m:
            m.mux.stamp_pane_title = AsyncMock(side_effect=RuntimeError("stamp failed"))
            with pytest.raises(RuntimeError, match="stamp failed"):
                await launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )

        m.adapter.close.assert_awaited_once_with("@5")
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")
        m.router.abort_topic_provisioning.assert_called_once_with(
            "claim-1", target_confirmed_absent=True
        )

    async def test_session_map_timeout_closes_target_before_unbinding_late_hook(
        self, tmp_path
    ) -> None:
        """A late hook cannot orphan the just-created target after timeout."""
        cleanup_order: list[str] = []

        with (
            _launch_env(supports_hook=True) as m,
            patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            m.adapter.close = AsyncMock(
                side_effect=lambda target_id: (
                    cleanup_order.append(f"close:{target_id}") or True
                )
            )
            m.router.unbind_thread.side_effect = lambda *_args, **_kwargs: (
                cleanup_order.append("unbind")
            )
            m.orchestration.clear_pending_creation.side_effect = lambda *_: (
                cleanup_order.append("clear-pending")
            )
            m.session_map.wait_for_session_map_entry = AsyncMock(return_value=False)

            result = await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42}),
                _request(cwd=str(tmp_path)),
            )

        assert result.success is False
        m.adapter.close.assert_not_awaited()
        m.router.abort_topic_provisioning.assert_called_once_with(
            "claim-1", target_confirmed_absent=True
        )
        assert cleanup_order == ["clear-pending"]
        assert "❌" in m.edit.call_args.args[1]

    @pytest.mark.parametrize("presence", [True, None], ids=["live", "unknown"])
    async def test_session_map_timeout_quarantines_target_without_killing(
        self, presence, tmp_path
    ) -> None:
        user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hi"}
        with (
            _launch_env(supports_hook=True) as m,
            patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=presence,
            ),
        ):
            m.session_map.wait_for_session_map_entry = AsyncMock(return_value=False)

            result = await launch_window(
                _make_query(),
                _make_context(user_data),
                _request(cwd=str(tmp_path)),
            )

        assert not result.success and "still starting" in (result.error_message or "")
        m.adapter.close.assert_not_awaited()
        m.router.abort_topic_provisioning.assert_not_called()
        m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")
        m.orchestration.clear_pending_creation.assert_called_once_with("@5")
        assert user_data[PENDING_THREAD_ID] == 42
        assert user_data[PENDING_THREAD_TEXT] == "hi"

    async def test_session_map_timeout_does_not_clear_pending_text_when_presence_unknown(
        self, tmp_path
    ) -> None:
        with _launch_env(supports_hook=True) as m:
            m.session_map.wait_for_session_map_entry = AsyncMock(return_value=False)

            with patch(
                "ccgram.multiplexer.reconciliation.window_presence",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await launch_window(
                    _make_query(),
                    _make_context({PENDING_THREAD_ID: 42}),
                    _request(cwd=str(tmp_path)),
                )

        assert not result.success
        m.adapter.close.assert_not_awaited()
        m.router.mark_provisioning_uncertain.assert_called_once_with("claim-1")

    async def test_pending_text_send_failure_reports_back_to_the_topic(
        self, tmp_path
    ) -> None:
        with (
            _launch_env(),
            patch(
                f"{_MODULE}send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(False, "pane is gone"),
            ),
            patch(f"{_MODULE}safe_send", new_callable=AsyncMock) as mock_safe_send,
        ):
            result = await launch_window(
                _make_query(),
                _make_context({PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "hi"}),
                _request(cwd=str(tmp_path), pending_text="hi"),
            )

        assert result.success is True
        mock_safe_send.assert_awaited_once()
        assert "pane is gone" in mock_safe_send.call_args.args[2]
        assert mock_safe_send.call_args.kwargs == {"message_thread_id": 42}
