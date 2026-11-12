import asyncio
from dataclasses import replace
import subprocess
from tempfile import TemporaryFile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest
from libtmux.exc import LibTmuxException

from ccgram.handlers.polling.window_tick.observe import _native_agent_status
from ccgram.multiplexer import agent_status_cache
from ccgram.multiplexer.base import (
    AgentStatus,
    ForegroundInfo,
    Multiplexer,
    PaneInfo,
    WindowRef,
)
from ccgram.multiplexer.tmux import TmuxManager
from ccgram.multiplexer.herdr import HerdrError
from ccgram.multiplexer.vim_state import notify_vim_insert_seen, reset_vim_state
from ccgram.session_protocol.backend import CreationOptions, MultiplexerSessionBackend
from ccgram.session_protocol.models import Action, ProtocolError
from ccgram.session_protocol.runtime import ProtocolRuntime
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.telegram import adapter_for
from ccgram.window_state_ports import identity_state
from ccgram.window_state_ports.identity_state import IdentityProjection, get_identity
from ccgram.window_state_store import WindowState, WindowStateStore


@pytest.fixture
def git_probes(monkeypatch, tmp_path):
    outputs = {
        ("rev-parse", "--is-inside-work-tree"): "true",
        ("rev-parse", "--is-bare-repository"): "false",
        ("rev-parse", "--abbrev-ref", "HEAD"): "main",
        ("rev-parse", "--git-dir"): str(tmp_path / ".git"),
        ("rev-parse", "--show-toplevel"): str(tmp_path),
    }

    def run(argv, **kwargs):
        args = tuple(argv[3:] if argv[1] == "-C" else argv[1:])
        return subprocess.CompletedProcess(argv, 0, outputs.get(args, ""), "")

    boundary = MagicMock(side_effect=run)
    monkeypatch.setattr("ccgram.worktrees.subprocess.run", boundary)
    return boundary, run


@pytest.mark.parametrize("probe", [("branch", "--list"), ("worktree", "list")])
async def test_worktree_branch_probe_timeout_is_structured(
    mux, tmp_path, git_probes, probe
):
    boundary, run = git_probes

    def timeout(argv, **kwargs):
        if tuple(argv[3:5]) == probe:
            raise subprocess.TimeoutExpired(argv, 5)
        return run(argv, **kwargs)

    boundary.side_effect = timeout
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {"directory": str(tmp_path), "provider": "claude", "worktree": True}
        )
    assert error.value.status == 503
    assert error.value.code == "backend_unavailable"
    mux.create_window.assert_not_awaited()


@pytest.mark.parametrize(
    "workspace", [None, "", 42, "bad\x00id", "bad\ud800id", "bad\udfffid"]
)
@pytest.mark.parametrize("worktree", [False, True])
async def test_invalid_workspace_is_rejected_before_creation(
    mux, tmp_path, git_probes, workspace, worktree
):
    boundary, _ = git_probes
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {
                "directory": str(tmp_path),
                "provider": "claude",
                "workspaceId": workspace,
                "worktree": worktree,
            }
        )
    assert error.value.status == 400
    assert error.value.code == "invalid_request"
    boundary.assert_not_called()
    mux.create_window.assert_not_awaited()
    mux.create_topic_target.assert_not_awaited()
    mux.create_worktree_window.assert_not_awaited()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("worktree", [False, True])
async def test_valid_workspace_reaches_selected_creation_backend(
    mux, tmp_path, git_probes, native, worktree
):
    mux.capabilities.native_topic_targets = native
    mux.create_window.return_value = (True, "", "", "@1")
    mux.create_topic_target.return_value = SimpleNamespace(target_id="@1")
    target = await MultiplexerSessionBackend(mux, "/tmp").create(
        {
            "directory": str(tmp_path),
            "provider": "claude",
            "workspaceId": "workspace-1",
            "worktree": worktree,
        }
    )
    assert target.id == "@1"
    boundary = mux.create_topic_target if native else mux.create_window
    assert boundary.await_args.kwargs["workspace_id"] == "workspace-1"


async def test_trusted_creation_options_keep_yolo_and_provider_resume_args(
    mux, tmp_path, monkeypatch
):
    mux.create_window.return_value = (True, "", "", "@1")
    record_approval = MagicMock()
    monkeypatch.setattr(identity_state, "set_window_approval_mode", record_approval)
    resume_id = "00000000-0000-4000-8000-000000000001"

    await MultiplexerSessionBackend(mux, "/tmp").create(
        {"directory": str(tmp_path), "provider": "claude"},
        trusted_options=CreationOptions(approval_mode="yolo", resume_id=resume_id),
    )

    kwargs = mux.create_window.await_args.kwargs
    assert "--dangerously-skip-permissions" in kwargs["launch_command"]
    assert kwargs["agent_args"] == f"--resume {resume_id}"
    record_approval.assert_called_once_with("@1", "yolo")


async def test_telegram_creation_returns_native_target_and_opaque_session(
    mux, store, tmp_path
):
    mux.list_windows_for_reconciliation.return_value = [
        WindowRef("@9", "created", "/tmp")
    ]
    mux.create_window.return_value = (True, "", "", "@9")
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})

    created = await adapter.create({"directory": str(tmp_path), "provider": "claude"})

    assert created.target_id == "@9"
    assert created.session.id != created.target_id
    assert created.session.id.startswith("s_")
    assert created.title == created.session.title == "created"


async def test_untrusted_remote_creation_still_rejects_yolo_mode(mux, tmp_path):
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {
                "directory": str(tmp_path),
                "provider": "claude",
                "mode": "yolo",
            }
        )

    assert error.value.code == "unsupported_action"
    mux.create_window.assert_not_awaited()


@pytest.mark.parametrize("failure", ["timeout", "os_error", "nonzero"])
async def test_failed_worktree_checkout_requires_inspection_before_retry(
    mux, tmp_path, git_probes, failure
):
    boundary, run = git_probes

    def checkout(argv, **kwargs):
        if argv[3:5] == ["worktree", "add"]:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 30)
            if failure == "os_error":
                raise OSError("checkout failed")
            return subprocess.CompletedProcess(argv, 128, "", "checkout failed")
        return run(argv, **kwargs)

    boundary.side_effect = checkout
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {"directory": str(tmp_path), "provider": "claude", "worktree": True}
        )
    assert error.value.status == 503
    assert error.value.code == "backend_unavailable"
    assert "inspect the terminal before retrying" in str(error.value)
    mux.create_window.assert_not_awaited()


@pytest.mark.parametrize(
    "title", [1, True, {"text": "title"}, ["title"], "ignored title"]
)
async def test_worktree_creation_ignores_undeclared_title(
    mux, tmp_path, git_probes, title
):
    mux.create_window.return_value = (True, "", "", "@1")
    target = await MultiplexerSessionBackend(mux, "/tmp").create(
        {
            "directory": str(tmp_path),
            "provider": "claude",
            "worktree": True,
            "title": title,
        }
    )
    assert target.id == "@1"
    assert mux.create_window.await_args.args[0] == str(
        tmp_path.parent / f"{tmp_path.name}.worktrees" / "ccg-agent-1"
    )


@pytest.mark.parametrize("branch", [None, "", False, 0, {}, []])
async def test_worktree_creation_rejects_present_invalid_branch(
    mux, tmp_path, git_probes, branch
):
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {
                "directory": str(tmp_path),
                "provider": "claude",
                "worktree": True,
                "branch": branch,
            }
        )
    assert error.value.status == 400
    assert error.value.code == "invalid_request"
    mux.create_window.assert_not_awaited()


async def test_tmux_session_setup_failure_returns_structured_creation_error(
    tmp_path, monkeypatch
):
    manager = TmuxManager(session_name="creation-regression")
    server = MagicMock()
    server.sessions.get.return_value = None
    server.new_session.side_effect = LibTmuxException("cannot start server")
    monkeypatch.setattr(manager, "_server", server)
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(manager, "/tmp").create(
            {"directory": str(tmp_path), "provider": "claude"}
        )
    assert error.value.status == 503
    assert error.value.code == "backend_unavailable"
    assert "inspect the terminal before retrying" in str(error.value)
    server.new_session.assert_called_once()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "failure", [HerdrError("deleted workspace"), OSError("offline")]
)
async def test_creation_errors_are_structured_and_warn_against_retry(
    mux, tmp_path, native, failure
):
    mux.capabilities.native_topic_targets = native
    boundary = mux.create_topic_target if native else mux.create_window
    boundary.side_effect = failure
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {"directory": str(tmp_path), "provider": "claude"}
        )
    assert error.value.status == 503
    assert error.value.code == "backend_unavailable"
    assert "inspect the terminal before retrying" in str(error.value)


@pytest.mark.parametrize("native", [False, True])
async def test_disappearing_created_target_warns_against_retry(mux, tmp_path, native):
    mux.capabilities.native_topic_targets = native
    mux.create_topic_target.return_value = SimpleNamespace(target_id="@missing")
    mux.create_window.return_value = (True, "", "", "@missing")
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {"directory": str(tmp_path), "provider": "claude"}
        )
    assert error.value.status == 503
    assert "inspect the terminal before retrying" in str(error.value)


@pytest.mark.parametrize("stored,reported", [("ab-CD", "AB-cd"), ("AB-cd", "ab-CD")])
async def test_tracked_ineligible_target_preserves_identity_and_backend_id(
    mux, monkeypatch, tmp_path, stored, reported
):
    state = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _: None
    )
    path = tmp_path / "transcript.jsonl"
    state.window_states[stored] = WindowState(
        session_id="conv", provider_name="claude", transcript_path=str(path)
    )
    monkeypatch.setattr(identity_state, "get_identity", get_identity)
    monkeypatch.setattr(identity_state, "window_store", state)
    mux.list_windows_for_reconciliation.return_value = [
        WindowRef(reported, "project", "/tmp", topic_eligible=False)
    ]
    targets = await MultiplexerSessionBackend(mux, "/tmp").targets()
    assert len(targets) == 1
    assert targets[0].id == reported
    assert targets[0].conversation_id == "conv"
    assert targets[0].transcript == path
    mux.foreground.assert_awaited_once_with(reported)


async def test_native_probe_failure_is_per_target_and_recovers(mux, monkeypatch):
    monkeypatch.setattr(identity_state, "get_identity", lambda _: None)
    mux.list_windows_for_reconciliation.return_value = [
        WindowRef("@1", "first", "/tmp"),
        WindowRef("@2", "second", "/tmp"),
    ]
    mux.agent_status.side_effect = [
        RuntimeError("probe failed"),
        AgentStatus("working"),
        AgentStatus("done"),
        AgentStatus("idle"),
    ]
    backend = MultiplexerSessionBackend(mux, "/tmp")

    targets = await backend.targets()
    assert [target.id for target in targets] == ["@1", "@2"]
    assert targets[0].status.source == "screen"
    assert targets[1].status.source == "native"
    assert targets[1].status.state == "working"

    recovered = await backend.targets()
    assert [target.status.source for target in recovered] == ["native", "native"]
    assert [target.status.state for target in recovered] == ["done", "idle"]


async def test_native_probe_cancellation_propagates(mux, monkeypatch):
    monkeypatch.setattr(identity_state, "get_identity", lambda _: None)
    mux.agent_status.side_effect = asyncio.CancelledError
    backend = MultiplexerSessionBackend(mux, "/tmp")

    with pytest.raises(asyncio.CancelledError):
        await backend._target(WindowRef("@1", "project", "/tmp"))


async def test_protocol_runtime_refresh_recovers_after_native_probe_failure(
    mux, store, monkeypatch
):
    monkeypatch.setattr(identity_state, "get_identity", lambda _: None)
    mux.agent_status.side_effect = [
        RuntimeError("probe failed"),
        AgentStatus("working"),
    ]
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr("ccgram.session_protocol.runtime.asyncio.sleep", sleep)

    with TemporaryFile() as lock:
        runtime = ProtocolRuntime(store, service, lock)
        with pytest.raises(asyncio.CancelledError):
            await runtime.poll("")

    assert mux.agent_status.await_count == 2
    assert sleep.await_count == 2
    assert next(iter(service._snapshots.values())).status.state == "working"


@pytest.fixture
def clean_agent_status_cache():
    agent_status_cache.reset()
    yield
    agent_status_cache.reset()


async def test_runtime_refresh_and_telegram_share_optional_negative_probe(
    mux, store, monkeypatch, clean_agent_status_cache
):
    now = [0.0]
    monkeypatch.setattr(agent_status_cache, "_clock", lambda: now[0])
    mux.capabilities.native_agent_status = True
    mux.capabilities.supports_event_stream = False
    mux.agent_status.return_value = None
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr("ccgram.session_protocol.runtime.asyncio.sleep", sleep)

    with TemporaryFile() as lock:
        runtime = ProtocolRuntime(store, service, lock)
        with pytest.raises(asyncio.CancelledError):
            await runtime.poll("")

    with patch("ccgram.handlers.polling.window_tick.observe.tmux_manager", mux):
        assert await _native_agent_status("@1") is None
    assert mux.agent_status.await_count == 1

    # A backend's None can include an expected transport failure. Optional
    # enrichment may stay negative for at most five seconds, then must recover.
    now[0] = 5.0
    mux.agent_status.return_value = AgentStatus("working", custom_status="fresh")
    await service.refresh()
    assert next(iter(service._snapshots.values())).status.state == "working"
    with patch("ccgram.handlers.polling.window_tick.observe.tmux_manager", mux):
        status = await _native_agent_status("@1")
    assert status is not None and status.display_label == "fresh"
    assert mux.agent_status.await_count == 3


@pytest.mark.parametrize("cached_positive", [False, True])
async def test_bulk_event_stream_status_stays_live_without_target_pushes(
    mux, clean_agent_status_cache, cached_positive
):
    mux.capabilities.native_agent_status = True
    mux.capabilities.supports_event_stream = True
    mux.agent_status.side_effect = [AgentStatus("working"), AgentStatus("idle")]
    if cached_positive:
        agent_status_cache.set_status("@1", AgentStatus("done"))
    backend = MultiplexerSessionBackend(mux, "/tmp")

    first = await backend.targets()
    second = await backend.targets()

    assert first[0].status.state == "working"
    assert second[0].status.state == "idle"
    assert mux.agent_status.await_count == 2


async def test_active_inspect_bypasses_bulk_negative_status_cache(
    mux, clean_agent_status_cache
):
    mux.capabilities.native_agent_status = True
    mux.capabilities.supports_event_stream = False
    mux.agent_status.side_effect = [None, AgentStatus("working")]
    backend = MultiplexerSessionBackend(mux, "/tmp")

    await backend.targets()
    inspected = await backend.inspect("@1")

    assert inspected.status.state == "working"
    assert mux.agent_status.await_count == 2


@pytest.fixture
def mux(monkeypatch):
    monkeypatch.setattr("ccgram.config.config.prompt_mode", "wrap")
    mux = AsyncMock(spec=Multiplexer)
    mux.capabilities = SimpleNamespace(
        native_worktrees=False,
        native_topic_targets=False,
        native_agent_status=False,
        supports_event_stream=False,
    )
    mux.list_windows_for_reconciliation.return_value = [
        WindowRef("@1", "project", "/tmp")
    ]
    mux.foreground.return_value = ForegroundInfo(0, 0, ["claude"], "/tmp")
    mux.list_panes.return_value = [PaneInfo("%1", 0, True, "claude", "/tmp", 80, 24)]
    mux.agent_status.return_value = None
    mux.capture_pane.return_value = "Ready"
    mux.capture_pane_by_id.return_value = "Ready"
    mux.get_pane_title.return_value = ""
    mux.send_to_pane.return_value = True
    monkeypatch.setattr(
        identity_state,
        "get_identity",
        lambda _: IdentityProjection(
            "@1", "claude", "conv", "/tmp", None, "project", "normal"
        ),
    )
    return mux


async def test_verified_pidless_foreground_can_act_and_argv_change_invalidates_binding(
    mux, store
):
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]
    result, status = await service.execute(
        session.id,
        Action(
            str(uuid4()), session.binding_version, "terminal.key", {"keys": ["Escape"]}
        ),
    )
    assert status == 200 and result.outcome == "sent"
    mux.send_to_pane.assert_awaited_once()
    mux.send_to_pane.reset_mock()
    mux.foreground.return_value = ForegroundInfo(
        0, 0, ["claude", "--resume", "other"], "/tmp"
    )
    result, status = await service.execute(
        session.id,
        Action(
            str(uuid4()), session.binding_version, "terminal.key", {"keys": ["Enter"]}
        ),
    )
    assert status == 409 and result.outcome == "rejected"
    mux.send_to_pane.assert_not_awaited()


async def test_terminal_input_bypasses_agent_message_submission_workarounds(mux, store):
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]

    pasted, _ = await service.execute(
        session.id,
        Action(str(uuid4()), session.binding_version, "terminal.paste", {"text": "y"}),
    )
    submitted, _ = await service.execute_internal(
        session.id,
        Action(str(uuid4()), session.binding_version, "terminal.submit", {"text": "y"}),
    )
    messaged, _ = await service.execute(
        session.id,
        Action(
            str(uuid4()), session.binding_version, "message.send", {"text": "hello"}
        ),
    )

    assert [pasted.outcome, submitted.outcome, messaged.outcome] == [
        "sent",
        "sent",
        "sent",
    ]
    assert [call.kwargs["raw"] for call in mux.send_to_pane.await_args_list] == [
        True,
        True,
        False,
    ]


async def test_pi_followup_text_failure_does_not_send_alt_enter(mux, store):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["pi"], "/tmp")
    mux.send_to_pane.return_value = False
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]

    result, status = await service.execute_internal(
        session.id,
        Action(
            str(uuid4()),
            session.binding_version,
            "message.followup",
            {"text": "continue"},
        ),
    )

    assert status == 200 and result.outcome == "unknown"
    mux.send_to_pane.assert_awaited_once_with(
        "%1",
        "continue",
        enter=False,
        literal=True,
        window_id="@1",
        raw=False,
    )


async def test_pi_followup_key_failure_is_reported_after_text(mux, store, monkeypatch):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["pi"], "/tmp")
    mux.send_to_pane.side_effect = [True, False]
    sleep = AsyncMock()
    monkeypatch.setattr("ccgram.session_protocol.backend.asyncio.sleep", sleep)
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]

    result, status = await service.execute_internal(
        session.id,
        Action(
            str(uuid4()),
            session.binding_version,
            "message.followup",
            {"text": "continue"},
        ),
    )

    assert status == 200 and result.outcome == "unknown"
    sleep.assert_awaited_once_with(0.5)
    assert mux.send_to_pane.await_args_list == [
        call(
            "%1",
            "continue",
            enter=False,
            literal=True,
            window_id="@1",
            raw=False,
        ),
        call("%1", "M-Enter", enter=False, literal=False, window_id="@1"),
    ]


async def test_explicit_pane_is_revalidated_before_terminal_input(mux):
    sibling = PaneInfo("%2", 1, False, "claude", "/tmp", 80, 24)
    mux.list_panes.return_value = [
        PaneInfo("%1", 0, True, "claude", "/tmp", 80, 24),
        sibling,
    ]
    backend = MultiplexerSessionBackend(mux, "/tmp")
    target = await backend.inspect("@1")
    selected = await backend.select_pane(target, "%2")

    assert await backend.send_text(selected, "y", enter=True)
    mux.send_to_pane.assert_awaited_once_with(
        "%2",
        "y",
        enter=True,
        literal=True,
        window_id="@1",
        raw=False,
    )

    mux.list_panes.return_value = [
        PaneInfo("%1", 0, True, "claude", "/tmp", 80, 24),
        PaneInfo("%2", 1, False, "bash", "/tmp", 80, 24),
    ]
    mux.send_to_pane.reset_mock()
    with pytest.raises(ProtocolError) as error:
        await backend.send_text(selected, "danger", enter=True)

    assert error.value.code == "binding_conflict"
    mux.send_to_pane.assert_not_awaited()


@pytest.mark.parametrize("foreground", [None, ForegroundInfo(23, 23, [], "/tmp")])
async def test_unknown_foreground_does_not_advertise_actions(mux, store, foreground):
    mux.foreground.return_value = foreground
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]
    assert session.capabilities == ("session.close", "terminal.view")


@pytest.mark.parametrize(
    "argv,screen,allowed",
    [
        (["bash", "./script.sh"], "⌘0⌘", False),
        (["bash", "-c", "read value"], "⌘0⌘", False),
        (["bash"], "⌘0⌘read value", False),
        (["bash"], "Enter value:", False),
        (["bash"], "⌘0⌘", True),
    ],
)
async def test_resume_requires_a_bare_shell_and_idle_prompt(
    mux, store, argv, screen, allowed
):
    mux.foreground.return_value = ForegroundInfo(23, 23, argv, "/tmp")
    mux.capture_pane.return_value = screen
    mux.capture_pane_by_id.return_value = screen
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]
    assert ("session.resume" in session.capabilities) is allowed


async def test_resume_revalidates_shell_before_injection(mux):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
    mux.capture_pane.return_value = mux.capture_pane_by_id.return_value = "⌘0⌘"
    backend = MultiplexerSessionBackend(mux, "/tmp")
    target = await backend.inspect("@1")
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash", "script.sh"], "/tmp")
    with pytest.raises(ProtocolError):
        await backend.resume(target, "conv")
    mux.send_to_pane.assert_not_awaited()


async def test_gemini_working_title_is_used_for_snapshot(mux):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["gemini"], "/tmp")
    mux.get_pane_title.return_value = "Working: ✦"
    target = await MultiplexerSessionBackend(mux, "/tmp").inspect("@1")
    assert target.status.state == "working"


async def test_selected_pane_prompt_ignores_active_pane_title(mux):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["gemini"], "/tmp")
    mux.list_panes.return_value = [
        PaneInfo("%1", 0, True, "gemini", "/tmp", 80, 24),
        PaneInfo("%2", 1, False, "gemini", "/tmp", 80, 24),
    ]
    mux.get_pane_title.return_value = "Working: ✦"
    mux.capture_pane_by_id.return_value = (
        "Action Required\n"
        "? Shell echo hello\n"
        "Allow execution of: 'echo hello'?\n"
        "1. Allow once\n"
        "2. No, suggest changes (esc"
    )
    backend = MultiplexerSessionBackend(mux, "/tmp")
    target = await backend.inspect("@1")
    selected = await backend.select_pane(target, "%2")
    mux.get_pane_title.reset_mock()

    prompt = await backend.prompt(selected)
    assert prompt is not None
    mux.get_pane_title.assert_not_awaited()

    active_prompt = await backend.prompt(target)
    assert active_prompt is None
    mux.get_pane_title.assert_awaited_once_with("@1")


async def test_native_worktree_rejects_selected_workspace(mux, tmp_path):
    mux.capabilities.native_worktrees = True
    with pytest.raises(ProtocolError, match="workspace") as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {
                "directory": str(tmp_path),
                "provider": "claude",
                "worktree": True,
                "workspaceId": "selected",
            }
        )
    assert error.value.status == 422
    mux.create_worktree_window.assert_not_awaited()


@pytest.mark.parametrize(
    "kind,raw", [("message.send", False), ("session.resume", True)]
)
async def test_action_uses_agent_submission_or_raw_shell_input(
    mux, store, kind, raw, monkeypatch
):
    conversation = str(uuid4())
    identity = identity_state.get_identity("@1")
    assert identity is not None
    monkeypatch.setattr(
        identity_state,
        "get_identity",
        lambda _: replace(identity, session_id=conversation),
    )
    if raw:
        mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
        mux.capture_pane.return_value = mux.capture_pane_by_id.return_value = "⌘0⌘"
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]
    data = {"conversationId": conversation} if raw else {"text": "hello"}
    result, status = await service.execute(
        session.id, Action(str(uuid4()), session.binding_version, kind, data)
    )
    assert status == 200 and result.outcome == "sent"
    assert mux.send_to_pane.await_args.kwargs["raw"] is raw
    assert mux.send_to_pane.await_args.kwargs["window_id"] == "@1"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello", [("i", False, True), ("hello", False, True), ("", True, False)]),
        (
            "!ls",
            [
                ("i", False, True),
                ("!", False, True),
                ("ls", False, True),
                ("", True, False),
            ],
        ),
    ],
)
async def test_tmux_safe_message_stays_on_pinned_pane(monkeypatch, text, expected):
    manager = TmuxManager.__new__(TmuxManager)
    pane, other = MagicMock(), MagicMock()
    pane.pane_id, other.pane_id = "%1", "%2"
    pane.capture_pane.return_value = ["-- NORMAL --"]
    window = MagicMock()
    window.window_id, window.panes, window.active_pane = "@1", [pane, other], other
    session = MagicMock()
    session.windows.__iter__.side_effect = lambda: iter([window])
    session.windows.get.return_value = window
    manager.get_session = MagicMock(return_value=session)
    reset_vim_state()
    notify_vim_insert_seen("@1")
    sleeps = AsyncMock()
    monkeypatch.setattr("ccgram.multiplexer.tmux.asyncio.sleep", sleeps)
    try:
        assert await manager.send_to_pane("%1", text, window_id="@1", raw=False)
        assert pane.send_keys.call_args_list == [
            call(value, enter=enter, literal=literal)
            for value, enter, literal in expected
        ]
        other.send_keys.assert_not_called()
        assert call(0.5) in sleeps.await_args_list
    finally:
        reset_vim_state()


async def test_antigravity_bound_history_survives_shared_core_cutover(
    mux, store, tmp_path, monkeypatch
):
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        '{"id":"u1","type":"USER","content":"question"}\n'
        '{"id":"a1","type":"ASSISTANT","content":"answer"}\n'
    )
    mux.foreground.return_value = ForegroundInfo(0, 0, ["agy"], "/tmp")
    monkeypatch.setattr(
        identity_state,
        "get_identity",
        lambda _: IdentityProjection(
            "@1", "antigravity", "conv", "/tmp", path, "project", "normal"
        ),
    )
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    session = (await service.list_sessions())[0]
    page = await service.history(session.id, None, 20)
    assert session.provider == "antigravity"
    assert session.conversation is not None
    assert session.conversation.id == "conv"
    assert [(message.role, message.parts[0].text) for message in page.messages] == [
        ("user", "question"),
        ("agent", "answer"),
    ]


@pytest.mark.parametrize("clear", [True, False])
async def test_prompt_setup_uses_selected_pane_and_raw_submission(mux, store, clear):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
    mux.list_panes.return_value = [PaneInfo("%1", 0, True, "bash", "/tmp", 80, 24)]
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    target, snapshot = await adapter.resolve("@1")

    success, _ = await adapter.setup_prompt(
        "@1",
        "setup command",
        clear=clear,
        pane_id=target.pane_id,
        expected_binding_version=snapshot.binding_version,
    )

    assert success
    assert mux.send_to_pane.await_args_list == [
        call("%1", "C-c", enter=False, literal=False, window_id="@1"),
        call("%1", "setup command", enter=True, literal=True, window_id="@1", raw=True),
        *(
            [call("%1", "clear", enter=True, literal=True, window_id="@1", raw=True)]
            if clear
            else []
        ),
    ]


@pytest.mark.parametrize("change", ["provider", "pane", "script", "unavailable"])
async def test_prompt_setup_revalidates_before_second_write(mux, store, change):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
    mux.list_panes.return_value = [PaneInfo("%1", 0, True, "bash", "/tmp", 80, 24)]
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    target, snapshot = await adapter.resolve("@1")

    async def change_after_interrupt(*args, **kwargs):
        if change == "provider":
            mux.foreground.return_value = ForegroundInfo(24, 24, ["claude"], "/tmp")
        elif change == "pane":
            mux.list_panes.return_value = []
        elif change == "script":
            mux.foreground.return_value = ForegroundInfo(
                23, 23, ["bash", "script.sh"], "/tmp"
            )
        else:
            mux.foreground.return_value = None
        return True

    mux.send_to_pane.side_effect = change_after_interrupt
    success, _ = await adapter.setup_prompt(
        "@1",
        "setup command",
        clear=True,
        pane_id=target.pane_id,
        expected_binding_version=snapshot.binding_version,
    )

    assert not success
    mux.send_to_pane.assert_awaited_once_with(
        "%1", "C-c", enter=False, literal=False, window_id="@1"
    )


@pytest.mark.parametrize("change", ["binding", "provider", "pane"])
async def test_prompt_setup_stale_resolution_sends_nothing(mux, store, change):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
    mux.list_panes.return_value = [PaneInfo("%1", 0, True, "bash", "/tmp", 80, 24)]
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    target, snapshot = await adapter.resolve("@1")
    if change == "binding":
        mux.foreground.return_value = ForegroundInfo(24, 24, ["bash"], "/tmp")
    elif change == "provider":
        mux.foreground.return_value = ForegroundInfo(24, 24, ["claude"], "/tmp")
    else:
        mux.list_panes.return_value = []

    success, _ = await adapter.setup_prompt(
        "@1",
        "setup command",
        clear=True,
        pane_id=target.pane_id,
        expected_binding_version=snapshot.binding_version,
    )

    assert not success
    mux.send_to_pane.assert_not_awaited()


async def test_close_can_kill_confirmed_window_without_foreground_or_capture(
    mux, store
):
    mux.foreground.return_value = None
    mux.capture_pane_by_id.return_value = None
    mux.kill_window.return_value = True
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    _, snapshot = await adapter.resolve("@1")
    assert snapshot.capabilities == ("session.close", "terminal.view")

    success, _ = await adapter.close("@1")

    assert success
    mux.kill_window.assert_awaited_once_with("@1")


@pytest.mark.parametrize("windows", [None, []])
async def test_close_never_kills_when_native_presence_is_unconfirmed(
    mux, store, windows
):
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    await adapter.resolve("@1")
    mux.list_windows_for_reconciliation.return_value = windows

    success, _ = await adapter.close("@1")

    assert not success
    mux.kill_window.assert_not_awaited()


@pytest.mark.parametrize("clear", [True, False])
async def test_prompt_setup_without_pane_handles_uses_native_target(mux, store, clear):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
    mux.list_panes.return_value = []
    mux.send.return_value = True
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    target, snapshot = await adapter.resolve("@1")
    assert target.pane_id == ""

    success, _ = await adapter.setup_prompt(
        "@1",
        "setup command",
        clear=clear,
        pane_id=None,
        expected_binding_version=snapshot.binding_version,
    )

    assert success
    assert mux.send.await_args_list == [
        call("@1", "C-c", enter=False, literal=False, raw=True),
        call("@1", "setup command", enter=True, literal=True, raw=True),
        *([call("@1", "clear", enter=True, literal=True, raw=True)] if clear else []),
    ]
    mux.send_to_pane.assert_not_awaited()


@pytest.mark.parametrize("change", ["process", "script", "unavailable"])
@pytest.mark.parametrize("after_write", [1, 2])
async def test_prompt_setup_without_panes_rechecks_every_write(
    mux, store, change, after_write
):
    mux.foreground.return_value = ForegroundInfo(23, 23, ["bash"], "/tmp")
    mux.list_panes.return_value = []
    service = LocalSessionService(MultiplexerSessionBackend(mux, "/tmp"), store)
    adapter = adapter_for({"session_service": service})
    _, snapshot = await adapter.resolve("@1")

    async def change_after_write(*args, **kwargs):
        if mux.send.await_count == after_write:
            mux.foreground.return_value = (
                ForegroundInfo(24, 24, ["bash"], "/tmp")
                if change == "process"
                else ForegroundInfo(23, 23, ["bash", "script.sh"], "/tmp")
                if change == "script"
                else None
            )
        return True

    mux.send.side_effect = change_after_write
    success, _ = await adapter.setup_prompt(
        "@1",
        "setup command",
        clear=True,
        pane_id=None,
        expected_binding_version=snapshot.binding_version,
    )

    assert not success
    assert mux.send.await_count == after_write
    mux.send_to_pane.assert_not_awaited()


async def test_trusted_launch_command_is_used_as_is(mux, tmp_path):
    mux.capabilities.native_topic_targets = False
    mux.create_window.return_value = (True, "", "", "@1")
    await MultiplexerSessionBackend(mux, "/tmp").create(
        {"directory": str(tmp_path), "provider": "claude"},
        trusted_options=CreationOptions(launch_command="/admin/start-friend.sh XY"),
    )
    kwargs = mux.create_window.await_args.kwargs
    assert kwargs["launch_command"] == "/admin/start-friend.sh XY"
    assert kwargs["agent_args"] == ""


@pytest.mark.parametrize(
    "options",
    [
        CreationOptions(approval_mode="yolo", launch_command="/x"),
        CreationOptions(resume_id="r", launch_command="/x"),
        CreationOptions(continue_session=True, launch_command="/x"),
    ],
)
async def test_launch_command_refuses_mode_resume_and_continue(mux, tmp_path, options):
    with pytest.raises(ProtocolError) as error:
        await MultiplexerSessionBackend(mux, "/tmp").create(
            {"directory": str(tmp_path), "provider": "claude"},
            trusted_options=options,
        )
    assert error.value.status in (400, 422)
    mux.create_window.assert_not_awaited()
