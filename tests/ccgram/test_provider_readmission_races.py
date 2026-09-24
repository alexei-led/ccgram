"""Exercise provider readmission through real map and state operations."""

import json
import os
import time
from unittest.mock import AsyncMock

import pytest

import ccgram.session_monitor as monitor_module
from ccgram.config import config
from ccgram.handlers import agent_command
from ccgram.session_map import (
    SessionMapSync,
    get_session_map_sync,
    install_session_map_sync,
)
from ccgram.thread_router import ThreadRouter, get_thread_router, install_thread_router
from ccgram.window_state_store import (
    WindowState,
    WindowStateStore,
    get_window_store,
    install_window_store,
)

WINDOW = "@71"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    old_store, old_sync, old_router = (
        get_window_store(),
        get_session_map_sync(),
        get_thread_router(),
    )
    sync = SessionMapSync(schedule_save=lambda: None)

    def clear_entry(window_id):
        sync.clear_session_map_entry(window_id)

    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=clear_entry
    )
    router = ThreadRouter(
        schedule_save=lambda: None,
        has_window_state=store.has_window,
        default_group_id=-100,
    )
    install_window_store(store)
    install_session_map_sync(sync)
    install_thread_router(router)
    monkeypatch.setattr(config, "multiplexer_name", "tmux")
    monkeypatch.setattr(config, "tmux_session_name", "ccgram")
    monkeypatch.setattr(config, "session_map_file", tmp_path / "session_map.json")
    monkeypatch.setattr(config, "monitor_state_file", tmp_path / "monitor.json")
    parent = tmp_path / ".claude/projects/test/primary.jsonl"
    parent.parent.mkdir(parents=True)
    parent.write_text('{"type":"assistant"}\n')
    os.utime(parent, (time.time() - 2, time.time() - 2))
    store.window_states[WINDOW] = WindowState(
        provider_name="claude",
        initial_provider_name="claude",
        session_id="primary",
        transcript_path=str(parent),
        cwd=str(tmp_path),
        provider_manual_override=True,
    )
    router.bind_thread(1, 71, WINDOW, chat_id=-100)
    config.session_map_file.write_text(
        json.dumps(
            {
                f"ccgram:{WINDOW}": {
                    "session_id": "primary",
                    "provider_name": "claude",
                    "transcript_path": str(parent),
                    "cwd": str(tmp_path),
                    "window_name": "test",
                }
            }
        )
    )
    monkeypatch.setattr(
        "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup", AsyncMock()
    )
    yield store, sync, parent
    install_window_store(old_store)
    install_session_map_sync(old_sync)
    install_thread_router(old_router)


@pytest.mark.parametrize("selection", ["claude", "auto"])
async def test_same_provider_selection_does_not_replace_primary_with_nested(
    runtime, monkeypatch, selection
):
    store, _sync, parent = runtime
    nested = parent.with_name("nested.jsonl")
    nested.write_text('{"type":"assistant"}\n')
    config.session_map_file.write_text(
        json.dumps(
            {
                f"ccgram:{WINDOW}": {
                    "session_id": "nested",
                    "provider_name": "claude",
                    "transcript_path": str(nested),
                    "cwd": str(parent.parent),
                    "window_name": "test",
                }
            }
        )
    )
    monkeypatch.setattr(
        agent_command, "_redetect_provider", AsyncMock(return_value="claude")
    )

    provider, _reply = await agent_command._apply_switch(WINDOW, selection)

    assert provider == "claude"
    assert store.window_states[WINDOW].session_id == "primary"
    assert store.window_states[WINDOW].transcript_path == str(parent)


@pytest.mark.parametrize("switch_at", ["hook", "post_hook_read"])
async def test_monitor_does_not_reload_snapshot_from_before_auto(
    runtime, monkeypatch, switch_at
):
    store, sync, _parent = runtime
    monitor = monitor_module.SessionMonitor(poll_interval=0)
    real_read = monitor_module.read_session_map_raw
    hook_finished = False
    snapshot_race_triggered = False
    monkeypatch.setattr(
        agent_command, "_redetect_provider", AsyncMock(return_value="shell")
    )

    async def select_auto():
        provider, _reply = await agent_command._apply_switch(WINDOW, "auto")
        assert provider == "shell"
        assert store.window_states[WINDOW].provider_name == "shell"
        assert json.loads(config.session_map_file.read_text()) == {}

    async def read_map():
        nonlocal snapshot_race_triggered
        snapshot = await real_read()
        if (
            switch_at == "post_hook_read"
            and hook_finished
            and not snapshot_race_triggered
        ):
            snapshot_race_triggered = True
            await select_auto()
        return snapshot

    async def process_hooks():
        nonlocal hook_finished
        if switch_at == "hook":
            await select_auto()
        hook_finished = True
        monitor._running = False

    monkeypatch.setattr(monitor_module, "read_session_map_raw", read_map)
    monkeypatch.setattr(
        monitor_module, "list_windows_for_reconciliation", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(monitor, "_cleanup_all_stale_sessions", AsyncMock())
    monkeypatch.setattr(monitor, "_read_hook_events", process_hooks)
    monkeypatch.setattr(
        monitor, "_detect_and_cleanup_changes", AsyncMock(return_value={})
    )
    monkeypatch.setattr(monitor, "check_for_updates", AsyncMock(return_value=[]))
    monitor._running = True

    await monitor._monitor_loop()

    assert store.window_states[WINDOW].provider_name == "shell"
    assert store.window_states[WINDOW].session_id == ""
    assert not store.window_states[WINDOW].provider_manual_override
