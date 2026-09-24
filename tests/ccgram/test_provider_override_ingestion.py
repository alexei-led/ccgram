"""Manual routing choices must survive independent hook ingestion paths."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from ccgram.config import config
from ccgram.session import session_manager  # noqa: F401 — wire the real state ports
from ccgram.session_map import SessionMapSync, observed_provider, parse_session_map
from ccgram.session_monitor import SessionMonitor, TrackedSession
from ccgram.handlers.messaging_pipeline.message_task import ContentTask
from ccgram.handlers.messaging_pipeline import message_queue
from ccgram.providers.pi import PiProvider
from ccgram.window_state_ports import identity_state
from ccgram.transcript_reader import _StartupBoundary, _prefix_digest
from ccgram.window_state_store import (
    WindowState,
    WindowStateStore,
    get_window_store,
    install_window_store,
)

WINDOW = "@876"


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"provider_name": None, "transcript_path": None},
        {"provider_name": [], "transcript_path": {}},
    ],
)
def test_missing_or_malformed_provider_metadata_is_unknown(info):
    assert observed_provider(info) == ""


@pytest.fixture
def selected(monkeypatch):
    previous = get_window_store()
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
    )
    install_window_store(store)
    monkeypatch.setattr(config, "multiplexer_name", "tmux")
    state = WindowState(
        provider_name="shell",
        provider_manual_override=True,
        session_id="selected-session",
        cwd="/selected",
        window_name="selected",
    )
    store.window_states[WINDOW] = state
    yield state
    install_window_store(previous)


def incoming(provider="pi", path="/home/test/.pi/agent/sessions/project/session.jsonl"):
    return {
        "session_id": "hook-session",
        "provider_name": provider,
        "cwd": "/incoming",
        "window_name": "incoming",
        "transcript_path": path,
    }


async def test_manual_shell_rejects_pi_hook_without_any_identity_mutation(selected):
    before = selected.to_dict()
    prefix = f"{config.tmux_session_name}:"
    raw = {prefix + WINDOW: incoming()}
    sync = SessionMapSync(schedule_save=lambda: None)
    await sync.load_session_map(raw)
    assert selected.to_dict() == before
    assert parse_session_map(raw, prefix) == {}


@pytest.mark.parametrize("chosen,allowed", [("shell", False), ("pi", True)])
async def test_hook_events_obey_manual_provider_before_dispatch(
    selected, monkeypatch, tmp_path, chosen, allowed
):
    selected.provider_name = chosen
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "ts": 1,
                "event": "Stop",
                "session_id": "hook-session",
                "window_key": f"{config.tmux_session_name}:{WINDOW}",
                "data": {"provider_name": "pi"},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(config, "events_file", path)
    monkeypatch.setattr(config, "monitor_state_file", tmp_path / "monitor.json")
    monitor = SessionMonitor()
    callback = AsyncMock()
    monitor.set_hook_event_callback(callback)
    await monitor._read_hook_events()
    assert callback.await_count == int(allowed)
    assert monitor.state.events_offset == path.stat().st_size


@pytest.mark.parametrize("chosen,allowed", [("shell", False), ("pi", True)])
def test_transcript_delivery_obeys_manual_provider(
    selected, monkeypatch, tmp_path, chosen, allowed
):
    selected.provider_name = chosen
    monkeypatch.setattr(config, "monitor_state_file", tmp_path / "monitor.json")
    provider = PiProvider()
    entry = provider.parse_transcript_line(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "live reply"}],
                },
            }
        )
    )
    assert entry is not None
    messages = []
    SessionMonitor()._transcript_reader._append_provider_messages(
        "hook-session",
        [entry],
        provider,
        {},
        WINDOW,
        messages,
    )
    assert [m.text for m in messages] == (["live reply"] if allowed else [])
    if allowed:
        assert messages[0].provider_name == "pi"


@pytest.mark.parametrize("already_tracked", [False, True])
async def test_rejected_transcript_does_not_consume_any_offset(
    selected, monkeypatch, tmp_path, already_tracked
):
    monkeypatch.setattr(config, "monitor_state_file", tmp_path / "monitor.json")
    path = tmp_path / ".pi/agent/sessions/project/live.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "must wait"}],
                },
            }
        )
        + "\n"
    )
    monitor = SessionMonitor()
    if already_tracked:
        monitor.state.update_session(
            TrackedSession(
                session_id="hook-session", file_path=str(path), last_byte_offset=0
            )
        )
    output = []
    await monitor._process_session_file("hook-session", path, output, window_id=WINDOW)
    tracked = monitor.state.get_session("hook-session")
    assert output == []
    if already_tracked:
        assert (
            tracked is not None
            and tracked.last_byte_offset == 0
            and tracked.parsed_offset == -1
        )
    else:
        assert tracked is None


async def test_manual_switch_during_file_read_preserves_offset(
    selected, monkeypatch, tmp_path
):
    selected.provider_name = "pi"
    monkeypatch.setattr(config, "monitor_state_file", tmp_path / "monitor.json")
    path = tmp_path / ".pi/agent/sessions/project/live.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "must wait"}],
                },
            }
        )
        + "\n"
    )
    monitor = SessionMonitor()
    tracked = TrackedSession(
        session_id="hook-session", file_path=str(path), last_byte_offset=0
    )
    monitor.state.update_session(tracked)
    output = []
    reading = asyncio.create_task(
        monitor._process_session_file("hook-session", path, output, window_id=WINDOW)
    )
    asyncio.get_running_loop().call_soon(setattr, selected, "provider_name", "shell")
    await reading
    assert output == []
    assert tracked.parsed_offset == -1


async def test_manual_switch_during_generation_probe_restores_offset_and_boundary(
    selected, monkeypatch, tmp_path
):
    selected.provider_name = "pi"
    monkeypatch.setattr(config, "monitor_state_file", tmp_path / "monitor.json")
    path = tmp_path / ".pi/agent/sessions/project/live.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("old transcript\n")
    old_size = path.stat().st_size
    old_digest = _prefix_digest(path, old_size)
    path.write_text("replacement transcript longer than the original\n")
    stat = path.stat()
    monitor = SessionMonitor()
    reader = monitor._transcript_reader
    tracked = TrackedSession(
        session_id="hook-session",
        file_path=str(path),
        last_byte_offset=old_size,
        parsed_offset=old_size,
    )
    monitor.state.update_session(tracked)
    reader._file_prefixes["hook-session"] = (old_size, old_digest)
    boundary = _StartupBoundary(size=old_size, device=stat.st_dev, inode=stat.st_ino)
    reader._startup_file_boundaries["hook-session"] = boundary
    output = []
    reading = asyncio.create_task(
        monitor._process_session_file("hook-session", path, output, window_id=WINDOW)
    )
    asyncio.get_running_loop().call_soon(setattr, selected, "provider_name", "shell")
    await reading
    assert output == []
    assert tracked.parsed_offset == old_size
    assert reader._startup_file_boundaries["hook-session"] == boundary


@pytest.mark.parametrize(
    "manual,source_provider,current_provider,dropped",
    [
        (True, "pi", "shell", True),
        (False, "pi", "shell", False),
        (True, "shell", "pi", True),
        (True, "pi", "pi", False),
        (False, "claude", "claude", False),
        (True, None, "shell", False),
    ],
)
def test_queued_content_is_retired_only_when_provider_changes(
    selected, monkeypatch, manual, source_provider, current_provider, dropped
):
    selected.provider_name = current_provider
    selected.provider_manual_override = manual
    selected.session_id = "rolled-session"
    monkeypatch.setattr(message_queue, "is_window_live", lambda _wid: True)
    task = ContentTask(
        window_id=WINDOW,
        parts=("queued",),
        source_session_id="old-session",
        source_provider_name=source_provider,
    )
    assert message_queue._is_stale_task(1, task) is dropped


def test_override_lookup_matches_case_variant_hook_window_id(selected):
    store = get_window_store()
    del store.window_states[WINDOW]
    store.window_states["ABCD-1234"] = selected
    selected.provider_name = "shell"
    selected.provider_manual_override = True
    assert not identity_state.accepts_provider_observation("abcd-1234", "pi")
    assert identity_state.accepts_provider_observation("abcd-1234", "shell")


async def test_same_provider_hook_remains_live_with_manual_selection(selected):
    selected.provider_name = "pi"
    prefix = f"{config.tmux_session_name}:"
    raw = {prefix + WINDOW: incoming()}
    await SessionMapSync(schedule_save=lambda: None).load_session_map(raw)
    assert selected.provider_name == "pi" and selected.provider_manual_override
    assert selected.session_id == "hook-session"
    assert parse_session_map(raw, prefix)[WINDOW]["session_id"] == "hook-session"


async def test_auto_mode_still_adopts_incoming_provider(selected):
    selected.provider_manual_override = False
    prefix = f"{config.tmux_session_name}:"
    await SessionMapSync(schedule_save=lambda: None).load_session_map(
        {prefix + WINDOW: incoming()}
    )
    assert selected.provider_name == "pi"
    assert selected.session_id == "hook-session"


async def test_transcript_identity_cannot_spoof_manual_shell(selected):
    before = selected.to_dict()
    prefix = f"{config.tmux_session_name}:"
    raw = {prefix + WINDOW: incoming(provider="shell")}
    await SessionMapSync(schedule_save=lambda: None).load_session_map(raw)
    assert selected.to_dict() == before
    assert parse_session_map(raw, prefix) == {}
