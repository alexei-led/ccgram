"""Tests for SessionMonitor."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.monitor_state import TrackedSession
from ccgram.multiplexer.base import MultiplexerCapabilities, WindowRef
from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider
from ccgram.session import SessionManager
from ccgram.session_monitor import NewWindowEvent, SessionMonitor
from ccgram.thread_router import thread_router
from ccgram.window_state_store import window_store


@pytest.fixture
def monitor(tmp_path) -> SessionMonitor:
    return SessionMonitor(
        projects_path=tmp_path / "projects",
        poll_interval=0.1,
        state_file=tmp_path / "monitor_state.json",
    )


class TestPendingToolsCleanup:
    async def test_cleanup_stale_removes_pending_tools(
        self, monitor: SessionMonitor
    ) -> None:
        monitor._pending_tools["stale-session"] = {"tool_1": {"name": "Read"}}
        monitor.state.update_session(
            TrackedSession(session_id="stale-session", file_path="/fake/path")
        )

        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value={},
        ):
            await monitor._cleanup_all_stale_sessions()

        assert "stale-session" not in monitor._pending_tools

    async def test_detect_changes_removes_pending_tools(
        self, monitor: SessionMonitor
    ) -> None:
        old_sid = "old-session"
        new_sid = "new-session"

        monitor._pending_tools[old_sid] = {"tool_1": {"name": "Write"}}
        monitor._last_session_map = {
            "my-window": {"session_id": old_sid, "cwd": "/a", "window_name": ""}
        }
        monitor.state.update_session(
            TrackedSession(session_id=old_sid, file_path="/fake/path")
        )

        new_map = {"my-window": {"session_id": new_sid, "cwd": "/a", "window_name": ""}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes()

        assert old_sid not in monitor._pending_tools


class TestNewWindowDetection:
    async def test_callback_fires_for_new_window(self, monitor: SessionMonitor) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monitor._last_session_map = {}

        new_map = {"@5": {"session_id": "s1", "cwd": "/proj", "window_name": "proj"}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes()

        cb.assert_called_once()
        event = cb.call_args[0][0]
        assert isinstance(event, NewWindowEvent)
        assert event.window_id == "@5"
        assert event.session_id == "s1"
        assert event.window_name == "proj"

    async def test_startup_does_not_trigger_callback(
        self, monitor: SessionMonitor
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        initial_map = {"@0": {"session_id": "s0", "cwd": "/a", "window_name": "a"}}
        monitor._last_session_map = initial_map

        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=initial_map,
        ):
            await monitor._detect_and_cleanup_changes()

        cb.assert_not_called()

    async def test_callback_error_does_not_crash(self, monitor: SessionMonitor) -> None:
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        monitor.set_new_window_callback(cb)
        monitor._last_session_map = {}

        new_map = {"@1": {"session_id": "s1", "cwd": "/x", "window_name": "x"}}
        with patch.object(
            monitor,
            "_load_current_session_map",
            spec=True,
            new_callable=AsyncMock,
            return_value=new_map,
        ):
            await monitor._detect_and_cleanup_changes()

        cb.assert_called_once()


_TMUX_CAPS = MultiplexerCapabilities(
    name="tmux",
    ids_stable_across_restart=True,
    exposes_pane_tty=True,
    native_agent_status=False,
    read_max_lines=None,
    self_identify_env="TMUX_PANE",
    supports_event_stream=False,
    native_worktrees=False,
)
_HERDR_CAPS = MultiplexerCapabilities(
    name="herdr",
    ids_stable_across_restart=False,
    exposes_pane_tty=False,
    native_agent_status=True,
    read_max_lines=1000,
    self_identify_env="HERDR_PANE_ID",
    supports_event_stream=True,
    native_worktrees=True,
)


def _winref(window_id: str, command: str) -> WindowRef:
    return WindowRef(
        window_id=window_id,
        window_name=window_id,
        cwd="/proj",
        pane_current_command=command,
    )


class TestEmitUnboundWindowEvents:
    """The unbound-window discovery path is capability-gated (Task 10)."""

    @pytest.fixture
    def wired(self, monkeypatch) -> None:
        # Wire thread_router via a real SessionManager and start empty.
        thread_router.reset()
        window_store.window_states.clear()
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()

    async def test_tmux_surfaces_every_unbound_window(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_TMUX_CAPS),
        )

        windows = [_winref("@1", "zsh"), _winref("@2", "claude")]
        await monitor._emit_unbound_window_events(windows, known_window_ids=set())

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {"@1", "@2"}

    async def test_herdr_surfaces_only_agent_panes(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        # w2:p1 + w2:p2 are agent panes (a tab split); w3:p1 is a bare shell.
        windows = [
            _winref("w2:p1", "claude"),
            _winref("w2:p2", "claude"),
            _winref("w3:p1", ""),
        ]
        await monitor._emit_unbound_window_events(windows, known_window_ids=set())

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {"w2:p1", "w2:p2"}

    async def test_skips_known_and_bound_windows(
        self, monitor: SessionMonitor, wired, monkeypatch
    ) -> None:
        thread_router.bind_thread(100, 1, "w2:p2")
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)
        monkeypatch.setattr(
            "ccgram.session_monitor.tmux_manager",
            SimpleNamespace(capabilities=_HERDR_CAPS),
        )

        windows = [
            _winref("w2:p1", "claude"),  # already in session_map (known)
            _winref("w2:p2", "claude"),  # already bound to a topic
            _winref("w2:p3", "claude"),  # genuinely new → surfaces
        ]
        await monitor._emit_unbound_window_events(windows, known_window_ids={"w2:p1"})

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {"w2:p3"}


class TestEmitKnownUnboundWindowEvents:
    """Steady-state self-heal: session_map windows not bound to a topic retry on each poll."""

    @pytest.fixture
    def wired(self, monkeypatch) -> None:
        thread_router.reset()
        window_store.window_states.clear()
        monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
        monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
        SessionManager()

    async def test_known_unbound_window_surfaces(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """A tab in session_map but not bound fires NewWindowEvent (self-heal path)."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {
                "session_id": "S1",
                "cwd": "/repo",
                "window_name": "agent",
            }
        }
        live_window_ids = {"w1:t1"}  # tab is live

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_called_once()
        event = cb.call_args[0][0]
        assert isinstance(event, NewWindowEvent)
        assert event.window_id == "w1:t1"
        assert event.session_id == "S1"
        assert event.window_name == "agent"

    async def test_bound_window_not_re_fired(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """A tab already bound to a topic is skipped (no spam)."""
        thread_router.bind_thread(100, 42, "w1:t1")
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/repo", "window_name": "agent"}
        }
        live_window_ids = {"w1:t1"}

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_not_called()

    async def test_dead_window_not_surfaced(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """A session_map entry for a window not in live_window_ids is skipped."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w9:t9": {"session_id": "S9", "cwd": "/gone", "window_name": "dead"}
        }
        live_window_ids: set[str] = set()  # tab not alive / __*__-filtered

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_not_called()

    async def test_star_tab_not_surfaced(self, monitor: SessionMonitor, wired) -> None:
        """__*__ tabs are absent from live_window_ids (filtered by list_windows) — never adopted."""
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        # Simulate a __*__ tab somehow in session_map; list_windows filtered it out
        current_map = {
            "w0:t0": {"session_id": "S0", "cwd": "/self", "window_name": "__main__"}
        }
        live_window_ids: set[str] = set()  # __*__ absent from list_windows output

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        cb.assert_not_called()

    async def test_multiple_windows_mixed(self, monitor: SessionMonitor, wired) -> None:
        """Only unbound live tabs surface; bound and dead are skipped."""
        thread_router.bind_thread(100, 1, "w1:t1")  # already bound
        cb = AsyncMock(spec=lambda event: None)
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/a", "window_name": "bound"},
            "w2:t2": {"session_id": "S2", "cwd": "/b", "window_name": "unbound"},
            "w3:t3": {"session_id": "S3", "cwd": "/c", "window_name": "dead"},
        }
        live_window_ids = {"w1:t1", "w2:t2"}  # w3:t3 not live

        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        surfaced = {c.args[0].window_id for c in cb.call_args_list}
        assert surfaced == {"w2:t2"}

    async def test_no_callback_is_noop(self, monitor: SessionMonitor, wired) -> None:
        """No callback registered — returns without error."""
        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/repo", "window_name": "agent"}
        }
        # No callback set — must not raise
        await monitor._emit_known_unbound_window_events(current_map, {"w1:t1"})

    async def test_callback_error_does_not_crash(
        self, monitor: SessionMonitor, wired
    ) -> None:
        """Callback error is caught and logged; loop continues."""
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        monitor.set_new_window_callback(cb)

        current_map = {
            "w1:t1": {"session_id": "S1", "cwd": "/a", "window_name": "a"},
            "w2:t2": {"session_id": "S2", "cwd": "/b", "window_name": "b"},
        }
        live_window_ids = {"w1:t1", "w2:t2"}

        # Should not raise despite the callback error
        await monitor._emit_known_unbound_window_events(current_map, live_window_ids)

        assert cb.call_count == 2


class TestLoadCurrentSessionMapBackend:
    """The monitor's session_map reader must honor the active backend prefix.

    Regression: under herdr the hook writes ``herdr:<wN:pM>`` keys; a tmux-only
    ``ccgram:`` prefix silently dropped every herdr session so none was tracked.
    """

    async def test_herdr_keys_surface(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        raw = {
            "herdr:w2:p1": {
                "session_id": "S1",
                "cwd": "/repo",
                "window_name": "agent",
                "transcript_path": "",
                "provider_name": "claude",
            }
        }
        result = await monitor._load_current_session_map(raw)
        assert result.get("w2:p1", {}).get("session_id") == "S1"

    async def test_tmux_skips_herdr_keys(
        self, monitor: SessionMonitor, monkeypatch
    ) -> None:
        from ccgram.config import config

        monkeypatch.setattr(config, "multiplexer_name", "tmux")
        raw = {"herdr:w2:p1": {"session_id": "S1", "cwd": "/repo"}}
        assert await monitor._load_current_session_map(raw) == {}


class TestPerWindowProviderResolution:
    async def test_process_session_file_passes_window_id(self, tmp_path) -> None:
        """_process_session_file uses window_id for per-window provider resolution."""
        session_file = tmp_path / "transcript.jsonl"
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n'
        session_file.write_text(line)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-pw",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages = []
        await monitor._process_session_file(
            "sess-pw", session_file, new_messages, window_id="@42"
        )
        assert len(new_messages) == 1
        assert "hello" in new_messages[0].text

    async def test_process_session_file_prefers_transcript_provider_when_stale(
        self, tmp_path
    ) -> None:
        """A stale hookful provider should not suppress Codex transcript parsing."""
        session_file = (
            tmp_path / ".codex" / "sessions" / "2026" / "03" / "23" / "transcript.jsonl"
        )
        session_file.parent.mkdir(parents=True)
        session_file.write_text(
            '{"timestamp":"2026-03-23T00:00:00Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello codex"}]}}\n'
        )

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-stale",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages = []
        with (
            patch(
                "ccgram.transcript_reader.get_provider_for_window",
                return_value=ClaudeProvider(),
            ),
            patch(
                "ccgram.transcript_reader.registry.is_valid",
                return_value=True,
            ),
            patch(
                "ccgram.transcript_reader.registry.get",
                return_value=CodexProvider(),
            ),
        ):
            await monitor._process_session_file(
                "sess-stale", session_file, new_messages, window_id="@42"
            )

        assert len(new_messages) == 1
        assert new_messages[0].text == "hello codex"

    async def test_check_for_updates_maps_session_to_window(self, tmp_path) -> None:
        """check_for_updates passes correct window_id to _process_session_file."""
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )

        captured_window_ids = []
        original = monitor._process_session_file

        async def spy(session_id, file_path, new_messages, window_id=""):
            captured_window_ids.append(window_id)
            return await original(
                session_id, file_path, new_messages, window_id=window_id
            )

        monitor._process_session_file = spy

        current_map = {
            "@7": {
                "session_id": "sess-map",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        await monitor.check_for_updates(current_map)
        assert "@7" in captured_window_ids


class TestReadNewLines:
    async def test_truncation_resets_offset(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        session_file.write_text(
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        )

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="t1",
            file_path=str(session_file),
            last_byte_offset=99999,
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        assert tracked.last_byte_offset < 99999
        assert len(entries) >= 1

    async def test_incremental_read_from_offset(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        line1 = '{"type":"assistant","message":{"content":[{"type":"text","text":"first"}]}}\n'
        line2 = '{"type":"assistant","message":{"content":[{"type":"text","text":"second"}]}}\n'
        session_file.write_text(line1 + line2)

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="t1",
            file_path=str(session_file),
            last_byte_offset=len(line1.encode()),
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        assert len(entries) == 1

    async def test_partial_line_stops_reading(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        good_line = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}\n'
        )
        session_file.write_text(good_line + '{"type":"ass')

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="t1", file_path=str(session_file), last_byte_offset=0
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        assert len(entries) == 1
        assert tracked.last_byte_offset == len(good_line.encode())


class TestCorruptedOffset:
    async def test_corrupted_offset_recovers(self, tmp_path) -> None:
        session_file = tmp_path / "test.jsonl"
        line1 = '{"type":"assistant","message":{"content":[{"type":"text","text":"first"}]}}\n'
        line2 = '{"type":"assistant","message":{"content":[{"type":"text","text":"second"}]}}\n'
        session_file.write_text(line1 + line2)

        monitor = SessionMonitor(
            projects_path=tmp_path,
            state_file=tmp_path / "ms.json",
        )
        # Set offset to mid-line1 (corrupted)
        tracked = TrackedSession(
            session_id="t1",
            file_path=str(session_file),
            last_byte_offset=10,
        )
        entries = await monitor._read_new_lines(tracked, session_file)
        # Should recover: skip rest of line1, read line2
        assert len(entries) == 1
        text = entries[0].get("message", {}).get("content", [{}])[0].get("text", "")
        assert text == "second"


class TestCheckForUpdates:
    async def test_new_session_initializes_to_eof_fallback(self, tmp_path) -> None:
        """Fallback path: entries without transcript_path use scan_projects."""
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        resolved = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-new.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": resolved,
            "entries": [
                {
                    "sessionId": "sess-new",
                    "fullPath": str(session_file),
                    "projectPath": resolved,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {"session_id": "sess-new", "cwd": resolved, "window_name": "proj"},
        }
        with patch.object(
            monitor,
            "_get_active_cwds",
            spec=True,
            new_callable=AsyncMock,
            return_value={resolved},
        ):
            msgs = await monitor.check_for_updates(current_map)

        assert msgs == []
        tracked = monitor.state.get_session("sess-new")
        assert tracked is not None
        assert tracked.last_byte_offset == session_file.stat().st_size

    async def test_new_session_initializes_to_eof_direct(self, tmp_path) -> None:
        """Primary path: entries with transcript_path are read directly."""
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-direct",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        msgs = await monitor.check_for_updates(current_map)

        assert msgs == []
        tracked = monitor.state.get_session("sess-direct")
        assert tracked is not None
        assert tracked.last_byte_offset == session_file.stat().st_size

    async def test_unchanged_mtime_skips_read(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproj"
        work_dir.mkdir()
        resolved = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproj"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": resolved,
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(session_file),
                    "projectPath": resolved,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        file_size = session_file.stat().st_size
        tracked = TrackedSession(
            session_id="sess-1",
            file_path=str(session_file),
            last_byte_offset=file_size,
        )
        monitor.state.update_session(tracked)
        monitor._file_mtimes["sess-1"] = session_file.stat().st_mtime

        current_map = {
            "@0": {"session_id": "sess-1", "cwd": resolved, "window_name": "proj"},
        }
        with (
            patch.object(
                monitor,
                "_get_active_cwds",
                spec=True,
                new_callable=AsyncMock,
                return_value={resolved},
            ),
            patch.object(
                monitor._transcript_reader,
                "_read_new_lines",
                spec=True,
                new_callable=AsyncMock,
            ) as mock_read,
        ):
            await monitor.check_for_updates(current_map)

        mock_read.assert_not_called()

    async def test_same_mtime_but_larger_size_triggers_read(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        projects_path.mkdir()

        session_file = tmp_path / "sess-1.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        original_mtime = session_file.stat().st_mtime

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-1",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)
        monitor._file_mtimes["sess-1"] = original_mtime

        # Append content without changing mtime (simulate sub-second write)
        with open(session_file, "a") as f:
            f.write(
                '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
            )
        os.utime(session_file, (original_mtime, original_mtime))

        current_map = {
            "@0": {
                "session_id": "sess-1",
                "cwd": str(tmp_path),
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        with patch.object(
            monitor._transcript_reader,
            "_read_new_lines",
            spec=True,
            new_callable=AsyncMock,
        ) as mock_read:
            await monitor.check_for_updates(current_map)

        mock_read.assert_called_once()

    async def test_direct_path_reads_new_content(self, tmp_path) -> None:
        """Primary path reads new content from transcript_path."""
        session_file = tmp_path / "transcript.jsonl"
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n'
        session_file.write_text(line)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        # Pre-track at offset 0 so it reads the content
        tracked = TrackedSession(
            session_id="sess-d",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        current_map = {
            "@1": {
                "session_id": "sess-d",
                "cwd": "/proj",
                "window_name": "proj",
                "transcript_path": str(session_file),
            },
        }
        msgs = await monitor.check_for_updates(current_map)

        assert len(msgs) == 1
        assert msgs[0].session_id == "sess-d"
        assert "hello" in msgs[0].text


class TestCheckForUpdatesExceptionResilience:
    async def test_error_in_one_session_does_not_block_others(self, tmp_path) -> None:
        good_file = tmp_path / "good.jsonl"
        good_file.write_text(
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        )
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-bad",
                "cwd": "/proj",
                "window_name": "bad",
                "transcript_path": str(bad_file),
            },
            "@1": {
                "session_id": "sess-good",
                "cwd": "/proj2",
                "window_name": "good",
                "transcript_path": str(good_file),
            },
        }

        original = monitor._process_session_file

        async def _blow_up(session_id, *args, **kwargs):
            if session_id == "sess-bad":
                raise TypeError("simulated provider bug")
            return await original(session_id, *args, **kwargs)

        with patch.object(monitor, "_process_session_file", side_effect=_blow_up):
            await monitor.check_for_updates(current_map)

        assert monitor.state.get_session("sess-good") is not None
        assert monitor.state.get_session("sess-bad") is None

    async def test_error_in_direct_session_still_saves_state(self, tmp_path) -> None:
        good_file = tmp_path / "good.jsonl"
        good_file.write_text('{"type":"summary"}\n')
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        current_map = {
            "@0": {
                "session_id": "sess-good",
                "cwd": "/proj",
                "window_name": "good",
                "transcript_path": str(good_file),
            },
            "@1": {
                "session_id": "sess-bad",
                "cwd": "/proj2",
                "window_name": "bad",
                "transcript_path": str(bad_file),
            },
        }

        original = monitor._process_session_file

        async def _blow_up(session_id, *args, **kwargs):
            if session_id == "sess-bad":
                raise ValueError("corrupt transcript")
            return await original(session_id, *args, **kwargs)

        with patch.object(monitor, "_process_session_file", side_effect=_blow_up):
            await monitor.check_for_updates(current_map)

        assert monitor.state.get_session("sess-good") is not None


class TestActivityTracking:
    def test_get_last_activity_returns_none_for_unknown(
        self, monitor: SessionMonitor
    ) -> None:
        assert monitor.get_last_activity("unknown-session") is None

    async def test_get_last_activity_updated_after_new_entries(self, tmp_path) -> None:
        session_file = tmp_path / "transcript.jsonl"
        line = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        )
        session_file.write_text(line)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-act",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages: list = []
        await monitor._process_session_file(
            "sess-act", session_file, new_messages, window_id="@1"
        )
        last = monitor.get_last_activity("sess-act")
        assert last is not None
        assert last > 0

    async def test_get_last_activity_not_updated_without_entries(
        self, tmp_path
    ) -> None:
        session_file = tmp_path / "transcript.jsonl"
        session_file.write_text("")

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="sess-empty",
            file_path=str(session_file),
            last_byte_offset=0,
        )
        monitor.state.update_session(tracked)

        new_messages: list = []
        await monitor._process_session_file(
            "sess-empty", session_file, new_messages, window_id="@1"
        )
        assert monitor.get_last_activity("sess-empty") is None


class TestScanProjects:
    def test_scan_projects_sync_reads_index(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproject"
        work_dir.mkdir()
        resolved_cwd = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproject"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-123.jsonl"
        session_file.write_text('{"type":"summary"}\n')

        index = {
            "originalPath": resolved_cwd,
            "entries": [
                {
                    "sessionId": "sess-123",
                    "fullPath": str(session_file),
                    "projectPath": resolved_cwd,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        active_cwds = {resolved_cwd}
        result = monitor._scan_projects_sync(active_cwds)

        assert len(result) == 1
        assert result[0].session_id == "sess-123"

    def test_scan_projects_sync_picks_up_unindexed_jsonl(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        work_dir = tmp_path / "myproject"
        work_dir.mkdir()
        resolved_cwd = str(work_dir.resolve())

        proj_dir = projects_path / "-tmp-myproject"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "orphan-sess.jsonl"
        jsonl.write_text(json.dumps({"cwd": resolved_cwd}) + "\n")

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        active_cwds = {resolved_cwd}
        result = monitor._scan_projects_sync(active_cwds)

        assert len(result) == 1
        assert result[0].session_id == "orphan-sess"

    def test_scan_projects_sync_filters_by_active_cwds(self, tmp_path) -> None:
        projects_path = tmp_path / "projects"
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        resolved_other = str(other_dir.resolve())

        proj_dir = projects_path / "-tmp-other"
        proj_dir.mkdir(parents=True)

        session_file = proj_dir / "sess-456.jsonl"
        session_file.write_text('{"type":"summary"}\n')
        index = {
            "originalPath": resolved_other,
            "entries": [
                {
                    "sessionId": "sess-456",
                    "fullPath": str(session_file),
                    "projectPath": resolved_other,
                }
            ],
        }
        (proj_dir / "sessions-index.json").write_text(json.dumps(index))

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        active_cwds = {str((tmp_path / "myproject").resolve())}
        result = monitor._scan_projects_sync(active_cwds)

        assert len(result) == 0

    def test_scan_projects_sync_skips_unindexed_jsonl_without_cwd(
        self, tmp_path
    ) -> None:
        projects_path = tmp_path / "projects"
        proj_dir = projects_path / "-tmp-my-project"
        proj_dir.mkdir(parents=True)

        jsonl = proj_dir / "orphan.jsonl"
        jsonl.write_text('{"type":"summary"}\n')

        monitor = SessionMonitor(
            projects_path=projects_path,
            state_file=tmp_path / "ms.json",
        )
        # active_cwds value is irrelevant — the skip happens before cwd matching
        active_cwds = {"anything"}
        result = monitor._scan_projects_sync(active_cwds)
        assert result == []

    def test_scan_projects_sync_skips_missing_dir(self, tmp_path) -> None:
        monitor = SessionMonitor(
            projects_path=tmp_path / "nonexistent",
            state_file=tmp_path / "ms.json",
        )
        result = monitor._scan_projects_sync({"/tmp/something"})
        assert result == []


class TestGeminiTranscriptReading:
    """Test _read_new_lines delegation for Gemini with supports_incremental_read=True."""

    _GEMINI_META = {
        "sessionId": "g1",
        "projectHash": "h1",
    }
    _GEMINI_MESSAGES = [
        {"type": "user", "content": [{"text": "hello"}]},
        {"type": "gemini", "content": [{"text": "hi there"}]},
        {"type": "user", "content": [{"text": "thanks"}]},
    ]

    def _write_jsonl(self, path, meta, messages):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    async def test_gemini_reads_jsonl_incrementally(self, tmp_path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        self._write_jsonl(transcript, self._GEMINI_META, self._GEMINI_MESSAGES)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        tracked = TrackedSession(
            session_id="g1",
            file_path=str(transcript),
            last_byte_offset=0,
        )

        with patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=_make_gemini_provider(),
        ):
            # First read: gets everything
            entries = await monitor._read_new_lines(tracked, transcript, window_id="@5")
            assert len(entries) == 4  # meta + 3 messages
            assert entries[0]["sessionId"] == "g1"
            assert entries[1]["type"] == "user"

            # Second read: nothing new
            entries = await monitor._read_new_lines(tracked, transcript, window_id="@5")
            assert len(entries) == 0

            # Third read: append a message
            with open(transcript, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"type": "gemini", "content": [{"text": "bye"}]}) + "\n"
                )

            entries = await monitor._read_new_lines(tracked, transcript, window_id="@5")
            assert len(entries) == 1
            assert entries[0]["type"] == "gemini"

    async def test_gemini_end_to_end_process_session(self, tmp_path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        self._write_jsonl(transcript, self._GEMINI_META, self._GEMINI_MESSAGES)

        monitor = SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "ms.json",
        )
        # We start tracking with offset pointing at the end of the file
        # (simulating a session that was already active at startup)
        st = transcript.stat()
        tracked = TrackedSession(
            session_id="g1",
            file_path=str(transcript),
            last_byte_offset=st.st_size,
        )
        monitor.state.update_session(tracked)

        # Append new message
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"type": "gemini", "content": [{"text": "new!"}]}) + "\n"
            )

        new_messages: list = []
        with patch(
            "ccgram.transcript_reader.get_provider_for_window",
            return_value=_make_gemini_provider(),
        ):
            await monitor._process_session_file(
                "g1", transcript, new_messages, window_id="@5"
            )

        assert len(new_messages) == 1
        assert new_messages[0].text == "new!"
        assert new_messages[0].role == "assistant"


def _make_gemini_provider():
    from ccgram.providers.gemini import GeminiProvider

    return GeminiProvider()
