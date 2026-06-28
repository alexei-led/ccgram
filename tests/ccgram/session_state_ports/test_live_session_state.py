"""Unit tests for session_state_ports.live_session_state.

Covers:
  - missing window (no snapshot, no wait header, no session, no activity)
  - hookless provider (session_lifecycle has no entry for window)
  - stale transcript path (session resolved but monitor returns no activity)
  - task-summary defaults (ClaudeTaskSnapshot totals when items is empty tuple)
  - full LiveSessionSnapshot construction
  - has_task_snapshot true/false branches

Patching strategy: the port functions use lazy imports, so patches target the
source modules (``ccgram.claude_task_state``, ``ccgram.session_lifecycle``,
``ccgram.session_monitor``) rather than names inside the port module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ccgram.claude_task_state import ClaudeTaskSnapshot
from ccgram.session_state_ports.live_session_state import (
    LiveSessionSnapshot,
    get_last_activity_ts,
    get_live_session_snapshot,
    get_session_id,
    get_task_snapshot,
    get_wait_header,
    has_task_snapshot,
)

# Patch targets — source modules, not the port module, because the port
# uses lazy imports (the names are not module-level attributes of the port).
_TASK_SNAPSHOT_FN = "ccgram.claude_task_state.get_claude_task_snapshot"
_WAIT_HEADER_FN = "ccgram.claude_task_state.get_claude_wait_header"
_TASK_STATE_STORE = "ccgram.claude_task_state.claude_task_state"
_SESSION_LIFECYCLE = "ccgram.session_lifecycle.session_lifecycle"
_GET_ACTIVE_MONITOR = "ccgram.session_monitor.get_active_monitor"


# ---------------------------------------------------------------------------
# get_task_snapshot
# ---------------------------------------------------------------------------


def test_get_task_snapshot_missing_window() -> None:
    """Returns None when window has no tracked tasks."""
    with patch(_TASK_SNAPSHOT_FN, return_value=None):
        assert get_task_snapshot("@99") is None


def test_get_task_snapshot_returns_snapshot() -> None:
    """Returns the ClaudeTaskSnapshot when tasks exist."""
    snap = ClaudeTaskSnapshot(items=(), done_count=0, open_count=0)
    with patch(_TASK_SNAPSHOT_FN, return_value=snap):
        assert get_task_snapshot("@1") is snap


# ---------------------------------------------------------------------------
# ClaudeTaskSnapshot defaults (task-summary defaults test)
# ---------------------------------------------------------------------------


def test_task_snapshot_defaults_empty_items() -> None:
    """ClaudeTaskSnapshot with no items reports zero totals."""
    snap = ClaudeTaskSnapshot(items=(), done_count=0, open_count=0)
    assert snap.total_count == 0
    assert snap.done_count == 0
    assert snap.open_count == 0
    assert snap.active_task_id is None


# ---------------------------------------------------------------------------
# has_task_snapshot
# ---------------------------------------------------------------------------


def test_has_task_snapshot_false_when_missing() -> None:
    mock_store = MagicMock()
    mock_store.has_snapshot.return_value = False
    with patch(_TASK_STATE_STORE, mock_store):
        assert has_task_snapshot("@99") is False


def test_has_task_snapshot_true_when_present() -> None:
    mock_store = MagicMock()
    mock_store.has_snapshot.return_value = True
    with patch(_TASK_STATE_STORE, mock_store):
        assert has_task_snapshot("@1") is True


# ---------------------------------------------------------------------------
# get_wait_header
# ---------------------------------------------------------------------------


def test_get_wait_header_none_when_missing() -> None:
    with patch(_WAIT_HEADER_FN, return_value=None):
        assert get_wait_header("@99") is None


def test_get_wait_header_returns_value() -> None:
    with patch(_WAIT_HEADER_FN, return_value="Waiting for input"):
        assert get_wait_header("@1") == "Waiting for input"


# ---------------------------------------------------------------------------
# get_session_id — hookless provider
# ---------------------------------------------------------------------------


def test_get_session_id_none_for_hookless_provider() -> None:
    """Hookless provider windows have no session_map entry → session_id is None."""
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = None
    with patch(_SESSION_LIFECYCLE, mock_lc):
        assert get_session_id("@hookless") is None


def test_get_session_id_returns_session() -> None:
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = "uuid-abc"
    with patch(_SESSION_LIFECYCLE, mock_lc):
        assert get_session_id("@1") == "uuid-abc"


# ---------------------------------------------------------------------------
# get_last_activity_ts — stale transcript / no monitor
# ---------------------------------------------------------------------------


def test_get_last_activity_ts_none_when_no_session() -> None:
    """No session_id → no activity timestamp (hookless / stale transcript path)."""
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = None
    with (
        patch(_SESSION_LIFECYCLE, mock_lc),
        patch(_GET_ACTIVE_MONITOR, return_value=None),
    ):
        assert get_last_activity_ts("@99") is None


def test_get_last_activity_ts_none_when_monitor_not_started() -> None:
    """Monitor not yet running → returns None without raising."""
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = "uuid-xyz"
    with (
        patch(_SESSION_LIFECYCLE, mock_lc),
        patch(_GET_ACTIVE_MONITOR, return_value=None),
    ):
        assert get_last_activity_ts("@1") is None


def test_get_last_activity_ts_stale_transcript_no_activity() -> None:
    """Monitor running but session has never been active → returns None."""
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = "uuid-xyz"
    mock_monitor = MagicMock()
    mock_monitor.get_last_activity.return_value = None
    with (
        patch(_SESSION_LIFECYCLE, mock_lc),
        patch(_GET_ACTIVE_MONITOR, return_value=mock_monitor),
    ):
        assert get_last_activity_ts("@1") is None
        mock_monitor.get_last_activity.assert_called_once_with("uuid-xyz")


def test_get_last_activity_ts_returns_timestamp() -> None:
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = "uuid-xyz"
    mock_monitor = MagicMock()
    mock_monitor.get_last_activity.return_value = 12345.6
    with (
        patch(_SESSION_LIFECYCLE, mock_lc),
        patch(_GET_ACTIVE_MONITOR, return_value=mock_monitor),
    ):
        assert get_last_activity_ts("@1") == pytest.approx(12345.6)


# ---------------------------------------------------------------------------
# get_live_session_snapshot — full construction
# ---------------------------------------------------------------------------


def test_get_live_session_snapshot_all_none() -> None:
    """Fully-missing window → all fields are None, no exception."""
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = None
    with (
        patch(_SESSION_LIFECYCLE, mock_lc),
        patch(_GET_ACTIVE_MONITOR, return_value=None),
        patch(_TASK_SNAPSHOT_FN, return_value=None),
        patch(_WAIT_HEADER_FN, return_value=None),
    ):
        snap = get_live_session_snapshot("@99")

    assert isinstance(snap, LiveSessionSnapshot)
    assert snap.window_id == "@99"
    assert snap.session_id is None
    assert snap.task_snapshot is None
    assert snap.wait_header is None
    assert snap.last_activity_ts is None


def test_get_live_session_snapshot_populated() -> None:
    """All fields populated when all sources return values."""
    task_snap = ClaudeTaskSnapshot(items=(), done_count=0, open_count=0)
    mock_lc = MagicMock()
    mock_lc.resolve_session_id.return_value = "uuid-abc"
    mock_monitor = MagicMock()
    mock_monitor.get_last_activity.return_value = 99.9
    with (
        patch(_SESSION_LIFECYCLE, mock_lc),
        patch(_GET_ACTIVE_MONITOR, return_value=mock_monitor),
        patch(_TASK_SNAPSHOT_FN, return_value=task_snap),
        patch(_WAIT_HEADER_FN, return_value="Waiting for input"),
    ):
        snap = get_live_session_snapshot("@1")

    assert snap.window_id == "@1"
    assert snap.session_id == "uuid-abc"
    assert snap.task_snapshot is task_snap
    assert snap.wait_header == "Waiting for input"
    assert snap.last_activity_ts == pytest.approx(99.9)


def test_get_live_session_snapshot_propagates_delegate_exception() -> None:
    """If a delegate raises, the exception propagates from get_live_session_snapshot."""
    with (
        patch(_TASK_SNAPSHOT_FN, side_effect=RuntimeError("delegate failure")),
        pytest.raises(RuntimeError, match="delegate failure"),
    ):
        get_live_session_snapshot("@test-window")


def test_live_session_snapshot_is_frozen() -> None:
    """LiveSessionSnapshot is immutable (frozen=True)."""
    snap = LiveSessionSnapshot(
        window_id="@1",
        session_id=None,
        task_snapshot=None,
        wait_header=None,
        last_activity_ts=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        snap.window_id = "@2"  # type: ignore[misc]
