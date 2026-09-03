"""Tests for multiplexer/tmux.py — Task 2 (tmux as the first backend).

Covers:
- tmux ``MultiplexerCapabilities`` pinned as a full snapshot (the Task 5
  characterization guard that the tmux behavior contract is unchanged).
- ``TmuxManager`` binds to the ``Multiplexer`` type.
- One round-trip per Protocol wrapper method (neutral value types in/out),
  with the libtmux/subprocess legacy methods mocked.
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.multiplexer.base import (
    CaptureResult,
    ForegroundInfo,
    Multiplexer,
    PaneDims,
    WindowRef,
)
from ccgram.config import config
from ccgram.multiplexer.tmux import TmuxManager


@pytest.fixture
def mgr() -> TmuxManager:
    """A fresh TmuxManager (does not touch the global singleton's state)."""
    return TmuxManager(session_name="ccgram-test")


# ── Capabilities ───────────────────────────────────────────────────────


def test_capabilities_full_snapshot(mgr: TmuxManager) -> None:
    """Characterization guard (Task 5): the entire tmux capability surface is
    locked. Any change to a flag is a behavior change and must fail here."""
    assert asdict(mgr.capabilities) == {
        "name": "tmux",
        "ids_stable_across_restart": True,
        "exposes_pane_tty": True,
        "native_agent_status": False,
        "read_max_lines": None,
        "self_identify_env": "TMUX_PANE",
        "supports_event_stream": False,
        "native_worktrees": False,
        "supports_display_name_rebind": True,
        "supports_workspace_selection": False,
        "native_topic_targets": False,
    }


class _FakeWindow:
    def __init__(self, window_id: str, window_name: str) -> None:
        self.window_id = window_id
        self.window_name = window_name

    @property
    def active_pane(self):
        return None


# ── Protocol conformance ───────────────────────────────────────────────


def test_tmux_manager_typed_as_multiplexer(mgr: TmuxManager) -> None:
    """A TmuxManager binds to the Multiplexer type (pyright structural check)."""
    backend: Multiplexer = mgr
    assert backend.capabilities.name == "tmux"


# ── Round-trips per wrapper method ─────────────────────────────────────


async def test_ensure_session_calls_get_or_create(mgr: TmuxManager) -> None:
    with patch.object(mgr, "get_or_create_session") as create:
        await mgr.ensure_session()
        create.assert_called_once_with()


async def test_reconciliation_listing_returns_empty_without_session(
    mgr: TmuxManager,
) -> None:
    """tmux answered, and the configured session is genuinely not there."""
    with patch("ccgram.multiplexer.tmux.fetch_objs", return_value=[]):
        assert await mgr.list_windows_for_reconciliation() == []


async def test_reconciliation_listing_is_unknown_when_tmux_cannot_be_listed(
    mgr: TmuxManager,
) -> None:
    """Through the real libtmux Server, because the stub cannot show this.

    ``Server.sessions`` wraps its listing in ``except Exception: pass`` and
    returns an empty QueryList, so a broken tmux looks exactly like a server
    with no sessions. A test that patches ``sessions.get`` to raise is testing
    a stub of the dependency, not the dependency: it passes either way. This
    one fails the listing where libtmux actually performs it, which is the
    shape a real outage has, and the answer must be unknown - the audit closes
    topics and prunes state on a confirmed empty listing.
    """
    import libtmux.server

    mgr._server = libtmux.server.Server()

    def _boom(*_args, **_kwargs):
        raise OSError("tmux unavailable")

    with (
        patch("libtmux.server.fetch_objs", _boom),
        patch("ccgram.multiplexer.tmux.fetch_objs", _boom),
    ):
        assert await mgr.list_windows_for_reconciliation() is None

    assert mgr._server is None


async def test_reconciliation_listing_is_complete(mgr: TmuxManager) -> None:
    """Every window in the session is listed, adoptable or not.

    The listing answers "which windows exist", and absence from it is what
    marks a binding a ghost for /sync Fix to close. The UI listing narrows to
    what discovery may adopt; this one must not, or renaming a bound window to
    the hidden ``_`` form silently makes it a closure candidate.
    """
    session = MagicMock()
    session.windows = [
        _FakeWindow("@0", config.tmux_main_window_name),
        _FakeWindow("@4", "_paused"),
        _FakeWindow("@5", "my-project"),
    ]

    server = MagicMock()
    server.sessions.get.return_value = session
    mgr._server = server

    windows = await mgr.list_windows_for_reconciliation()

    assert windows is not None
    assert [w.window_id for w in windows] == ["@0", "@4", "@5"]
    assert [w.topic_eligible for w in windows] == [False, False, True]


def test_reconciliation_window_keeps_identity_when_pane_query_fails() -> None:
    class FailedPaneWindow:
        window_id = "@7"
        window_name = "project"

        @property
        def active_pane(self):
            raise OSError("pane unavailable")

    window = TmuxManager._window_ref_for_reconciliation(FailedPaneWindow())  # type: ignore[arg-type]

    assert window is not None
    assert window.window_id == "@7"
    assert window.cwd == ""


@pytest.mark.parametrize(
    ("window_name", "window_id"),
    [
        ("_paused", "@4"),
        (config.tmux_main_window_name, "@0"),
    ],
)
def test_reconciliation_keeps_windows_list_windows_hides(
    window_name: str, window_id: str
) -> None:
    """A window the UI listing hides is still live, and must still be listed.

    Dropping it made a bound window vanish from liveness the moment it was
    renamed to the hidden form — and a binding whose window is absent from
    this listing is a ghost that /sync Fix offers to close. It comes back
    ineligible so discovery still refuses it.
    """
    window = TmuxManager._window_ref_for_reconciliation(  # type: ignore[arg-type]
        _FakeWindow(window_id, window_name)
    )

    assert window.window_id == window_id
    assert window.topic_eligible is False


def test_reconciliation_keeps_ccgram_own_window() -> None:
    with patch.object(config, "own_window_id", "@9"):
        window = TmuxManager._window_ref_for_reconciliation(  # type: ignore[arg-type]
            _FakeWindow("@9", "ccgram")
        )

    assert window.window_id == "@9"
    assert window.topic_eligible is False


def test_reconciliation_marks_an_ordinary_window_adoptable() -> None:
    window = TmuxManager._window_ref_for_reconciliation(  # type: ignore[arg-type]
        _FakeWindow("@5", "my-project")
    )

    assert window.topic_eligible is True


async def test_find_window_by_id_returns_windowref(mgr: TmuxManager) -> None:
    win = WindowRef(window_id="@3", window_name="proj", cwd="/tmp")
    mgr.list_windows_for_reconciliation = AsyncMock(return_value=[win])  # type: ignore[method-assign]
    result = await mgr.find_window_by_id("@3")
    assert result is win
    assert isinstance(result, WindowRef)


async def test_find_window_by_id_missing_returns_none(mgr: TmuxManager) -> None:
    mgr.list_windows_for_reconciliation = AsyncMock(return_value=[])  # type: ignore[method-assign]
    assert await mgr.find_window_by_id("@99") is None


async def test_find_window_by_id_resolves_a_window_list_windows_hides(
    mgr: TmuxManager,
) -> None:
    """Addressing a known id is not discovery.

    Every handler resolves an already-bound window through here. When this
    read the adoption-filtered listing, renaming a bound window to the hidden
    ``_`` form made it unreachable: no text delivered, no screenshot, no
    toolbar, and nothing in the logs pointing at the rename.
    """
    hidden = WindowRef(
        window_id="@3", window_name="_paused", cwd="/tmp", topic_eligible=False
    )
    mgr.list_windows = AsyncMock(return_value=[])  # type: ignore[method-assign]
    mgr.list_windows_for_reconciliation = AsyncMock(return_value=[hidden])  # type: ignore[method-assign]

    assert await mgr.find_window_by_id("@3") is hidden


async def test_find_window_by_id_none_listing_is_not_a_match(
    mgr: TmuxManager,
) -> None:
    """An unconfirmed listing is not proof the window is gone, but it is also
    not a window to drive: callers get None and retry on the next tick."""
    mgr.list_windows_for_reconciliation = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.find_window_by_id("@3") is None


async def test_capture_pane_plain_returns_text(mgr: TmuxManager) -> None:
    mgr._capture_pane_plain = AsyncMock(return_value="hello world")  # type: ignore[method-assign]
    result = await mgr.capture_pane("@0")
    assert result == "hello world"
    mgr._capture_pane_plain.assert_awaited_once_with("@0")


async def test_capture_pane_ansi_returns_text(mgr: TmuxManager) -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"hello world\n", b""))
    proc.returncode = 0
    create = AsyncMock(return_value=proc)
    with patch("asyncio.create_subprocess_exec", create):
        result = await mgr.capture_pane("@0", with_ansi=True)

    assert result == "hello world"
    awaited = create.await_args
    assert awaited is not None
    argv = awaited.args
    assert argv[:2] == ("tmux", "capture-pane")
    assert "-e" in argv
    assert "-p" in argv
    assert argv[argv.index("-t") + 1] == "@0"


async def test_capture_pane_none_passthrough(mgr: TmuxManager) -> None:
    mgr._capture_pane_plain = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.capture_pane("@0") is None


async def test_capture_scrollback_no_clamp_for_tmux(mgr: TmuxManager) -> None:
    mgr.capture_pane_scrollback = AsyncMock(return_value="line1\nline2")  # type: ignore[method-assign]
    result = await mgr.capture_scrollback("@0", lines=5000)
    assert isinstance(result, CaptureResult)
    assert result.text == "line1\nline2"
    # tmux read_max_lines is None → never truncates, history passed through.
    assert result.truncated is False
    mgr.capture_pane_scrollback.assert_awaited_once_with("@0", history=5000)


async def test_capture_scrollback_none_passthrough(mgr: TmuxManager) -> None:
    mgr.capture_pane_scrollback = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.capture_scrollback("@0") is None


async def test_send_forwards_to_send_keys(mgr: TmuxManager) -> None:
    mgr.send_keys = AsyncMock(return_value=True)  # type: ignore[method-assign]
    ok = await mgr.send("@0", "hi", enter=False, literal=True, raw=True)
    assert ok is True
    mgr.send_keys.assert_awaited_once_with(
        "@0", "hi", enter=False, literal=True, raw=True
    )


async def test_send_to_pane_forwards(mgr: TmuxManager) -> None:
    mgr.send_keys_to_pane = AsyncMock(return_value=True)  # type: ignore[method-assign]
    ok = await mgr.send_to_pane("%2", "hi", enter=True, literal=True, window_id="@0")
    assert ok is True
    mgr.send_keys_to_pane.assert_awaited_once_with(
        "%2", "hi", enter=True, literal=True, window_id="@0"
    )


async def test_stamp_pane_title_sets_tmux_title(mgr: TmuxManager) -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    create = AsyncMock(return_value=proc)
    with patch("asyncio.create_subprocess_exec", create):
        await mgr.stamp_pane_title("@0", "claude")

    awaited = create.await_args
    assert awaited is not None
    assert awaited.args == (
        "tmux",
        "select-pane",
        "-t",
        f"{mgr.session_name}:@0",
        "-T",
        "ccgram:claude",
    )


async def test_pane_dims_parses_dimensions(mgr: TmuxManager, monkeypatch) -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"220:50\n", b""))
    proc.returncode = 0
    create = AsyncMock(return_value=proc)
    monkeypatch.setattr("asyncio.create_subprocess_exec", create)

    dims = await mgr.pane_dims("@0")
    assert dims == PaneDims(width=220, height=50)


async def test_pane_dims_returns_none_on_error(mgr: TmuxManager, monkeypatch) -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b"no server"))
    proc.returncode = 1
    monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    assert await mgr.pane_dims("@0") is None


async def test_foreground_builds_info(mgr: TmuxManager) -> None:
    win = WindowRef(
        window_id="@0", window_name="proj", cwd="/work", pane_tty="/dev/ttys003"
    )
    mgr.find_window_by_id = AsyncMock(return_value=win)  # type: ignore[method-assign]
    mgr._ps_foreground = AsyncMock(return_value=(321, 321, ["claude", "--continue"]))  # type: ignore[method-assign]

    info = await mgr.foreground("@0")
    assert info == ForegroundInfo(
        pid=321,
        pgid=321,
        argv=["claude", "--continue"],
        cwd="/work",
        tty="/dev/ttys003",
    )


async def test_foreground_none_without_tty(mgr: TmuxManager) -> None:
    win = WindowRef(window_id="@0", window_name="proj", cwd="/work", pane_tty="")
    mgr.find_window_by_id = AsyncMock(return_value=win)  # type: ignore[method-assign]
    assert await mgr.foreground("@0") is None


async def test_foreground_none_when_window_gone(mgr: TmuxManager) -> None:
    mgr.find_window_by_id = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.foreground("@0") is None


# ── _parse_ps_line (pure) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(
            "321 321 S+ claude --continue",
            (321, 321, ["claude", "--continue"]),
            id="foreground-group-leader",
        ),
        pytest.param(
            "555 321 S+ node x",
            (555, 321, ["node", "x"]),
            id="foreground-non-leader",
        ),
        pytest.param("100 100 Ss bash", None, id="background"),
        pytest.param("garbage", None, id="too-few-columns"),
        pytest.param("abc def S+ claude", None, id="non-numeric-pid"),
    ],
)
def test_parse_ps_line(line: str, expected: tuple | None) -> None:
    assert TmuxManager._parse_ps_line(line) == expected
