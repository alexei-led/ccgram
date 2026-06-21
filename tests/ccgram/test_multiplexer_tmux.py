"""Tests for multiplexer/tmux.py — Task 2 (tmux as the first backend).

Covers:
- tmux ``MultiplexerCapabilities`` are pinned to the design values
  (per-field and as a full snapshot — the Task 5 characterization guard that
  the tmux behavior contract is unchanged).
- ``TmuxManager`` satisfies the ``Multiplexer`` Protocol structurally.
- One round-trip per Protocol wrapper method (neutral value types in/out),
  with the libtmux/subprocess legacy methods mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ccgram.multiplexer.base import (
    CaptureResult,
    ForegroundInfo,
    Multiplexer,
    PaneDims,
    WindowRef,
)
from ccgram.multiplexer.tmux import TmuxManager, tmux_manager


@pytest.fixture
def mgr() -> TmuxManager:
    """A fresh TmuxManager (does not touch the global singleton's state)."""
    return TmuxManager(session_name="ccgram-test")


# ── Capabilities ───────────────────────────────────────────────────────


class TestTmuxCapabilities:
    def test_capabilities_pinned(self, mgr: TmuxManager) -> None:
        caps = mgr.capabilities
        assert caps.name == "tmux"
        assert caps.ids_stable_across_restart is True
        assert caps.exposes_pane_tty is True
        assert caps.native_agent_status is False
        assert caps.read_max_lines is None
        assert caps.self_identify_env == "TMUX_PANE"
        assert caps.supports_event_stream is False

    def test_capabilities_is_frozen(self, mgr: TmuxManager) -> None:
        with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
            mgr.capabilities.name = "other"  # type: ignore[misc]

    def test_capabilities_full_snapshot(self, mgr: TmuxManager) -> None:
        """Characterization guard (Task 5): the entire tmux capability surface
        is locked. Any change to a flag is a behavior change and must fail here.
        """
        from dataclasses import asdict

        assert asdict(mgr.capabilities) == {
            "name": "tmux",
            "ids_stable_across_restart": True,
            "exposes_pane_tty": True,
            "native_agent_status": False,
            "read_max_lines": None,
            "self_identify_env": "TMUX_PANE",
            "supports_event_stream": False,
        }


# ── Protocol conformance ───────────────────────────────────────────────


def test_tmux_manager_satisfies_protocol() -> None:
    """The singleton is a structural Multiplexer (runtime_checkable)."""
    assert isinstance(tmux_manager, Multiplexer)


def test_tmux_manager_typed_as_multiplexer(mgr: TmuxManager) -> None:
    """A TmuxManager binds to the Multiplexer type (pyright structural check)."""
    backend: Multiplexer = mgr
    assert backend.capabilities.name == "tmux"


# ── Round-trips per wrapper method ─────────────────────────────────────


async def test_ensure_session_calls_get_or_create(mgr: TmuxManager) -> None:
    import unittest.mock as mock

    with mock.patch.object(mgr, "get_or_create_session") as create:
        await mgr.ensure_session()
        create.assert_called_once_with()


async def test_find_window_returns_windowref(mgr: TmuxManager) -> None:
    win = WindowRef(window_id="@3", window_name="proj", cwd="/tmp")
    mgr.find_window_by_id = AsyncMock(return_value=win)  # type: ignore[method-assign]
    result = await mgr.find_window("@3")
    assert result is win
    assert isinstance(result, WindowRef)


async def test_find_window_missing_returns_none(mgr: TmuxManager) -> None:
    mgr.find_window_by_id = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.find_window("@99") is None


@pytest.mark.parametrize("ansi", [False, True])
async def test_capture_wraps_text(mgr: TmuxManager, ansi: bool) -> None:
    mgr.capture_pane = AsyncMock(return_value="hello world")  # type: ignore[method-assign]
    result = await mgr.capture("@0", ansi=ansi)
    assert isinstance(result, CaptureResult)
    assert result.text == "hello world"
    assert result.truncated is False
    mgr.capture_pane.assert_awaited_once_with("@0", with_ansi=ansi)


async def test_capture_none_passthrough(mgr: TmuxManager) -> None:
    mgr.capture_pane = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.capture("@0") is None


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


async def test_set_title_forwards(mgr: TmuxManager) -> None:
    mgr.stamp_pane_title = AsyncMock()  # type: ignore[method-assign]
    await mgr.set_title("@0", "claude")
    mgr.stamp_pane_title.assert_awaited_once_with("@0", "claude")


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


class TestParsePsLine:
    def test_foreground_leader(self) -> None:
        # pid == pgid, "+" foreground stat
        line = "321 321 S+ claude --continue"
        assert TmuxManager._parse_ps_line(line) == (321, 321, ["claude", "--continue"])

    def test_non_foreground_skipped(self) -> None:
        assert TmuxManager._parse_ps_line("100 100 Ss bash") is None

    def test_malformed_line(self) -> None:
        assert TmuxManager._parse_ps_line("garbage") is None

    def test_non_numeric_pid(self) -> None:
        assert TmuxManager._parse_ps_line("abc def S+ claude") is None

    def test_foreground_non_leader(self) -> None:
        # foreground but pid != pgid → still parsed (fallback candidate)
        assert TmuxManager._parse_ps_line("555 321 S+ node x") == (
            555,
            321,
            ["node", "x"],
        )
