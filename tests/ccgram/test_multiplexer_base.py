"""Tests for multiplexer/base.py — Task 1 fitness gate (F3).

Covers:
- Value type construction and field defaults.
- MultiplexerCapabilities construction and immutability.
- Multiplexer Protocol structural checks.
- F3: multiplexer.base imports no I/O module (no subprocess, libtmux, asyncio.subprocess).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from ccgram.multiplexer.base import (
    CaptureResult,
    ForegroundInfo,
    MultiplexerCapabilities,
    Multiplexer,
    PaneDims,
    PaneInfo,
    WindowRef,
)

# ── Value type construction ────────────────────────────────────────────


class TestWindowRef:
    def test_required_fields(self) -> None:
        w = WindowRef(window_id="@0", window_name="mywin", cwd="/tmp")
        assert w.window_id == "@0"
        assert w.window_name == "mywin"
        assert w.cwd == "/tmp"

    def test_optional_field_defaults(self) -> None:
        w = WindowRef(window_id="@0", window_name="x", cwd="/")
        assert w.pane_current_command == ""
        assert w.pane_tty == ""
        assert w.pane_width == 0
        assert w.pane_height == 0

    def test_all_fields(self) -> None:
        w = WindowRef(
            window_id="@5",
            window_name="project",
            cwd="/home/user/project",
            pane_current_command="claude",
            pane_tty="/dev/ttys003",
            pane_width=220,
            pane_height=50,
        )
        assert w.pane_width == 220
        assert w.pane_height == 50
        assert w.pane_tty == "/dev/ttys003"


class TestPaneInfo:
    def test_construction(self) -> None:
        p = PaneInfo(
            pane_id="%3",
            index=0,
            active=True,
            command="claude",
            path="/tmp",
            width=200,
            height=40,
        )
        assert p.pane_id == "%3"
        assert p.active is True
        assert p.width == 200

    def test_herdr_style_id(self) -> None:
        p = PaneInfo(
            pane_id="w2:p1",
            index=0,
            active=True,
            command="claude",
            path="/tmp",
            width=80,
            height=24,
        )
        assert p.pane_id == "w2:p1"


class TestCaptureResult:
    def test_defaults(self) -> None:
        r = CaptureResult(text="hello")
        assert r.text == "hello"
        assert r.truncated is False

    def test_truncated(self) -> None:
        r = CaptureResult(text="lots of text", truncated=True)
        assert r.truncated is True


class TestForegroundInfo:
    def test_required_fields(self) -> None:
        f = ForegroundInfo(
            pid=1234, pgid=1234, argv=["claude", "--continue"], cwd="/tmp"
        )
        assert f.pid == 1234
        assert f.argv == ["claude", "--continue"]
        assert f.tty == ""  # default

    def test_with_tty(self) -> None:
        f = ForegroundInfo(pid=42, pgid=42, argv=["bash"], cwd="/", tty="/dev/ttys001")
        assert f.tty == "/dev/ttys001"


class TestPaneDims:
    def test_construction(self) -> None:
        d = PaneDims(width=220, height=50)
        assert d.width == 220
        assert d.height == 50


# ── MultiplexerCapabilities ────────────────────────────────────────────


class TestMultiplexerCapabilities:
    def _tmux_caps(self) -> MultiplexerCapabilities:
        return MultiplexerCapabilities(
            name="tmux",
            ids_stable_across_restart=True,
            exposes_pane_tty=True,
            native_agent_status=False,
            read_max_lines=None,
            self_identify_env="TMUX_PANE",
            supports_event_stream=False,
        )

    def _herdr_caps(self) -> MultiplexerCapabilities:
        return MultiplexerCapabilities(
            name="herdr",
            ids_stable_across_restart=False,
            exposes_pane_tty=False,
            native_agent_status=True,
            read_max_lines=1000,
            self_identify_env="HERDR_PANE_ID",
            supports_event_stream=True,
        )

    def test_tmux_caps(self) -> None:
        caps = self._tmux_caps()
        assert caps.name == "tmux"
        assert caps.ids_stable_across_restart is True
        assert caps.exposes_pane_tty is True
        assert caps.native_agent_status is False
        assert caps.read_max_lines is None
        assert caps.self_identify_env == "TMUX_PANE"
        assert caps.supports_event_stream is False

    def test_herdr_caps(self) -> None:
        caps = self._herdr_caps()
        assert caps.name == "herdr"
        assert caps.ids_stable_across_restart is False
        assert caps.exposes_pane_tty is False
        assert caps.native_agent_status is True
        assert caps.read_max_lines == 1000
        assert caps.self_identify_env == "HERDR_PANE_ID"
        assert caps.supports_event_stream is True

    def test_immutable(self) -> None:
        caps = self._tmux_caps()
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            caps.name = "other"  # type: ignore[misc]


# ── F3: multiplexer.base imports no I/O module ────────────────────────

_FORBIDDEN_IO_MODULES = frozenset(
    {
        "subprocess",
        "asyncio.subprocess",
        "libtmux",
        "libtmux.exc",
        "socket",
        "fcntl",
        "termios",
    }
)

_BASE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "ccgram" / "multiplexer" / "base.py"
)


def _collect_imports(path: Path) -> list[str]:
    """Return all module names imported at module level in *path*."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_f3_base_imports_no_io_module() -> None:
    """multiplexer.base must not import any I/O or backend library.

    This is the F3 fitness assertion: the core contract layer is pure.
    """
    imports = _collect_imports(_BASE_PATH)
    violations = [m for m in imports if m in _FORBIDDEN_IO_MODULES]
    assert not violations, (
        f"multiplexer/base.py imports forbidden I/O modules: {violations}. "
        "Keep the core contract layer pure — no subprocess, libtmux, or asyncio.subprocess."
    )


def test_f3_base_imports_no_backend_submodule() -> None:
    """multiplexer.base must not import any concrete backend module."""
    imports = _collect_imports(_BASE_PATH)
    backend_imports = [
        m
        for m in imports
        if "multiplexer.tmux" in m
        or "multiplexer.herdr" in m
        or "multiplexer.registry" in m
    ]
    assert not backend_imports, (
        f"multiplexer/base.py imports concrete backend(s): {backend_imports}. "
        "The core layer must not depend on adapters."
    )


# ── Protocol structural check ──────────────────────────────────────────


def test_multiplexer_is_runtime_checkable() -> None:
    """Multiplexer must be @runtime_checkable for isinstance checks."""

    # A class with all required methods satisfies the Protocol structurally.
    # We verify the protocol itself has the @runtime_checkable decorator by
    # confirming isinstance() doesn't raise TypeError.
    class _Fake:
        pass

    # Should not raise — just returns False for a non-implementing class.
    result = isinstance(_Fake(), Multiplexer)
    assert result is False  # doesn't implement it, but the check itself works


def test_multiplexer_protocol_has_expected_methods() -> None:
    """All contract methods declared in the design are present on Multiplexer."""
    expected = {
        "capabilities",
        "ensure_session",
        "list_windows",
        "find_window",
        "capture",
        "capture_scrollback",
        "pane_dims",
        "send",
        "send_to_pane",
        "kill_window",
        "rename_window",
        "list_panes",
        "create_window",
        "set_title",
        "foreground",
    }
    actual = {name for name in dir(Multiplexer) if not name.startswith("_")}
    missing = expected - actual
    assert not missing, f"Multiplexer Protocol is missing methods: {missing}"


# ── Clean-import check ─────────────────────────────────────────────────


def test_multiplexer_base_clean_import() -> None:
    """Importing ccgram.multiplexer.base in a warm interpreter must succeed."""
    # Force re-import to ensure we test the actual module state.
    mod = importlib.import_module("ccgram.multiplexer.base")
    assert hasattr(mod, "Multiplexer")
    assert hasattr(mod, "MultiplexerCapabilities")
    assert hasattr(mod, "WindowRef")
    assert hasattr(mod, "PaneInfo")
    assert hasattr(mod, "CaptureResult")
    assert hasattr(mod, "ForegroundInfo")
    assert hasattr(mod, "PaneDims")
