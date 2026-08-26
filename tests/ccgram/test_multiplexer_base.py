"""Tests for multiplexer/base.py — Task 1 fitness gate (F3).

Covers:
- Value type field defaults (the part of the dataclass shape callers rely on).
- MultiplexerCapabilities immutability.
- Multiplexer Protocol structural checks.
- F3: multiplexer.base imports no I/O module (no subprocess, libtmux, asyncio.subprocess).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from ccgram.multiplexer.base import (
    CaptureResult,
    ForegroundInfo,
    MultiplexerCapabilities,
    Multiplexer,
    WindowRef,
)

# ── Value type defaults ────────────────────────────────────────────────


class TestValueTypeDefaults:
    """Optional fields callers read without setting them."""

    def test_window_ref_optional_fields(self) -> None:
        window = WindowRef(window_id="@0", window_name="x", cwd="/")
        assert window.pane_current_command == ""
        assert window.pane_tty == ""
        assert window.pane_width == 0
        assert window.pane_height == 0
        assert window.alias_window_ids == ()

    def test_capture_result_is_untruncated_by_default(self) -> None:
        assert CaptureResult(text="hello").truncated is False

    def test_foreground_info_has_no_tty_by_default(self) -> None:
        info = ForegroundInfo(pid=1234, pgid=1234, argv=["claude"], cwd="/tmp")
        assert info.tty == ""


# ── MultiplexerCapabilities ────────────────────────────────────────────


def test_capabilities_are_immutable() -> None:
    caps = MultiplexerCapabilities(
        name="tmux",
        ids_stable_across_restart=True,
        exposes_pane_tty=True,
        native_agent_status=False,
        read_max_lines=None,
        self_identify_env="TMUX_PANE",
        supports_event_stream=False,
        native_worktrees=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
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
    """@runtime_checkable — isinstance() answers instead of raising TypeError."""

    class _Fake:
        pass

    assert isinstance(_Fake(), Multiplexer) is False


def test_multiplexer_protocol_has_expected_methods() -> None:
    """All contract methods declared in the design are present on Multiplexer."""
    expected = {
        "capabilities",
        "ensure_session",
        "list_windows",
        "capture_scrollback",
        "pane_dims",
        "send",
        "send_to_pane",
        "kill_window",
        "rename_window",
        "list_panes",
        "create_window",
        "foreground",
        "find_window_by_id",
        "capture_pane",
        "stamp_pane_title",
    }
    actual = {name for name in dir(Multiplexer) if not name.startswith("_")}
    missing = expected - actual
    assert not missing, f"Multiplexer Protocol is missing methods: {missing}"
