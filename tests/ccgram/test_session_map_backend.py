"""Backend-aware session_map key scheme regression tests.

The hook writes herdr keys as ``herdr:<wN:pM>`` (the backend name) while tmux
keys are ``<tmux_session_name>:<@id>``. Readers must mirror the active backend's
prefix and id form, else herdr session entries are silently skipped (no
transcript monitoring / message delivery) or purged as "old format".
"""

import pytest

from ccgram.config import config
from ccgram.session_map import (
    is_backend_window_id,
    parse_session_map,
    session_map_prefix,
)


@pytest.fixture
def herdr_backend(monkeypatch):
    monkeypatch.setattr(config, "multiplexer_name", "herdr")


@pytest.fixture
def tmux_backend(monkeypatch):
    monkeypatch.setattr(config, "multiplexer_name", "tmux")


def _entry(session_id: str) -> dict[str, str]:
    return {
        "session_id": session_id,
        "cwd": "/repo",
        "window_name": "agent",
        "transcript_path": "",
        "provider_name": "claude",
    }


def test_prefix_tmux_uses_session_name(tmux_backend) -> None:
    assert session_map_prefix() == f"{config.tmux_session_name}:"


def test_prefix_herdr_uses_backend_name(herdr_backend) -> None:
    assert session_map_prefix() == "herdr:"


def test_is_backend_window_id_tmux(tmux_backend) -> None:
    assert is_backend_window_id("@12")
    # herdr-shaped + legacy window-name keys are old format on tmux → purged.
    assert not is_backend_window_id("w2:p1")
    assert not is_backend_window_id("my-project")


def test_is_backend_window_id_herdr(herdr_backend) -> None:
    assert is_backend_window_id("w2:p1")
    assert not is_backend_window_id("")


def test_parse_session_map_surfaces_herdr_entry(herdr_backend) -> None:
    """The monitor's read path must see hook-written herdr keys."""
    raw = {"herdr:w2:p1": _entry("S1")}
    parsed = parse_session_map(raw, session_map_prefix())
    assert "w2:p1" in parsed
    assert parsed["w2:p1"]["session_id"] == "S1"


def test_parse_session_map_tmux_skips_other_backend(tmux_backend) -> None:
    """A tmux run ignores stale herdr-prefixed entries (no cross-backend leak)."""
    raw = {"herdr:w2:p1": _entry("S1")}
    assert parse_session_map(raw, session_map_prefix()) == {}
