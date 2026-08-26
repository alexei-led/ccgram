"""Backend-aware session_map key scheme regression tests.

The hook writes Herdr keys as ``herdr:<opaque-session-target>`` while tmux
keys are ``<tmux_session_name>:<@id>``. Readers must mirror the active backend's
prefix and preserve the opaque target, else Herdr session entries are silently
skipped (no transcript monitoring / message delivery) or purged as old format.
"""

import pytest

from ccgram.config import config
from ccgram.session_monitor import SessionMonitor
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


_HERDR_TARGET = "herdr-session-v1-" + "a" * 64


@pytest.mark.parametrize(
    ("window_id", "accepted"),
    [
        pytest.param("@12", True, id="tmux-window-id"),
        # herdr-shaped and legacy window-name keys are old format on tmux → purged.
        pytest.param("w2:p1", False, id="herdr-locator"),
        pytest.param("my-project", False, id="legacy-window-name"),
        pytest.param("", False, id="empty"),
    ],
)
def test_is_backend_window_id_tmux(tmux_backend, window_id, accepted) -> None:
    assert is_backend_window_id(window_id) is accepted


@pytest.mark.parametrize(
    ("window_id", "accepted"),
    [
        pytest.param(_HERDR_TARGET, True, id="session-target"),
        pytest.param("herdr-session-v1-target", False, id="not-a-digest"),
        pytest.param("herdr-session-v1-" + "A" * 64, False, id="uppercase-digest"),
        pytest.param("herdr-session-v1-" + "a" * 63, False, id="short-digest"),
        pytest.param("w2:p1", False, id="raw-pane-locator"),
        pytest.param("", False, id="empty"),
    ],
)
def test_is_backend_window_id_herdr(herdr_backend, window_id, accepted) -> None:
    assert is_backend_window_id(window_id) is accepted


def test_parse_session_map_surfaces_herdr_entry(herdr_backend) -> None:
    """The monitor's read path must see hook-written Herdr keys."""
    raw = {f"herdr:{_HERDR_TARGET}": _entry("S1")}
    parsed = parse_session_map(raw, session_map_prefix())
    assert _HERDR_TARGET in parsed
    assert parsed[_HERDR_TARGET]["session_id"] == "S1"


def test_parse_session_map_herdr_rejects_raw_and_legacy_ids(herdr_backend) -> None:
    raw = {
        "herdr:w2:t1": _entry("raw-tab"),
        "herdr:w2:p1": _entry("raw-pane"),
        "herdr:herdr-session-v1-" + "A" * 64: _entry("bad-digest"),
    }
    assert parse_session_map(raw, session_map_prefix()) == {}


async def test_session_monitor_rejects_raw_herdr_ids(herdr_backend) -> None:
    raw = {"herdr:w2:t1": _entry("raw-tab")}
    assert await SessionMonitor()._load_current_session_map(raw) == {}


def test_parse_session_map_tmux_skips_other_backend(tmux_backend) -> None:
    """A tmux run ignores stale Herdr-prefixed entries (no cross-backend leak)."""
    raw = {f"herdr:{_HERDR_TARGET}": _entry("S1")}
    assert parse_session_map(raw, session_map_prefix()) == {}
