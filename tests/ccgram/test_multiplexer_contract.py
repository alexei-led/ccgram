"""F2 contract test — one test, run against each registered backend.

The ``Multiplexer`` contract (design "The Multiplexer contract") is enforced by
this parametrized test. The tmux leg is always active; the herdr leg activates
once herdr is registered (Task 7) and a socket is present, otherwise it skips.

The Protocol is ``runtime_checkable``, so ``isinstance`` verifies the backend
exposes every contract method. The capability shape is checked structurally so
a backend can't silently drop a flag callers gate on.
"""

from __future__ import annotations

import inspect

import pytest

from ccgram.multiplexer import UnknownMultiplexerError, get_multiplexer
from ccgram.multiplexer.base import Multiplexer, MultiplexerCapabilities

# Every backend the contract should hold for. Unregistered names skip.
CANDIDATE_BACKENDS = ["tmux", "herdr"]

# The full method surface every backend must expose (Protocol + transitional).
CONTRACT_METHODS = (
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
    "find_window_by_id",
    "capture_pane",
    "capture_pane_by_id",
    "capture_pane_scrollback",
    "send_keys",
    "send_keys_to_pane",
    "get_pane_title",
    "stamp_pane_title",
)


def _backend_or_skip(name: str) -> Multiplexer:
    try:
        return get_multiplexer(name)
    except UnknownMultiplexerError:
        pytest.skip(f"multiplexer backend {name!r} not registered")
    except (OSError, RuntimeError) as exc:  # e.g. herdr socket unavailable
        pytest.skip(f"multiplexer backend {name!r} unavailable: {exc}")


@pytest.fixture(params=CANDIDATE_BACKENDS)
def backend(request: pytest.FixtureRequest) -> Multiplexer:
    return _backend_or_skip(request.param)


def test_backend_satisfies_protocol(backend: Multiplexer) -> None:
    assert isinstance(backend, Multiplexer)


def test_backend_exposes_every_contract_method(backend: Multiplexer) -> None:
    for method in CONTRACT_METHODS:
        attr = getattr(backend, method, None)
        assert attr is not None, f"missing contract method {method!r}"
        assert callable(attr), f"contract method {method!r} is not callable"
        assert inspect.iscoroutinefunction(attr), f"{method!r} must be async"


def test_backend_capabilities_shape(backend: Multiplexer) -> None:
    caps = backend.capabilities
    assert isinstance(caps, MultiplexerCapabilities)
    assert isinstance(caps.name, str) and caps.name
    assert isinstance(caps.ids_stable_across_restart, bool)
    assert isinstance(caps.exposes_pane_tty, bool)
    assert isinstance(caps.native_agent_status, bool)
    assert caps.read_max_lines is None or isinstance(caps.read_max_lines, int)
    assert isinstance(caps.self_identify_env, str) and caps.self_identify_env
    assert isinstance(caps.supports_event_stream, bool)


def test_tmux_capability_values() -> None:
    """tmux capability flags are pinned (design "MultiplexerCapabilities")."""
    caps = get_multiplexer("tmux").capabilities
    assert caps.name == "tmux"
    assert caps.ids_stable_across_restart is True
    assert caps.exposes_pane_tty is True
    assert caps.native_agent_status is False
    assert caps.read_max_lines is None
    assert caps.self_identify_env == "TMUX_PANE"
    assert caps.supports_event_stream is False


def test_herdr_capability_values() -> None:
    """herdr capability flags are pinned (design "MultiplexerCapabilities").

    Resolving the backend touches no socket (the constructor is I/O-free), so
    this runs in the unit suite even without a running herdr.
    """
    caps = get_multiplexer("herdr").capabilities
    assert caps.name == "herdr"
    assert caps.ids_stable_across_restart is False
    assert caps.exposes_pane_tty is False
    assert caps.native_agent_status is True
    assert caps.read_max_lines == 1000
    assert caps.self_identify_env == "HERDR_PANE_ID"
    assert caps.supports_event_stream is True
