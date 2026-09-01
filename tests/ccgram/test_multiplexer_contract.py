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
CANDIDATE_BACKENDS = ["tmux", "herdr", "agterm"]

# The full method surface every backend must expose (Protocol + transitional).
CONTRACT_METHODS = (
    "ensure_session",
    "list_windows",
    # Reached by getattr in multiplexer/reconciliation.py, which raises when a
    # backend lacks it, so a backend can pass every other check here and still
    # bring the bot down at startup and on every monitor cycle.
    "list_windows_for_reconciliation",
    "capture_scrollback",
    "pane_dims",
    "send",
    "send_to_pane",
    "kill_window",
    "rename_window",
    "list_panes",
    "create_window",
    "create_topic_target",
    "create_worktree_window",
    "foreground",
    "agent_status",
    "split_window",
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


def test_backend_watch_events_is_async_generator(backend: Multiplexer) -> None:
    # watch_events streams (async generator), so it is checked here rather than
    # in CONTRACT_METHODS (which asserts plain coroutine functions).
    assert inspect.isasyncgenfunction(backend.watch_events), (
        "watch_events must be an async generator"
    )


async def test_tmux_watch_events_yields_nothing() -> None:
    """tmux has no event stream — watch_events is an empty async iterator."""
    events = [event async for event in get_multiplexer("tmux").watch_events([])]
    assert events == []


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
    assert isinstance(caps.native_worktrees, bool)
    assert isinstance(caps.supports_workspace_selection, bool)
    assert isinstance(caps.native_topic_targets, bool)


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
    assert caps.native_worktrees is False
    assert caps.supports_workspace_selection is False
    assert caps.native_topic_targets is False


async def test_tmux_agent_status_returns_none() -> None:
    """tmux has no native agent status — agent_status() always returns None."""
    status = await get_multiplexer("tmux").agent_status("@0")
    assert status is None


async def test_tmux_create_worktree_window_unsupported() -> None:
    """tmux has no native worktrees — create_worktree_window() fails cleanly."""
    ok, msg, name, win_id = await get_multiplexer("tmux").create_worktree_window(
        "/repo", "/repo.worktrees/x", "ccg/x"
    )
    assert ok is False
    assert (name, win_id) == ("", "")
    assert "tmux" in msg


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
    assert caps.native_worktrees is True
    assert caps.supports_workspace_selection is True
    assert caps.native_topic_targets is True


# ── window identity matching (WindowRef.matches / window_presence) ─────


def test_window_ref_matches_is_case_insensitive() -> None:
    """agterm UUIDs round-trip through callers that may lowercase them.

    Its own ``_find_session`` case-folds for exactly this reason, so a
    presence check that compared ids directly would report a bound window
    gone — and every caller of ``window_presence`` takes a destructive branch
    on a confirmed absence.
    """
    from ccgram.multiplexer.base import WindowRef

    ref = WindowRef(window_id="ABC-DEF", window_name="proj", cwd="/p")

    assert ref.matches("abc-def")
    assert ref.matches("ABC-DEF")
    assert not ref.matches("other")


def test_window_ref_matches_superseded_identities() -> None:
    """The value type's contract, ahead of any backend using it.

    No backend publishes ``alias_window_ids`` today, and herdr deliberately
    leaves them empty across a session re-key. This pins what the matcher owes
    a backend that starts: a window persisted under a superseded id must read
    as present, or the destructive guards fire on a live window.
    """
    from ccgram.multiplexer.base import WindowRef

    ref = WindowRef(
        window_id="new-id",
        window_name="proj",
        cwd="/p",
        alias_window_ids=("OLD-ID",),
        legacy_alias_window_ids=("MIGRATION-ONLY",),
    )

    assert ref.matches("old-id")
    assert ref.matches("new-id")
    # Legacy aliases are declared non-actionable, and window_snapshot hands
    # the matched ref to callers that drive it.
    assert not ref.matches("migration-only")


async def test_window_presence_finds_a_lowercased_agterm_id() -> None:
    """End to end through the seam helper, not just the dataclass."""
    from ccgram.multiplexer.base import WindowRef
    from ccgram.multiplexer.reconciliation import window_presence

    class _Backend:
        async def list_windows_for_reconciliation(self) -> list[WindowRef]:
            return [WindowRef(window_id="9F1C2D3E-4A5B", window_name="agent", cwd="/p")]

    assert await window_presence("9f1c2d3e-4a5b", _Backend()) is True
    assert await window_presence("no-such-id", _Backend()) is False


async def test_window_presence_is_none_when_the_backend_cannot_answer() -> None:
    from ccgram.multiplexer.reconciliation import window_presence

    class _Backend:
        async def list_windows_for_reconciliation(self) -> None:
            return None

    assert await window_presence("@5", _Backend()) is None
