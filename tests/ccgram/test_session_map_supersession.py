"""Session-map behaviour around windows whose identity is still settling.

A backend that derives window identity from facts arriving over time (Herdr
publishes an agent session after the pane exists) leaves a freshly created
window in a state the session map cannot tell apart from an abandoned one:
no hook entry yet, and no topic binding until the creation flow finishes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from ccgram.handlers.topics.topic_orchestration import (
    is_pending_creation,
    pending_creation_transaction,
)
from ccgram.session import session_manager as _session_manager  # noqa: F401  (wires window_store)
from ccgram.session_map import (
    SessionMapSync,
    _reset_in_flight_window_predicate_for_testing,
    register_in_flight_window_predicate,
)
from ccgram.thread_router import thread_router
from ccgram.window_state_store import WindowState, window_store


@pytest.fixture(autouse=True)
def _unwired_predicate() -> Iterator[None]:
    _reset_in_flight_window_predicate_for_testing()
    yield
    _reset_in_flight_window_predicate_for_testing()


@pytest.fixture
def sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SessionMapSync:
    monkeypatch.setattr(
        "ccgram.config.config.session_map_file", tmp_path / "session_map.json"
    )
    monkeypatch.setattr("ccgram.config.config.multiplexer_name", "herdr")
    return SessionMapSync(schedule_save=lambda: None)


def _write_map(path: Path, key: str) -> None:
    path.write_text(
        json.dumps(
            {
                key: {
                    "session_id": "sid-1",
                    "cwd": "/repo",
                    "window_name": "repo",
                    "transcript_path": "/repo/t.jsonl",
                    "provider_name": "claude",
                }
            }
        )
    )


class TestWaitFollowsSupersession:
    """The hook writes under the id the window has *now*. A wait pinned to the
    id creation minted times out on a key nothing will ever write, and the
    creation flow then tears down a perfectly healthy session."""

    async def test_finds_the_entry_under_the_superseded_id(
        self, sync: SessionMapSync, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_map(tmp_path / "session_map.json", "herdr:canonical")
        monkeypatch.setattr(sync, "load_session_map", _noop_load)

        found = await sync.wait_for_session_map_entry(
            "provisional",
            timeout=1.0,
            interval=0.01,
            resolve_window_id=lambda _wid: "canonical",
        )

        assert found

    async def test_times_out_without_a_resolver(
        self, sync: SessionMapSync, tmp_path: Path
    ) -> None:
        _write_map(tmp_path / "session_map.json", "herdr:canonical")

        assert not await sync.wait_for_session_map_entry(
            "provisional", timeout=0.05, interval=0.01
        )


async def _noop_load(session_map: dict) -> None:
    return None


class TestStaleSweepSparesInFlightCreations:
    """Dropping the state of a window mid-creation discards the cwd, provider,
    approval mode and origin the flow just wrote; the window returns
    re-derived and, having lost its ccgram origin, outside its lifecycle."""

    def test_keeps_a_window_a_creation_flow_owns(self, sync: SessionMapSync) -> None:
        window_store.window_states["@9"] = WindowState(cwd="/repo")
        register_in_flight_window_predicate(lambda wid: wid == "@9")

        try:
            removed = sync._remove_stale_window_states(
                valid_wids=set(), old_format_sids=set()
            )
        finally:
            window_store.window_states.pop("@9", None)

        assert not removed

    def test_still_removes_a_window_nothing_owns(self, sync: SessionMapSync) -> None:
        window_store.window_states["@9"] = WindowState(cwd="/repo")
        register_in_flight_window_predicate(lambda _wid: False)

        try:
            removed = sync._remove_stale_window_states(
                valid_wids=set(), old_format_sids=set()
            )
        finally:
            window_store.window_states.pop("@9", None)

        assert removed

    def test_transaction_does_not_protect_an_unrelated_window(
        self, sync: SessionMapSync
    ) -> None:
        window_store.window_states["@9"] = WindowState(cwd="/repo")
        register_in_flight_window_predicate(is_pending_creation)

        try:
            with pending_creation_transaction():
                removed = sync._remove_stale_window_states(
                    valid_wids=set(), old_format_sids=set()
                )
        finally:
            window_store.window_states.pop("@9", None)

        assert removed


class TestStaleSweepRespectsChatScopedBindings:
    """_remove_stale_window_states must use all_bound_window_ids(), not just
    thread_bindings.  set_group_chat_id() moves a binding from thread_bindings
    into chat_thread_bindings; the old code only iterated thread_bindings and
    so treated every chat-scoped window as unbound, deleting its state every
    poll cycle (the bug that erased Pi/Codex/Gemini window states in production).
    """

    @pytest.fixture(autouse=True)
    def _restore_thread_router(self) -> Iterator[None]:
        """Snapshot and restore all ThreadRouter dicts to prevent test leakage."""
        saved_bindings = {k: dict(v) for k, v in thread_router.thread_bindings.items()}
        saved_chat = dict(thread_router.chat_thread_bindings)
        saved_chat_w2t = dict(thread_router._chat_window_to_thread)
        saved_w2t = dict(thread_router._window_to_thread)
        saved_group = dict(thread_router.group_chat_ids)
        yield
        thread_router.thread_bindings.clear()
        thread_router.thread_bindings.update(
            {k: dict(v) for k, v in saved_bindings.items()}
        )
        thread_router.chat_thread_bindings.clear()
        thread_router.chat_thread_bindings.update(saved_chat)
        thread_router._chat_window_to_thread.clear()
        thread_router._chat_window_to_thread.update(saved_chat_w2t)
        thread_router._window_to_thread.clear()
        thread_router._window_to_thread.update(saved_w2t)
        thread_router.group_chat_ids.clear()
        thread_router.group_chat_ids.update(saved_group)

    def test_chat_scoped_binding_survives_sweep(self, sync: SessionMapSync) -> None:
        """A window promoted to chat scope by set_group_chat_id must not be swept.

        The old code only read thread_bindings, which set_group_chat_id empties.
        Demonstrate the blindness in the test: after promotion thread_bindings is
        empty while all_bound_window_ids() still sees the window.
        """
        register_in_flight_window_predicate(lambda _wid: False)
        window_store.window_states["@42"] = WindowState(cwd="/project")

        # First bind via thread_bindings, then promote to chat scope.
        thread_router.thread_bindings[1] = {100: "@42"}
        thread_router.set_group_chat_id(1, 100, 999)

        # Demonstrate the old-code blindness: thread_bindings is now empty.
        old_code_bound = {
            wid
            for user_bindings in thread_router.thread_bindings.values()
            for wid in user_bindings.values()
            if wid
        }
        assert old_code_bound == set()  # old code would miss this window
        assert thread_router.all_bound_window_ids() == {"@42"}  # fix sees it

        try:
            removed = sync._remove_stale_window_states(
                valid_wids=set(), old_format_sids=set()
            )
        finally:
            window_store.window_states.pop("@42", None)

        assert not removed  # window must survive

    def test_legacy_thread_binding_still_survives_sweep(
        self, sync: SessionMapSync
    ) -> None:
        """thread_bindings (legacy path) must still protect a bound window."""
        register_in_flight_window_predicate(lambda _wid: False)
        window_store.window_states["@43"] = WindowState(cwd="/project")
        thread_router.thread_bindings[2] = {200: "@43"}

        try:
            removed = sync._remove_stale_window_states(
                valid_wids=set(), old_format_sids=set()
            )
        finally:
            window_store.window_states.pop("@43", None)
            thread_router.thread_bindings.pop(2, None)

        assert not removed

    def test_genuinely_unbound_window_still_removed(self, sync: SessionMapSync) -> None:
        """The fix must not make the guard too broad; unbound windows are swept."""
        register_in_flight_window_predicate(lambda _wid: False)
        window_store.window_states["@44"] = WindowState(cwd="/project")

        try:
            removed = sync._remove_stale_window_states(
                valid_wids=set(), old_format_sids=set()
            )
        finally:
            window_store.window_states.pop("@44", None)

        assert removed
