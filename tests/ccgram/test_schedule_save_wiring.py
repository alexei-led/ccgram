"""Tests for required _schedule_save callbacks on persistence singletons.

All four state singletons — ``WindowStateStore`` (F2.1),
``ThreadRouter`` (F2.2), ``UserPreferences`` (F2.3) and
``SessionMapSync`` (F2.4) — are constructor-injected. Their
``schedule_save`` callbacks are required arguments, so a singleton
cannot be built without explicit wiring. The legacy ``unwired_save``
fallback was removed in F2.5.

This guarantees that any test or caller that builds a singleton without
wiring it fails loudly instead of silently dropping the save.
"""

from __future__ import annotations

import pytest

from ccgram.session_map import SessionMapSync
from ccgram.thread_router import ThreadRouter
from ccgram.user_preferences import UserPreferences
from ccgram.window_state_store import WindowStateStore


class TestWindowStateStoreRequiresCallbacks:
    def test_constructor_requires_schedule_save(self) -> None:
        with pytest.raises(TypeError, match="schedule_save"):
            WindowStateStore(  # type: ignore[call-arg]
                on_hookless_provider_switch=lambda _wid: None,
            )

    def test_constructor_requires_on_hookless_provider_switch(self) -> None:
        with pytest.raises(TypeError, match="on_hookless_provider_switch"):
            WindowStateStore(  # type: ignore[call-arg]
                schedule_save=lambda: None,
            )

    def test_constructor_wires_schedule_save(self) -> None:
        calls: list[int] = []
        store = WindowStateStore(
            schedule_save=lambda: calls.append(1),
            on_hookless_provider_switch=lambda _wid: None,
        )
        store.set_pane_lifecycle_notify("@1", True)
        assert calls == [1]

    def test_constructor_wires_hookless_provider_switch(self) -> None:
        seen: list[str] = []
        store = WindowStateStore(
            schedule_save=lambda: None,
            on_hookless_provider_switch=seen.append,
        )
        state = store.get_window_state("@1")
        state.provider_name = "claude"
        store.set_window_provider("@1", "shell", new_provider_supports_hook=False)
        assert seen == ["@1"]


class TestUserPreferencesRequiresCallback:
    def test_constructor_requires_schedule_save(self) -> None:
        with pytest.raises(TypeError, match="schedule_save"):
            UserPreferences()  # type: ignore[call-arg]

    def test_constructor_wires_schedule_save_for_mru(self) -> None:
        calls: list[int] = []
        prefs = UserPreferences(schedule_save=lambda: calls.append(1))
        prefs.update_user_mru(100, "/tmp/proj")
        assert calls == [1]

    def test_constructor_wires_schedule_save_for_star(self) -> None:
        calls: list[int] = []
        prefs = UserPreferences(schedule_save=lambda: calls.append(1))
        prefs.toggle_user_star(100, "/tmp/proj")
        assert calls == [1]

    def test_constructor_wires_schedule_save_for_offset(self) -> None:
        calls: list[int] = []
        prefs = UserPreferences(schedule_save=lambda: calls.append(1))
        prefs.update_user_window_offset(100, "@1", 42)
        assert calls == [1]

    def test_from_dict_does_not_trigger_save(self) -> None:
        calls: list[int] = []
        prefs = UserPreferences(schedule_save=lambda: calls.append(1))
        prefs.from_dict(
            {"user_window_offsets": {"100": {"@1": 42}}, "user_dir_favorites": {}}
        )
        assert calls == []


class TestThreadRouterRequiresCallbacks:
    def test_constructor_requires_schedule_save(self) -> None:
        with pytest.raises(TypeError, match="schedule_save"):
            ThreadRouter(  # type: ignore[call-arg]
                has_window_state=lambda _wid: False,
            )

    def test_constructor_requires_has_window_state(self) -> None:
        with pytest.raises(TypeError, match="has_window_state"):
            ThreadRouter(  # type: ignore[call-arg]
                schedule_save=lambda: None,
            )

    def test_constructor_wires_schedule_save(self) -> None:
        calls: list[int] = []
        router = ThreadRouter(
            schedule_save=lambda: calls.append(1),
            has_window_state=lambda _wid: False,
        )
        router.bind_thread(100, 1, "@1")
        assert calls == [1]

    def test_constructor_wires_has_window_state(self) -> None:
        # When has_window_state returns True, unbind_thread must NOT
        # remove the display name (the WindowState still references it).
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: True,
        )
        router.bind_thread(100, 1, "@1", window_name="proj")
        router.unbind_thread(100, 1)
        assert router.get_display_name("@1") == "proj"

    def test_unbind_drops_display_name_when_window_state_absent(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        router.bind_thread(100, 1, "@1", window_name="proj")
        router.unbind_thread(100, 1)
        # Falls back to window_id when display name was removed.
        assert router.get_display_name("@1") == "@1"


class TestSessionMapSyncRequiresCallback:
    def test_constructor_requires_schedule_save(self) -> None:
        with pytest.raises(TypeError, match="schedule_save"):
            SessionMapSync()  # type: ignore[call-arg]

    def test_constructor_wires_schedule_save(self) -> None:
        calls: list[int] = []
        sync = SessionMapSync(schedule_save=lambda: calls.append(1))
        sync._schedule_save()
        assert calls == [1]


class TestSessionManagerWiresAllSingletons:
    def test_post_init_wires_all_schedule_save_callbacks(self) -> None:
        # SessionManager.__post_init__ wires every singleton's _schedule_save
        # to its own _save_state. After construction, calling _schedule_save
        # on any singleton must NOT raise.
        from ccgram.session import SessionManager

        sm = SessionManager()
        from ccgram.session_map import session_map_sync
        from ccgram.thread_router import thread_router
        from ccgram.user_preferences import user_preferences
        from ccgram.window_state_store import get_window_store

        for singleton in (
            get_window_store(),
            thread_router,
            user_preferences,
            session_map_sync,
        ):
            assert singleton._schedule_save is not None
            # _save_state may schedule via asyncio if a loop is running,
            # otherwise saves synchronously — either path is fine here.
            singleton._schedule_save()

        del sm

    def test_set_window_provider_triggers_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end check that SessionManager-level mutations propagate
        # through the wired stores into StatePersistence.
        from ccgram.session import SessionManager

        sm = SessionManager()
        saves: list[None] = []
        monkeypatch.setattr(
            sm._persistence, "schedule_save", lambda: saves.append(None)
        )
        sm.set_window_provider("@99", "claude")
        assert saves, "set_window_provider must trigger a debounced save"
        del sm

    def test_thread_router_bind_triggers_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.session import SessionManager

        sm = SessionManager()
        saves: list[None] = []
        monkeypatch.setattr(
            sm._persistence, "schedule_save", lambda: saves.append(None)
        )
        sm._thread_router.bind_thread(100, 1, "@1")
        assert saves, "ThreadRouter.bind_thread must trigger a debounced save"
        del sm

    def test_user_preferences_star_triggers_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.session import SessionManager

        sm = SessionManager()
        saves: list[None] = []
        monkeypatch.setattr(
            sm._persistence, "schedule_save", lambda: saves.append(None)
        )
        sm._user_preferences.toggle_user_star(100, "/tmp/proj")
        assert saves, "UserPreferences.toggle_user_star must trigger a save"
        del sm


class TestGetWindowStore:
    def test_returns_installed_store(self) -> None:
        from ccgram.session import SessionManager
        from ccgram.window_state_store import get_window_store

        sm = SessionManager()
        store = get_window_store()
        assert store is sm._window_store
        del sm


class TestGetThreadRouter:
    def test_returns_installed_router(self) -> None:
        from ccgram.session import SessionManager
        from ccgram.thread_router import get_thread_router

        sm = SessionManager()
        router = get_thread_router()
        assert router is sm._thread_router
        del sm


class TestGetUserPreferences:
    def test_returns_installed_prefs(self) -> None:
        from ccgram.session import SessionManager
        from ccgram.user_preferences import get_user_preferences

        sm = SessionManager()
        prefs = get_user_preferences()
        assert prefs is sm._user_preferences
        del sm


class TestGetSessionMapSync:
    def test_returns_installed_sync(self) -> None:
        from ccgram.session import SessionManager
        from ccgram.session_map import get_session_map_sync

        sm = SessionManager()
        sync = get_session_map_sync()
        assert sync is sm._session_map_sync
        del sm
