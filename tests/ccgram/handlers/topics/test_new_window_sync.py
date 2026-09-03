"""Integration tests for manual tmux window -> Telegram topic sync (TASK-033).

Verifies the full chain: session_map update -> SessionMonitor detection ->
handle_new_window -> topic creation -> binding established.
Unlike the fully-mocked unit tests in test_topic_orchestration.py, these wire the
monitor's detection logic through to handle_new_window with a real SessionManager
and a real ThreadRouter (disk I/O mocked) to verify end-to-end state changes.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topics.topic_orchestration import handle_new_window
from ccgram.session import SessionManager
from ccgram.thread_router import thread_router
from ccgram.session_monitor import NewWindowEvent, SessionMonitor


@pytest.fixture
def sm(monkeypatch: pytest.MonkeyPatch) -> SessionManager:
    """SessionManager with disk I/O disabled."""
    thread_router.reset()
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


@pytest.fixture
def monitor(tmp_path) -> SessionMonitor:
    return SessionMonitor(
        projects_path=tmp_path / "projects",
        poll_interval=0.1,
        state_file=tmp_path / "monitor_state.json",
    )


def _make_topic(thread_id: int = 999) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id
    return topic


def _map_entry(
    session_id: str, cwd: str = "/proj", window_name: str = "proj"
) -> dict[str, str]:
    return {"session_id": session_id, "cwd": cwd, "window_name": window_name}


def _wire_orchestration(
    monitor: SessionMonitor,
    sm: SessionManager,
    bot: AsyncMock,
    *,
    group_id: int | None,
    allowed_users: set[int],
    captured: list[NewWindowEvent] | None = None,
) -> None:
    """Route the monitor's new-window callback into the real orchestration handler."""

    async def on_new_window(event: NewWindowEvent) -> None:
        if captured is not None:
            captured.append(event)
        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager", sm),
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_config.group_id = group_id
            mock_config.allowed_users = allowed_users
            await handle_new_window(event, bot)

    monitor.set_new_window_callback(on_new_window)


async def _detect(monitor: SessionMonitor, session_map: dict) -> None:
    # Adoption is gated on the backend's verdict from the current listing, which
    # the monitor loop derives before reaching this call. These windows are live
    # and on tmux every live window is adoptable, so mirror that here.
    with patch.object(
        monitor,
        "_load_current_session_map",
        spec=True,
        new_callable=AsyncMock,
        return_value=session_map,
    ):
        await monitor._detect_and_cleanup_changes(adoptable_window_ids=set(session_map))


class TestNewWindowSyncWithBindings:
    """Full flow: existing bindings provide target group, monitor detects new window."""

    async def test_monitor_detect_triggers_topic_and_binding(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        user_id, existing_thread, existing_window = 100, 5, "@1"
        group_chat, new_window, new_thread = -100200, "@7", 77

        thread_router.thread_bindings = {user_id: {existing_thread: existing_window}}
        thread_router.group_chat_ids = {f"{user_id}:{existing_thread}": group_chat}

        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(new_thread))
        captured: list[NewWindowEvent] = []
        _wire_orchestration(
            monitor,
            sm,
            bot,
            group_id=None,
            allowed_users={user_id},
            captured=captured,
        )
        monitor._last_session_map = {existing_window: _map_entry("old-sess")}

        await _detect(
            monitor,
            {
                existing_window: _map_entry("old-sess"),
                new_window: _map_entry(
                    "new-sess", "/home/user/new-project", "new-project"
                ),
            },
        )

        assert [(e.window_id, e.window_name) for e in captured] == [
            (new_window, "new-project")
        ]
        bot.create_forum_topic.assert_called_once_with(
            chat_id=group_chat, name="new-project"
        )
        assert thread_router.get_window_for_thread(user_id, new_thread) == new_window
        assert thread_router.resolve_chat_id(user_id, new_thread) == group_chat
        assert thread_router.window_display_names.get(new_window) == "new-project"

    async def test_multiple_groups_get_separate_topics(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        user_a, user_b = 100, 200
        group_a, group_b = -100100, -100200

        thread_router.thread_bindings = {user_a: {1: "@1"}, user_b: {2: "@2"}}
        thread_router.group_chat_ids = {f"{user_a}:1": group_a, f"{user_b}:2": group_b}

        topic_counter = iter([50, 60])
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(
            side_effect=lambda **kw: _make_topic(next(topic_counter))
        )
        _wire_orchestration(
            monitor, sm, bot, group_id=None, allowed_users={user_a, user_b}
        )
        monitor._last_session_map = {}

        await _detect(monitor, {"@9": _map_entry("s1")})

        assert bot.create_forum_topic.call_count == 2
        assert {
            call.kwargs["chat_id"] for call in bot.create_forum_topic.call_args_list
        } == {group_a, group_b}

    async def test_same_user_does_not_create_orphan_topics_in_multiple_groups(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        user_id, new_window = 100, "@9"
        thread_router.thread_bindings = {user_id: {1: "@1", 2: "@2"}}
        thread_router.group_chat_ids = {
            f"{user_id}:1": -100100,
            f"{user_id}:2": -100200,
        }

        topic_counter = iter([50, 60])
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(
            side_effect=lambda **kw: _make_topic(next(topic_counter))
        )
        _wire_orchestration(monitor, sm, bot, group_id=None, allowed_users={user_id})
        monitor._last_session_map = {}

        await _detect(monitor, {new_window: _map_entry("s1")})

        assert bot.create_forum_topic.call_count == 2
        for chat_id in (-100100, -100200):
            matches = [
                window_id
                for (bound_chat, _thread_id), window_id in (
                    (key[1:], value)
                    for key, value in thread_router.chat_thread_bindings.items()
                    if key[0] == user_id
                )
                if bound_chat == chat_id
            ]
            assert matches == [new_window]


class TestNewWindowSyncColdStart:
    """Cold-start: no existing bindings, CCGRAM_GROUP_ID drives topic creation."""

    async def test_cold_start_with_group_id_creates_and_binds(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        group_id, user_id, new_window, new_thread = -100500, 12345, "@3", 42

        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(new_thread))
        _wire_orchestration(
            monitor, sm, bot, group_id=group_id, allowed_users={user_id}
        )
        monitor._last_session_map = {}

        await _detect(
            monitor,
            {new_window: _map_entry("fresh-sess", "/home/user/project", "project")},
        )

        bot.create_forum_topic.assert_called_once_with(chat_id=group_id, name="project")
        assert thread_router.get_window_for_thread(user_id, new_thread) == new_window
        assert thread_router.resolve_chat_id(user_id, new_thread) == group_id

    async def test_cold_start_without_group_id_skips(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        bot = AsyncMock()
        _wire_orchestration(monitor, sm, bot, group_id=None, allowed_users=set())
        monitor._last_session_map = {}

        await _detect(monitor, {"@4": _map_entry("s1")})

        bot.create_forum_topic.assert_not_called()
        assert thread_router.thread_bindings == {}


class TestNewWindowSyncEdgeCases:
    """Edge cases in the sync flow."""

    async def test_already_bound_window_skips_topic_creation(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        user_id, window_id, thread_id = 100, "@5", 10

        thread_router.thread_bindings = {user_id: {thread_id: window_id}}
        thread_router._rebuild_reverse_index()
        thread_router.group_chat_ids = {f"{user_id}:{thread_id}": -100100}

        bot = AsyncMock()
        _wire_orchestration(monitor, sm, bot, group_id=-100100, allowed_users={user_id})
        monitor._last_session_map = {}

        await _detect(monitor, {window_id: _map_entry("s-new")})

        bot.create_forum_topic.assert_not_called()

    async def test_topic_name_falls_back_to_cwd(
        self, monitor: SessionMonitor, sm: SessionManager
    ) -> None:
        group_id, user_id = -100500, 12345

        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(88))
        _wire_orchestration(
            monitor, sm, bot, group_id=group_id, allowed_users={user_id}
        )
        monitor._last_session_map = {}

        await _detect(monitor, {"@6": _map_entry("s1", "/home/user/cool-project", "")})

        bot.create_forum_topic.assert_called_once_with(
            chat_id=group_id, name="cool-project"
        )

    @pytest.mark.parametrize(
        ("already_bound", "expected_events"),
        [(False, ["@1"]), (True, [])],
        ids=["unbound_window_fires", "bound_window_stays_quiet"],
    )
    async def test_session_id_change_fires_only_for_unbound_window(
        self, monitor: SessionMonitor, already_bound: bool, expected_events: list[str]
    ) -> None:
        events: list[NewWindowEvent] = []

        async def on_new_window(event: NewWindowEvent) -> None:
            events.append(event)

        monitor.set_new_window_callback(on_new_window)
        monitor._last_session_map = {"@1": _map_entry("old-sess")}

        with patch("ccgram.thread_router.thread_router") as mock_router:
            mock_router.has_window.return_value = already_bound
            await _detect(monitor, {"@1": _map_entry("new-sess")})

        assert [event.window_id for event in events] == expected_events
