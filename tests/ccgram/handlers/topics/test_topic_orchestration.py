from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut

from ccgram.multiplexer.base import WindowRef
from ccgram.thread_router import ThreadRouter
from ccgram.handlers.topics.topic_orchestration import (
    collect_target_chats,
    create_topic_in_chat,
    _is_pending_user_creation,
    _is_window_already_bound,
    _topic_create_retry_until,
    _pending_user_creations,
    _window_topic_locks,
    adopt_unbound_windows,
    clear_pending_creation,
    handle_new_window,
    is_pending_creation,
)
from ccgram.session_monitor import NewWindowEvent


@pytest.fixture(autouse=True)
def _clear_retry_state():
    _topic_create_retry_until.clear()
    _pending_user_creations.clear()
    _window_topic_locks.clear()
    yield
    _topic_create_retry_until.clear()
    _pending_user_creations.clear()
    _window_topic_locks.clear()


@pytest.fixture(autouse=True)
def _mock_tmux():
    mock_window = MagicMock()
    mock_window.pane_current_command = ""
    with patch("ccgram.handlers.topics.topic_orchestration.tmux_manager") as mock_tmux:
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
        # New creation targets are live; old same-name targets remain absent.
        mock_tmux.list_windows_for_reconciliation = AsyncMock(
            return_value=[
                WindowRef(wid, "project", "/p")
                for wid in ("@10", "@5", "@3", "@new", "herdr-session-v1-new")
            ]
        )
        yield mock_tmux


def _make_event(
    window_id: str = "@10",
    session_id: str = "sess-1",
    window_name: str = "my-project",
    cwd: str = "/home/user/my-project",
) -> NewWindowEvent:
    return NewWindowEvent(
        window_id=window_id,
        session_id=session_id,
        window_name=window_name,
        cwd=cwd,
    )


def _make_topic(thread_id: int = 999) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id
    return topic


def _configure_provisioning(mock_router: MagicMock) -> None:
    claim = MagicMock()
    claim.claim_id = "claim-1"
    claim.topic_mark_owner = None
    mock_router.get_topic_provisioning.return_value = claim
    mock_router.begin_topic_provisioning.return_value = claim
    mock_router.commit_topic_provisioning.return_value = True


class TestIsWindowAlreadyBound:
    @pytest.mark.parametrize("has_window", [True, False])
    def test_reflects_router_binding(self, has_window: bool):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.has_window.return_value = has_window
            assert _is_window_already_bound("@5") is has_window
        mock_router.has_window.assert_called_once_with("@5")


class TestCollectTargetChats:
    def test_from_bindings(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings.return_value = [
                (1, 100, "@0"),
            ]
            mock_router.resolve_chat_id.return_value = -1001
            result = collect_target_chats("@5")
            assert result == {-1001}

    def test_fallback_to_group_chat_ids(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {1: -2002}
            result = collect_target_chats("@5")
            assert result == {-2002}

    def test_fallback_to_config_group_id(self):
        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}
            mock_config.group_id = -3003
            result = collect_target_chats("@5")
            assert result == {-3003}

    def test_no_chats_available(self):
        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}
            mock_config.group_id = None
            result = collect_target_chats("@5")
            assert result == set()

    def test_skips_positive_ids_without_private_topic_capability(self):
        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_private_topic_chat_ids.return_value = []
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {"100:5": 100}
            mock_config.group_id = None
            result = collect_target_chats("@5")
            assert result == set()

    def test_includes_private_chat_with_observed_topic_metadata(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.iter_private_topic_chat_ids.return_value = [100]
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}

            assert collect_target_chats("@5") == {100}


class TestHandleNewWindow:
    async def test_serializes_concurrent_creation_for_same_window(self) -> None:
        active = 0
        max_active = 0

        async def locked_handler(
            _event: NewWindowEvent, _client: AsyncMock, **_kwargs: object
        ) -> bool:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return True

        event = _make_event()
        with patch(
            "ccgram.handlers.topics.topic_orchestration._handle_new_window_locked",
            side_effect=locked_handler,
        ):
            assert await asyncio.gather(
                handle_new_window(event, AsyncMock()),
                handle_new_window(event, AsyncMock()),
            ) == [True, True]

        assert max_active == 1
        assert _window_topic_locks == {}

    async def test_targeted_creation_ignores_another_users_binding(self) -> None:
        event = _make_event(window_id="@2", window_name="proj")
        bot = AsyncMock()
        bot.create_forum_topic.return_value = _make_topic(thread_id=77)

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = True
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([(200, 2, "@2")])
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {100, 200}
            created = await handle_new_window(
                event,
                bot,
                target_user_id=100,
                target_chat_id=-100100,
            )

        assert created is True
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 77)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="proj"
        )

    async def test_private_topic_creation_requires_enabled_bot_capability(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.get_me.return_value = MagicMock(has_topics_enabled=False)

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
        ):
            _configure_provisioning(mock_tr)
            created = await handle_new_window(
                event, bot, target_user_id=100, target_chat_id=100
            )

        assert created is False
        bot.get_me.assert_awaited_once_with()
        bot.create_forum_topic.assert_not_called()
        mock_tr.bind_thread.assert_not_called()

    async def test_private_topic_creation_uses_enabled_bot_capability(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.get_me.return_value = MagicMock(has_topics_enabled=True)
        bot.create_forum_topic.return_value = _make_topic(thread_id=42)

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
        ):
            _configure_provisioning(mock_tr)
            created = await handle_new_window(
                event, bot, target_user_id=100, target_chat_id=100
            )

        assert created is True
        bot.get_me.assert_awaited_once_with()
        bot.create_forum_topic.assert_awaited_once_with(chat_id=100, name="my-project")
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 42)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="my-project"
        )

    async def test_skips_already_bound(self):
        event = NewWindowEvent(
            window_id="@0", session_id="s1", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()
        with patch(
            "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
            return_value=True,
        ):
            await handle_new_window(event, bot)
        bot.create_forum_topic.assert_not_called()

    async def test_skips_when_no_chats(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.collect_target_chats",
                return_value=set(),
            ),
        ):
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()

    async def test_rate_limit_backoff(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()
        _topic_create_retry_until[-1001] = time.monotonic() + 60

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.collect_target_chats",
                return_value={-1001},
            ),
        ):
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()

    async def test_group_id_fallback_creates_topic_and_binds_allowed_user(
        self,
    ) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.resolve_chat_id.return_value = 12345
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="my-project"
        )
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 42)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="my-project"
        )

    async def test_existing_binding_supplies_chat_and_owner(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_config.topic_discovery_mode = "auto"
            bindings = [(100, 5, "@1")]
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.side_effect = [
                iter(bindings),
                iter(bindings),
                iter(bindings),
            ]
            mock_tr.resolve_chat_id.return_value = -100200

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="my-project"
        )
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 77)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="my-project"
        )

    async def test_topic_name_falls_back_to_cwd_dirname(self) -> None:
        event = _make_event(window_name="", cwd="/home/user/cool-project")
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="cool-project"
        )

    async def test_telegram_error_logged_not_raised(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=TelegramError("API error"))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

    async def test_retry_after_sets_backoff_and_skips_immediate_retry(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=RetryAfter(27))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.time.monotonic",
                side_effect=lambda: 100.0,
            ),
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.side_effect = [iter([]) for _ in range(6)]
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="my-project"
        )

    async def test_retry_after_can_propagate_after_cleanup(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=RetryAfter(27))
        session = MagicMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ) as retry_until,
            patch(
                "ccgram.handlers.topics.topic_orchestration.time.monotonic",
                return_value=100.0,
            ),
            pytest.raises(RetryAfter),
        ):
            await create_topic_in_chat(
                bot,
                -100500,
                "@propagate",
                "project",
                user_id=100,
                propagate_retry_after=True,
            )

        bot.create_forum_topic.assert_awaited_once_with(
            chat_id=-100500,
            name="project",
        )
        assert router.iter_topic_provisionings() == []
        assert retry_until[-100500] > 100.0

    async def test_retries_after_backoff_expires(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(
            side_effect=[RetryAfter(3), _make_topic(thread_id=42)]
        )
        clock = [100.0]

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.time.monotonic",
                side_effect=lambda: clock[0],
            ),
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.side_effect = [iter([]) for _ in range(6)]
            mock_tr.resolve_chat_id.return_value = 12345
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)
            clock[0] = 106.0
            await handle_new_window(event, bot)

        assert bot.create_forum_topic.call_count == 2
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 42)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="my-project"
        )

    async def test_uses_group_chat_ids_when_no_bindings(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.group_chat_ids = {"100:5": -100200}
            mock_config.group_id = None
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="my-project"
        )

    async def test_same_name_window_gets_a_fresh_topic(self) -> None:
        event = _make_event(window_id="@3", window_name="reflex-gh")
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=501))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([(100, 120014, "@1")])
            mock_tr.resolve_chat_id.return_value = -100200
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            # Only the new target is live; the old same-name target is gone.
            mock_tmux.list_windows_for_reconciliation = AsyncMock(
                return_value=[WindowRef(event.window_id, event.window_name, event.cwd)]
            )

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_awaited_once_with(
            chat_id=-100200,
            name="reflex-gh",
        )
        mock_tr.unbind_thread.assert_not_called()
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 501)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="reflex-gh"
        )

    async def test_same_name_fresh_topic_preserves_old_binding(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        router.bind_thread(100, 42, "@old", window_name="reflex-gh", chat_id=-100200)
        bot = AsyncMock()
        bot.create_forum_topic.return_value = _make_topic(thread_id=504)
        session = MagicMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
        ):
            assert await handle_new_window(
                _make_event(window_id="@new", window_name="reflex-gh"), bot
            )

        assert router.get_window_for_chat_thread(-100200, 42) == "@old"
        assert router.get_window_for_chat_thread(-100200, 504) == "@new"
        bot.create_forum_topic.assert_awaited_once_with(
            chat_id=-100200,
            name="reflex-gh",
        )

    async def test_herdr_same_name_window_gets_a_fresh_topic(self) -> None:
        event = _make_event(window_id="herdr-session-v1-new", window_name="reflex-gh")
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=78))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            # Only the new target is live; the old same-name target is gone.
            mock_tmux.list_windows_for_reconciliation = AsyncMock(
                return_value=[WindowRef(event.window_id, event.window_name, event.cwd)]
            )
            # Only the new target is live; the old same-name target is gone.
            mock_tmux.list_windows_for_reconciliation = AsyncMock(
                return_value=[WindowRef(event.window_id, event.window_name, event.cwd)]
            )
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([(100, 120014, "old")])
            mock_tr.resolve_chat_id.return_value = -100200
            mock_tr.get_display_name.return_value = "reflex-gh"
            mock_config.group_id = -100200
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {100}

            assert await handle_new_window(event, bot)

        bot.create_forum_topic.assert_awaited_once_with(
            chat_id=-100200, name="reflex-gh"
        )
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 78)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="reflex-gh"
        )

    async def test_dead_same_name_topic_stays_bound_when_new_topic_is_created(
        self,
    ) -> None:
        event = _make_event(window_id="@3", window_name="reflex-gh")
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.side_effect = [
                iter([(100, 120014, "@1")]),
                iter([]),
                iter([]),
            ]
            mock_tr.get_display_name.return_value = "reflex-gh"
            mock_tr.resolve_chat_id.return_value = -100200
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            # Only the new target is live; the old same-name target is gone.
            mock_tmux.list_windows_for_reconciliation = AsyncMock(
                return_value=[WindowRef(event.window_id, event.window_name, event.cwd)]
            )
            mock_tr.group_chat_ids = {"100:120014": -100200}
            mock_config.group_id = None
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {100}

            await handle_new_window(event, bot)

        mock_tr.unbind_thread.assert_not_called()
        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="reflex-gh"
        )
        mock_tr.attach_provisioning_topic.assert_called_once_with("claim-1", 77)
        mock_tr.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="reflex-gh"
        )


class TestCreateForumTopicTransientRetry:
    async def test_timed_out_retries_then_succeeds(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(
            side_effect=[TimedOut("blip"), _make_topic(thread_id=42)]
        )

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        assert bot.create_forum_topic.call_count == 2

    async def test_timed_out_exhausts_retries_then_logs(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=TimedOut("persistent"))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_tr.has_window.return_value = False
            _configure_provisioning(mock_tr)
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.topic_discovery_mode = "auto"
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        # Original attempt + 1 retry = 2 calls
        assert bot.create_forum_topic.call_count == 2
        assert not _is_pending_user_creation(event.window_id)
        assert -100500 in _topic_create_retry_until


class TestDurableTopicProvisioning:
    async def test_retry_claim_is_requeued_after_definitive_failure(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        claim = router.begin_topic_provisioning(
            100,
            -100500,
            thread_id=42,
            target_id="@retry",
            kind="replacement",
        )
        prepared = router.prepare_topic_recreation(claim.claim_id)
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=BadRequest("denied"))
        session = MagicMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ),
        ):
            assert (
                await create_topic_in_chat(
                    bot,
                    -100500,
                    "@retry",
                    "project",
                    user_id=100,
                    claim_id=prepared.claim_id,
                )
                is False
            )

        current = router.get_topic_provisioning(claim.claim_id)
        assert current is not None
        assert current.thread_id is None
        assert current.retry_thread_id == 42
        assert current.retry_at > 0
        assert current.uncertain is False
        assert router.owns_topic_provisioning(claim.claim_id) is False

    async def test_retry_claim_stays_uncertain_after_network_failure(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        claim = router.begin_topic_provisioning(
            100,
            -100500,
            thread_id=42,
            target_id="@uncertain",
            kind="replacement",
        )
        prepared = router.prepare_topic_recreation(claim.claim_id)
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=NetworkError("offline"))
        session = MagicMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ),
        ):
            assert (
                await create_topic_in_chat(
                    bot,
                    -100500,
                    "@uncertain",
                    "project",
                    user_id=100,
                    claim_id=prepared.claim_id,
                )
                is False
            )

        current = router.get_topic_provisioning(claim.claim_id)
        assert current is not None
        assert current.thread_id is None
        assert current.retry_thread_id == 42
        assert current.retry_at == 0.0
        assert current.uncertain is True
        assert router.owns_topic_provisioning(claim.claim_id) is False

    async def test_checkpoint_failure_aborts_before_remote_creation(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        bot = AsyncMock()
        session = MagicMock()
        session.flush_state.side_effect = OSError("disk full")

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
        ):
            with pytest.raises(OSError, match="disk full"):
                await create_topic_in_chat(
                    bot,
                    -100500,
                    "@checkpoint",
                    "project",
                    user_id=100,
                )

            bot.create_forum_topic.assert_not_awaited()
            assert router.iter_topic_provisionings() == []
            assert router.owns_topic_provisioning("not-a-claim") is False

            session.flush_state.side_effect = None
            bot.create_forum_topic.return_value = _make_topic(thread_id=502)
            assert await create_topic_in_chat(
                bot,
                -100500,
                "@checkpoint",
                "project",
                user_id=100,
            )

        bot.create_forum_topic.assert_awaited_once_with(
            chat_id=-100500,
            name="project",
        )

    async def test_response_checkpoint_failure_retains_exact_uncertain_claim(
        self,
    ) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        bot = AsyncMock()
        bot.create_forum_topic.return_value = _make_topic(thread_id=503)
        session = MagicMock()
        session.flush_state.side_effect = [
            None,
            OSError("disk full"),
            OSError("disk full"),
        ]

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
        ):
            with pytest.raises(OSError, match="disk full"):
                await create_topic_in_chat(
                    bot,
                    -100500,
                    "@response-checkpoint",
                    "project",
                    user_id=100,
                )

            claims = router.iter_topic_provisionings()
            assert len(claims) == 1
            assert claims[0].thread_id == 503
            assert claims[0].uncertain is True
            assert is_pending_creation("@response-checkpoint")

            assert (
                await create_topic_in_chat(
                    bot,
                    -100500,
                    "@response-checkpoint",
                    "project",
                    user_id=100,
                )
                is False
            )

        bot.create_forum_topic.assert_awaited_once_with(
            chat_id=-100500,
            name="project",
        )

    async def test_cancellation_retains_late_topic_response_for_locked_recovery(
        self,
    ) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def create_topic(**_kwargs: object) -> MagicMock:
            started.set()
            await release.wait()
            return _make_topic(thread_id=501)

        bot = AsyncMock()
        bot.create_forum_topic.side_effect = create_topic
        session = MagicMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
        ):
            creation = asyncio.create_task(
                create_topic_in_chat(
                    bot,
                    -100500,
                    "@cancelled",
                    "project",
                    user_id=100,
                )
            )
            await started.wait()
            creation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await creation

            claims = router.iter_topic_provisionings()
            assert len(claims) == 1
            assert claims[0].uncertain is True
            assert is_pending_creation("@cancelled")

            release.set()
            for _ in range(3):
                await asyncio.sleep(0)

            (claim,) = router.iter_topic_provisionings()
            assert claim.thread_id == 501
            assert claim.uncertain is True
            assert router.get_window_for_chat_thread(-100500, 501) is None

    async def test_lost_response_stays_pending_after_marker_expiry(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: False,
        )
        started = asyncio.Event()
        stop = asyncio.Event()

        async def create_topic(**_kwargs: object) -> MagicMock:
            started.set()
            await stop.wait()
            raise asyncio.CancelledError

        bot = AsyncMock()
        bot.create_forum_topic.side_effect = create_topic
        session = MagicMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.thread_router", router),
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager", session
            ),
        ):
            creation = asyncio.create_task(
                create_topic_in_chat(
                    bot,
                    -100500,
                    "@lost-response",
                    "project",
                    user_id=100,
                )
            )
            await started.wait()
            creation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await creation

            clear_pending_creation("@lost-response")
            assert is_pending_creation("@lost-response")
            assert router.iter_topic_provisionings()[0].uncertain is True

            stop.set()
            for _ in range(3):
                await asyncio.sleep(0)

            assert len(router.iter_topic_provisionings()) == 1
            assert router.iter_topic_provisionings()[0].uncertain is True


class TestAdoptUnboundWindows:
    @staticmethod
    def _audit(*categories: str) -> MagicMock:
        audit = MagicMock()
        audit.issues = [MagicMock(category=c) for c in categories]
        return audit

    async def test_adopts_orphaned_windows(self):
        bot = AsyncMock()
        mock_window = MagicMock(window_id="@0", window_name="test")

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ) as mock_adopt,
        ):
            mock_tmux.list_windows_for_reconciliation = AsyncMock(
                return_value=[mock_window]
            )
            mock_sm.audit_state.return_value = self._audit("orphaned_window")

            await adopt_unbound_windows(bot)

        mock_adopt.assert_called_once()

    async def test_no_orphans_skips_adoption(self):
        bot = AsyncMock()

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ) as mock_adopt,
        ):
            mock_tmux.list_windows_for_reconciliation = AsyncMock(return_value=[])
            mock_sm.audit_state.return_value = self._audit()

            await adopt_unbound_windows(bot)

        mock_adopt.assert_not_called()


class TestAdoptUnboundWindowsRespectsEligibility:
    """Startup orphan repair creates topics, so it takes the backend's verdict.

    Adoption narrows, liveness does not: narrowing the live set would turn a
    present-but-excluded window into a fixable ghost, which is what the first
    attempt at this did.
    """

    async def test_an_ineligible_window_is_live_but_not_adoptable(self):
        bot = AsyncMock()
        refused = WindowRef(
            window_id="OUT-OF-SCOPE",
            window_name="elsewhere",
            cwd="/other",
            pane_current_command="claude",
            topic_eligible=False,
        )
        allowed = WindowRef(
            window_id="IN-SCOPE",
            window_name="agent",
            cwd="/proj",
            pane_current_command="claude",
        )

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ),
        ):
            mock_tmux.list_windows_for_reconciliation = AsyncMock(
                return_value=[refused, allowed]
            )
            audit = MagicMock()
            audit.issues = []
            mock_sm.audit_state.return_value = audit

            await adopt_unbound_windows(bot)

            live_ids, live_pairs, adoptable = mock_sm.audit_state.call_args[0]

        # Liveness stays complete: a present-but-excluded window is not a ghost.
        assert live_ids == {"OUT-OF-SCOPE", "IN-SCOPE"}
        assert {wid for wid, _ in live_pairs} == {"OUT-OF-SCOPE", "IN-SCOPE"}
        # Only adoption is narrowed.
        assert adoptable == {"IN-SCOPE"}


class TestMarkedStartupAdoption:
    async def test_startup_audit_keeps_live_windows_and_only_adopts_marked(
        self, monkeypatch
    ):
        from ccgram.config import config

        monkeypatch.setattr(config, "topic_discovery_mode", "marked")
        windows = [
            WindowRef(
                "marked", "one", "/p", topic_marked=True, topic_mark_owner="owner"
            ),
            WindowRef("unmarked", "two", "/p", topic_marked=False),
            WindowRef("unknown", "three", "/p", topic_marked=None),
            WindowRef(
                "pending", "four", "/p", topic_marked=True, topic_mark_owner="owner"
            ),
        ]
        monkeypatch.setattr(
            "ccgram.handlers.topics.topic_orchestration.lifecycle_state.is_topic_detach_pending",
            lambda window_id, *, owner: window_id == "pending",
        )
        with (
            patch("ccgram.handlers.topics.topic_orchestration.tmux_manager") as mux,
            patch("ccgram.handlers.topics.topic_orchestration.session_manager") as sm,
        ):
            mux.list_windows_for_reconciliation = AsyncMock(return_value=windows)
            sm.audit_state.return_value = MagicMock(issues=[])
            await adopt_unbound_windows(AsyncMock())
            live, _, adoptable = sm.audit_state.call_args.args

        assert live == {"marked", "unmarked", "unknown", "pending"}
        assert adoptable == {"marked"}


class TestAdoptUnboundWindowsNeedsAConfirmedListing:
    """Startup repair creates topics; it must not run on a guess."""

    async def test_unavailable_listing_adopts_nothing(self):
        bot = AsyncMock()

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ) as mock_adopt,
        ):
            mock_tmux.list_windows_for_reconciliation = AsyncMock(return_value=None)

            await adopt_unbound_windows(bot)

        mock_adopt.assert_not_called()
        mock_sm.audit_state.assert_not_called()


class TestStillAdoptable:
    """The point-of-use check behind every automatic adoption."""

    @staticmethod
    def _ref(window_id: str, *, eligible: bool):
        return WindowRef(
            window_id=window_id,
            window_name="proj",
            cwd="/proj",
            topic_eligible=eligible,
        )

    async def _verdict(self, windows) -> bool:
        from ccgram.handlers.topics.topic_orchestration import still_adoptable

        with patch(
            "ccgram.handlers.topics.topic_orchestration.tmux_manager"
        ) as mock_tmux:
            mock_tmux.list_windows_for_reconciliation = AsyncMock(return_value=windows)
            return await still_adoptable("@5")

    async def test_eligible_window_passes(self):
        assert await self._verdict([self._ref("@5", eligible=True)]) is True

    async def test_window_that_left_scope_is_refused(self):
        assert await self._verdict([self._ref("@5", eligible=False)]) is False

    async def test_window_that_went_away_is_refused(self):
        assert await self._verdict([self._ref("@9", eligible=True)]) is False

    async def test_unconfirmed_listing_is_refused(self):
        assert await self._verdict(None) is False


class TestOwnedAutomaticAdmission:
    @pytest.fixture
    def runtime(self, monkeypatch, _mock_tmux):
        from ccgram.config import config
        from ccgram.window_state_store import (
            WindowStateStore,
            get_window_store,
            install_window_store,
        )

        previous = get_window_store()
        store = WindowStateStore(
            schedule_save=lambda: None, on_hookless_provider_switch=lambda wid: None
        )
        install_window_store(store)
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda wid: wid in store.window_states,
        )
        monkeypatch.setattr(
            "ccgram.handlers.topics.topic_orchestration.thread_router", router
        )
        monkeypatch.setattr(
            "ccgram.handlers.topics.topic_orchestration.session_manager", MagicMock()
        )
        monkeypatch.setattr(config, "topic_discovery_mode", "marked")
        monkeypatch.setattr(config, "allowed_users", {100})
        monkeypatch.setattr(config, "group_id", -100500)
        _mock_tmux.list_windows_for_reconciliation.return_value = [
            WindowRef(
                "@10", "project", "/p", topic_marked=True, topic_mark_owner="owned"
            )
        ]
        _mock_tmux.set_topic_mark = AsyncMock()
        _mock_tmux.find_window_by_id = AsyncMock(
            return_value=WindowRef(
                "@10", "project", "/p", topic_marked=True, topic_mark_owner="owned"
            )
        )
        monkeypatch.setattr("ccgram.multiplexer.multiplexer", _mock_tmux)
        try:
            yield router, store, _mock_tmux
        finally:
            install_window_store(previous)

    async def test_automatic_claim_owner_is_durable_before_telegram_can_change_bound_observation(
        self, runtime, tmp_path, monkeypatch
    ):
        import json
        from ccgram.handlers.topics import topic_orchestration as mod
        from ccgram.window_state_ports import lifecycle_state

        router, store, mux = runtime
        router.bind_thread(100, 77, "@10", chat_id=-100501)
        checkpoint = tmp_path / "state.json"
        monkeypatch.setattr(
            mod.session_manager,
            "flush_state",
            lambda: checkpoint.write_text(
                json.dumps({"router": router.to_dict(), "store": store.to_dict()})
            ),
        )

        async def create(**_kwargs):
            saved = json.loads(checkpoint.read_text())
            assert (
                saved["router"]["topic_provisioning"][0]["topic_mark_owner"] == "owned"
            )
            lifecycle_state.observe_topic_mark_owner("@10", "replacement")
            mux.list_windows_for_reconciliation.return_value = [
                WindowRef(
                    "@10",
                    "replacement",
                    "/p",
                    topic_marked=True,
                    topic_mark_owner="replacement",
                )
            ]
            return _make_topic(501)

        client = AsyncMock()
        client.create_forum_topic.side_effect = create
        assert not await create_topic_in_chat(
            client, -100500, "@10", "project", user_id=100, automatic=True
        )
        restored = ThreadRouter(
            schedule_save=lambda: None, has_window_state=lambda _wid: False
        )
        restored.from_dict(json.loads(checkpoint.read_text())["router"])
        (claim,) = restored.iter_topic_provisionings()
        assert claim.topic_mark_owner == "owned" and claim.thread_id == 501
        assert restored.get_window_for_chat_thread(-100500, 501) is None
        assert restored.get_window_for_chat_thread(-100501, 77) == "@10"
        assert store.window_states["@10"].topic_mark_owner == "replacement"
        mux.set_topic_mark.assert_not_awaited()

    async def test_automatic_bind_captures_owner_without_setting_mark(self, runtime):
        router, store, mux = runtime
        client = AsyncMock()
        client.create_forum_topic.return_value = _make_topic(501)

        assert await handle_new_window(_make_event(), client)

        assert store.window_states["@10"].topic_mark_owner == "owned"
        assert router.get_window_for_chat_thread(-100500, 501) == "@10"
        mux.set_topic_mark.assert_not_awaited()

    async def test_auto_mode_records_unmarked_owner_without_changing_selection(
        self, runtime, monkeypatch
    ):
        from ccgram.config import config

        router, store, mux = runtime
        monkeypatch.setattr(config, "topic_discovery_mode", "auto")
        mux.list_windows_for_reconciliation.return_value = [
            WindowRef(
                "@10", "project", "/p", topic_marked=False, topic_mark_owner="owned"
            )
        ]
        client = AsyncMock()
        client.create_forum_topic.return_value = _make_topic(501)

        assert await handle_new_window(_make_event(), client)

        assert store.window_states["@10"].topic_mark_owner == "owned"
        assert router.get_window_for_chat_thread(-100500, 501) == "@10"
        mux.set_topic_mark.assert_not_awaited()

    async def test_detach_while_waiting_for_creation_lock_prevents_remote_create(
        self, runtime
    ):
        from ccgram.handlers.topics.topic_orchestration import _window_topic_lock
        from ccgram.window_state_ports import lifecycle_state

        router, _, _ = runtime
        client = AsyncMock()
        async with _window_topic_lock("@10"):
            task = asyncio.create_task(handle_new_window(_make_event(), client))
            await asyncio.sleep(0)
            lifecycle_state.set_topic_selection(
                "@old", owner="owned", pending=False, detached=True
            )

        assert await task is False
        client.create_forum_topic.assert_not_awaited()
        assert router.iter_topic_provisionings() == []

    @pytest.mark.parametrize("change", ["detach", "owner", "unmark"])
    async def test_new_pending_or_changed_selection_cannot_commit_remote_response(
        self, runtime, change
    ):
        from ccgram.window_state_ports import lifecycle_state

        router, _, mux = runtime

        async def create(**kwargs):
            if change == "detach":
                lifecycle_state.set_topic_selection(
                    "@old", owner="owned", pending=False, detached=True
                )
            else:
                mux.list_windows_for_reconciliation.return_value = [
                    WindowRef(
                        "@10",
                        "project",
                        "/p",
                        topic_marked=change != "unmark",
                        topic_mark_owner="different" if change == "owner" else "owned",
                    )
                ]
            return _make_topic(501)

        client = AsyncMock()
        client.create_forum_topic.side_effect = create

        assert await handle_new_window(_make_event(), client) is False

        assert router.get_window_for_chat_thread(-100500, 501) is None
        (claim,) = router.iter_topic_provisionings()
        assert claim.thread_id == 501
        assert claim.uncertain

    async def test_late_automatic_response_is_retained_for_locked_recovery(
        self, runtime
    ):
        router, _, _ = runtime
        started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def create(**kwargs):
            started.set()
            await release.wait()
            completed.set()
            return _make_topic(501)

        client = AsyncMock()
        client.create_forum_topic.side_effect = create
        task = asyncio.create_task(handle_new_window(_make_event(), client))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await completed.wait()
        await asyncio.sleep(0)

        assert router.get_window_for_chat_thread(-100500, 501) is None
        (claim,) = router.iter_topic_provisionings()
        assert claim.thread_id == 501
        assert claim.uncertain

    @pytest.mark.parametrize("late", [False, True])
    @pytest.mark.parametrize(
        ("change", "mode"),
        [("observed", "marked"), ("native", "marked"), ("unknown", "auto")],
    )
    async def test_explicit_response_cannot_bind_reused_native_owner(
        self, runtime, late, change, mode, monkeypatch
    ):
        from ccgram.config import config
        from ccgram.window_state_ports import lifecycle_state

        monkeypatch.setattr(config, "topic_discovery_mode", mode)
        router, _, mux = runtime
        lifecycle_state.record_topic_mark_owner("@10", "owned")
        started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def create(**kwargs):
            started.set()
            await release.wait()
            if change == "observed":
                lifecycle_state.observe_topic_mark_owner("@10", "replacement")
            else:
                mux.find_window_by_id.return_value = WindowRef(
                    "@10",
                    "project",
                    "/p",
                    topic_marked=True,
                    topic_mark_owner="replacement" if change == "native" else None,
                )
            completed.set()
            return _make_topic(501)

        client = AsyncMock()
        client.create_forum_topic.side_effect = create
        task = asyncio.create_task(
            handle_new_window(
                _make_event(), client, target_user_id=100, target_chat_id=-100500
            )
        )
        await started.wait()
        if late:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        await completed.wait()
        if late:
            await asyncio.sleep(0)
        else:
            assert await task is False

        assert router.get_window_for_chat_thread(-100500, 501) is None
        (claim,) = router.iter_topic_provisionings()
        assert claim.thread_id == 501
        assert claim.uncertain

    @pytest.mark.parametrize(
        "initial_owner,final_owner,admitted",
        [
            pytest.param("owned", None, False, id="known-owner-lost"),
            pytest.param("owned", "replacement", False, id="known-owner-replaced"),
            pytest.param("owned", "owned", True, id="stable-unmarked-owner"),
            pytest.param(None, None, True, id="initially-ownerless"),
            pytest.param(
                None, "newly-confirmed", True, id="initially-unknown-confirmed"
            ),
        ],
    )
    async def test_auto_creation_preserves_captured_owner_after_telegram_await(
        self, runtime, monkeypatch, initial_owner, final_owner, admitted
    ):
        from ccgram.config import config
        from ccgram.handlers.topics import topic_orchestration
        from ccgram.providers import _ensure_registered
        from ccgram.window_state_ports import lifecycle_state

        router, store, mux = runtime
        lifecycle_state.record_topic_mark_owner("@10", initial_owner)
        store.window_states["@10"].provider_name = "claude"
        _ensure_registered()
        monkeypatch.setattr(config, "topic_discovery_mode", "auto")
        initial = WindowRef(
            "@10", "project", "/p", topic_marked=False, topic_mark_owner=initial_owner
        )
        final = WindowRef(
            "@10", "project", "/p", topic_marked=False, topic_mark_owner=final_owner
        )
        mux.list_windows_for_reconciliation.return_value = [initial]
        mux.find_window_by_id.return_value = initial
        client = AsyncMock()

        async def create(**kwargs):
            mux.list_windows_for_reconciliation.return_value = [final]
            mux.find_window_by_id.return_value = final
            return _make_topic(501)

        client.create_forum_topic.side_effect = create

        assert await handle_new_window(_make_event(), client) is admitted
        mux.set_topic_mark.assert_not_awaited()
        client.delete_forum_topic.assert_not_awaited()
        if admitted:
            assert router.get_window_for_chat_thread(-100500, 501) == "@10"
            assert store.window_states["@10"].topic_mark_owner == final_owner
            assert router.iter_topic_provisionings() == []
            return

        assert store.window_states["@10"].topic_mark_owner == initial_owner
        assert router.get_window_for_chat_thread(-100500, 501) is None
        (claim,) = router.iter_topic_provisionings()
        assert claim.thread_id == 501 and claim.uncertain
        assert not router.owns_topic_provisioning(claim.claim_id)
        restored = ThreadRouter(
            schedule_save=lambda: None, has_window_state=store.has_window
        )
        restored.from_dict(router.to_dict())
        await self._recover_exact_topic(
            restored,
            client,
            mux,
            initial,
            topic_orchestration.session_manager,
            monkeypatch,
        )

    @pytest.mark.parametrize("automatic", [False, True])
    @pytest.mark.parametrize("failure", ["cancel", "checkpoint"])
    async def test_post_creation_failure_preserves_exact_claim(
        self, runtime, tmp_path, monkeypatch, automatic, failure
    ):
        import json

        from ccgram.handlers.topics import topic_orchestration
        from ccgram.providers import _ensure_registered
        from ccgram.window_state_ports import lifecycle_state

        router, store, mux = runtime
        lifecycle_state.record_topic_mark_owner("@10", "owned")
        store.window_states["@10"].provider_name = "claude"
        _ensure_registered()
        created, recheck = asyncio.Event(), asyncio.Event()
        blocked = asyncio.Event()
        native = WindowRef(
            "@10", "project", "/p", topic_marked=True, topic_mark_owner="owned"
        )

        async def observe(*args):
            if created.is_set():
                recheck.set()
                if failure == "cancel":
                    await blocked.wait()
            return native

        async def listing():
            return [await observe()]

        mux.find_window_by_id.side_effect = observe
        mux.list_windows_for_reconciliation.side_effect = listing
        checkpoint = tmp_path / "claims.json"
        failed = False
        checkpoint_router = router

        def flush():
            nonlocal failed
            if (
                failure == "checkpoint"
                and created.is_set()
                and (recheck.is_set() or not automatic)
                and not failed
            ):
                failed = True
                raise OSError("checkpoint full")
            checkpoint.write_text(
                json.dumps(checkpoint_router.to_dict()), encoding="utf-8"
            )

        monkeypatch.setattr(topic_orchestration.session_manager, "flush_state", flush)

        async def create(**kwargs):
            created.set()
            return _make_topic(501)

        client = AsyncMock()
        client.create_forum_topic.side_effect = create
        task = asyncio.create_task(
            handle_new_window(
                _make_event(),
                client,
                **(
                    {}
                    if automatic
                    else {"target_user_id": 100, "target_chat_id": -100500}
                ),
            )
        )
        if failure == "cancel":
            await recheck.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(OSError, match="checkpoint full"):
                await task

        restored = ThreadRouter(
            schedule_save=lambda: None, has_window_state=lambda wid: False
        )
        restored.from_dict(json.loads(checkpoint.read_text(encoding="utf-8")))
        (claim,) = restored.iter_topic_provisionings()
        assert claim.thread_id == 501
        assert claim.uncertain is True
        assert not router.owns_topic_provisioning(claim.claim_id)
        assert restored.get_window_for_chat_thread(-100500, 501) is None
        assert router.get_window_for_chat_thread(-100500, 501) is None
        client.delete_forum_topic.assert_not_awaited()
        mux.kill_window.assert_not_called()
        assert _window_topic_locks == {}

        checkpoint_router = restored
        await self._recover_exact_topic(
            restored,
            client,
            mux,
            native,
            topic_orchestration.session_manager,
            monkeypatch,
        )

    async def _recover_exact_topic(
        self, restored, client, mux, native, session, monkeypatch
    ):
        from types import SimpleNamespace

        from ccgram.handlers.topics import topic_provisioning_recovery

        mux.find_window_by_id.side_effect = None
        mux.find_window_by_id.return_value = native
        mux.list_windows_for_reconciliation.side_effect = None
        mux.list_windows_for_reconciliation.return_value = [native]
        native_backend = SimpleNamespace(
            capabilities=SimpleNamespace(name="tmux"),
            list_windows_for_reconciliation=AsyncMock(return_value=[native]),
        )
        adapter = MagicMock()
        adapter.resolve = AsyncMock(
            return_value=(
                SimpleNamespace(
                    foreground_argv=("claude",), provider="claude", available=True
                ),
                None,
            )
        )
        monkeypatch.setattr(topic_provisioning_recovery, "multiplexer", mux)
        monkeypatch.setattr(
            topic_provisioning_recovery,
            "session_manager",
            session,
        )
        outcomes = await topic_provisioning_recovery.recover_topic_provisioning(
            client, router=restored, backend=native_backend, adapter=adapter
        )
        assert outcomes == {"bound": 1}
        assert restored.get_window_for_chat_thread(-100500, 501) == "@10"
        assert restored.iter_topic_provisionings() == []
        client.create_forum_topic.assert_awaited_once()
        client.delete_forum_topic.assert_not_awaited()

    async def test_owner_quarantine_survives_conversation_rekey(self, runtime):
        from ccgram.window_state_store import WindowState

        router, store, _ = runtime
        store.window_states["@old"] = WindowState(topic_mark_owner="replacement-P")
        router.begin_topic_provisioning(
            100,
            -100500,
            target_id="@old",
            kind="target_for_topic",
            topic_mark_owner="owned",
        )
        client = AsyncMock()

        assert await handle_new_window(_make_event(), client) is False
        client.create_forum_topic.assert_not_awaited()
        assert "@10" not in store.window_states


class TestMarkedPointOfUse:
    @pytest.mark.parametrize(
        ("marked", "pending", "expected"),
        [
            (True, False, True),
            (False, False, False),
            (None, False, False),
            (True, True, False),
        ],
    )
    async def test_fresh_listing_controls_admission(
        self, monkeypatch, marked, pending, expected
    ):
        from ccgram.config import config
        from ccgram.handlers.topics.topic_orchestration import still_adoptable

        monkeypatch.setattr(config, "topic_discovery_mode", "marked")
        monkeypatch.setattr(
            "ccgram.handlers.topics.topic_orchestration.lifecycle_state.is_topic_detach_pending",
            lambda window_id, *, owner: pending,
        )
        with patch("ccgram.handlers.topics.topic_orchestration.tmux_manager") as mux:
            mux.list_windows_for_reconciliation = AsyncMock(
                return_value=[
                    WindowRef(
                        "@5",
                        "proj",
                        "/p",
                        topic_marked=marked,
                        topic_mark_owner="owner",
                    )
                ]
            )
            assert await still_adoptable("@5") is expected


class TestAdoptionIdentityFoldsCase:
    """An orphan issue carries the persisted id, which may differ in case.

    agterm reports UUIDs uppercase while a caller may round-trip one
    lowercased. Comparing raw meant the refreshed listing looked as though the
    window had gone, so /sync Fix skipped creating its topic on every attempt.
    """

    async def test_a_case_variant_id_is_still_adoptable(self) -> None:
        from ccgram.handlers.topics.topic_orchestration import still_adoptable

        live = WindowRef(
            window_id="9F1C2D3E-4A5B", window_name="proj", cwd="/p", topic_eligible=True
        )
        with patch(
            "ccgram.handlers.topics.topic_orchestration.tmux_manager"
        ) as mock_tmux:
            mock_tmux.list_windows_for_reconciliation = AsyncMock(return_value=[live])
            assert await still_adoptable("9f1c2d3e-4a5b") is True

    async def test_an_ineligible_case_variant_is_still_refused(self) -> None:
        """Folding case must not soften the eligibility half of the check."""
        from ccgram.handlers.topics.topic_orchestration import still_adoptable

        live = WindowRef(
            window_id="9F1C2D3E-4A5B",
            window_name="proj",
            cwd="/p",
            topic_eligible=False,
        )
        with patch(
            "ccgram.handlers.topics.topic_orchestration.tmux_manager"
        ) as mock_tmux:
            mock_tmux.list_windows_for_reconciliation = AsyncMock(return_value=[live])
            assert await still_adoptable("9f1c2d3e-4a5b") is False
