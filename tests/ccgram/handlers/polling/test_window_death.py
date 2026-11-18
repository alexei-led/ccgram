"""Behavior of the public Telegram window-death entrypoint."""

from unittest.mock import AsyncMock, MagicMock, call, patch

from ccgram.handlers.polling.window_death import handle_window_death
from ccgram.thread_router import ThreadRouter


async def test_routes_current_recipients_even_when_deletion_unbinds_them():
    bot = MagicMock()
    router = ThreadRouter(
        schedule_save=lambda: None,
        has_window_state=lambda _wid: False,
    )
    router.bind_thread(7, 42, "old")
    router.bind_thread(7, 99, "other")

    async def delete_binding(_bot, user_id, thread_id, window_id):
        assert router.unbind_thread(user_id, thread_id) == window_id

    notify = AsyncMock(side_effect=delete_binding)
    with (
        patch("ccgram.thread_router.thread_router", router),
        patch(
            "ccgram.handlers.polling.window_death._handle_dead_window_notification",
            notify,
        ),
    ):
        router.bind_thread(7, 42, "target")
        router.bind_thread(8, 43, "target")
        await handle_window_death(bot, "old")
        notify.assert_not_awaited()
        await handle_window_death(bot, "target")

    assert notify.await_args_list == [
        call(bot, 7, 42, "target"),
        call(bot, 8, 43, "target"),
    ]
