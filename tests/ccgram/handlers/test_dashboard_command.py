"""Tests for /dashboard: native WebApp button delivered via private chat."""

from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import Forbidden

from ccgram.handlers.dashboard_command import dashboard_command

MOD = "ccgram.handlers.dashboard_command"


def _update(user_id=7, thread_id=42):
    u = MagicMock()
    u.id = user_id
    m = MagicMock()
    m.reply_text = AsyncMock()
    upd = MagicMock()
    upd.effective_user = u
    upd.message = m
    ctx = MagicMock()
    ctx.bot.send_message = AsyncMock()
    return upd, m, ctx


async def test_dms_native_button_for_bound_topic():
    upd, msg, ctx = _update()
    button = MagicMock(name="btn")
    with (
        patch(f"{MOD}.config") as cfg,
        patch(f"{MOD}.get_thread_id", return_value=42),
        patch(f"{MOD}.thread_router") as tr,
        patch(f"{MOD}.build_dashboard_button", return_value=button) as bdb,
        patch(f"{MOD}.safe_reply", new_callable=AsyncMock) as sr,
    ):
        cfg.is_user_allowed.return_value = True
        cfg.miniapp_base_url = "https://x.example"
        tr.get_window_for_thread.return_value = "w1"
        tr.get_display_name.return_value = "proj"
        await dashboard_command(upd, ctx)
    bdb.assert_called_once_with("w1", 7)
    kb = ctx.bot.send_message.await_args.kwargs["reply_markup"].inline_keyboard
    assert [btn for row in kb for btn in row] == [button]
    assert ctx.bot.send_message.await_args.kwargs["chat_id"] == 7
    sr.assert_awaited_once()


async def test_unbound_topic_replies_error():
    upd, msg, ctx = _update()
    with (
        patch(f"{MOD}.config") as cfg,
        patch(f"{MOD}.get_thread_id", return_value=42),
        patch(f"{MOD}.thread_router") as tr,
        patch(f"{MOD}.safe_reply", new_callable=AsyncMock) as sr,
    ):
        cfg.is_user_allowed.return_value = True
        cfg.miniapp_base_url = "https://x.example"
        tr.get_window_for_thread.return_value = None
        await dashboard_command(upd, ctx)
    assert sr.await_args is not None
    assert "not bound" in sr.await_args.args[1]
    ctx.bot.send_message.assert_not_awaited()


async def test_dm_failure_hints_private_start():
    upd, msg, ctx = _update()
    ctx.bot.send_message.side_effect = Forbidden("bot blocked")
    with (
        patch(f"{MOD}.config") as cfg,
        patch(f"{MOD}.get_thread_id", return_value=42),
        patch(f"{MOD}.thread_router") as tr,
        patch(f"{MOD}.build_dashboard_button", return_value=MagicMock()),
        patch(f"{MOD}.safe_reply", new_callable=AsyncMock) as sr,
    ):
        cfg.is_user_allowed.return_value = True
        cfg.miniapp_base_url = "https://x.example"
        tr.get_window_for_thread.return_value = "w1"
        tr.get_display_name.return_value = "proj"
        await dashboard_command(upd, ctx)
    assert sr.await_args is not None
    assert "/start" in sr.await_args.args[1]
