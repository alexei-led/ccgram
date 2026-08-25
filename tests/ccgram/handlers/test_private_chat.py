"""Tests for the private-chat /start greeting."""

from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.private_chat import private_start_command

MOD = "ccgram.handlers.private_chat"


async def test_private_start_replies_welcome():
    u = MagicMock()
    u.id = 7
    m = MagicMock()
    m.reply_text = AsyncMock()
    upd = MagicMock()
    upd.effective_user = u
    upd.message = m
    with (
        patch(f"{MOD}.config") as cfg,
        patch(f"{MOD}.safe_reply", new_callable=AsyncMock) as sr,
    ):
        cfg.is_user_allowed.return_value = True
        await private_start_command(upd, MagicMock())
    assert sr.await_args is not None
    assert "topics" in sr.await_args.args[1]


async def test_private_start_ignores_disallowed_user():
    u = MagicMock()
    u.id = 8
    upd = MagicMock()
    upd.effective_user = u
    upd.message = MagicMock()
    with (
        patch(f"{MOD}.config") as cfg,
        patch(f"{MOD}.safe_reply", new_callable=AsyncMock) as sr,
    ):
        cfg.is_user_allowed.return_value = False
        await private_start_command(upd, MagicMock())
    sr.assert_not_awaited()
