"""The /commands entrypoint opens a topic-aware panel after authorization."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.commands import commands_command

_CO = "ccgram.handlers.commands"


@pytest.mark.parametrize("thread_id", [None, 1, 42])
async def test_commands_opens_the_current_topic_panel(thread_id):
    update = MagicMock()
    update.effective_user.id = 100
    update.message.message_thread_id = thread_id
    with (
        patch(f"{_CO}.config.is_user_allowed", return_value=True),
        patch(
            f"{_CO}.sync_scoped_menu_for_text_context", new_callable=AsyncMock
        ) as sync,
        patch(f"{_CO}.send_command_panel", new_callable=AsyncMock) as show,
    ):
        await commands_command(update, MagicMock())
    sync.assert_awaited_once_with(update, 100)
    show.assert_awaited_once_with(update.message, 100)


@pytest.mark.parametrize("missing", ["user", "message", "authorization"])
async def test_invalid_commands_request_does_not_open_panel(missing):
    update = MagicMock()
    if missing == "user":
        update.effective_user = None
    if missing == "message":
        update.message = None
    with (
        patch(f"{_CO}.config.is_user_allowed", return_value=missing != "authorization"),
        patch(f"{_CO}.send_command_panel", new_callable=AsyncMock) as show,
        patch(
            f"{_CO}.sync_scoped_menu_for_text_context", new_callable=AsyncMock
        ) as sync,
    ):
        await commands_command(update, MagicMock())
    show.assert_not_awaited()
    sync.assert_not_awaited()
