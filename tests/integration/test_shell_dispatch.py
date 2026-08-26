"""Integration tests for PTB dispatch routing to the shell handler.

Text in a shell-bound topic must reach ``handle_text_message`` unchanged —
including a ``!``-prefixed raw command, which must not be mistaken for a
Telegram bot command — and ``sh:``-prefixed callbacks must reach the shell
callback handler only after the allowlist check passes.
"""

from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.callback_data import CB_SHELL_RUN

pytestmark = pytest.mark.integration


async def test_shell_callback_dispatches_to_shell_handler(
    dispatch_app, make_callback_update
) -> None:
    update = make_callback_update(f"{CB_SHELL_RUN}@0", bot=dispatch_app.bot)

    with (
        patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_callback",
            new_callable=AsyncMock,
        ) as mock_shell_cb,
        patch(
            "ccgram.handlers.callback_registry.config.is_user_allowed",
            return_value=True,
        ),
    ):
        await dispatch_app.process_update(update)

    mock_shell_cb.assert_awaited_once()
    assert mock_shell_cb.call_args[0][2].startswith(CB_SHELL_RUN)


async def test_shell_callback_from_unauthorized_user_is_refused(
    dispatch_app, make_callback_update
) -> None:
    """The allowlist gate lives in dispatch(), so the handler never runs."""
    update = make_callback_update(f"{CB_SHELL_RUN}@0", bot=dispatch_app.bot)

    with (
        patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_callback",
            new_callable=AsyncMock,
        ) as mock_shell_cb,
        patch(
            "ccgram.handlers.callback_registry.config.is_user_allowed",
            return_value=False,
        ),
    ):
        await dispatch_app.process_update(update)

    mock_shell_cb.assert_not_awaited()


async def test_unregistered_callback_prefix_reaches_no_handler(
    dispatch_app, make_callback_update
) -> None:
    update = make_callback_update("zz:nothing:@0", bot=dispatch_app.bot)

    with (
        patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_callback",
            new_callable=AsyncMock,
        ) as mock_shell_cb,
        patch(
            "ccgram.handlers.callback_registry.config.is_user_allowed",
            return_value=True,
        ),
    ):
        await dispatch_app.process_update(update)

    mock_shell_cb.assert_not_awaited()
