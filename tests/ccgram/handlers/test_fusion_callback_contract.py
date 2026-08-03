"""Regression coverage for lossless callback targets on Herdr sessions."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.agent_command import _build_keyboard
from ccgram.handlers.callback_tokens import _tokens, resolve_callback_data
from ccgram.handlers.interactive.interactive_ui import _build_interactive_keyboard
from ccgram.handlers.live.live_view import build_live_keyboard
from ccgram.handlers.live.pane_callbacks import (
    build_pane_buttons,
    build_pane_lifecycle_button,
)
from ccgram.handlers.live.screenshot_callbacks import build_screenshot_keyboard
from ccgram.handlers.recovery.recovery_banner import build_recovery_keyboard
from ccgram.handlers.recovery.resume_picker import (
    _SessionEntry,
    _build_empty_resume_keyboard,
    _build_resume_picker_keyboard,
)
from ccgram.handlers.shell.shell_commands import _build_approval_keyboard
from ccgram.handlers.shell.shell_prompt_orchestrator import _show_offer_keyboard

HERDR_TARGET = "herdr-session-v1-" + "a" * 64


@pytest.fixture(autouse=True)
def _clear_callback_tokens() -> Generator[None, None, None]:
    _tokens.clear()
    yield
    _tokens.clear()


def _callback_data(markup: InlineKeyboardMarkup) -> list[str]:
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if isinstance(button.callback_data, str)
    ]


def _assert_lossless_callbacks(callbacks: list[str]) -> None:
    target_callbacks = [data for data in callbacks if data not in {"rec:x", "rec:p:0"}]
    assert target_callbacks
    assert all(len(data.encode("utf-8")) <= 64 for data in target_callbacks)
    for data in target_callbacks:
        payload = resolve_callback_data(
            data,
            7,
            lambda user_id, wid: user_id == 7 and wid == HERDR_TARGET,
        )
        assert payload is not None
        assert HERDR_TARGET in payload


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _build_approval_keyboard(HERDR_TARGET, is_dangerous=False),
        lambda: _build_keyboard(HERDR_TARGET, current="claude"),
        lambda: _build_interactive_keyboard(HERDR_TARGET, pane_id="%7"),
        lambda: build_screenshot_keyboard(HERDR_TARGET, pane_id="%7"),
        lambda: build_live_keyboard(HERDR_TARGET, pane_id="%7"),
        lambda: InlineKeyboardMarkup([build_pane_buttons(HERDR_TARGET, "%7", False)]),
        lambda: InlineKeyboardMarkup(
            [[build_pane_lifecycle_button(HERDR_TARGET, enabled=True)]]
        ),
        lambda: build_recovery_keyboard(HERDR_TARGET),
        lambda: _build_empty_resume_keyboard(HERDR_TARGET),
        lambda: _build_resume_picker_keyboard(
            [_SessionEntry("session", "summary")], HERDR_TARGET
        ),
    ],
    ids=[
        "shell-approval",
        "agent-picker",
        "interactive",
        "screenshot",
        "live",
        "pane-buttons",
        "pane-lifecycle",
        "recovery-banner",
        "empty-resume-picker",
        "resume-picker",
    ],
)
def test_builders_losslessly_round_trip_herdr_target(builder) -> None:
    _assert_lossless_callbacks(_callback_data(builder()))


async def test_shell_offer_losslessly_round_trips_herdr_target() -> None:
    with patch(
        "ccgram.handlers.shell.shell_prompt_orchestrator.safe_send",
        new_callable=AsyncMock,
        return_value=MagicMock(),
    ) as safe_send:
        await _show_offer_keyboard(
            HERDR_TARGET, client=MagicMock(), chat_id=1, thread_id=2
        )

    assert safe_send.await_args is not None
    markup = safe_send.await_args.kwargs["reply_markup"]
    _assert_lossless_callbacks(_callback_data(markup))
