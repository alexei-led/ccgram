"""Real Pi captures must surface menus, not transcript lookalikes."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.interactive.interactive_ui import handle_interactive_ui
from ccgram.providers.pi import PiProvider
from ccgram.multiplexer.base import WindowRef

FIXTURES = Path(__file__).parent / "fixtures" / "pi_menus"


@pytest.mark.parametrize(
    ("name", "ui_type"),
    [
        ("thinking", "PiThinking"),
        ("model", "PiModel"),
        ("settings", "PiSettings"),
        ("theme", "PiSelect"),
        ("scoped-models", "PiScopedModels"),
        ("modes", "PiSelect"),
        ("ccgram-input", "PiInput"),
        ("ccgram-editor", "PiEditor"),
        ("login", "PiSelect"),
        ("logout", "PiSelect"),
        ("tree", "PiTree"),
    ],
)
@pytest.mark.parametrize("backend", ["tmux", "agterm"])
def test_live_menu_capture(name: str, ui_type: str, backend: str) -> None:
    capture = (FIXTURES / f"{name}.txt").read_text()
    if backend == "agterm":
        capture = "\n".join(line.ljust(189) for line in capture.splitlines())
    status = PiProvider().parse_terminal_status(capture)
    assert status is not None and status.is_interactive
    assert status.ui_type == ui_type
    assert not status.ui_advisory
    assert "Router initialized" not in status.raw_text
    assert "fix-pi-interactive-commands" not in status.raw_text


@pytest.mark.parametrize("name", ["thinking", "settings", "scoped-models"])
def test_old_menu_above_idle_editor_is_not_interactive(name: str) -> None:
    capture = (FIXTURES / f"{name}.txt").read_text()
    idle_editor = "\n" + "─" * 80 + "\n >\n" + "─" * 80 + "\nstatus"
    assert PiProvider().parse_terminal_status(capture + idle_editor) is None


def test_editor_newline_uses_verified_shift_enter() -> None:
    capture = (FIXTURES / "ccgram-editor.txt").read_text()
    status = PiProvider().parse_terminal_status(capture)
    assert status is not None and status.menu is not None
    newline = next(c for c in status.menu.controls if c.label == "New line")
    assert newline.keys == ("S-Enter",)


def test_unadvertised_editor_newline_key_is_not_invented() -> None:
    capture = (
        (FIXTURES / "ccgram-editor.txt").read_text().replace("shift+enter", "alt+enter")
    )
    status = PiProvider().parse_terminal_status(capture)
    assert status is not None and status.menu is not None
    assert not any(c.label == "New line" for c in status.menu.controls)


def test_empty_filtered_settings_remain_interactive() -> None:
    screen = (
        "─" * 80
        + """
> no-such-setting

  No matching settings
  Type to search · Enter/Space to change · Esc to cancel
"""
        + "─" * 80
    )
    status = PiProvider().parse_terminal_status(screen)
    assert status is not None and status.ui_type == "PiSettings"


def test_standard_extension_selector_without_command_allowlist() -> None:
    screen = (
        "─" * 80
        + """
 Choose a mode

 → 1. Safe
   2. Fast

 ↑↓ navigate  enter select  escape/ctrl+c cancel
"""
        + "─" * 80
    )
    status = PiProvider().parse_terminal_status(screen)
    assert status is not None and status.ui_type == "PiSelect"


def test_fork_selector_header_outside_frame_is_preserved() -> None:
    border = "─" * 80
    screen = (
        "Fork from Message\n"
        "Select a user message to copy the active path up to that point into a new session\n"
        + border
        + "\n› Non-secret fork target\n  Message 1 of 1\n"
        + border
    )
    status = PiProvider().parse_terminal_status(screen)
    assert status is not None and status.ui_type == "PiSelect"
    assert "Fork from Message" in status.raw_text


def test_empty_searchable_settings_submenu_keeps_controls() -> None:
    border = "─" * 80
    screen = (
        border + "\nSelect model\n> missing\nNo matching items\n"
        "Type to filter · Enter to select · Esc to go back\n" + border
    )
    status = PiProvider().parse_terminal_status(screen)
    assert status is not None and status.ui_type == "PiSelect"


async def test_polling_preserves_already_rendered_pi_grid() -> None:
    from ccgram.handlers.polling.window_tick.observe import _resolve_status

    capture = "\n".join(
        line.ljust(189) for line in (FIXTURES / "settings.txt").read_text().splitlines()
    )
    window = WindowRef("pi-grid", "Pi", "/tmp", pane_width=80, pane_height=24)
    with patch(
        "ccgram.handlers.polling.window_tick.observe._get_provider",
        return_value=PiProvider(),
    ):
        status = await _resolve_status("pi-grid", capture, window)
    assert status is not None and status.ui_type == "PiSettings"
    assert "Auto-compact" in status.raw_text


async def test_settings_reaches_telegram_with_value_controls() -> None:
    client = AsyncMock()
    client.send_message.return_value = MagicMock(message_id=8821)
    mux = AsyncMock()
    mux.find_window_by_id.return_value = MagicMock(window_id="pi-menu")
    mux.capture_pane.return_value = (FIXTURES / "settings.txt").read_text()
    with (
        patch("ccgram.handlers.interactive.interactive_ui.tmux_manager", mux),
        patch(
            "ccgram.handlers.interactive.interactive_ui.get_window_provider",
            return_value="pi",
        ),
        patch(
            "ccgram.handlers.interactive.interactive_ui.rate_limit_send", AsyncMock()
        ),
    ):
        assert await handle_interactive_ui(
            client, 8821, "pi-menu", thread_id=8822, chat_id=-8821
        )
    sent = client.send_message.call_args.kwargs
    assert "Auto-compact" in sent["text"]
    assert "Reply to this message" in sent["text"]
    buttons = [b for row in sent["reply_markup"].inline_keyboard for b in row]
    assert {"↑", "↓", "↑5", "↓5", "⏎ Enter", "⎋ Esc", "Clear search"} <= {
        b.text for b in buttons
    }
    assert all(len(str(b.callback_data).encode()) <= 64 for b in buttons)
    assert not any(str(b.callback_data).startswith("aq:pick:") for b in buttons)
