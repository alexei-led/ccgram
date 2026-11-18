"""Codex native command-menu captures must expose only verified controls."""

from pathlib import Path

import pytest

from ccgram.providers.codex import CodexProvider
from ccgram.providers.codex_tui import parse_codex_menu
from ccgram.providers.base import TerminalControl

FIXTURES = Path(__file__).parent / "fixtures" / "codex_menus"


@pytest.mark.parametrize(
    ("name", "ui_type"),
    [
        ("model", "CodexModel"),
        ("model-narrow", "CodexModel"),
        ("reasoning", "CodexReasoning"),
        ("advanced-reasoning", "CodexAdvancedReasoning"),
        ("permissions", "CodexPermissions"),
        ("skills", "CodexSkills"),
        ("statusline", "CodexStatusline"),
        ("statusline-narrow", "CodexStatusline"),
        ("statusline-scrolled", "CodexStatusline"),
        ("statusline-empty", "CodexStatusline"),
        ("review", "CodexReview"),
        ("keymap", "CodexKeymap"),
        ("keymap-empty", "CodexKeymap"),
        ("title", "CodexTitle"),
        ("title-empty", "CodexTitle"),
        ("theme", "CodexTheme"),
        ("theme-empty", "CodexTheme"),
        ("config", "CodexConfig"),
        ("config-search", "CodexConfig"),
        ("config-empty", "CodexConfig"),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
def test_verified_native_menu_capture(name: str, ui_type: str, padded: bool) -> None:
    capture = (FIXTURES / f"{name}.txt").read_text()
    if padded:
        capture = "\n".join(line.ljust(189) for line in capture.splitlines())
    status = CodexProvider().parse_terminal_status(capture)

    assert status is not None
    assert status.is_interactive
    assert status.ui_type == ui_type
    assert status.display_label == ui_type
    assert status.menu is not None
    assert not status.ui_advisory


@pytest.mark.parametrize(
    "name",
    [
        "statusline",
        "statusline-empty",
        "keymap",
        "keymap-empty",
        "title",
        "title-empty",
        "theme",
        "theme-empty",
        "config-search",
        "config-empty",
    ],
)
def test_searchable_menus_accept_text(name: str) -> None:
    status = CodexProvider().parse_terminal_status(
        (FIXTURES / f"{name}.txt").read_text()
    )

    assert status is not None and status.menu is not None
    assert status.menu.text_input


@pytest.mark.parametrize(
    ("name", "control"),
    [
        ("statusline", TerminalControl("Toggle", ("Space",))),
        ("statusline", TerminalControl("Move left", ("Left",))),
        ("statusline", TerminalControl("Move right", ("Right",))),
        ("keymap", TerminalControl("Previous group", ("Left",))),
        ("keymap", TerminalControl("Next group", ("Right",))),
        ("model", TerminalControl("Select", ("Enter",))),
        ("permissions", TerminalControl("Apply selected policy", ("Enter",))),
        ("statusline", TerminalControl("Save", ("Enter",))),
        ("title", TerminalControl("Save", ("Enter",))),
        ("theme", TerminalControl("Select", ("Enter",))),
        ("model", TerminalControl("Cancel", ("Escape",))),
        ("reasoning", TerminalControl("More reasoning", ("5",))),
    ],
)
def test_controls_use_verified_native_keys(name: str, control: TerminalControl) -> None:
    status = CodexProvider().parse_terminal_status(
        (FIXTURES / f"{name}.txt").read_text()
    )

    assert status is not None and status.menu is not None
    assert control in status.menu.controls


@pytest.mark.parametrize(
    "name", ["unsupported-personality", "unsupported-mode", "slash"]
)
def test_unrecognized_slash_commands_are_not_menus(name: str) -> None:
    capture = (FIXTURES / f"{name}.txt").read_text()

    assert parse_codex_menu(capture) is None
    assert CodexProvider().parse_terminal_status(capture) is None


def test_closed_menu_in_scrollback_is_not_treated_as_active() -> None:
    capture = (FIXTURES / "model.txt").read_text()
    idle_editor = "\n› Ask Codex to do anything\n  Ready · session"
    assert parse_codex_menu(capture + idle_editor) is None
    assert CodexProvider().parse_terminal_status(capture + idle_editor) is None


def test_statusline_search_has_filter_empty_state() -> None:
    status = CodexProvider().parse_terminal_status(
        (FIXTURES / "statusline-empty.txt").read_text()
    )

    assert status is not None
    assert status.ui_type == "CodexStatusline"
    assert "no matches" in status.raw_text


@pytest.mark.parametrize(
    "footer",
    [
        "enter default · s session · esc back",
        "enter apply · s session · esc back",
    ],
)
@pytest.mark.parametrize(
    ("name", "ui_type"),
    [
        ("reasoning", "CodexReasoning"),
        ("advanced-reasoning", "CodexAdvancedReasoning"),
    ],
)
def test_reasoning_footer_variants_keep_their_meaning(
    footer: str, name: str, ui_type: str
) -> None:
    capture = (FIXTURES / f"{name}.txt").read_text()
    capture = capture.replace(capture.splitlines()[-1].strip(), footer)
    status = CodexProvider().parse_terminal_status(capture)
    assert status is not None and status.menu is not None
    assert status.ui_type == ui_type
    assert footer in status.raw_text
    assert not status.menu.text_input
    label = (
        "Set default" if footer.startswith("enter default") else "Apply selected effort"
    )
    assert TerminalControl(label, ("Enter",)) in status.menu.controls
    assert TerminalControl("Use this session", ("s",)) in status.menu.controls


def test_prose_containing_menu_title_and_footer_is_not_a_menu() -> None:
    capture = (FIXTURES / "model.txt").read_text()
    capture = capture.replace(
        "Select Model and Effort", "Documentation: Select Model and Effort"
    )
    assert parse_codex_menu(capture) is None


def test_quoted_menu_is_not_active_chrome() -> None:
    capture = (FIXTURES / "model.txt").read_text()
    quoted = "\n".join("> " + line for line in capture.splitlines())
    assert CodexProvider().parse_terminal_status(quoted) is None


def test_model_current_default_marker_and_footer_are_preserved() -> None:
    capture = (FIXTURES / "model.txt").read_text()
    status = CodexProvider().parse_terminal_status(capture)
    assert status is not None
    assert "› 3. GPT-6-Astra (current)" in status.raw_text
    assert status.raw_text.endswith("enter select · esc back")


def test_reasoning_control_requires_its_visible_native_option() -> None:
    capture = (FIXTURES / "reasoning.txt").read_text()
    capture = "\n".join(
        line for line in capture.splitlines() if "More reasoning…" not in line
    )
    status = CodexProvider().parse_terminal_status(capture)
    assert status is not None and status.menu is not None
    assert TerminalControl("More reasoning", ("5",)) not in status.menu.controls


def test_config_search_is_an_explicit_mode_not_a_typing_toggle() -> None:
    for name, accepts_text in (
        ("config", False),
        ("config-search", True),
        ("config-empty", True),
        ("config", False),
    ):
        status = CodexProvider().parse_terminal_status(
            (FIXTURES / f"{name}.txt").read_text()
        )
        assert status is not None and status.menu is not None
        assert status.menu.text_input is accepts_text
        search = TerminalControl("Search", ("/",))
        assert (search in status.menu.controls) is (not accepts_text)
        assert "Changes save automatically" in status.raw_text


def test_numbered_picker_rejects_text_input() -> None:
    status = CodexProvider().parse_terminal_status((FIXTURES / "model.txt").read_text())
    assert status is not None and status.menu is not None
    assert not status.menu.text_input


@pytest.mark.parametrize(
    ("current", "ui_type"),
    [
        ("review", "CodexReview"),
        ("permissions", "CodexPermissions"),
        ("skills", "CodexSkills"),
        ("theme", "CodexTheme"),
    ],
)
def test_current_menu_wins_when_old_menu_has_the_same_footer(
    current: str, ui_type: str
) -> None:
    old = (FIXTURES / "model.txt").read_text()
    active = (FIXTURES / f"{current}.txt").read_text()
    status = CodexProvider().parse_terminal_status(old + active)
    assert status is not None and status.ui_type == ui_type
    assert "Select Model and Effort" not in status.raw_text


def test_old_menu_does_not_claim_a_new_unknown_selector() -> None:
    old = (FIXTURES / "model.txt").read_text()
    unknown = "\nUnknown selector\n› 1. First\n  2. Second\nenter select · esc back\n"
    assert parse_codex_menu(old + unknown) is None
    status = CodexProvider().parse_terminal_status(old + unknown)
    assert status is None or status.menu is None


@pytest.mark.parametrize(
    "body",
    ["Waiting for shortcut input", "› Control binding\nPress a new shortcut"],
)
def test_keymap_child_editor_cannot_reuse_root_menu_metadata(body: str) -> None:
    root = (FIXTURES / "keymap.txt").read_text()
    header = root.split("Type to search shortcuts", 1)[0]
    footer = root.splitlines()[-1]
    child = header + body + "\n" + footer
    assert parse_codex_menu(child) is None


@pytest.mark.parametrize("name", ["skills", "review", "config", "theme-empty"])
def test_no_activation_for_unknown_child_or_empty_selection(name: str) -> None:
    status = CodexProvider().parse_terminal_status(
        (FIXTURES / f"{name}.txt").read_text()
    )
    assert status is not None and status.menu is not None
    assert all("Enter" not in control.keys for control in status.menu.controls)


def test_reasoning_submenu_footer_exposes_select_not_session_write() -> None:
    capture = (FIXTURES / "reasoning.txt").read_text()
    capture = capture.replace(
        capture.splitlines()[-1].strip(), "enter select · esc back"
    )
    capture = capture.replace("› 3.", "  3.").replace("  5.", "› 5.")
    status = CodexProvider().parse_terminal_status(capture)
    assert status is not None and status.menu is not None
    assert TerminalControl("Select", ("Enter",)) in status.menu.controls
    assert TerminalControl("Use this session", ("s",)) not in status.menu.controls


def test_keymap_root_does_not_expose_unverified_editor_writes() -> None:
    status = CodexProvider().parse_terminal_status(
        (FIXTURES / "keymap.txt").read_text()
    )
    assert status is not None and status.menu is not None
    assert {key for control in status.menu.controls for key in control.keys} == {
        "Up",
        "Down",
        "Left",
        "Right",
        "Escape",
    }


@pytest.mark.parametrize(
    ("name", "footer"),
    [
        (
            "statusline",
            "space toggle · ←/→ reorder · enter save · esc cancel",
        ),
        (
            "keymap",
            "left/right group · enter edit shortcut · * custom · - unbound · esc close",
        ),
        (
            "config",
            "left/right group · / search · enter open · esc clear search/close",
        ),
    ],
)
def test_wrapped_footer_stays_attached_to_the_active_menu(
    name: str, footer: str
) -> None:
    capture = (FIXTURES / f"{name}.txt").read_text()
    split_at = footer.rfind(" · ")
    wrapped = capture.replace(
        footer, footer[: split_at + 3] + "\n  " + footer[split_at + 3 :]
    )
    status = CodexProvider().parse_terminal_status(wrapped)
    assert status is not None and status.menu is not None
    assert (
        "enter save" in status.raw_text
        or "enter edit shortcut" in status.raw_text
        or "enter open" in status.raw_text
    )


def test_wrapped_old_footer_does_not_claim_a_new_unknown_selector() -> None:
    old = (
        (FIXTURES / "model.txt")
        .read_text()
        .replace("enter select · esc back", "enter select ·\n  esc back")
    )
    unknown = "\nUnknown selector\n› 1. First\n  2. Second\nenter select · esc back\n"
    assert parse_codex_menu(old + unknown) is None


def test_multiple_st_terminated_osc_sequences_preserve_visible_menu() -> None:
    capture = (FIXTURES / "model.txt").read_text()
    wrapped = (
        chr(27)
        + "]0;before"
        + chr(27)
        + "\\"
        + capture
        + chr(27)
        + "]0;after"
        + chr(27)
        + "\\"
    )
    status = parse_codex_menu(wrapped)
    assert status is not None and status.ui_type == "CodexModel"
    assert "Select Model and Effort" in status.raw_text


def test_ansi_csi_sequences_do_not_hide_the_active_menu() -> None:
    capture = (FIXTURES / "model.txt").read_text()
    ansi = capture.replace(
        "Select Model and Effort",
        "\x1b[1;36mSelect Model and Effort\x1b[0m",
    ).replace(
        "› 3. GPT-6-Astra (current)",
        "\x1b[1m› 10. GPT-6-Astra (current)\x1b[0m",
    )
    ansi = ansi.replace("  4. GPT-6-Sol", "  11. GPT-6-Sol")
    status = CodexProvider().parse_terminal_status(ansi)
    assert status is not None and status.menu is not None
    assert status.ui_type == "CodexModel"
    assert "› 10. GPT-6-Astra (current)" in status.raw_text
    assert "\x1b" not in status.raw_text


def test_later_active_menu_wins_over_old_menu() -> None:
    model = (FIXTURES / "model.txt").read_text()
    statusline = (FIXTURES / "statusline.txt").read_text()
    status = CodexProvider().parse_terminal_status(model + statusline)
    assert status is not None and status.ui_type == "CodexStatusline"
    assert "Select Model and Effort" not in status.raw_text
