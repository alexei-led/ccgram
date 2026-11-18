"""Sanitized native Claude 2.1.312 captures and non-menu regressions."""

from pathlib import Path

import pytest

from ccgram.providers.claude import ClaudeProvider

_FIXTURES = Path(__file__).parent / "fixtures" / "claude_menus"
_MENU_CASES = [
    pytest.param(
        "model",
        "ClaudeModel",
        False,
        (
            ("Previous model", ("Up",)),
            ("Next model", ("Down",)),
            ("Set as default", ("Enter",)),
            ("Use this session", ("s",)),
            ("Cancel", ("Escape",)),
        ),
        id="model",
    ),
    pytest.param(
        "effort",
        "ClaudeEffort",
        False,
        (
            ("Decrease effort", ("Left",)),
            ("Increase effort", ("Right",)),
            ("Confirm", ("Enter",)),
            ("Use this session", ("s",)),
            ("Cancel", ("Escape",)),
        ),
        id="effort",
    ),
    pytest.param(
        "permissions",
        "ClaudePermissions",
        False,
        (
            ("Previous tab", ("Left",)),
            ("Next tab", ("Right",)),
            ("Select", ("Down",)),
            ("Cancel", ("Escape",)),
        ),
        id="permissions",
    ),
    pytest.param(
        "skills",
        "ClaudeSkills",
        False,
        (
            ("Cycle skill state", ("Enter",)),
            ("Search", ("/",)),
            ("Sort", ("t",)),
            ("Escape", ("Escape",)),
        ),
        id="skills",
    ),
    pytest.param(
        "skills_empty_filter",
        "ClaudeSkills",
        True,
        (
            ("Down", ("Down",)),
            ("Select", ("Enter",)),
            ("Clear filter", ("Escape",)),
        ),
        id="skills-empty-filter",
    ),
]


@pytest.mark.parametrize(("name", "ui_type", "text_input", "controls"), _MENU_CASES)
@pytest.mark.parametrize("padded", [False, True], ids=["trimmed", "agterm-padded"])
def test_native_command_menu(name, ui_type, text_input, controls, padded):
    screen = (_FIXTURES / f"{name}.txt").read_text()
    if padded:
        screen = "\n".join(line.ljust(189) for line in screen.splitlines())

    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is not None
    assert status.is_interactive
    assert status.ui_type == ui_type
    assert status.ui_advisory is False
    assert status.menu is not None
    assert tuple((c.label, c.keys) for c in status.menu.controls) == controls
    assert status.menu.text_input is text_input


@pytest.mark.parametrize("name", sorted(path.stem for path in _FIXTURES.glob("*.txt")))
def test_menu_in_scrollback_does_not_override_idle_composer(name):
    screen = (_FIXTURES / f"{name}.txt").read_text()
    screen += "\n" + "─" * 80 + "\n❯ \n" + "─" * 80 + "\n-- INSERT --"

    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is None or status.menu is None


@pytest.mark.parametrize(
    "screen",
    [
        "Previous Effort output\n←/→ to adjust · Enter to confirm · s for this session only · Esc to cancel",
        "Previous Skills output\n98 skills · enter/space to cycle, / to search, t to sort, Esc to close",
        "Output mentioned Enter to set as default · s to use this session only · Esc to cancel",
        "Select model\n1. First\n2. Second",
    ],
)
def test_partial_menu_does_not_match_or_crash(screen):
    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is None or status.menu is None


def test_model_picker_excludes_previous_terminal_content_and_ansi():
    screen = (
        "unrelated previous output\n\x1b[32m"
        + (_FIXTURES / "model.txt").read_text()
        + "\x1b[0m"
    )

    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is not None
    assert status.ui_type == "ClaudeModel"
    assert status.raw_text.startswith("Select model")
    assert "unrelated previous output" not in status.raw_text
    assert "\x1b" not in status.raw_text


def test_wrapped_model_footer_retains_explicit_default_and_session_choices():
    screen = (
        (_FIXTURES / "model.txt")
        .read_text()
        .replace("· s to use this session only", "·\ns to use this session only")
    )

    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is not None and status.menu is not None
    controls = {c.label: c.keys for c in status.menu.controls}
    assert controls["Set as default"] == ("Enter",)
    assert controls["Use this session"] == ("s",)


@pytest.mark.parametrize(
    ("screen", "ui_type"),
    [
        (
            "Do you want to proceed?\n\n❯ Yes\n  No\n\nEsc to cancel\n",
            "PermissionPrompt",
        ),
        ("☐ First option\n☐ Second option\nEnter to select\n", "AskUserQuestion"),
    ],
)
def test_existing_prompts_do_not_get_command_menu_metadata(screen, ui_type):
    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is not None
    assert status.ui_type == ui_type
    assert status.menu is None


def test_plan_dialog_does_not_get_command_menu_metadata(sample_pane_exit_plan):
    status = ClaudeProvider().parse_terminal_status(sample_pane_exit_plan)

    assert status is not None
    assert status.ui_type == "ExitPlanMode"
    assert status.menu is None


@pytest.mark.parametrize(
    ("name", "ui_type", "text_input", "expected_controls"),
    [
        (
            "settings_search",
            "ClaudeSettings",
            True,
            {"Select": ("Down",), "Tabs": ("Up",), "Clear / leave search": ("Escape",)},
        ),
        (
            "settings_empty_filter",
            "ClaudeSettings",
            True,
            {"Select": ("Down",), "Tabs": ("Up",), "Clear / leave search": ("Escape",)},
        ),
        (
            "settings_list",
            "ClaudeSettings",
            False,
            {"Change setting": ("Enter",), "Search": ("/",), "Close": ("Escape",)},
        ),
        (
            "settings_tabs",
            "ClaudeSettings",
            False,
            {
                "Previous tab": ("Left",),
                "Next tab": ("Right",),
                "Return": ("Down",),
                "Close": ("Escape",),
            },
        ),
        ("settings_usage", "ClaudeSettings", False, {"Close": ("Escape",)}),
        (
            "theme",
            "ClaudeTheme",
            False,
            {
                "Previous theme": ("Up",),
                "Next theme": ("Down",),
                "Apply theme": ("Enter",),
                "Cancel": ("Escape",),
            },
        ),
        (
            "theme_narrow",
            "ClaudeTheme",
            False,
            {
                "Previous theme": ("Up",),
                "Next theme": ("Down",),
                "Apply theme": ("Enter",),
                "Cancel": ("Escape",),
            },
        ),
        (
            "help",
            "ClaudeHelp",
            False,
            {"Previous tab": ("Left",), "Next tab": ("Right",), "Cancel": ("Escape",)},
        ),
        (
            "help_commands",
            "ClaudeHelp",
            False,
            {"Previous tab": ("Left",), "Next tab": ("Right",), "Cancel": ("Escape",)},
        ),
        ("rewind_empty", "ClaudeRewind", False, {"Cancel": ("Escape",)}),
    ],
)
def test_disposable_profile_menu_states(name, ui_type, text_input, expected_controls):
    screen = (_FIXTURES / f"{name}.txt").read_text()
    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is not None
    assert status.ui_type == ui_type
    assert status.is_interactive and not status.ui_advisory
    assert status.menu is not None
    assert status.menu.text_input is text_input
    assert {c.label: c.keys for c in status.menu.controls} == expected_controls


@pytest.mark.parametrize(
    "name",
    sorted(path.stem for path in _FIXTURES.glob("*.txt")),
)
def test_ordinary_output_after_menu_footer_retires_confirmed_menu(name):
    screen = (_FIXTURES / f"{name}.txt").read_text()
    screen += "\nOrdinary output after the menu closed.\n"

    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is None or status.menu is None


@pytest.mark.parametrize("rows", ["", "1. Default\n2. Opus 5.5 ✔\n"])
def test_model_title_and_footer_without_selection_are_not_confirmed(rows):
    screen = (
        "Select model\n"
        + rows
        + "Enter to set as default · s to use this session only · Esc to cancel\n"
    )

    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is None or status.menu is None


@pytest.mark.parametrize(
    "screen",
    [
        "Effort\n←/→ to adjust · Enter to confirm · s for this session only · Esc to cancel",
        "Theme\nEnter to select · Esc to cancel",
        "Permissions  Recently denied   Allow\n←/→ to switch · ↓ to select · Esc to cancel",
        "Settings  Status   Config   Usage   Stats\nType to filter · Enter/↓ to select · ↑ to tabs · Esc to clear",
        "Help  General   Commands   Custom commands\nEsc to cancel",
        "Skills\n98 skills · enter/space to cycle, / to search, t to sort, Esc to close\nPlugin skills are managed via /plugin",
    ],
)
def test_command_titles_and_hints_need_native_body_evidence(screen):
    status = ClaudeProvider().parse_terminal_status(screen)

    assert status is None or status.menu is None
