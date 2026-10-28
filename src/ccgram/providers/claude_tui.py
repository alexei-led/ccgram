"""Verified Claude command menus; ordinary dialogs retain the legacy parser."""

import re

from .base import StatusUpdate, TerminalControl, TerminalMenu

_TITLE = re.compile(
    r"^(Select model|Effort|Skills|Theme|Rewind|Permissions(?:  +.*)?|"
    r"Settings  +Status.*|Help  +General.*)$"
)
_COMPOSER_BORDER = re.compile(r"^─{8,}$")
_MODEL_FOOTER = "Enter to set as default · s to use this session only · Esc to cancel"
_EFFORT_FOOTER = (
    "←/→ to adjust · Enter to confirm · s for this session only · Esc to cancel"
)
_PERMISSIONS_FOOTER = "←/→ to switch · ↓ to select · Esc to cancel"
_SKILLS_FOOTER = "enter/space to cycle, / to search, t to sort, Esc to close"
_SKILLS_FILTER_FOOTER = "type to filter · ↓/enter to select · esc to clear"
_ROW_CURSOR = re.compile(r"^\s*❯\s+\S", re.MULTILINE)
_MODEL_CURSOR = re.compile(r"^\s*❯\s+\d+\.\s+\S", re.MULTILINE)
_SKILLS_TRAILERS = (
    "Plugin skills are managed via /plugin",
    "/deep-research runs a workflow, not your claude.ai skill.",
)
_CANCEL = TerminalControl("Cancel", ("Escape",))
_CLOSE = TerminalControl("Close", ("Escape",))


def _settings_menu(hints: str) -> TerminalMenu | None:
    if hints.endswith("Type to filter · Enter/↓ to select · ↑ to tabs · Esc to clear"):
        return TerminalMenu(
            controls=(
                TerminalControl("Select", ("Down",)),
                TerminalControl("Tabs", ("Up",)),
                TerminalControl("Clear / leave search", ("Escape",)),
            ),
            text_input=True,
        )
    if hints.endswith("Enter/Space to change · / to search · Esc to close"):
        return TerminalMenu(
            controls=(
                TerminalControl("Change setting", ("Enter",)),
                TerminalControl("Search", ("/",)),
                _CLOSE,
            )
        )
    if hints.endswith("←/→/tab to switch · ↓ to return · Esc to close"):
        return TerminalMenu(
            controls=(
                TerminalControl("Previous tab", ("Left",)),
                TerminalControl("Next tab", ("Right",)),
                TerminalControl("Return", ("Down",)),
                _CLOSE,
            )
        )
    if "Total cost:" in hints and hints.endswith("Esc to cancel"):
        return TerminalMenu(controls=(_CLOSE,))
    return None


def _picker_menu(title: str, hints: str, body: str) -> tuple[str, TerminalMenu] | None:
    controls: tuple[TerminalControl, ...]
    text_input = False
    if (
        title == "Select model"
        and hints.endswith(_MODEL_FOOTER)
        and _MODEL_CURSOR.search(body)
    ):
        ui_type = "ClaudeModel"
        controls = (
            TerminalControl("Previous model", ("Up",)),
            TerminalControl("Next model", ("Down",)),
            TerminalControl("Set as default", ("Enter",)),
            TerminalControl("Use this session", ("s",)),
            _CANCEL,
        )
    elif title == "Effort" and hints.endswith(_EFFORT_FOOTER) and "▲" in body:
        ui_type = "ClaudeEffort"
        controls = (
            TerminalControl("Decrease effort", ("Left",)),
            TerminalControl("Increase effort", ("Right",)),
            TerminalControl("Confirm", ("Enter",)),
            TerminalControl("Use this session", ("s",)),
            _CANCEL,
        )
    elif (
        title.startswith("Permissions")
        and hints.endswith(_PERMISSIONS_FOOTER)
        and "⌕" in body
    ):
        ui_type = "ClaudePermissions"
        controls = (
            TerminalControl("Previous tab", ("Left",)),
            TerminalControl("Next tab", ("Right",)),
            TerminalControl("Select", ("Down",)),
            _CANCEL,
        )
    elif title == "Skills" and _SKILLS_FILTER_FOOTER in hints.lower():
        ui_type = "ClaudeSkills"
        text_input = True
        controls = (
            TerminalControl("Down", ("Down",)),
            TerminalControl("Select", ("Enter",)),
            TerminalControl("Clear filter", ("Escape",)),
        )
    elif title == "Skills" and _SKILLS_FOOTER.lower() in hints.lower():
        ui_type = "ClaudeSkills"
        controls = (
            TerminalControl("Cycle skill state", ("Enter",)),
            TerminalControl("Search", ("/",)),
            TerminalControl("Sort", ("t",)),
            TerminalControl("Escape", ("Escape",)),
        )
    elif (
        title == "Theme"
        and hints.endswith("Enter to select · Esc to cancel")
        and _ROW_CURSOR.search(body)
    ):
        ui_type = "ClaudeTheme"
        controls = (
            TerminalControl("Previous theme", ("Up",)),
            TerminalControl("Next theme", ("Down",)),
            TerminalControl("Apply theme", ("Enter",)),
            _CANCEL,
        )
    elif (
        title.startswith("Help  ")
        and hints.endswith("Esc to cancel")
        and "For more help:" in hints
    ):
        ui_type = "ClaudeHelp"
        controls = (
            TerminalControl("Previous tab", ("Left",)),
            TerminalControl("Next tab", ("Right",)),
            _CANCEL,
        )
    elif (
        title == "Rewind"
        and "Nothing to rewind to yet." in hints
        and hints.endswith("Esc to cancel")
    ):
        ui_type = "ClaudeRewind"
        controls = (_CANCEL,)
    else:
        return None
    return ui_type, TerminalMenu(controls=controls, text_input=text_input)


def parse_claude_command_menu(pane_text: str) -> StatusUpdate | None:
    """Require the last exact menu title and its native control hints."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", pane_text)
    lines = [line.rstrip() for line in clean.splitlines()]
    titles = [i for i, line in enumerate(lines) if _TITLE.fullmatch(line.strip())]
    if not titles:
        return None
    body_lines = lines[titles[-1] :]
    # These captured modals have no interior composer border. A later composer
    # retires old menu output, even when capture includes scrollback.
    if any(_COMPOSER_BORDER.fullmatch(line.strip()) for line in body_lines):
        return None
    body = "\n".join(body_lines).strip()
    title = body_lines[0].strip()
    hints = " ".join(body.split())
    if title == "Skills" and ("⌕" not in body or not hints.endswith(_SKILLS_TRAILERS)):
        return None
    if title.startswith("Settings  "):
        if "⌕" not in body and "Total cost:" not in hints:
            return None
        menu = _settings_menu(hints)
        if menu is None:
            return None
        result = ("ClaudeSettings", menu)
    else:
        result = _picker_menu(title, hints, body)
    if result is None:
        return None
    ui_type, menu = result
    return StatusUpdate(
        raw_text=body,
        display_label=ui_type,
        is_interactive=True,
        ui_type=ui_type,
        menu=menu,
    )
