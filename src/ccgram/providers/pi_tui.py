"""Recognize Pi's active, bordered selectors without guessing from transcript prose."""

import re

from .base import StatusUpdate, TerminalControl, TerminalMenu

PI_QUESTION_FOOTER = "↑↓ navigate • Enter select • Esc cancel • number quick-select"

_FRAME_BORDERS = 2
_TREE_BORDERS = 3
_EDITOR_BORDERS = 4
_BORDER = re.compile(r"^\s*─{8,}\s*$")
_CURSOR = re.compile(r"^\s*[→›]\s+\S", re.MULTILINE)
_SELECT_HINT = re.compile(
    r"^\s*(?:Enter to select[^\n]*Escape/Ctrl\+C to cancel|"
    r"(?:Type to filter · )?Enter to select · Esc to go back|"
    r"↑↓ navigate  enter select  escape/ctrl\+c cancel)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_INPUT_HINT = re.compile(
    r"^\s*enter submit[^\n]*escape/ctrl\+c cancel[^\n]*$", re.MULTILINE
)


def _active_block(pane_text: str) -> str | None:
    # agterm returns fixed-width rows; tmux normally strips their padding.
    lines = [
        line.rstrip()
        for line in re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", pane_text).splitlines()
    ]
    borders = [i for i, line in enumerate(lines) if _BORDER.fullmatch(line)]
    if not borders:
        return None
    start, end = borders[-2:] if len(borders) >= _FRAME_BORDERS else (-1, borders[-1])
    # The tree has an extra separator between its search header and entries.
    if len(borders) >= _TREE_BORDERS:
        header = "\n".join(lines[borders[-3] + 1 : start])
        if "Session Tree" in header and "Type to search:" in header:
            start = borders[-3]
    body = "\n".join(lines[start + 1 : end]).strip()
    before_start = borders[-3] + 1 if len(borders) >= _TREE_BORDERS else 0
    header = "\n".join(lines[before_start:start]).strip()
    if header.startswith("Fork from Message") and re.search(
        r"^\s*Message \d+ of \d+\s*$", body, re.MULTILINE
    ):
        body = header + "\n" + body
    # The standard extension editor embeds two more borders inside its frame.
    if (
        "newline" in body
        and _INPUT_HINT.search(body)
        and len(borders) >= _EDITOR_BORDERS
    ):
        body = "\n".join(lines[borders[-4] + 1 : end]).strip()
    return body


def parse_pi_ui(pane_text: str) -> StatusUpdate | None:
    """Match current Pi UI contracts; a later idle editor retires old menus.

    Custom extension screens without these standard hints deliberately fall
    back to /screenshot and /toolbar instead of latching interactive mode.
    """
    body = _active_block(pane_text)
    if not body:
        return None
    ui_type = _ui_type(body)
    if ui_type is None:
        return None
    return StatusUpdate(
        raw_text=body,
        display_label=ui_type,
        is_interactive=True,
        ui_type=ui_type,
        menu=_menu(ui_type, body) if ui_type.startswith("Pi") else None,
    )


def _menu(ui_type: str, body: str) -> TerminalMenu:
    controls = [
        TerminalControl("↑5", ("Up",) * 5),
        TerminalControl("↑", ("Up",)),
        TerminalControl("↓", ("Down",)),
        TerminalControl("↓5", ("Down",) * 5),
        TerminalControl("←", ("Left",)),
        TerminalControl("␣ Space", ("Space",)),
        TerminalControl("→", ("Right",)),
        TerminalControl("⇥ Tab", ("Tab",)),
    ]
    text_input = (
        ui_type != "PiSelect"
        or "Type to filter" in body
        or body.startswith("Select provider to logout:")
    )
    if text_input:
        controls.extend(
            [
                TerminalControl("Clear search", ("C-u",)),
                TerminalControl("⌫", ("BSpace",)),
            ]
        )
    if ui_type in {"PiModel", "PiThinking", "PiScopedModels"}:
        controls.append(
            TerminalControl(
                "Save scope" if ui_type == "PiScopedModels" else "Save default",
                ("C-s",),
            )
        )
    if ui_type == "PiScopedModels":
        controls.extend(
            [
                TerminalControl("All matching", ("C-a",)),
                TerminalControl("None matching", ("C-x",)),
                TerminalControl("Provider", ("C-p",)),
                TerminalControl("Move ↑", ("M-Up",)),
                TerminalControl("Move ↓", ("M-Down",)),
            ]
        )
    if ui_type == "PiEditor" and "shift+enter" in body:
        controls.append(TerminalControl("New line", ("S-Enter",)))
    controls.extend(
        [
            TerminalControl("⎋ Esc", ("Escape",)),
            TerminalControl("⏎ Enter", ("Enter",)),
        ]
    )
    return TerminalMenu(tuple(controls), text_input=text_input)


def _ui_type(body: str) -> str | None:
    if PI_QUESTION_FOOTER in body:
        return (
            "AskUserQuestion"
            if re.search(r"^\s*→\s+\d+\.\s", body, re.MULTILINE)
            else None
        )
    if "Type to search · Enter/Space to change · Esc to cancel" in body:
        return "PiSettings"
    if (
        body.startswith("Model Configuration\n")
        and "Enter toggle" in body
        and "Ctrl+S save" in body
    ):
        return "PiScopedModels"
    if "Session Tree" in body and "Type to search:" in body:
        return "PiTree"
    if _INPUT_HINT.search(body):
        return "PiEditor" if "newline" in body else "PiInput"
    if _SELECT_HINT.search(body) or body.startswith(
        ("Select provider to logout:", "Fork from Message\n")
    ):
        return _select_type(body)
    return None


def _select_type(body: str) -> str | None:
    if body.startswith("Thinking Level\n"):
        return "PiThinking"
    if body.startswith("Scope:"):
        return "PiModel"
    searchable = "Type to filter · Enter to select · Esc to go back" in body
    return "PiSelect" if _CURSOR.search(body) or searchable else None
