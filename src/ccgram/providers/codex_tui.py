"""Codex 0.164 command menus, anchored to active headings and native footers."""

import re
from dataclasses import dataclass

from .base import StatusUpdate, TerminalControl, TerminalMenu

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
_NAVIGATION = (
    TerminalControl("Up", ("Up",)),
    TerminalControl("Down", ("Down",)),
    TerminalControl("Cancel", ("Escape",)),
)
_SELECTED_OPTION_RE = re.compile(r"^\s*›\s+\d+\.\s+\S")
_KEYMAP_SUMMARY_RE = re.compile(r"^\s*\d+ actions, \d+ customized, \d+ unbound\.$")
_KEYMAP_ROW_RE = re.compile(r"^\s*›\s+\S+\s{2,}.+\s{2,}\S")
_REASONING_ACTIONS = {
    "enter select · esc back": (TerminalControl("Select", ("Enter",)),),
    "enter default · s session · esc back": (
        TerminalControl("Set default", ("Enter",)),
        TerminalControl("Use this session", ("s",)),
    ),
    "enter apply · s session · esc back": (
        TerminalControl("Apply selected effort", ("Enter",)),
        TerminalControl("Use this session", ("s",)),
    ),
}
_REASONING_FOOTERS = tuple(_REASONING_ACTIONS)
_MAX_MENU_LINES = 80


@dataclass(frozen=True, slots=True)
class _MenuSpec:
    title: str
    ui_type: str
    footers: tuple[str, ...] = ("enter select · esc back",)
    controls: tuple[TerminalControl, ...] = _NAVIGATION
    text_input: bool = False


_SPECS = (
    _MenuSpec(
        "Select Model and Effort",
        "CodexModel",
        controls=(*_NAVIGATION, TerminalControl("Select", ("Enter",))),
    ),
    _MenuSpec(
        "Select Reasoning Level for ",
        "CodexReasoning",
        _REASONING_FOOTERS,
    ),
    _MenuSpec("Advanced Reasoning", "CodexAdvancedReasoning", _REASONING_FOOTERS[1:]),
    _MenuSpec(
        "Update Model Permissions",
        "CodexPermissions",
        controls=(*_NAVIGATION, TerminalControl("Apply selected policy", ("Enter",))),
    ),
    _MenuSpec("Skills", "CodexSkills"),
    _MenuSpec("Select a review preset", "CodexReview"),
    _MenuSpec(
        "Configure Status Line",
        "CodexStatusline",
        ("space toggle · ←/→ reorder · enter save · esc cancel",),
        (
            *_NAVIGATION,
            TerminalControl("Toggle", ("Space",)),
            TerminalControl("Move left", ("Left",)),
            TerminalControl("Move right", ("Right",)),
            TerminalControl("Save", ("Enter",)),
        ),
        text_input=True,
    ),
    _MenuSpec(
        "Configure Terminal Title",
        "CodexTitle",
        ("space toggle · ←/→ reorder · enter save · esc cancel",),
        (*_NAVIGATION, TerminalControl("Save", ("Enter",))),
        text_input=True,
    ),
    _MenuSpec(
        "Select Syntax Theme",
        "CodexTheme",
        controls=_NAVIGATION,
        text_input=True,
    ),
    _MenuSpec(
        "Configuration",
        "CodexConfig",
        (
            "left/right group · / search · enter open · esc clear search/close",
            "enter open · esc clear search",
            "left/right group · / search · esc clear search/close",
        ),
        (*_NAVIGATION, TerminalControl("Search", ("/",))),
    ),
    _MenuSpec(
        "Keymap",
        "CodexKeymap",
        ("left/right group · enter edit shortcut · * custom · - unbound · esc close",),
        (
            *_NAVIGATION,
            TerminalControl("Previous group", ("Left",)),
            TerminalControl("Next group", ("Right",)),
        ),
        text_input=True,
    ),
)
_MENU_FOOTERS = frozenset(footer for spec in _SPECS for footer in spec.footers)
_NORMALIZED_MENU_FOOTERS = frozenset(footer.lower() for footer in _MENU_FOOTERS)


def parse_codex_menu(pane_text: str) -> StatusUpdate | None:
    """Ignore old menus above a composer and preserve empty filtered menus."""
    normalized = _ANSI_OSC_RE.sub("", _ANSI_CSI_RE.sub("", pane_text))
    lines = [line.rstrip() for line in normalized.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return None

    active_footer = _active_footer(lines)
    if active_footer is None:
        return None
    footer, footer_start = active_footer
    candidates: list[tuple[int, _MenuSpec, str]] = []
    for spec in _SPECS:
        if footer not in spec.footers:
            continue
        title_index = next(
            (
                index
                for index in range(
                    len(lines) - 2, max(-1, len(lines) - _MAX_MENU_LINES), -1
                )
                if _matches_title(lines[index].strip(), spec)
            ),
            None,
        )
        if title_index is not None:
            candidates.append((title_index, spec, footer))
    if not candidates:
        return None

    title_index, spec, footer = max(candidates, key=lambda candidate: candidate[0])
    content = lines[title_index:]
    if _has_prior_menu_footer(lines[title_index:footer_start]):
        return None
    if not _has_menu_body(content, spec):
        return None
    return StatusUpdate(
        raw_text="\n".join(content).strip(),
        display_label=spec.ui_type,
        is_interactive=True,
        ui_type=spec.ui_type,
        menu=_terminal_menu(spec, content, footer),
    )


def _normalize_footer(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _active_footer(lines: list[str]) -> tuple[str, int] | None:
    for start in range(max(0, len(lines) - 4), len(lines)):
        candidate = _normalize_footer(" ".join(lines[start:]))
        for footer in _MENU_FOOTERS:
            if candidate == _normalize_footer(footer):
                return footer, start
    return None


def _has_prior_menu_footer(lines: list[str]) -> bool:
    for start in range(len(lines)):
        for end in range(start + 1, min(len(lines), start + 4) + 1):
            if (
                _normalize_footer(" ".join(lines[start:end]))
                in _NORMALIZED_MENU_FOOTERS
            ):
                return True
    return False


def _terminal_menu(spec: _MenuSpec, lines: list[str], footer: str) -> TerminalMenu:
    controls = spec.controls
    text_input = spec.text_input
    if spec.ui_type == "CodexConfig":
        text_input = any(line.strip().startswith("Search ›") for line in lines)
        if text_input:
            controls = _NAVIGATION
    if spec.ui_type in {"CodexReasoning", "CodexAdvancedReasoning"}:
        controls = (*controls, *_REASONING_ACTIONS[footer])
    if spec.ui_type == "CodexTheme" and any(
        line.lstrip().startswith("› ") for line in lines
    ):
        controls = (*controls, TerminalControl("Select", ("Enter",)))
    if spec.ui_type == "CodexReasoning" and any(
        line.strip().removeprefix("› ").startswith("5. More reasoning…")
        for line in lines
    ):
        controls = (*controls, TerminalControl("More reasoning", ("5",)))
    return TerminalMenu(controls, text_input=text_input)


def _matches_title(line: str, spec: _MenuSpec) -> bool:
    if spec.ui_type == "CodexReasoning":
        return line.startswith(spec.title) and bool(line.removeprefix(spec.title))
    if spec.ui_type == "CodexTheme":
        return line.split("  ", 1)[0] == spec.title
    return line == spec.title


def _has_menu_body(lines: list[str], spec: _MenuSpec) -> bool:
    text = "\n".join(lines)
    if spec.ui_type == "CodexStatusline":
        return "Select which items to display in the status line." in text and (
            "[x]" in text or "[ ]" in text or "no matches" in text
        )
    if spec.ui_type == "CodexTitle":
        return "Select which items to display in the terminal title." in text
    if spec.ui_type == "CodexTheme":
        return (
            "Custom .tmTheme files can be added to the ~/.codex/themes directory."
            in text
        )
    if spec.ui_type == "CodexConfig":
        return "Changes save automatically" in text
    if spec.ui_type == "CodexKeymap":
        return (
            "All configurable shortcuts." in text
            and any(_KEYMAP_SUMMARY_RE.match(line) for line in lines)
            and any(
                _KEYMAP_ROW_RE.match(line) or line.strip() == "no matches"
                for line in lines
            )
        )
    return any(_SELECTED_OPTION_RE.match(line) for line in lines)
