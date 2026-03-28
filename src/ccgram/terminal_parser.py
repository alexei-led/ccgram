"""Terminal output parser — detects Claude Code UI elements in pane text.

Parses captured tmux pane content to detect:
  - Interactive UIs (AskUserQuestion, ExitPlanMode, Permission Prompt,
    RestoreCheckpoint) via regex-based UIPattern matching with top/bottom
    delimiters.
  - Status line (spinner characters + working text) by scanning from bottom up.

All Claude Code text patterns live here. To support a new UI type or
a changed Claude Code version, edit UI_PATTERNS / STATUS_SPINNERS.

Key functions: extract_interactive_content(), parse_status_line(),
strip_pane_chrome(), extract_bash_output(), detect_remote_control().
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccgram.screen_buffer import ScreenBuffer


@dataclass
class InteractiveUIContent:
    """Content extracted from an interactive UI."""

    content: str  # The extracted display content
    name: str = ""  # Pattern name that matched (e.g. "AskUserQuestion")


@dataclass(frozen=True)
class UIPattern:
    """A text-marker pair that delimits an interactive UI region.

    Extraction scans lines top-down: the first line matching any `top` pattern
    marks the start, the first subsequent line matching any `bottom` pattern
    marks the end.  Both boundary lines are included in the extracted content.

    ``top`` and ``bottom`` are tuples of compiled regexes — any single match
    is sufficient.  This accommodates wording changes across Claude Code
    versions (e.g. a reworded confirmation prompt).

    When ``context_above`` > 0, the extracted block includes up to that many
    non-blank lines above the top marker.  This lets structural patterns
    (e.g. matching ``❯`` as top) still display the question/description that
    precedes the selection area.
    """

    name: str  # Descriptive label (not used programmatically)
    top: tuple[re.Pattern[str], ...]
    bottom: tuple[re.Pattern[str], ...]
    min_gap: int = 2  # minimum lines between top and bottom (inclusive)
    context_above: int = 0  # extra lines above top marker to include in content


# ── UI pattern definitions (order matters — first match wins) ────────────

UI_PATTERNS: list[UIPattern] = [
    UIPattern(
        name="ExitPlanMode",
        top=(
            re.compile(r"^\s*Would you like to proceed\?"),
            # v2.1.29+: longer prefix that may wrap across lines
            re.compile(r"^\s*Claude has written up a plan"),
        ),
        bottom=(
            re.compile(r"^\s*ctrl-g to edit in "),
            re.compile(r"^\s*Esc to (cancel|exit)"),
        ),
    ),
    UIPattern(
        name="AskUserQuestion",
        top=(re.compile(r"^\s*←\s+[☐✔☒]"),),  # Multi-tab: no bottom needed
        bottom=(),
        min_gap=1,
    ),
    UIPattern(
        name="AskUserQuestion",
        top=(re.compile(r"^\s*[☐✔☒]"),),  # Single-tab: bottom required
        bottom=(re.compile(r"^\s*Enter to select"),),
        min_gap=1,
    ),
    UIPattern(
        name="PermissionPrompt",
        top=(
            re.compile(r"^\s*Do you want to proceed\?"),
            re.compile(r"^\s*Do you want to make this edit"),
            re.compile(r"^\s*Network request outside of sandbox"),
            re.compile(r"^\s*This command requires approval"),
            re.compile(r"^\s*Allow .+ to"),
        ),
        bottom=(re.compile(r"^\s*Esc to cancel"),),
    ),
    UIPattern(
        name="RestoreCheckpoint",
        top=(re.compile(r"^\s*Restore the code"),),
        bottom=(re.compile(r"^\s*Enter to continue"),),
    ),
    UIPattern(
        name="Settings",
        top=(re.compile(r"^\s*Settings:"),),
        bottom=(
            re.compile(r"Esc to (cancel|exit)"),
            re.compile(r"^\s*Type to filter"),
        ),
    ),
    UIPattern(
        name="SelectModel",
        top=(re.compile(r"^\s*Select model"),),
        bottom=(
            re.compile(r"Enter to confirm"),
            re.compile(r"^\s*Esc to exit"),
        ),
    ),
    # ── Structural catch-all (MUST be last — catches anything above) ─
    # Ink's SelectInput renders ❯ (U+276F) as the selection cursor for
    # the highlighted option.  Combined with a bottom action hint OR a
    # non-selected list item, this catches ANY selection UI.
    # context_above=10 pulls in the question/description text above the
    # cursor.  min_gap=1 for compact prompts.
    UIPattern(
        name="SelectionUI",
        top=(re.compile(r"^\s*[❯›]\s"),),
        bottom=(
            re.compile(r"^\s*Esc to (cancel|exit)"),
            re.compile(r"^\s*Enter to (select|confirm|continue)"),
            re.compile(r"^\s*ctrl-g to edit"),
            re.compile(r"(?i)^\s*Press enter to (confirm|select|continue|submit)"),
            re.compile(r"(?i)^\s*enter to (submit|confirm|select)"),
            # Non-selected list items (e.g. /remote-control has no footer)
            re.compile(r"^\s+\d+\.\s"),
        ),
        min_gap=1,
        context_above=10,
    ),
]

# Catch-all must be last — it would shadow more specific patterns above.
# (Not an assert — must survive python -O.)
if UI_PATTERNS[-1].name != "SelectionUI":
    raise RuntimeError("catch-all SelectionUI pattern must be last in UI_PATTERNS")


# ── Post-processing ──────────────────────────────────────────────────────

_RE_LONG_DASH = re.compile(r"^─{5,}$")

# Minimum number of "─" characters to recognize a line as a separator
_MIN_SEPARATOR_WIDTH = 20

# Maximum length of a chrome line (prompt, status bar) between separators.
# Lines longer than this are considered actual output content.
_MAX_CHROME_LINE_LENGTH = 80


def _shorten_separators(text: str) -> str:
    """Replace lines of 5+ ─ characters with exactly ─────."""
    return "\n".join(
        "─────" if _RE_LONG_DASH.match(line) else line for line in text.split("\n")
    )


# ── Core extraction ──────────────────────────────────────────────────────


def _context_start(lines: list[str], top_idx: int, context_above: int) -> int:
    """Find the first non-blank line within *context_above* lines above *top_idx*."""
    if context_above <= 0:
        return top_idx
    for k in range(max(0, top_idx - context_above), top_idx):
        if lines[k].strip():
            return k
    return top_idx


def _try_extract(lines: list[str], pattern: UIPattern) -> InteractiveUIContent | None:
    """Try to extract content matching a single UI pattern.

    When ``pattern.bottom`` is empty, the region extends from the top marker
    to the last non-empty line (used for multi-tab AskUserQuestion where the
    bottom delimiter varies by tab).
    """
    top_idx: int | None = None
    bottom_idx: int | None = None

    for i, line in enumerate(lines):
        if top_idx is None:
            if any(p.search(line) for p in pattern.top):
                top_idx = i
        elif pattern.bottom and any(p.search(line) for p in pattern.bottom):
            bottom_idx = i
            break

    if top_idx is None:
        return None

    # No bottom patterns → use last non-empty line as boundary
    if not pattern.bottom:
        for i in range(len(lines) - 1, top_idx, -1):
            if lines[i].strip():
                bottom_idx = i
                break

    if bottom_idx is None or bottom_idx - top_idx < pattern.min_gap:
        return None

    display_start = _context_start(lines, top_idx, pattern.context_above)
    content = "\n".join(lines[display_start : bottom_idx + 1]).rstrip()
    return InteractiveUIContent(content=_shorten_separators(content), name=pattern.name)


# ── Bottom-up fallback ───────────────────────────────────────────────────

# Action hints that reliably mark the bottom of any Claude Code interactive UI.
_ACTION_HINT_RE = re.compile(
    r"(?i)^\s*("
    r"Esc to (cancel|exit)"
    r"|Enter to (select|confirm|continue|submit)"
    r"|ctrl-g to edit"
    r"|Type to filter"
    r"|Press enter to (confirm|select|continue|submit)"
    r")"
)

# Maximum lines to scan upward from the action hint.
_BOTTOM_UP_MAX_SCAN = 30

# Maximum non-blank lines from terminal bottom to consider the action hint
# part of a currently-active UI (not leftover output from an earlier prompt).
_BOTTOM_UP_MAX_DEPTH = 5

# Consecutive blank lines that signal a section break (UI boundary).
_SECTION_BREAK_BLANKS = 2

# Minimum lines between top and bottom for a valid UI block.
_BOTTOM_UP_MIN_GAP = 2


_CHECKBOX_CHARS_RE = re.compile(r"[☐✔☒]")
_CURSOR_CHARS_RE = re.compile(r"[❯›]\s")


def _infer_ui_name(lines: list[str]) -> str:
    """Infer interactive UI type from content characteristics."""
    for line in lines:
        if _CHECKBOX_CHARS_RE.search(line):
            return "AskUserQuestion"
        if _CURSOR_CHARS_RE.search(line):
            return "SelectionUI"
    return "InteractiveUI"


def _try_extract_bottom_up(lines: list[str]) -> InteractiveUIContent | None:
    """Fallback: detect interactive UI by action hints near terminal bottom.

    Anchors on action-hint lines ("Esc to cancel", "Enter to confirm", etc.)
    in the last few non-blank lines, then scans upward to find the UI boundary.
    Resilient to title/wording changes — only depends on stable action hints.
    """
    # Find action hint near the bottom of the terminal
    bottom_idx: int | None = None
    non_blank_seen = 0
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            non_blank_seen += 1
            if non_blank_seen > _BOTTOM_UP_MAX_DEPTH:
                break
            if _ACTION_HINT_RE.search(lines[i]):
                bottom_idx = i
                break

    if bottom_idx is None:
        return None

    # Scan upward from the action hint to find the top boundary.
    # Stop at: two consecutive blank lines (section break), or max scan limit.
    top_idx = bottom_idx
    consecutive_blank = 0
    scan_floor = max(0, bottom_idx - _BOTTOM_UP_MAX_SCAN)
    for i in range(bottom_idx - 1, scan_floor - 1, -1):
        if not lines[i].strip():
            consecutive_blank += 1
            if consecutive_blank >= _SECTION_BREAK_BLANKS:
                # Two blank lines = section break; UI starts after them.
                top_idx = i + consecutive_blank
                break
        else:
            consecutive_blank = 0
            top_idx = i

    if bottom_idx - top_idx < _BOTTOM_UP_MIN_GAP:
        return None

    content = "\n".join(lines[top_idx : bottom_idx + 1]).rstrip()
    name = _infer_ui_name(lines[top_idx : bottom_idx + 1])
    return InteractiveUIContent(content=_shorten_separators(content), name=name)


# ── Public API ───────────────────────────────────────────────────────────


def extract_interactive_content(
    pane_text: str | list[str],
    patterns: list[UIPattern] | None = None,
) -> InteractiveUIContent | None:
    """Extract content from an interactive UI in terminal output.

    Tries each UI pattern in declaration order; first match wins.
    Falls back to bottom-up detection (action hints near terminal bottom)
    when no pattern matches — resilient to title/wording changes.

    ``pane_text`` can be a raw string (split on newlines) or a pre-split
    list of lines (e.g. from ScreenBuffer.display).

    ``patterns`` defaults to ``UI_PATTERNS`` (Claude Code).  Providers with
    different terminal UIs pass their own pattern list.
    """
    if not pane_text:
        return None

    lines = pane_text if isinstance(pane_text, list) else pane_text.strip().split("\n")
    for pattern in patterns or UI_PATTERNS:
        result = _try_extract(lines, pattern)
        if result:
            return result

    # Bottom-up fallback: detect by action hints near terminal bottom
    return _try_extract_bottom_up(lines)


def parse_from_screen(screen: ScreenBuffer) -> InteractiveUIContent | None:
    """Detect interactive UI content using pyte-rendered screen lines.

    Uses the ScreenBuffer's rendered lines (ANSI-stripped by pyte) and
    cursor position. Falls back to the same regex patterns used by
    extract_interactive_content().
    """
    lines = screen.display
    cursor_row = screen.cursor_row

    # Trim trailing empty lines, but don't go past the cursor row
    end = max(cursor_row + 1, 1)
    for i in range(len(lines) - 1, cursor_row, -1):
        if lines[i].strip():
            end = i + 1
            break

    active_lines = lines[:end]
    if not active_lines:
        return None

    return extract_interactive_content(active_lines)


def parse_status_from_screen(screen: ScreenBuffer) -> str | None:
    """Extract status line using pyte-rendered screen lines and cursor position.

    Uses the ScreenBuffer's clean rendered output for more robust status
    detection — ANSI escapes are already stripped by pyte, so the spinner
    character check is more reliable.
    """
    lines = screen.display

    # Trim trailing empty lines
    last_nonempty = len(lines) - 1
    while last_nonempty >= 0 and not lines[last_nonempty].strip():
        last_nonempty -= 1
    if last_nonempty < 0:
        return None

    active_lines = lines[: last_nonempty + 1]

    # Reuse the existing parse_status_line logic on the joined text
    return parse_status_line("\n".join(active_lines), pane_rows=screen.rows)


# ── Status line parsing ─────────────────────────────────────────────────

# Spinner characters Claude Code uses in its status line (fast-path lookup)
STATUS_SPINNERS = frozenset(["·", "✻", "✽", "✶", "✳", "✢"])

# Box-drawing range U+2500–U+257F and other known non-spinner symbols
_BRAILLE_START = 0x2800
_BRAILLE_END = 0x28FF
_NON_SPINNER_RANGES = ((0x2500, 0x257F),)  # box-drawing characters
_NON_SPINNER_CHARS = frozenset("─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬>|+<=~")

# Unicode categories that spinner characters typically belong to.
# So = Symbol Other (✻, ✽, ✶, ✳, ✢, ☐, ✔, ☒)
# Sm = Symbol Math (∘, ⊛)
# Note: Po (Punctuation Other) is excluded — it includes common ASCII chars
# like !, #, %, @, *, / that would cause false positives.
_SPINNER_CATEGORIES = frozenset({"So", "Sm"})


def is_likely_spinner(char: str) -> bool:
    """Check if a character is likely a spinner symbol.

    Uses a two-tier approach:
    1. Fast-path: check the known STATUS_SPINNERS frozenset
    2. Fallback: use Unicode category matching (So, Sm, Braille)
       while excluding box-drawing and other non-spinner characters
    """
    if not char:
        return False
    if char in STATUS_SPINNERS:
        return True
    if char in _NON_SPINNER_CHARS:
        return False
    cp = ord(char)
    for start, end in _NON_SPINNER_RANGES:
        if start <= cp <= end:
            return False
    # Braille Patterns block U+2800–U+28FF
    if _BRAILLE_START <= cp <= _BRAILLE_END:
        return True
    category = unicodedata.category(char)
    return category in _SPINNER_CATEGORIES


def parse_status_line(pane_text: str, *, pane_rows: int | None = None) -> str | None:
    """Extract the Claude Code status line from terminal output.

    The status line sits above a chrome separator (a line of ``─`` characters).
    Scans from the bottom up for separators, then checks the lines immediately
    above each separator for a spinner character.

    When ``pane_rows`` is provided, the separator scan is limited to the
    bottom 40% of the screen as an optimization.

    Returns the text after the spinner, or None if no status line found.
    """
    if not pane_text:
        return None

    lines = pane_text.strip().split("\n")

    # Determine scan range: either bottom 40% of screen or all lines
    if pane_rows is not None:
        scan_limit = max(int(pane_rows * 0.4), 16)
        scan_start = max(len(lines) - scan_limit, 0)
    else:
        scan_start = 0

    # Scan separators from bottom up within the scan range.
    # Claude Code 4.6 renders two separators around the prompt line;
    # the spinner sits above the upper one, possibly with a blank line between.
    for i in range(len(lines) - 1, scan_start - 1, -1):
        if not _is_separator(lines[i]):
            continue
        # Check up to 2 lines above the separator (skip blanks).
        for offset in (1, 2):
            j = i - offset
            if j < scan_start:
                break
            candidate = lines[j].strip()
            if not candidate:
                continue  # skip blank line
            if is_likely_spinner(candidate[0]):
                return candidate[1:].strip()
            break  # non-blank, non-spinner → stop looking above this separator

    return None


# ── Status display formatting ──────────────────────────────────────────

# Keyword → short label mapping for status display in Telegram.
# First match wins; checked against the first word, then full string as fallback.
_STATUS_KEYWORDS: list[tuple[str, str]] = [
    ("think", "\U0001f9e0 thinking\u2026"),
    ("reason", "\U0001f9e0 thinking\u2026"),
    ("test", "\U0001f9ea testing\u2026"),
    ("read", "\U0001f4d6 reading\u2026"),
    ("edit", "\u270f\ufe0f editing\u2026"),
    ("writ", "\U0001f4dd writing\u2026"),
    ("search", "\U0001f50d searching\u2026"),
    ("grep", "\U0001f50d searching\u2026"),
    ("glob", "\U0001f4c2 searching\u2026"),
    ("install", "\U0001f4e6 installing\u2026"),
    ("runn", "\u26a1 running\u2026"),
    ("bash", "\u26a1 running\u2026"),
    ("execut", "\u26a1 running\u2026"),
    ("compil", "\U0001f3d7\ufe0f building\u2026"),
    ("build", "\U0001f3d7\ufe0f building\u2026"),
    ("lint", "\U0001f9f9 linting\u2026"),
    ("format", "\U0001f9f9 formatting\u2026"),
    ("deploy", "\U0001f680 deploying\u2026"),
    ("fetch", "\U0001f310 fetching\u2026"),
    ("download", "\u2b07\ufe0f downloading\u2026"),
    ("upload", "\u2b06\ufe0f uploading\u2026"),
    ("commit", "\U0001f4be committing\u2026"),
    ("push", "\u2b06\ufe0f pushing\u2026"),
    ("pull", "\u2b07\ufe0f pulling\u2026"),
    ("clone", "\U0001f4cb cloning\u2026"),
    ("debug", "\U0001f41b debugging\u2026"),
    ("delet", "\U0001f5d1\ufe0f deleting\u2026"),
    ("creat", "\u2728 creating\u2026"),
    ("check", "\u2705 checking\u2026"),
    ("updat", "\U0001f504 updating\u2026"),
    ("analyz", "\U0001f52c analyzing\u2026"),
    ("analys", "\U0001f52c analyzing\u2026"),
    ("pars", "\U0001f50d parsing\u2026"),
    ("verif", "\u2705 verifying\u2026"),
]

_DEFAULT_STATUS = "\u2699\ufe0f working\u2026"


def format_status_display(raw_status: str) -> str:
    """Convert raw Claude Code status text to a short display label.

    Matches the first word first (so "Writing tests" → "writing…", not "testing…"),
    then falls back to scanning the full string. Returns "working…" if nothing matches.
    """
    lower = raw_status.lower()
    first_word = lower.split(maxsplit=1)[0] if lower else ""
    for keyword, label in _STATUS_KEYWORDS:
        if keyword in first_word:
            return label
    for keyword, label in _STATUS_KEYWORDS:
        if keyword in lower:
            return label
    return _DEFAULT_STATUS


# ── Remote Control detection ──────────────────────────────────────────

_RC_MARKER = "Remote Control active"


def detect_remote_control(lines: list[str]) -> bool:
    """Detect 'Remote Control active' in the status bar below chrome separators."""
    boundary = find_chrome_boundary(lines)
    if boundary is None:
        return False
    return any(_RC_MARKER in line for line in lines[boundary:])


# ── Pane chrome stripping & bash output extraction ─────────────────────


def _is_separator(line: str) -> bool:
    """Check if a line is a chrome separator (all ─ chars, wide enough)."""
    stripped = line.strip()
    return len(stripped) >= _MIN_SEPARATOR_WIDTH and all(c == "─" for c in stripped)


def find_chrome_boundary(lines: list[str]) -> int | None:
    """Find the topmost separator row of Claude Code's bottom chrome.

    Scans from the bottom upward (limited to last 20 lines), looking for the
    first separator that has only chrome content below it (more separators,
    prompt chars, status bar).
    Returns the line index of that separator, or None if no chrome found.
    """
    if not lines:
        return None

    # Limit separator scan to the bottom 20 lines to avoid false matches
    # from content separators (e.g. markdown tables) higher up.
    scan_start = max(len(lines) - 20, 0)

    # Find all separator indices, scanning from bottom up
    separator_indices: list[int] = []
    for i in range(len(lines) - 1, scan_start - 1, -1):
        if _is_separator(lines[i]):
            separator_indices.append(i)

    if not separator_indices:
        return None

    # The topmost separator is the chrome boundary.
    # Walk the separators (already sorted bottom-up) and find the one
    # where everything between consecutive separators is chrome (prompt, status).
    # The topmost separator in a contiguous chrome block is our boundary.
    boundary = separator_indices[0]  # start with the bottommost

    for idx in separator_indices[1:]:
        # Check if the lines between this separator and the current boundary
        # are all chrome-like (empty, prompt, status bar, or short non-content).
        gap_is_chrome = True
        for j in range(idx + 1, boundary):
            line = lines[j].strip()
            if not line:
                continue
            # Chrome lines: prompt (❯), status bar info, short UI elements
            # Non-chrome: actual output content (longer meaningful text)
            # Heuristic: lines in chrome are typically short UI elements
            if len(line) > _MAX_CHROME_LINE_LENGTH:
                gap_is_chrome = False
                break
        if gap_is_chrome:
            boundary = idx
        else:
            break

    return boundary


def strip_pane_chrome(lines: list[str]) -> list[str]:
    """Strip Claude Code's bottom chrome (prompt area + status bar).

    The bottom of the pane looks like::

        ────────────────────────  (separator)
        ❯                        (prompt)
        ────────────────────────  (separator)
          [Opus 4.6] Context: 34%
          ⏵⏵ bypass permissions…

    Finds the topmost separator in the bottom chrome block and strips
    everything from there down.
    """
    boundary = find_chrome_boundary(lines)
    if boundary is not None:
        return lines[:boundary]
    return lines


def extract_bash_output(pane_text: str, command: str) -> str | None:
    """Extract ``!`` command output from a captured tmux pane.

    Searches from the bottom for the ``! <command>`` echo line, then
    returns that line and everything below it (including the ``⎿`` output).
    Returns *None* if the command echo wasn't found.
    """
    lines = strip_pane_chrome(pane_text.splitlines())

    # Find the last "! <command>" echo line (search from bottom).
    # Match on the first 10 chars of the command in case the line is truncated.
    cmd_idx: int | None = None
    match_prefix = command[:10]
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith(f"! {match_prefix}") or stripped.startswith(
            f"!{match_prefix}"
        ):
            cmd_idx = i
            break

    if cmd_idx is None:
        return None

    # Include the command echo line and everything after it
    raw_output = lines[cmd_idx:]

    # Strip trailing empty lines
    while raw_output and not raw_output[-1].strip():
        raw_output.pop()

    if not raw_output:
        return None

    return "\n".join(raw_output).strip()
