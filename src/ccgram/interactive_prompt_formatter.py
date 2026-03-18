"""Formatting helpers for interactive prompt text shown in Telegram.

Currently used for Codex interactive prompts to keep approval controls readable
when edit diffs are rendered in dense side-by-side terminal output.
"""

import re

_ACTION_HINT_RE = re.compile(
    r"(?i)^\s*(?:press\s+)?enter\s+to\s+(?:confirm|select|continue|submit)\b"
)
_ESC_HINT_RE = re.compile(r"(?i)^\s*esc\s+to\s+(?:cancel|exit)\b")
_OPTION_CURSOR_RE = re.compile(r"^\s*[❯›●○◉]\s+")
_OPTION_NUMBER_RE = re.compile(r"^\s*\d+\.\s+")
_INLINE_OPTION_RE = re.compile(r"\d+\.\s+.+?(?=(?:\s{2,}\d+\.\s)|$)")
_EDIT_PROMPT_RE = re.compile(r"(?i)do you want to make this edit")
_EDIT_FILE_RE = re.compile(r"(?i)make this edit(?: to)?\s+(.+?)\?\s*$")
_SIDE_BY_SIDE_MINUS_RE = re.compile(r"(?:^|\s)-\s+(.+?)(?=(?:\s+\d+\s+[+-]\s+)|$)")
_SIDE_BY_SIDE_PLUS_RE = re.compile(r"(?:^|\s)\+\s+(.+?)(?=(?:\s+\d+\s+[+-]\s+)|$)")
_SIDE_BY_SIDE_OLD_RE = re.compile(r"\b\d+\s+\d+\s+-\s+")
_SIDE_BY_SIDE_NEW_RE = re.compile(r"\b\d+\s+\+\s+")
_LONG_DASH_RE = re.compile(r"^[-─]{5,}$")

_MIN_INLINE_OPTIONS = 2
_MAX_PREVIEW_LINES = 4
_MAX_PREVIEW_CHARS = 120


def format_codex_interactive_prompt(raw_text: str, ui_type: str | None = None) -> str:
    """Format Codex interactive prompt text for Telegram readability."""
    _ = ui_type
    if not raw_text:
        return raw_text

    lines = [line.rstrip() for line in raw_text.splitlines()]
    if not lines:
        return raw_text

    normalized = _normalize_inline_numbered_options(lines)
    if not _is_edit_prompt(normalized):
        return "\n".join(normalized).strip()

    return _format_edit_prompt(normalized)


def _normalize_inline_numbered_options(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        split = _split_inline_numbered_options(line)
        if split:
            out.extend(split)
        else:
            out.append(line)
    return out


def _split_inline_numbered_options(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    matches = [m.group(0).strip() for m in _INLINE_OPTION_RE.finditer(stripped)]
    if len(matches) < _MIN_INLINE_OPTIONS:
        return None

    marker = ""
    if stripped and stripped[0] in {"❯", "›", "●", "○", "◉"}:
        marker = stripped[0]

    result: list[str] = []
    first_prefix = f"{marker} " if marker else ""
    for i, option in enumerate(matches):
        prefix = first_prefix if i == 0 else "  "
        result.append(f"{prefix}{option}")
    return result


def _is_edit_prompt(lines: list[str]) -> bool:
    return any(_EDIT_PROMPT_RE.search(line) for line in lines)


def _format_edit_prompt(lines: list[str]) -> str:
    control_start = _find_controls_start(lines)
    if control_start is None:
        return "\n".join(lines).strip()

    pre = lines[:control_start]
    controls = lines[control_start:]
    question = _extract_question(pre)
    file_path = _extract_file_path(question)
    added, removed = _count_changes(pre)
    previews = _extract_previews(pre)
    had_diff_blob = any(_looks_like_diff_blob(line) for line in pre)

    out: list[str] = []
    if question:
        out.append(question)
    if file_path:
        out.append(f"File: {file_path}")
    if added or removed:
        out.append(f"Changes: +{added} -{removed}")
    elif had_diff_blob:
        out.append("Changes: diff detected")

    if previews:
        out.append("Preview:")
        out.extend(f"  {line}" for line in previews)
    elif had_diff_blob:
        out.append("Diff preview omitted (wrapped output).")

    if out and controls:
        out.append("")
    out.extend(_trim_blank_edges(controls))
    return "\n".join(_squash_blank_runs(out)).strip()


def _find_controls_start(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if _is_option_line(line) or _is_action_line(line):
            return i
    return None


def _is_option_line(line: str) -> bool:
    return bool(_OPTION_CURSOR_RE.match(line) or _OPTION_NUMBER_RE.match(line))


def _is_action_line(line: str) -> bool:
    return bool(_ACTION_HINT_RE.match(line) or _ESC_HINT_RE.match(line))


def _extract_question(pre_lines: list[str]) -> str:
    for line in pre_lines:
        if _EDIT_PROMPT_RE.search(line):
            return line.strip()
    for line in pre_lines:
        stripped = line.strip()
        if stripped and not _LONG_DASH_RE.match(stripped):
            return stripped
    return ""


def _extract_file_path(question: str) -> str:
    if not question:
        return ""
    m = _EDIT_FILE_RE.search(question)
    if not m:
        return ""
    return m.group(1).strip("` ")


def _count_changes(lines: list[str]) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("+") and not stripped.startswith("+++"):
            added += 1
        if stripped.startswith("-") and not stripped.startswith("---"):
            removed += 1
        added += len(_SIDE_BY_SIDE_PLUS_RE.findall(line))
        removed += len(_SIDE_BY_SIDE_MINUS_RE.findall(line))
    return added, removed


_HEAD_TAIL_LINES = 2


def _extract_previews(lines: list[str]) -> list[str]:
    all_previews: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("+") and not stripped.startswith("+++"):
            _push_preview(all_previews, seen, _shorten(stripped))
            continue
        if stripped.startswith("-") and not stripped.startswith("---"):
            _push_preview(all_previews, seen, _shorten(stripped))
            continue

        for match in _SIDE_BY_SIDE_MINUS_RE.findall(line):
            text = _shorten(f"- {match.strip()}")
            _push_preview(all_previews, seen, text)
        for match in _SIDE_BY_SIDE_PLUS_RE.findall(line):
            text = _shorten(f"+ {match.strip()}")
            _push_preview(all_previews, seen, text)

    if len(all_previews) <= _MAX_PREVIEW_LINES:
        return all_previews

    head = all_previews[:_HEAD_TAIL_LINES]
    tail = all_previews[-_HEAD_TAIL_LINES:]
    omitted = len(all_previews) - _HEAD_TAIL_LINES * 2
    return head + [f"... ({omitted} more lines)"] + tail


def _push_preview(previews: list[str], seen: set[str], text: str) -> None:
    if not text:
        return
    normalized = text.strip()
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    previews.append(normalized)


def _shorten(text: str) -> str:
    if len(text) <= _MAX_PREVIEW_CHARS:
        return text
    return text[:_MAX_PREVIEW_CHARS] + "..."


def _looks_like_diff_blob(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith(("+", "-"))
        or _SIDE_BY_SIDE_OLD_RE.search(line)
        or _SIDE_BY_SIDE_NEW_RE.search(line)
    )


def _trim_blank_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _squash_blank_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    return out
