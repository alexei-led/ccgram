"""Markdown → Telegram MarkdownV2 conversion layer.

Wraps `telegramify_markdown` and adds special handling for expandable
blockquotes (delimited by sentinel tokens from providers.base).
Expandable quotes are escaped and formatted as Telegram >…|| syntax
separately, so the library doesn't mangle them.

Key function: convert_markdown(text) → MarkdownV2 string.
"""

import re

from telegramify_markdown import markdownify

from .providers.base import EXPANDABLE_QUOTE_END, EXPANDABLE_QUOTE_START

_EXPQUOTE_RE = re.compile(
    re.escape(EXPANDABLE_QUOTE_START) + r"([\s\S]*?)" + re.escape(EXPANDABLE_QUOTE_END)
)

# Characters that must be escaped in Telegram MarkdownV2 plain text
_MDV2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def _escape_mdv2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MDV2_ESCAPE_RE.sub(r"\\\1", text)


# Max rendered chars for a single expandable quote block.
# Leaves room for surrounding text within Telegram's 4096 char message limit.
_EXPQUOTE_MAX_RENDERED = 3800

# Minimum characters to bother including a partial line during truncation
_MIN_PARTIAL_LINE_LEN = 20


def _render_expandable_quote(m: re.Match[str]) -> str:
    """Render an expandable blockquote block in raw MarkdownV2.

    Truncates the rendered output to _EXPQUOTE_MAX_RENDERED chars
    to ensure the final message fits within Telegram's 4096 limit.
    """
    inner = m.group(1)
    escaped = _escape_mdv2(inner)
    lines = escaped.split("\n")
    # Build quoted lines, truncating if needed to stay within budget
    built: list[str] = []
    total_len = 0
    suffix = "\n>… \\(truncated\\)||"
    budget = _EXPQUOTE_MAX_RENDERED - len(suffix)
    truncated = False
    for line in lines:
        # +1 for ">" prefix, +1 for "\n" separator
        line_cost = 1 + len(line) + 1
        if total_len + line_cost > budget:
            # Try to fit a partial line
            remaining = budget - total_len - 2  # -2 for ">" and "\n"
            if remaining > _MIN_PARTIAL_LINE_LEN:
                built.append(f">{line[:remaining]}")
            truncated = True
            break
        built.append(f">{line}")
        total_len += line_cost
    if truncated:
        return "\n".join(built) + suffix
    return "\n".join(built) + "||"


_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
_INDENTED_CODE_RE = re.compile(r"(?<=\n\n)((?:    .+\n?)+)")
_INDENTED_LINE_RE = re.compile(r"^    ", re.MULTILINE)


def _strip_indented_code_blocks(text: str) -> str:
    """Strip 4-space indentation that CommonMark treats as code blocks.

    Claude Code uses fenced ``` blocks for code; indented blocks in its
    output are typically continuation text, not code.  Pyromark (CommonMark)
    converts 4-space-indented paragraphs into code blocks, so we strip
    the leading spaces before conversion.

    Fenced code blocks are left untouched — only non-fenced segments
    are processed.
    """
    # Split text into alternating (outside-fence, inside-fence) segments
    parts: list[str] = []
    inside_fence = False
    fence_marker = ""
    last_end = 0

    for m in _FENCE_RE.finditer(text):
        marker = m.group(1)
        if not inside_fence:
            # Entering a fenced block — process the preceding non-fenced text
            parts.append(_deindent(text[last_end : m.start()], last_end == 0))
            inside_fence = True
            fence_marker = marker  # e.g. "```" or "~~~~~"
            last_end = m.start()
        elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
            # Closing fence — keep fenced content verbatim
            end = m.end()
            parts.append(text[last_end:end])
            last_end = end
            inside_fence = False
            fence_marker = ""

    # Remaining text after last fence (or entire text if no fences)
    tail = text[last_end:]
    if inside_fence:
        # Unclosed fence — keep verbatim
        parts.append(tail)
    else:
        parts.append(_deindent(tail, last_end == 0))

    return "".join(parts)


def _deindent(text: str, is_start: bool) -> str:
    """Strip 4-space indented code blocks from a non-fenced text segment."""
    if is_start:
        text = re.sub(
            r"^((?:    .+\n?)+)",
            lambda m: _INDENTED_LINE_RE.sub("", m.group(0)),
            text,
        )
    return _INDENTED_CODE_RE.sub(
        lambda m: _INDENTED_LINE_RE.sub("", m.group(0)),
        text,
    )


def _markdownify(text: str) -> str:
    """Convert Markdown to Telegram MarkdownV2 via telegramify-markdown.

    Pre-strips indented code blocks so only fenced ``` blocks are code.
    """
    return markdownify(_strip_indented_code_blocks(text))


def convert_markdown(text: str) -> str:
    """Convert standard Markdown to Telegram MarkdownV2 format.

    Expandable blockquote sections (marked by sentinel tokens from
    TranscriptParser) are extracted, escaped, and formatted separately
    so that telegramify_markdown doesn't mangle the >...|| syntax.
    """
    # Extract expandable quote blocks before telegramify
    segments: list[tuple[bool, str]] = []  # (is_quote, content)
    last_end = 0
    for m in _EXPQUOTE_RE.finditer(text):
        if m.start() > last_end:
            segments.append((False, text[last_end : m.start()]))
        segments.append((True, m.group(0)))
        last_end = m.end()
    if last_end < len(text):
        segments.append((False, text[last_end:]))

    if not segments:
        return _markdownify(text)

    parts: list[str] = []
    for is_quote, segment in segments:
        if is_quote:
            parts.append(_EXPQUOTE_RE.sub(_render_expandable_quote, segment))
        else:
            parts.append(_markdownify(segment))
    return "".join(parts)
