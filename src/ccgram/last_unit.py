"""Last-unit extraction — slice last command+output from shell scrollback.

Provides ``extract_last_shell_block`` which isolates the most recent
command and its output from plain scrollback text using prompt markers.
ANSI escape sequences are stripped before marker matching but the original
colored lines are returned unchanged.
"""

from __future__ import annotations

import re

from .providers.shell_infra import match_prompt

# Strip ANSI escape sequences for marker matching purposes only.
# Covers full CSI range (cursor movement, SGR, private-mode such as
# bracketed paste \x1b[?2004h), OSC strings, and simple two-byte
# designators. tmux ``capture-pane -e`` can interleave these around prompt
# markers, so the marker regex must not trip on residual escape bytes.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][AB012]")


def _strip_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line)


def extract_last_shell_block(scrollback_text: str) -> str | None:
    """Slice last command+output between prompt markers.

    Scans from the bottom of *scrollback_text* for the last bare prompt
    (marker with empty trailing text after stripping ANSI), then scans upward
    for the most recent command echo (marker with non-empty trailing text).
    Returns lines from the command-echo through to the end of the scrollback,
    inclusive. Returns None if either pivot is not found (no markers, command
    still running, or only one marker present).
    """
    lines = scrollback_text.splitlines()
    if not lines:
        return None

    # Scan from bottom for last bare prompt (trailing_text empty after stripping ANSI)
    bare_idx: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        m = match_prompt(_strip_ansi(lines[i]))
        if m is not None and not _strip_ansi(m.trailing_text).strip():
            bare_idx = i
            break

    if bare_idx is None:
        return None

    # Scan upward from bare_idx for command echo (trailing_text non-empty after stripping)
    echo_idx: int | None = None
    for i in range(bare_idx - 1, -1, -1):
        m = match_prompt(_strip_ansi(lines[i]))
        if m is not None and _strip_ansi(m.trailing_text).strip():
            echo_idx = i
            break

    if echo_idx is None:
        return None

    return "\n".join(lines[echo_idx:])
