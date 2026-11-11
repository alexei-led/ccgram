"""Claude Code's wrapper around large pastes in transcript user messages.

A long message pasted into Claude Code (which is how Telegram text reaches a
Herdr terminal) is recorded in the transcript as

    <pasted_content id="cccd">
    the text
    </pasted_content id="cccd">

Comparisons with the text CCGram sent must look through that wrapper, or a
Telegram message is mistaken for terminal input (a 👤 echo, a desk switch).
"""

from __future__ import annotations

import re

_PASTE_TAG = re.compile(r'\n?</?pasted_content(?: id="[^"]*")?>\n?')


def unwrap_pasted_content(text: str) -> str:
    """``text`` without Claude Code's paste wrappers."""
    if "pasted_content" not in text:
        return text
    return _PASTE_TAG.sub("", text)
