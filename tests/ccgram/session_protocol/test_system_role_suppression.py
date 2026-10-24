"""System-role transcript entries must not become chat records.

The zai variant writes the harness context reminder
(``<total_tokens>N tokens left</total_tokens>``) as ``api-request-blob``
entries with ``message.role = "system"``. Upstream #330 admitted system
as a chat role at the parse layer, so after v4.17 the reminder flooded
Telegram topics as a delivered message per counter update (2026-10-24:
109 sends in one window, text_len 49 = the exact bare-marker length),
and the same records surfaced in /history and the mobile API. Pre-merge
parsers admitted only user/assistant; this pins that contract back at
the parse layer, through the real production entry shape.
"""

from __future__ import annotations

from ccgram.session_protocol.messages import (
    SourceEntry,
    TranscriptParserState,
    parse_records,
)
from ccgram.transcript_reader import render_protocol_records

_REMINDER = "<total_tokens>14999719 tokens left</total_tokens>"
_RAW_ENTRY = {
    "type": "api-request-blob",
    "uuid": "marker-1",
    "timestamp": "2026-10-24T06:00:00.000Z",
    "message": {"role": "system", "content": _REMINDER},
}


def test_zai_reminder_entry_yields_no_records():
    records, _ = parse_records((SourceEntry(0, _RAW_ENTRY),), "claude")
    assert list(records) == []


def test_user_and_agent_still_parse_and_render():
    user_entry = dict(
        _RAW_ENTRY, uuid="u", message={"role": "user", "content": "hello"}
    )
    agent_entry = dict(
        _RAW_ENTRY, uuid="a", message={"role": "assistant", "content": "hi"}
    )
    records, _ = parse_records(
        (SourceEntry(0, user_entry), SourceEntry(0, agent_entry)), "claude"
    )
    assert [record.role for _, record in records] == ["user", "agent"]
    rendered = render_protocol_records("s", "claude", records, TranscriptParserState())
    assert [m.text for m in rendered] == ["hello", "hi"]
