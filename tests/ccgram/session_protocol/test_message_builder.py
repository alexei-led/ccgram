"""The incremental builder must match one batch build at every chunk boundary."""

import json

import pytest

from ccgram.session_protocol import messages
from ccgram.session_protocol.messages import (
    MessageBuilder,
    SourceChangedError,
    SourceEntry,
    build_messages,
)


def _call(offset, uuid, *calls):
    blocks = [
        {"type": "tool_use", "id": call, "name": "Bash", "input": {"command": call}}
        for call in calls
    ]
    return SourceEntry(
        offset,
        {
            "type": "assistant",
            "uuid": uuid,
            "message": {"role": "assistant", "content": [*blocks]},
        },
    )


def _result(offset, uuid, call, output, *, failed=False):
    return SourceEntry(
        offset,
        {
            "type": "user",
            "uuid": uuid,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call,
                        "content": output,
                        "is_error": failed,
                    }
                ],
            },
        },
    )


def _prompt(offset, uuid, text):
    return SourceEntry(
        offset,
        {"type": "user", "uuid": uuid, "message": {"role": "user", "content": text}},
    )


def _mixed_transcript():
    return [
        _prompt(0, "p1", "list files"),
        _result(10, "r0", "orphan", "arrived before its call"),
        _call(20, "a1", "c1", "c2"),
        _result(30, "r1", "c1", "one"),
        _result(40, "r2", "c2", "two", failed=True),
        _call(50, "a2", "c1"),
        _result(60, "r3", "c1", "again"),
        _prompt(70, "p2", "thanks"),
        _call(80, "a3", "c3"),
    ]


def _summary(messages):
    return [
        (
            m.id,
            m.revision,
            [(p.tool_call_id, p.status, (p.output or {}).get("text")) for p in m.parts],
        )
        for m in messages
    ]


def test_every_chunk_boundary_matches_batch_build():
    entries = _mixed_transcript()
    expected = build_messages(entries, "conv", "claude")
    for cut in range(len(entries) + 1):
        builder = MessageBuilder("conv", "claude")
        builder.feed(entries[:cut])
        builder.feed(entries[cut:])
        assert builder.messages() == expected, f"cut at {cut}"


def test_one_record_at_a_time_matches_batch_build():
    entries = _mixed_transcript()
    builder = MessageBuilder("conv", "claude")
    for entry in entries:
        builder.feed([entry])
    assert builder.messages() == build_messages(entries, "conv", "claude")


def test_result_seen_before_its_call_still_attaches_to_the_call():
    messages = {m.id: m for m in build_messages(_mixed_transcript(), "conv", "claude")}
    assert [(p.tool_call_id, p.status) for p in messages["conv/a1"].parts] == [
        ("c1", "running"),
        ("c2", "failed"),
    ]
    assert messages["conv/a1"].parts[1].output == {"text": "two", "truncated": False}


def test_reclaimed_call_id_moves_its_results_to_the_latest_claim():
    messages = {m.id: m for m in build_messages(_mixed_transcript(), "conv", "claude")}
    assert messages["conv/a1"].parts[0].status == "running"
    latest = messages["conv/a2"].parts[0]
    assert (latest.status, latest.output) == (
        "completed",
        {"text": "again", "truncated": False},
    )


def test_revision_counts_distinct_source_records_for_the_message():
    messages = {m.id: m for m in build_messages(_mixed_transcript(), "conv", "claude")}
    assert messages["conv/a1"].revision == 2  # its record plus the c2 result
    assert messages["conv/a2"].revision == 3  # its record plus two results
    assert messages["conv/a3"].revision == 1


def test_result_never_attaches_to_a_part_with_another_call_id():
    entries = [
        _call(0, "same", "old-call"),
        _result(10, "r", "old-call", "stale output"),
        _call(20, "same", "new-call"),
    ]
    message = build_messages(entries, "conv", "claude")[0]
    assert [(p.tool_call_id, p.status) for p in message.parts] == [
        ("new-call", "running")
    ]
    assert message.revision == 3


def test_dropping_a_completed_tool_part_raises_the_revision():
    builder = MessageBuilder("conv", "claude")
    builder.feed([_call(0, "same", "c"), _result(10, "r", "c", "done")])
    before = builder.message("conv/same")
    text_only = {
        "type": "assistant",
        "uuid": "same",
        "message": {"role": "assistant", "content": "text only"},
    }
    builder.feed([SourceEntry(20, text_only)])
    after = builder.message("conv/same")
    assert after.revision > before.revision
    assert [p.type for p in after.parts] == ["text"]


def test_source_range_selects_messages_by_contributing_offsets():
    entries = _mixed_transcript()
    selected = build_messages(entries, "conv", "claude", source_range=(30, 31))
    assert [m.id for m in selected] == ["conv/a2"]
    assert build_messages(entries, "conv", "claude", source_range=(900, None)) == ()


@pytest.mark.parametrize("generation", [0, 3])
def test_generation_prefixes_message_ids(generation):
    messages = build_messages([_prompt(0, "p1", "hi")], "conv", "claude", generation)
    expected = "conv/p1" if generation == 0 else f"conv/g{generation}/p1"
    assert messages[0].id == expected


def test_position_and_ids_follow_first_observed_order():
    builder = MessageBuilder("conv", "claude")
    builder.feed(_mixed_transcript()[:3])
    builder.feed(_mixed_transcript()[3:])
    assert builder.ids == ["conv/p1", "conv/a1", "conv/a2", "conv/p2", "conv/a3"]
    assert builder.position("conv/p2") == 3
    assert builder.position("conv/missing") is None


def test_dirty_ids_report_changes_until_cleared():
    builder = MessageBuilder("conv", "claude")
    entries = _mixed_transcript()
    builder.feed(entries[:3])
    assert builder.dirty_ids() == ["conv/p1", "conv/a1"]
    assert builder.dirty_ids() == ["conv/p1", "conv/a1"]
    builder.clear_dirty()
    builder.feed(entries[3:4])
    assert builder.dirty_ids() == ["conv/a1"]
    builder.clear_dirty()
    assert builder.dirty_ids() == []


@pytest.fixture
def spill_every_value(monkeypatch):
    monkeypatch.setattr(messages, "SPILL_BYTES", 0)


def _line_reader(entries):
    """Byte-range reader over entries, keyed by each record's offset."""
    lines = {
        entry.offset: (json.dumps(entry.data) + "\n").encode() for entry in entries
    }
    return lines.__getitem__


def test_spilled_values_match_inline_build_at_every_chunk_boundary(spill_every_value):
    entries = _mixed_transcript()
    expected = build_messages(entries, "conv", "claude")
    for cut in range(len(entries) + 1):
        builder = MessageBuilder("conv", "claude", reader=_line_reader(entries))
        builder.feed(entries[:cut])
        builder.feed(entries[cut:])
        assert builder.messages() == expected, f"cut at {cut}"


def test_local_command_output_matches_inline_build(spill_every_value):
    entries = [
        _prompt(0, "c1", "<command-name>/cost</command-name>"),
        _prompt(10, "o1", "<local-command-stdout>$1.20 spent</local-command-stdout>"),
    ]
    builder = MessageBuilder("conv", "claude", reader=_line_reader(entries))
    builder.feed(entries)
    assert builder.messages() == build_messages(entries, "conv", "claude")


def test_materialized_values_are_read_once_while_cached(monkeypatch, spill_every_value):
    entries = [_prompt(0, "p1", "x" * 100)]
    lines = _line_reader(entries)
    reads: list[int] = []

    def reader(offset):
        reads.append(offset)
        return lines(offset)

    builder = MessageBuilder("conv", "claude", reader=reader)
    builder.feed(entries)
    builder.message("conv/p1")
    builder.message("conv/p1")
    assert reads == [0]


def test_cache_evicts_least_recently_used_values_past_its_byte_budget(
    monkeypatch, spill_every_value
):
    monkeypatch.setattr(messages, "CACHE_BYTES", 150)
    entries = [_prompt(0, "p1", "a" * 100), _prompt(10, "p2", "b" * 100)]
    lines = _line_reader(entries)
    reads: list[int] = []

    def reader(offset):
        reads.append(offset)
        return lines(offset)

    builder = MessageBuilder("conv", "claude", reader=reader)
    builder.feed(entries)
    builder.message("conv/p1")
    builder.message("conv/p2")
    builder.message("conv/p1")
    assert reads == [0, 10, 0]


def test_record_that_no_longer_parses_raises_source_changed(spill_every_value):
    entries = [_prompt(0, "p1", "x" * 100)]
    builder = MessageBuilder(
        "conv", "claude", reader=lambda offset: b'{"type": "system"}\n'
    )
    builder.feed(entries)
    with pytest.raises(SourceChangedError):
        builder.message("conv/p1")
