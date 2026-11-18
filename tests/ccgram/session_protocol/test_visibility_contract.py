import json
from dataclasses import replace
from typing import Any

import pytest

from ccgram.handlers.response_builder import build_response_parts
from ccgram.session_protocol.messages import (
    SourceEntry,
    build_messages,
    parse_records,
    read_entries,
)
from ccgram.session_protocol.models import wire
from ccgram.transcript_reader import render_protocol_records

PROVIDERS = ("claude", "pi", "codex", "gemini", "antigravity")
INTERNAL = "INTERNAL_CONTEXT_MUST_NOT_BE_PUBLIC"


def conversation(provider, role, text, key) -> dict[str, Any]:
    message = {"role": role, "content": [{"type": "text", "text": text}]}
    if provider == "codex":
        return {
            "type": "response_item",
            "payload": {"type": "message", "id": key, **message},
        }
    if provider == "gemini":
        return {
            "type": "gemini" if role == "assistant" else "user",
            "id": key,
            "content": text,
        }
    if provider == "antigravity":
        return {
            "type": "PLANNER_RESPONSE" if role == "assistant" else "USER_INPUT",
            "id": key,
            "content": text,
        }
    return {
        "type": "message" if provider == "pi" else role,
        "id": key,
        "uuid": key,
        "message": message,
    }


def diagnostic(provider, role="assistant") -> dict[str, Any]:
    message = {"role": role, "content": [{"type": "text", "text": INTERNAL}]}
    return {
        "type": "api-request-blob",
        "source": "MODEL",
        "role": role,
        "content": INTERNAL,
        "message": message,
        "payload": {"type": "message", **message},
    }


def projected_text(messages):
    return [
        (message.role, part.text)
        for message in messages
        for part in message.parts
        if part.type == "text"
    ]


@pytest.mark.parametrize("provider", (*PROVIDERS, "shell", "unknown-provider"))
@pytest.mark.parametrize("role", ("user", "assistant", "system"))
def test_internal_envelopes_are_never_conversation(provider, role):
    entries = [SourceEntry(0, diagnostic(provider, role))]

    records, state = parse_records(entries, provider)

    assert records == ()
    assert build_messages(entries, "conversation", provider) == ()
    assert render_protocol_records("conversation", provider, records, state) == []


@pytest.mark.parametrize("provider", ("claude", "pi", "codex"))
def test_provider_system_prompt_is_not_a_conversation_message(provider):
    row = conversation(provider, "assistant", INTERNAL, "internal")
    if provider == "codex":
        row["payload"]["role"] = "system"
    else:
        row["message"]["role"] = "system"
        if provider == "claude":
            row["type"] = "system"

    assert build_messages([SourceEntry(0, row)], "conversation", provider) == ()


@pytest.mark.parametrize("flag", ("isMeta", "isSidechain", "isCompactSummary"))
@pytest.mark.parametrize("role", ("user", "assistant"))
def test_claude_internal_conversation_markers_are_not_public(flag, role):
    row = conversation("claude", role, INTERNAL, "internal")
    row[flag] = True

    assert build_messages([SourceEntry(0, row)], "conversation", "claude") == ()


def test_claude_conflicting_envelope_and_message_role_is_not_public():
    row = conversation("claude", "user", INTERNAL, "internal")
    row["type"] = "assistant"

    assert build_messages([SourceEntry(0, row)], "conversation", "claude") == ()


def test_internal_tool_calls_cannot_poison_cross_batch_parser_state():
    row = diagnostic("claude")
    row["message"]["content"] = [
        {"type": "tool_use", "id": "injected", "name": INTERNAL, "input": {}}
    ]

    records, state = parse_records([SourceEntry(0, row)], "claude")

    assert records == ()
    assert state.tool_names == {}
    assert state.pending_tools == {}


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_live_history_and_replay_only_contain_the_actual_dialogue(
    provider, service, backend, tmp_path
):
    path = tmp_path / "conversation.jsonl"
    rows = [
        diagnostic(provider, "system"),
        conversation(provider, "user", "Hi", "question"),
        diagnostic(provider),
        conversation(provider, "assistant", "Hello", "answer"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backend.target = replace(backend.target, provider=provider, transcript=path)
    _, session = await service.for_target(backend.target.id)
    expected = [("user", "Hi"), ("agent", "Hello")]

    history = await service.history(session.id, None, 20)
    assert projected_text(history.messages) == expected
    entries = read_entries(path, provider)
    records, state = parse_records(entries, provider)
    rendered = render_protocol_records("conversation", provider, records, state)
    assert [(event.role, event.text) for event in rendered] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]
    assert [
        part
        for event in rendered
        for part in build_response_parts(
            event.text, event.is_complete, event.content_type, event.role
        )
    ] == ["👤 Hi", "Hello"]
    _, events = service.hub.replay(service.hub.epoch, 0)
    assert INTERNAL not in json.dumps([event.body() for event in events])
    assert INTERNAL not in json.dumps(wire(history))

    with path.open("a", encoding="utf-8") as stream:
        for row in [
            diagnostic(provider, "system"),
            conversation(provider, "user", "Next", "next-question"),
            diagnostic(provider),
            conversation(provider, "assistant", "Done", "next-answer"),
        ]:
            stream.write(json.dumps(row) + "\n")
    later = await service.history(session.id, None, 20)
    assert projected_text(later.messages) == expected + [
        ("user", "Next"),
        ("agent", "Done"),
    ]
    entries = read_entries(path, provider)
    assert projected_text(build_messages(entries, "conversation", provider)) == (
        expected + [("user", "Next"), ("agent", "Done")]
    )


def test_large_internal_prompt_cannot_become_29_telegram_messages():
    rows = [
        conversation("claude", "user", "Hi", "question"),
        diagnostic("claude", "user"),
        diagnostic("claude", "system"),
        conversation("claude", "assistant", "Hello", "answer"),
    ]
    rows[2]["message"]["content"][0]["text"] = (INTERNAL + "\n") * 2400
    records, state = parse_records(
        [SourceEntry(index, row) for index, row in enumerate(rows)], "claude"
    )
    rendered = render_protocol_records("conversation", "claude", records, state)

    assert [
        part
        for event in rendered
        for part in build_response_parts(
            event.text, event.is_complete, event.content_type, event.role
        )
    ] == ["👤 Hi", "Hello"]


@pytest.mark.parametrize(
    "prefix",
    (
        "<environment_context>",
        "<permissions scope='session'>",
        "<user_instructions>",
        "# AGENTS.md instructions for /workspace",
    ),
)
async def test_codex_injected_user_context_never_reaches_any_projection(
    prefix, service, backend, tmp_path
):
    path = tmp_path / "codex.jsonl"
    injected = conversation("codex", "user", prefix + "\n" + INTERNAL, "injected")
    rows = [
        injected,
        conversation("codex", "user", "Hi", "user"),
        conversation("codex", "assistant", "Hello", "assistant"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backend.target = replace(backend.target, provider="codex", transcript=path)
    _, session = await service.for_target(backend.target.id)
    history = await service.history(session.id, None, 20)
    assert projected_text(history.messages) == [("user", "Hi"), ("agent", "Hello")]
    records, state = parse_records(read_entries(path, "codex"), "codex")
    rendered = render_protocol_records("conversation", "codex", records, state)
    assert [(event.role, event.text) for event in rendered] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]
    _, events = service.hub.replay(service.hub.epoch, 0)
    assert INTERNAL not in json.dumps([event.body() for event in events])


@pytest.mark.parametrize(
    "text",
    (
        "Please read AGENTS.md",
        "Explain <environment_context>",
        "# AGENTS.md is a normal filename",
        "<environment_contextual>literal text",
    ),
)
def test_codex_real_user_mentions_are_not_removed(text):
    row = conversation("codex", "user", text, "user")
    assert projected_text(
        build_messages([SourceEntry(0, row)], "conversation", "codex")
    ) == [("user", text)]


def test_codex_tool_output_with_context_markup_remains_visible():
    output = (
        "<environment_context>Explicitly requested file contents</environment_context>"
    )
    rows = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "read",
                "name": "Read",
                "arguments": {"path": "AGENTS.md"},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "read",
                "output": output,
            },
        },
    ]
    messages = build_messages(
        [SourceEntry(i, row) for i, row in enumerate(rows)], "conversation", "codex"
    )
    assert len(messages) == 1
    assert messages[0].parts[0].output == {"text": output, "truncated": False}


def test_codex_injected_context_does_not_split_a_streamed_answer():
    rows = [
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "Hello"}},
        conversation("codex", "user", "<environment_context>" + INTERNAL, "injected"),
        {
            "type": "event_msg",
            "payload": {"type": "task_complete", "last_agent_message": "Hello"},
        },
    ]
    messages = build_messages(
        [SourceEntry(i, row) for i, row in enumerate(rows)], "conversation", "codex"
    )
    assert projected_text(messages) == [("agent", "Hello")]
