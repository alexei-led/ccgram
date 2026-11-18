import asyncio
import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from ccgram.session_protocol.backend import Prompt
from ccgram.session_protocol.events import EventHub
from ccgram.session_protocol.messages import SourceEntry, build_messages
from ccgram.session_protocol.models import ProtocolError
from ccgram.session_protocol.requests import PendingHook, RequestStore
from ccgram.session_protocol.service import LocalSessionService
from ccgram.session_protocol.store import ProtocolStore
from ccgram.session_protocol.transcript_source import TranscriptSource


@pytest.mark.parametrize("role", ["developer", "tool", "unknown"])
def test_codex_internal_roles_are_not_public_messages(role):
    entry = SourceEntry(
        0,
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "content": [{"type": "output_text", "text": "internal"}],
            },
        },
    )
    assert build_messages([entry], "conv", "codex") == ()


@pytest.mark.parametrize("key", ["output", "error", "result"])
def test_gemini_nested_function_response_is_preserved(key):
    entry = SourceEntry(
        0,
        {
            "type": "gemini",
            "id": "m",
            "toolCalls": [
                {
                    "id": "call",
                    "name": "shell",
                    "args": {},
                    "status": "success",
                    "result": [
                        {"functionResponse": {"response": {key: "actual output"}}}
                    ],
                }
            ],
        },
    )
    message = build_messages([entry], "conv", "gemini")[0]
    assert message.parts[0].output == {"text": "actual output", "truncated": False}


def question_hook(*, multi=False, labels=("Staging", "Staging + production")):
    return PendingHook(
        "conv",
        "question",
        "AskUserQuestion",
        {
            "questions": [
                {
                    "question": "Where?",
                    "multiSelect": multi,
                    "options": [{"label": label} for label in labels],
                }
            ]
        },
        time.time(),
    )


def test_question_answers_have_unique_exact_menu_keys():
    requests = RequestStore(EventHub())
    requests.observe_hook("s", question_hook())
    opened = requests.reconcile(
        "s",
        "conv",
        1,
        Prompt("Where?\n1. Staging\n2. Staging + production", "question"),
    )
    assert opened is not None
    assert opened.answer_keys == {"Staging": "1", "Staging + production": "2"}


@pytest.mark.parametrize(
    "multi,screen",
    [
        (True, "Where?\n1. Staging\n2. Staging + production"),
        (False, "Where?\nSelect all that apply\n1. Staging\n2. Staging + production"),
    ],
)
def test_multiselect_prompts_have_no_structured_controls(multi, screen):
    requests = RequestStore(EventHub())
    requests.observe_hook("s", question_hook(multi=multi))
    opened = requests.reconcile("s", "conv", 1, Prompt(screen, "question"))
    assert opened is not None
    assert opened.request.source == "screen"
    assert not opened.option_keys and not opened.answer_keys


def test_duplicate_question_labels_have_no_structured_controls():
    requests = RequestStore(EventHub())
    requests.observe_hook("s", question_hook(labels=("Same", "Same")))
    opened = requests.reconcile(
        "s", "conv", 1, Prompt("Where?\n1. Same\n2. Same", "question")
    )
    assert opened is not None and not opened.option_keys and not opened.answer_keys


def test_hook_cannot_be_reused_for_a_changed_prompt_and_fresh_hook_replaces_it():
    requests = RequestStore(EventHub())
    requests.observe_hook(
        "s",
        PendingHook(
            "conv", "permission", "Bash", {"command": "git status"}, time.time()
        ),
    )
    first = requests.reconcile(
        "s", "conv", 1, Prompt("git status\n1. Yes\n2. No", "permission")
    )
    assert first is not None and first.request.source == "hook"
    prompt = Prompt("git status --short\n1. Yes\n2. No", "permission")
    changed = requests.reconcile("s", "conv", 1, prompt)
    assert changed is not None and changed.request.source == "screen"
    requests.observe_hook(
        "s",
        PendingHook(
            "conv", "permission", "Bash", {"command": "git status --short"}, time.time()
        ),
    )
    fresh = requests.reconcile("s", "conv", 1, prompt)
    assert fresh is not None and fresh.request.source == "hook"
    assert fresh.request.tool is not None
    assert fresh.request.tool["input"] == {"command": "git status --short"}
    assert fresh.request.id != changed.request.id


def test_new_hook_replaces_tool_evidence_even_on_unchanged_screen():
    requests = RequestStore(EventHub())
    prompt = Prompt("/tmp/file\n1. Yes\n2. No", "permission")
    for content in ("old", "new"):
        requests.observe_hook(
            "s",
            PendingHook(
                "conv",
                "permission",
                "Edit",
                {"file_path": "/tmp/file", "content": content},
                time.time(),
            ),
        )
        opened = requests.reconcile("s", "conv", 1, prompt)
        assert opened is not None and opened.request.tool is not None
        assert opened.request.tool["input"]["content"] == content


@pytest.mark.parametrize(
    "command", ["git status --short", "git  status", "echo git status"]
)
def test_permission_evidence_requires_exact_command_text(command):
    requests = RequestStore(EventHub())
    requests.observe_hook(
        "s",
        PendingHook(
            "conv", "permission", "Bash", {"command": "git status"}, time.time()
        ),
    )
    opened = requests.reconcile(
        "s", "conv", 1, Prompt(f"{command}\n1. Yes\n2. No", "permission")
    )
    assert opened is not None and opened.request.source == "screen"


def test_consumed_hook_cannot_be_replayed_after_request_closes():
    requests = RequestStore(EventHub())
    hook = PendingHook(
        "conv", "permission", "Bash", {"command": "git status"}, time.time()
    )
    prompt = Prompt("git status\n1. Yes\n2. No", "permission")
    requests.observe_hook("s", hook)
    opened = requests.reconcile("s", "conv", 1, prompt)
    assert opened is not None and opened.request.source == "hook"
    requests.close("s", "answered")
    requests.observe_hook("s", hook)
    opened = requests.reconcile("s", "conv", 1, prompt)
    assert opened is not None and opened.request.source == "screen"


async def test_capture_failure_is_isolated_and_recovery_preserves_identity(
    service, backend
):
    healthy = backend.target
    other = replace(healthy, id="@2", conversation_id="other")
    backend.targets = AsyncMock(return_value=[other, healthy])
    unavailable = True

    async def prompt(target):
        if unavailable and target.id == "@2":
            raise ProtocolError(503, "backend_unavailable", "Capture unavailable")
        return None

    backend.prompt = prompt
    sessions = await service.list_sessions()
    assert len(sessions) == 2
    failed = next(s for s in sessions if s.status.state == "unknown")
    unavailable = False
    recovered = await service.get_session(failed.id)
    assert recovered.status.state == "idle"
    assert recovered.binding_version > failed.binding_version


def _source(store, path):
    return TranscriptSource(
        "conv", path, "conv", "claude", store.transcript_cursor("conv")
    )


async def test_rewrite_beyond_first_page_invalidates_history_and_survives_restart(
    tmp_path,
):
    path = tmp_path / "transcript"
    path.write_bytes(b"a" * 4096 + b"b" * 4096)
    db = tmp_path / "protocol.sqlite3"
    store = ProtocolStore(db)
    source = _source(store, path)
    await source.sync(store)
    assert source.generation == 0
    path.write_bytes(b"a" * 4096 + b"c" * 8192)
    await source.sync(store)
    assert source.generation == 1
    with path.open("ab") as stream:
        stream.write(b"d")
    await source.sync(store)
    assert source.generation == 1
    store.close()
    store = ProtocolStore(db)
    reopened = _source(store, path)
    await reopened.sync(store)
    assert reopened.generation == 1
    store.close()


async def test_unavailable_transcript_does_not_block_healthy_sessions(
    service, backend, tmp_path
):
    healthy = backend.target
    path = tmp_path / "missing"
    path.touch()
    unavailable = replace(healthy, id="@2", conversation_id="other", transcript=path)
    backend.targets = AsyncMock(return_value=[unavailable, healthy])
    await service.list_sessions()
    path.unlink()
    sessions = await service.list_sessions()
    assert len(sessions) == 2
    failed = next(
        s for s in sessions if s.conversation and s.conversation.id == "other"
    )
    assert failed.status.state == "unknown"
    assert set(failed.capabilities) == {"session.close", "terminal.view"}
    assert (
        await service.get_session(next(s.id for s in sessions if s.id != failed.id))
    ).provider == "claude"
    assert len((await LocalSessionService(backend, service.store).list_sessions())) == 2


async def test_current_and_refresh_cannot_publish_regressing_bindings(service, backend):
    session = (await service.list_sessions())[0]
    entered, release = asyncio.Event(), asyncio.Event()

    async def prompt(target):
        if target.conversation_id == "conv":
            entered.set()
            await release.wait()
        return None

    backend.prompt = prompt
    current = asyncio.create_task(service.current(session.id))
    await entered.wait()
    backend.target = replace(backend.target, conversation_id="replacement")
    refresh = asyncio.create_task(service.refresh())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(current, refresh)
    assert service._snapshots[session.id].binding_version == 2
    _, events = service.hub.replay(service.hub.epoch, 0)
    versions = [
        e.data["bindingVersion"]
        for e in events
        if e.type in {"session.snapshot", "session.updated"}
        and "bindingVersion" in e.data
    ]
    assert versions == sorted(versions)


async def test_multiselect_response_is_rejected_without_input(service, backend):
    session = (await service.list_sessions())[0]
    service.requests.observe_hook(session.id, question_hook(multi=True))
    backend.current_prompt = Prompt(
        "Where?\n1. Staging\n2. Staging + production", "question"
    )
    session = await service.get_session(session.id)
    from uuid import uuid4
    from ccgram.session_protocol.models import Action

    result, status = await service.execute(
        session.id,
        Action(
            str(uuid4()),
            session.binding_version,
            "request.respond",
            {"requestId": session.open_requests[0].id, "optionId": "1"},
        ),
    )
    assert status == 422 and result.outcome == "rejected"
    assert backend.sent == []


def test_bound_hook_keeps_controls_after_matching_ttl(monkeypatch):
    requests = RequestStore(EventHub())
    now = time.time()
    hook = PendingHook("conv", "permission", "Bash", {"command": "git status"}, now)
    prompt = Prompt("git status\n1. Yes\n2. No", "permission")
    requests.observe_hook("s", hook)
    first = requests.reconcile("s", "conv", 1, prompt)
    assert first is not None and first.option_keys
    monkeypatch.setattr("ccgram.session_protocol.requests.time.time", lambda: now + 31)
    current = requests.reconcile("s", "conv", 1, prompt)
    assert current is not None and current.request.id == first.request.id
    assert current.option_keys == first.option_keys


async def test_repeated_capture_failure_does_not_change_binding(service, backend):
    backend.prompt = AsyncMock(
        side_effect=ProtocolError(503, "backend_unavailable", "offline")
    )
    first = (await service.list_sessions())[0]
    second = await service.get_session(first.id)
    assert second.binding_version == first.binding_version
    assert second.status.state == "unknown"
