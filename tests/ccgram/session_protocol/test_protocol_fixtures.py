import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from ccgram.session_protocol.actions import parse_action
from ccgram.session_protocol.messages import SourceEntry, build_messages
from ccgram.session_protocol.models import wire

ROOT = Path(__file__).resolve().parents[3] / "protocol" / "v1"
# parents: session_protocol -> ccgram -> tests -> repository
SCHEMAS = {p.stem: json.loads(p.read_text()) for p in (ROOT / "schemas").glob("*.json")}
REGISTRY = Registry().with_resources(
    (value["$id"], Resource.from_contents(value)) for value in SCHEMAS.values()
)
MANIFEST = json.loads((ROOT / "fixtures" / "_manifest.json").read_text())


def validate(schema_name, value):
    Draft202012Validator(
        SCHEMAS[schema_name], registry=REGISTRY, format_checker=FormatChecker()
    ).validate(value)


@pytest.mark.parametrize("schema_name", SCHEMAS)
def test_shared_schemas_are_valid(schema_name):
    Draft202012Validator.check_schema(SCHEMAS[schema_name])


@pytest.mark.parametrize(
    "schema_name,filename", [(s, f) for s, files in MANIFEST.items() for f in files]
)
def test_every_golden_fixture_matches_shared_schema(schema_name, filename):
    validate(schema_name, json.loads((ROOT / "fixtures" / filename).read_text()))


def test_action_fixture_is_decoded_by_the_server():
    expected = json.loads((ROOT / "fixtures" / "action.json").read_text())
    assert wire(parse_action(expected)) == expected


def test_builder_output_matches_message_golden_fixture():
    timestamp = "2026-10-20T10:33:12Z"
    call = {
        "uuid": "entry",
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "content": [
                {"type": "text", "text": "Run the tests."},
                {
                    "type": "tool_use",
                    "id": "tool1",
                    "name": "Bash",
                    "input": {"command": "uv run pytest"},
                },
            ]
        },
    }
    result = {
        "uuid": "result",
        "type": "user",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": "tool1", "content": "passed"}
            ]
        },
    }
    actual = build_messages(
        [SourceEntry(0, call), SourceEntry(200, result)], "conv", "claude"
    )[0]
    assert wire(actual) == json.loads((ROOT / "fixtures" / "message.json").read_text())


@pytest.mark.parametrize("provider", ["claude", "codex", "gemini", "pi"])
async def test_real_service_output_matches_schema(service, backend, tmp_path, provider):
    from dataclasses import replace

    message = {"role": "assistant", "content": [{"type": "text", "text": "public"}]}
    if provider == "codex":
        entries = [
            {"type": "response_item", "payload": {"type": "message", **message}},
            {
                "type": "response_item",
                "payload": {"type": "message", **message, "role": "developer"},
            },
        ]
    elif provider == "gemini":
        entries = [{"type": "gemini", "id": "native", "content": "public"}]
    else:
        entries = [
            {
                "type": "assistant" if provider == "claude" else "message",
                "uuid": "native",
                "id": "native",
                "message": message,
            }
        ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    backend.target = replace(backend.target, provider=provider, transcript=path)
    session = (await service.list_sessions())[0]
    service.hub.subscribe()  # message events are published to subscribers only
    validate("Session", wire(session))
    history = await service.history(session.id, None, 50)
    assert len(history.messages) == 1 and history.messages[0].role == "agent"
    validate("History", wire(history))
    _, events = service.hub.replay(service.hub.epoch, 0)
    assert any(event.type == "message.upsert" for event in events)
    for event in events:
        validate("ServerFrame", event.body())


def test_openapi_and_asyncapi_references_resolve_locally():
    def check(node, document, current):
        if isinstance(node, dict):
            reference = node.get("$ref")
            if reference:
                file, _, fragment = reference.partition("#")
                target = (
                    json.loads((current.parent / file).read_text())
                    if file
                    else document
                )
                for segment in (
                    fragment.removeprefix("/").split("/") if fragment else []
                ):
                    target = target[segment.replace("~1", "/").replace("~0", "~")]
            for value in node.values():
                check(value, document, current)
        elif isinstance(node, list):
            for item in node:
                check(item, document, current)

    for filename in ("openapi.yaml", "asyncapi.yaml"):
        path = ROOT / filename
        document = json.loads(path.read_text())
        check(document, document, path)
    assert json.loads((ROOT / "openapi.yaml").read_text())["openapi"] == "3.1.0"
    assert json.loads((ROOT / "asyncapi.yaml").read_text())["asyncapi"] == "3.0.0"
