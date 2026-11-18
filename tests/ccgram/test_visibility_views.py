import json

import pytest

from ccgram.claude_task_state import ClaudeTaskStateStore
from ccgram.utils import read_session_metadata_from_jsonl


@pytest.mark.parametrize("flag", ("isMeta", "isSidechain", "isCompactSummary"))
def test_resume_summary_ignores_internal_user_context(tmp_path, flag):
    path = tmp_path / "session.jsonl"
    rows = [
        {
            "type": "user",
            flag: True,
            "cwd": "/workspace",
            "message": {"role": "user", "content": "INTERNAL_CONTEXT"},
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Actual request"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert read_session_metadata_from_jsonl(path) == ("/workspace", "Actual request")


@pytest.mark.parametrize("flag", ("isMeta", "isSidechain", "isCompactSummary"))
def test_internal_records_cannot_create_tasks_in_the_public_status_view(flag):
    store = ClaudeTaskStateStore()
    row = {
        "type": "assistant",
        flag: True,
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "hidden-task",
                    "name": "TodoWrite",
                    "input": {
                        "todos": [{"content": "INTERNAL_CONTEXT", "status": "pending"}]
                    },
                }
            ],
        },
    }

    assert not store.apply_entries("@test", "session", [row])
    assert store.get_snapshot("@test") is None


@pytest.mark.parametrize("flag", ("isMeta", "isSidechain", "isCompactSummary"))
def test_completion_summary_ignores_hidden_assistant_records(flag):
    from ccgram.llm.summarizer import _build_summary_context

    rows = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Actual answer"}]},
        },
        {
            "type": "assistant",
            flag: True,
            "message": {"content": [{"type": "text", "text": "INTERNAL_CONTEXT"}]},
        },
    ]
    context = _build_summary_context([json.dumps(row) for row in rows])
    assert "Actual answer" in context
    assert "INTERNAL_CONTEXT" not in context


@pytest.mark.parametrize(
    "row",
    [
        {
            "type": "api-request-blob",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "INTERNAL_CONTEXT"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "system",
                "content": [{"type": "text", "text": "INTERNAL_CONTEXT"}],
            },
        },
    ],
)
def test_completion_summary_rejects_wrong_envelopes_and_roles(row):
    from ccgram.llm.summarizer import _build_summary_context

    assert _build_summary_context([json.dumps(row)]) == ""
