"""Tests for Codex transcript status snapshot helper."""

import json

from ccgram.codex_status import (
    build_codex_status_snapshot,
    has_codex_assistant_output_since,
)


def _write_jsonl(path, entries) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_build_codex_status_snapshot_with_token_stats(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-03-02T17:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "sess-abc",
                    "cwd": "/Users/alexei/Workspace/repo",
                    "cli_version": "0.106.0",
                },
            },
            {
                "timestamp": "2026-03-02T17:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "model_context_window": 1000,
                        "total_token_usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 3,
                            "output_tokens": 4,
                            "reasoning_output_tokens": 2,
                            "total_tokens": 14,
                        },
                        "rate_limits": {
                            "primary": {"used_percent": 12.5, "resets_at": 1772485454},
                            "secondary": {
                                "used_percent": 20.0,
                                "resets_at": 1772792459,
                            },
                        },
                    },
                },
            },
        ],
    )

    result = build_codex_status_snapshot(
        str(transcript),
        display_name="repo",
        session_id="",
        cwd="",
    )

    assert result is not None
    assert "[repo] Codex status snapshot" in result
    assert "session: `sess-abc`" in result
    assert "tokens (total): in `10`" in result
    assert "context window: `14` / `1,000` (1.4%)" in result
    assert "primary limit: `12.5%` used" in result


def test_build_codex_status_snapshot_missing_file() -> None:
    result = build_codex_status_snapshot(
        "/tmp/ccgram-missing-codex-status.jsonl",
        display_name="repo",
    )
    assert result is None


def test_build_codex_status_snapshot_without_token_count(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-03-02T17:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "sess-xyz",
                    "cwd": "/tmp/repo",
                    "cli_version": "0.106.0",
                },
            }
        ],
    )

    result = build_codex_status_snapshot(
        str(transcript),
        display_name="repo",
    )

    assert result is not None
    assert "token stats: unavailable" in result


def test_has_codex_assistant_output_since_detects_response(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-03-02T17:00:00.000Z",
                "type": "input_item",
                "payload": {"role": "user", "content": "/status"},
            },
            {
                "timestamp": "2026-03-02T17:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "status text"}],
                },
            },
        ],
    )

    assert has_codex_assistant_output_since(str(transcript), 0) is True


def test_has_codex_assistant_output_since_handles_midline_offset(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-03-02T17:00:00.000Z",
                "type": "event_msg",
                "payload": {"type": "token_count", "info": {"total_token_usage": {}}},
            },
            {
                "timestamp": "2026-03-02T17:00:02.000Z",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "ready"},
            },
        ],
    )

    first_line = transcript.read_text(encoding="utf-8").splitlines()[0]
    mid_offset = len(first_line) // 2
    assert has_codex_assistant_output_since(str(transcript), mid_offset) is True
