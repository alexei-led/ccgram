import json

import pytest

from ccgram.providers import codex_status
from ccgram.providers.codex_status import (
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
                    },
                    "rate_limits": {
                        "primary": {"used_percent": 12.5, "resets_at": 1772485454},
                        "secondary": {"used_percent": 20.0, "resets_at": 1772792459},
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


def _meta_entry() -> dict:
    return {
        "timestamp": "2026-03-02T17:00:00.000Z",
        "type": "session_meta",
        "payload": {"id": "sess-big", "cwd": "/tmp/repo", "cli_version": "0.106.0"},
    }


def _token_entry(index: int) -> dict:
    return {
        "timestamp": f"2026-03-02T17:00:{index % 60:02d}.000Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "model_context_window": 1000,
                "total_token_usage": {"total_tokens": index},
            },
        },
    }


def _filler_entry(index: int) -> dict:
    return {
        "timestamp": f"2026-03-02T18:{index // 60 % 60:02d}:{index % 60:02d}.000Z",
        "type": "response_item",
        "payload": {"type": "reasoning", "text": "x" * 80},
    }


def test_snapshot_reads_rate_limits_nested_in_info_for_older_rollouts(
    tmp_path,
) -> None:
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [
            _meta_entry(),
            {
                "timestamp": "2026-03-02T17:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {"total_tokens": 14},
                        "rate_limits": {
                            "primary": {"used_percent": 7.0, "resets_at": 1772485454}
                        },
                    },
                },
            },
        ],
    )

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert "primary limit: `7.0%` used" in result
    assert "secondary limit" not in result


def test_snapshot_shows_primary_limit_when_secondary_is_null(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    entry = _token_entry(5)
    entry["payload"]["rate_limits"] = {
        "primary": {"used_percent": 71.0, "resets_at": 1772485454},
        "secondary": None,
    }
    _write_jsonl(transcript, [_meta_entry(), entry])

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert "primary limit: `71.0%` used" in result
    assert "secondary limit" not in result


@pytest.mark.parametrize("block_bytes", [32, 100, 1 << 20])
def test_snapshot_reads_latest_token_count_across_tail_blocks(
    tmp_path, monkeypatch, block_bytes
) -> None:
    monkeypatch.setattr(codex_status, "_READ_BLOCK_BYTES", block_bytes)
    transcript = tmp_path / "codex.jsonl"
    entries = [_meta_entry(), _token_entry(1), _token_entry(7)]
    entries += [_filler_entry(i) for i in range(40)]
    _write_jsonl(transcript, entries)

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert "session: `sess-big`" in result
    assert "transcript entries: `43`" in result
    assert "last transcript event: `2026-03-02T18:00:39.000Z`" in result
    assert "total `7`" in result


def test_snapshot_without_trailing_newline_counts_last_entry(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        json.dumps(_meta_entry()) + "\n" + json.dumps(_token_entry(3)),
        encoding="utf-8",
    )

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert "transcript entries: `2`" in result
    assert "total `3`" in result


def test_snapshot_ignores_token_count_older_than_the_tail_limit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(codex_status, "_TAIL_MAX_BYTES", 2048)
    monkeypatch.setattr(codex_status, "_READ_BLOCK_BYTES", 256)
    transcript = tmp_path / "codex.jsonl"
    entries = [_meta_entry(), _token_entry(5)]
    entries += [_filler_entry(i) for i in range(200)]
    _write_jsonl(transcript, entries)

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert "session: `sess-big`" in result
    assert "transcript entries: `202`" in result
    assert "last transcript event: `2026-03-02T18:03:19.000Z`" in result
    assert "token stats: unavailable" in result


def test_snapshot_parses_only_a_bounded_tail_of_a_large_transcript(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(codex_status, "_TAIL_MAX_BYTES", 4096)
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [_meta_entry(), *[_token_entry(i) for i in range(2000)]],
    )
    parsed_lines = []
    real_loads = json.loads

    def counting_loads(raw, *args, **kwargs):
        parsed_lines.append(raw)
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr(codex_status.json, "loads", counting_loads)

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert "transcript entries: `2,001`" in result
    assert "total `1,999`" in result
    assert len(parsed_lines) < 10


@pytest.mark.parametrize("content", ["", "\n\n", "not json\n[1, 2]\n"])
def test_snapshot_returns_none_without_json_object_entries(tmp_path, content) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(content, encoding="utf-8")

    assert build_codex_status_snapshot(str(transcript), display_name="repo") is None


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
