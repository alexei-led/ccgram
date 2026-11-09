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
                        "last_token_usage": {"total_tokens": 9},
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
    assert "context window: `9` / `1,000` (0.9%)" in result
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
                "last_token_usage": {"total_tokens": index // 2},
            },
            "rate_limits": {
                "primary": {"used_percent": 1.0, "resets_at": 1772485454},
                "secondary": None,
            },
        },
    }


def _filler_entry(index: int) -> dict:
    return {
        "timestamp": f"2026-03-02T18:{index // 60 % 60:02d}:{index % 60:02d}.000Z",
        "type": "response_item",
        "payload": {"type": "reasoning", "text": "x" * 80},
    }


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


def _token_count(info, rate_limits, second=0) -> dict:
    return {
        "timestamp": f"2026-03-02T17:00:{second:02d}.000Z",
        "type": "event_msg",
        "payload": {"type": "token_count", "info": info, "rate_limits": rate_limits},
    }


def _snapshot(tmp_path, *events) -> str:
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(transcript, [_meta_entry(), *events])
    result = build_codex_status_snapshot(str(transcript), display_name="repo")
    assert result is not None
    return result


def test_context_window_uses_the_last_request_not_the_running_total(tmp_path) -> None:
    info = {
        "model_context_window": 828_400,
        "total_token_usage": {"total_tokens": 897_184},
        "last_token_usage": {"total_tokens": 96_645},
    }

    result = _snapshot(tmp_path, _token_count(info, None))

    assert "total `897,184`" in result
    assert "context window: `96,645` / `828,400` (11.7%)" in result
    assert "108.3%" not in result


def test_context_window_line_is_omitted_without_last_token_usage(tmp_path) -> None:
    info = {"model_context_window": 1000, "total_token_usage": {"total_tokens": 5000}}

    result = _snapshot(tmp_path, _token_count(info, None))

    assert "total `5,000`" in result
    assert "context window" not in result


def test_newest_rate_limits_come_from_a_token_count_without_info(tmp_path) -> None:
    info = {
        "model_context_window": 1000,
        "total_token_usage": {"total_tokens": 40},
        "last_token_usage": {"total_tokens": 20},
    }
    old = {"primary": {"used_percent": 10.0, "resets_at": 1772485454}}
    new = {"primary": {"used_percent": 55.0, "resets_at": 1772485454}}

    result = _snapshot(tmp_path, _token_count(info, old, 1), _token_count(None, new, 2))

    assert "primary limit: `55.0%` used" in result
    assert "10.0%" not in result
    assert "total `40`" in result


def test_rate_limits_show_when_no_token_usage_exists(tmp_path) -> None:
    limits = {"primary": {"used_percent": 40.0, "resets_at": 1772485454}}

    result = _snapshot(tmp_path, _token_count(None, limits))

    assert "token stats: unavailable" in result
    assert "primary limit: `40.0%` used" in result


def test_newest_token_count_without_limits_does_not_show_older_limits(
    tmp_path,
) -> None:
    info = {"total_token_usage": {"total_tokens": 40}}
    old = {"primary": {"used_percent": 95.0, "resets_at": 1000000000}}

    result = _snapshot(
        tmp_path, _token_count(info, old, 1), _token_count(info, None, 2)
    )

    assert "limit:" not in result
    assert "total `40`" in result


def test_limits_object_without_windows_shows_no_limit_lines(tmp_path) -> None:
    info = {"total_token_usage": {"total_tokens": 40}}
    old = {"primary": {"used_percent": 95.0, "resets_at": 1000000000}}
    empty = {"limit_id": "codex", "primary": None, "secondary": None}

    result = _snapshot(
        tmp_path, _token_count(info, old, 1), _token_count(info, empty, 2)
    )

    assert "limit:" not in result


def test_null_used_percent_renders_as_unknown(tmp_path) -> None:
    limits = {"primary": {"used_percent": None, "resets_at": 1772485454}}

    result = _snapshot(tmp_path, _token_count(None, limits))

    assert "primary limit: `?%` used" in result
    assert "None" not in result


def test_snapshot_stops_at_the_newest_token_count_without_rate_limits(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(codex_status, "_TAIL_MAX_BYTES", 1 << 20)
    info = {"total_token_usage": {"total_tokens": 1}}
    transcript = tmp_path / "codex.jsonl"
    _write_jsonl(
        transcript,
        [_meta_entry(), *[_token_count(info, None, i % 60) for i in range(3000)]],
    )
    parsed_lines = []
    real_loads = json.loads

    def counting_loads(raw, *args, **kwargs):
        parsed_lines.append(raw)
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr(codex_status.json, "loads", counting_loads)

    result = build_codex_status_snapshot(str(transcript), display_name="repo")

    assert result is not None
    assert len(parsed_lines) < 10


@pytest.mark.parametrize(
    ("window_minutes", "label"),
    [
        pytest.param(300, "5h", id="five-hours"),
        pytest.param(10080, "weekly", id="week"),
        pytest.param(1440, "1d", id="day"),
        pytest.param(45, "45m", id="minutes"),
        pytest.param(0, "primary", id="zero-falls-back"),
        pytest.param(None, "primary", id="null-falls-back"),
        pytest.param("soon", "primary", id="non-numeric-falls-back"),
    ],
)
def test_rate_limit_lines_are_named_by_window(tmp_path, window_minutes, label) -> None:
    limits = {
        "primary": {
            "used_percent": 71.0,
            "window_minutes": window_minutes,
            "resets_at": 1772485454,
        }
    }

    result = _snapshot(tmp_path, _token_count(None, limits))

    assert f"- {label} limit: `71.0%` used" in result


def test_missing_window_minutes_falls_back_to_slot_names(tmp_path) -> None:
    limits = {
        "primary": {"used_percent": 1.0, "resets_at": 1772485454},
        "secondary": {"used_percent": 2.0, "resets_at": 1772792459},
    }

    result = _snapshot(tmp_path, _token_count(None, limits))

    assert "- primary limit: `1.0%` used" in result
    assert "- secondary limit: `2.0%` used" in result


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
