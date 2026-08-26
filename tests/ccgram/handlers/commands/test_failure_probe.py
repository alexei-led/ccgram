"""Tests for failure_probe — post-send transcript + pane delta probes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.commands.failure_probe import (
    _extract_pane_delta,
    _extract_probe_error_line,
    _maybe_send_command_failure_message,
    _probe_transcript_command_error,
)


_FP = "ccgram.handlers.commands.failure_probe"


class TestExtractProbeErrorLine:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "ok\nunrecognized command '/cost'\n",
                "unrecognized command '/cost'",
                id="unrecognized-command",
            ),
            pytest.param(
                "all good\nERROR executing command /x\n",
                "ERROR executing command /x",
                id="error-executing",
            ),
            pytest.param("all good\nstill fine\n", None, id="no-error-line"),
            pytest.param("", None, id="empty"),
        ],
    )
    def test_extraction(self, text: str, expected: str | None) -> None:
        assert _extract_probe_error_line(text) == expected


class TestExtractPaneDelta:
    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            pytest.param("line1\nline2", "line1\nline2\nline3", "line3", id="appended"),
            pytest.param("A\nB", "B\nC\nD", "C\nD", id="scrolled"),
            pytest.param("same", "same", "", id="unchanged"),
            pytest.param(None, "only after", "only after", id="no-baseline"),
            pytest.param(
                "abc", "xabcx\ndef", "xabcx\ndef", id="no-common-line-keeps-all"
            ),
        ],
    )
    def test_delta(self, before: str | None, after: str, expected: str) -> None:
        assert _extract_pane_delta(before, after) == expected


class TestProbeTranscriptCommandError:
    async def test_uses_incremental_reader_for_codex(self, tmp_path) -> None:
        transcript = tmp_path / "session.jsonl"
        prefix = "ok\n"
        suffix = "unknown command: /status\n"
        transcript.write_text(prefix + suffix, encoding="utf-8")

        provider = SimpleNamespace(
            capabilities=SimpleNamespace(supports_incremental_read=True),
            parse_transcript_line=lambda line: (
                {"text": line.strip()} if line.strip() else None
            ),
            parse_transcript_entries=lambda entries, pending_tools: (
                [
                    SimpleNamespace(role="assistant", text=entry["text"])
                    for entry in entries
                ],
                pending_tools,
            ),
            read_transcript_file=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotImplementedError("incremental only")
            ),
        )

        result = await _probe_transcript_command_error(
            provider,  # type: ignore[arg-type]
            str(transcript),
            len(prefix),
        )
        assert result == "unknown command: /status"

    async def test_whole_file_not_implemented_returns_none(self, tmp_path) -> None:
        transcript = tmp_path / "session.json"
        transcript.write_text("{}", encoding="utf-8")

        provider = SimpleNamespace(
            capabilities=SimpleNamespace(supports_incremental_read=False),
            read_transcript_file=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotImplementedError("not implemented")
            ),
            parse_transcript_entries=lambda entries, pending_tools: ([], pending_tools),
        )

        result = await _probe_transcript_command_error(provider, str(transcript), 0)  # type: ignore[arg-type]
        assert result is None


class TestMaybeSendCommandFailureMessage:
    async def test_surfaces_transcript_error(self) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        message = AsyncMock()

        with (
            patch(f"{_FP}.asyncio.sleep", new_callable=AsyncMock),
            patch(
                f"{_FP}._probe_transcript_command_error",
                new_callable=AsyncMock,
                return_value="unrecognized command '/foo'",
            ),
            patch(f"{_FP}.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            await _maybe_send_command_failure_message(
                message,
                "@1",
                "project",
                "/foo",
                provider=provider,  # type: ignore[arg-type]
                transcript_path="/tmp/codex.jsonl",
                since_offset=0,
                pane_before="",
            )

        mock_reply.assert_called_once()
        assert "failed" in mock_reply.call_args.args[1]
        assert "unrecognized command" in mock_reply.call_args.args[1]

    async def test_falls_back_to_pane_delta_when_transcript_has_no_error(self) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        message = AsyncMock()

        with (
            patch(f"{_FP}.asyncio.sleep", new_callable=AsyncMock),
            patch(
                f"{_FP}._probe_transcript_command_error",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.capture_pane",
                new_callable=AsyncMock,
                return_value="before\nunknown command: /foo",
            ),
            patch(f"{_FP}.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            await _maybe_send_command_failure_message(
                message,
                "@1",
                "project",
                "/foo",
                provider=provider,  # type: ignore[arg-type]
                transcript_path=None,
                since_offset=None,
                pane_before="before",
            )

        mock_reply.assert_called_once()
        assert "unknown command" in mock_reply.call_args.args[1]

    async def test_no_error_found_sends_no_message(self) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        message = AsyncMock()

        with (
            patch(f"{_FP}.asyncio.sleep", new_callable=AsyncMock),
            patch(
                f"{_FP}._probe_transcript_command_error",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.capture_pane",
                new_callable=AsyncMock,
                return_value="before\nall good",
            ),
            patch(f"{_FP}.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            await _maybe_send_command_failure_message(
                message,
                "@1",
                "project",
                "/help",
                provider=provider,  # type: ignore[arg-type]
                transcript_path=None,
                since_offset=None,
                pane_before="before",
            )

        mock_reply.assert_not_called()
