"""Tests for status_snapshot — /status and /stats fallback for non-native providers."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.commands.status_snapshot import (
    _maybe_send_status_snapshot,
    _status_snapshot_probe_offset,
)

_SS = "ccgram.handlers.commands.status_snapshot"


def _transcript(size: int = 4096, name: str = "/tmp/codex.jsonl") -> MagicMock:
    path = MagicMock(spec=Path)
    path.stat.return_value.st_size = size
    path.__str__ = MagicMock(return_value=name)
    return path


def _view(*, transcript_path: object, provider_name: str = "codex") -> SimpleNamespace:
    return SimpleNamespace(
        transcript_path=transcript_path,
        provider_name=provider_name,
        session_id="s",
        cwd="/c",
    )


def _provider(*, supported: bool = True, **extra: object) -> SimpleNamespace:
    return SimpleNamespace(
        capabilities=SimpleNamespace(supports_status_snapshot=supported), **extra
    )


@pytest.fixture
def snapshot_env() -> Iterator[SimpleNamespace]:
    with (
        patch(f"{_SS}.window_query") as window_query,
        patch(f"{_SS}.get_provider_for_window") as get_provider,
        patch(f"{_SS}.safe_reply", new_callable=AsyncMock) as reply,
        patch(f"{_SS}.asyncio.sleep", new_callable=AsyncMock),
    ):
        yield SimpleNamespace(
            window_query=window_query, get_provider=get_provider, reply=reply
        )


class TestStatusSnapshotProbeOffset:
    def test_returns_transcript_size_for_supported_provider(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        snapshot_env.window_query.view_window.return_value = _view(
            transcript_path=_transcript(4096)
        )
        snapshot_env.get_provider.return_value = _provider()

        assert _status_snapshot_probe_offset("@1", "/status") == 4096

    @pytest.mark.parametrize(
        ("command", "provider_name", "supported"),
        [
            pytest.param("/clear", "codex", True, id="not-a-status-command"),
            pytest.param("/status", "claude", False, id="provider-has-native-status"),
        ],
    )
    def test_returns_none(
        self,
        snapshot_env: SimpleNamespace,
        command: str,
        provider_name: str,
        supported: bool,
    ) -> None:
        snapshot_env.window_query.view_window.return_value = _view(
            transcript_path=None, provider_name=provider_name
        )
        snapshot_env.get_provider.return_value = _provider(supported=supported)

        assert _status_snapshot_probe_offset("@1", command) is None


class TestMaybeSendStatusSnapshot:
    async def test_emits_snapshot(self, snapshot_env: SimpleNamespace) -> None:
        message = AsyncMock()
        snapshot_env.window_query.view_window.return_value = _view(
            transcript_path=_transcript()
        )
        snapshot_env.get_provider.return_value = _provider(
            build_status_snapshot=MagicMock(return_value="snapshot body"),
            has_output_since=MagicMock(return_value=False),
        )

        await _maybe_send_status_snapshot(message, "@1", "p", "/status", since_offset=0)

        snapshot_env.reply.assert_called_once_with(message, "snapshot body")

    async def test_skips_snapshot_when_provider_already_answered(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        provider = _provider(
            build_status_snapshot=MagicMock(return_value="ignored"),
            has_output_since=MagicMock(return_value=True),
        )
        snapshot_env.window_query.view_window.return_value = _view(
            transcript_path=_transcript()
        )
        snapshot_env.get_provider.return_value = provider

        await _maybe_send_status_snapshot(
            AsyncMock(), "@1", "p", "/status", since_offset=0
        )

        snapshot_env.reply.assert_not_called()
        provider.build_status_snapshot.assert_not_called()

    async def test_reports_missing_transcript(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        snapshot_env.window_query.view_window.return_value = _view(transcript_path=None)
        snapshot_env.get_provider.return_value = _provider()

        await _maybe_send_status_snapshot(AsyncMock(), "@1", "p", "/status")

        snapshot_env.reply.assert_called_once()
        assert "no transcript path" in snapshot_env.reply.call_args.args[1]

    async def test_skips_for_non_status_command(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        await _maybe_send_status_snapshot(AsyncMock(), "@1", "p", "/clear")

        snapshot_env.window_query.view_window.assert_not_called()
        snapshot_env.reply.assert_not_called()

    async def test_skips_when_provider_has_native_status(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        snapshot_env.window_query.view_window.return_value = _view(
            transcript_path=None, provider_name="claude"
        )
        snapshot_env.get_provider.return_value = _provider(supported=False)

        await _maybe_send_status_snapshot(AsyncMock(), "@1", "p", "/status")

        snapshot_env.reply.assert_not_called()
