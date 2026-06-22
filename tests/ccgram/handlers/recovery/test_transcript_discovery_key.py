"""Regression: _find_and_register_transcript builds window_key from session_map_prefix().

Before this fix the key was hardcoded as f"{config.tmux_session_name}:{window_id}",
which never matched herdr session_map entries (keyed as "herdr:<tab_id>").
"""

from unittest.mock import MagicMock, patch

from ccgram.config import config
from ccgram.handlers.recovery.transcript_discovery import _find_and_register_transcript
from ccgram.window_state_ports import identity_state


def _identity(
    window_id: str = "@0", cwd: str = "/repo"
) -> identity_state.IdentityProjection:
    return identity_state.IdentityProjection(
        window_id=window_id,
        cwd=cwd,
        session_id="",
        transcript_path=None,
        provider_name="codex",
        window_name="agent",
        approval_mode="default",
    )


class TestFindAndRegisterTranscriptWindowKey:
    """window_key passed to provider.discover_transcript matches the active backend prefix."""

    async def _run(self, window_id: str) -> str | None:
        """Call _find_and_register_transcript and return the window_key seen by discover_transcript."""
        captured: list[str] = []
        provider = MagicMock()

        def _discover(cwd, window_key, *, max_age=None):
            captured.append(window_key)
            return None  # no transcript found — enough to verify the key

        provider.discover_transcript.side_effect = _discover

        with patch(
            "ccgram.handlers.recovery.transcript_discovery._session_id_already_bound",
            return_value=False,
        ):
            await _find_and_register_transcript(
                window_id,
                _identity(window_id=window_id),
                [("codex", provider)],
                pane_alive=True,
            )

        return captured[0] if captured else None

    async def test_tmux_backend_key(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "tmux")
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        key = await self._run("@7")
        assert key == "ccgram:@7"

    async def test_herdr_backend_key(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        key = await self._run("w2:p1")
        assert key == "herdr:w2:p1"
