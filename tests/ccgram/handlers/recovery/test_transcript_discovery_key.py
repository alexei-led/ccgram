"""Regression: _find_and_register_transcript builds window_key from session_map_prefix().

Before this fix the key was hardcoded as f"{config.tmux_session_name}:{window_id}",
which never matched herdr session_map entries (keyed as "herdr:<tab_id>").
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.config import config
from ccgram.handlers.recovery.transcript_discovery import (
    _bootstrap_identity,
    _find_and_register_transcript,
    discover_and_register_transcript,
)
from ccgram.multiplexer.base import WindowRef
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

    @pytest.mark.parametrize(
        ("backend", "window_id", "expected_key"),
        [
            ("tmux", "@7", "ccgram:@7"),
            ("herdr", "w2:p1", "herdr:w2:p1"),
        ],
    )
    async def test_window_key_uses_the_active_backend_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
        window_id: str,
        expected_key: str,
    ) -> None:
        monkeypatch.setattr(config, "multiplexer_name", backend)
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
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

        assert captured == [expected_key]


# ── Fix 3: _bootstrap_identity seeds window state for hookless providers ───


def _window_ref(
    window_id: str = "@9",
    cwd: str = "/project",
    pane_current_command: str = "pi",
) -> WindowRef:
    return WindowRef(
        window_id=window_id,
        window_name="tab",
        cwd=cwd,
        pane_current_command=pane_current_command,
    )


class TestBootstrapIdentity:
    """_bootstrap_identity creates window state for live windows ccgram has no record of."""

    async def test_w_none_returns_none_without_creating_state(self) -> None:
        # No window available: bootstrap cannot detect provider, returns None.
        result = await _bootstrap_identity("@9", None)
        assert result is None

    async def test_undetected_provider_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
            AsyncMock(return_value=None),
        )
        mock_sm = MagicMock()
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_manager",
            mock_sm,
        )

        result = await _bootstrap_identity(
            "@9", _window_ref(pane_current_command="bash")
        )

        assert result is None
        mock_sm.set_window_provider.assert_not_called()

    async def test_existing_session_map_entry_is_never_bootstrapped_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seeding state must not destroy the hook's entry for a live session.

        On a state-less window set_window_provider counts as a provider switch
        and clears the session_map entry, and hook.py refuses to recreate one
        from any non-SessionStart event. Bootstrapping a window the hook is
        already tracking would leave a running agent untracked until it was
        restarted, with its messages no longer reaching the topic.
        """
        detect = AsyncMock(return_value="claude")
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
            detect,
        )
        mock_sm = MagicMock()
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_manager",
            mock_sm,
        )
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ".session_map_entry_may_exist",
            AsyncMock(return_value=True),
        )

        result = await _bootstrap_identity("@9", _window_ref(cwd="/project"))

        assert result is None
        mock_sm.set_window_provider.assert_not_called()
        detect.assert_not_called()

    async def test_entry_written_during_the_probe_still_blocks_the_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SessionStart can land while the provider probe is in flight.

        The guard runs before an await; the hook writes from another process.
        Without a re-check the seed proceeds and clears the entry the hook has
        just made, and hook.py will not recreate one outside SessionStart.
        """
        seen: list[bool] = []

        async def _entry_appears(_window_id: str) -> bool:
            # Absent on the first check, present by the time the probe returns.
            seen.append(True)
            return len(seen) > 1

        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ".session_map_entry_may_exist",
            _entry_appears,
        )
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
            AsyncMock(return_value="claude"),
        )
        mock_sm = MagicMock()
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_manager", mock_sm
        )

        result = await _bootstrap_identity("@9", _window_ref(cwd="/project"))

        assert result is None
        mock_sm.set_window_provider.assert_not_called()

    async def test_shell_detection_writes_no_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare shell must not be stamped as the window's origin provider.

        ``set_window_provider`` seeds ``initial_provider_name`` from the value
        being written when the state is fresh, so bootstrapping "shell" would
        make ``_is_agent_origin`` False forever and the agent-exit recovery
        banner would never fire for that window again.
        """
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
            AsyncMock(return_value="shell"),
        )
        mock_sm = MagicMock()
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_manager",
            mock_sm,
        )

        result = await _bootstrap_identity(
            "@9", _window_ref(pane_current_command="zsh")
        )

        assert result is None
        mock_sm.set_window_provider.assert_not_called()

    async def test_detected_provider_creates_state_with_correct_cwd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
            AsyncMock(return_value="pi"),
        )
        mock_sm = MagicMock()
        mock_id_state = MagicMock()
        mock_id_state.get_identity.return_value = _identity("@9", cwd="/project")
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_manager",
            mock_sm,
        )
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.identity_state",
            mock_id_state,
        )

        result = await _bootstrap_identity("@9", _window_ref(cwd="/project"))

        mock_sm.set_window_provider.assert_called_once_with("@9", "pi", cwd="/project")
        assert result is not None

    async def test_discover_bootstraps_pi_and_passes_identity_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No window state → bootstrap creates it → discovery does not return False."""
        w = _window_ref(pane_current_command="pi", cwd="/project")
        set_provider_calls: list[tuple] = []

        # get_identity: first call (outer gate) returns None; second call (inside
        # _bootstrap_identity after set_window_provider) returns a valid identity.
        get_identity_seq = [None, _identity("@9")]
        mock_id_state = MagicMock()
        mock_id_state.get_identity.side_effect = lambda _: get_identity_seq.pop(0)

        mock_sm = MagicMock()
        mock_sm.set_window_provider.side_effect = lambda wid, prov, cwd=None: (
            set_provider_calls.append((wid, prov, cwd))
        )

        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.identity_state",
            mock_id_state,
        )
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.session_manager",
            mock_sm,
        )
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
            AsyncMock(return_value="pi"),
        )
        # Patch _detect_and_apply_provider to signal "agent exited", which makes
        # discover_and_register_transcript return True right after the gate —
        # unambiguous proof the identity gate was passed.
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery._detect_and_apply_provider",
            AsyncMock(return_value=True),
        )

        result = await discover_and_register_transcript("@9", _window=w)

        assert result is True  # gate passed, not False
        assert set_provider_calls == [("@9", "pi", "/project")]

    async def test_existing_identity_skips_bootstrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When window state already exists, _bootstrap_identity must not be called."""
        mock_id_state = MagicMock()
        mock_id_state.get_identity.return_value = _identity("@9")
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery.identity_state",
            mock_id_state,
        )
        monkeypatch.setattr(
            "ccgram.handlers.recovery.transcript_discovery._detect_and_apply_provider",
            AsyncMock(return_value=True),
        )

        w = _window_ref(pane_current_command="pi")
        with patch(
            "ccgram.handlers.recovery.transcript_discovery._bootstrap_identity"
        ) as spy:
            result = await discover_and_register_transcript("@9", _window=w)

        spy.assert_not_called()
        assert result is True
