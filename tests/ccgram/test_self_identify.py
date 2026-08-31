"""Tests for the backend-neutral hook identity resolver (Task 6).

Table-driven over the four cases the design calls out: tmux env, herdr env,
neither, and nested-session rejection (the last exercised through
``hook._locate_primary_window`` since nested detection is provider-gated there).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ccgram.multiplexer.self_identify import SelfIdentity, resolve_self_identity


def _fail_query(*_locator: str):
    raise AssertionError("tmux_query must not run without $TMUX_PANE")


class TestResolveSelfIdentity:
    @pytest.mark.parametrize(
        ("env", "tmux_result", "herdr_query", "expected"),
        [
            (
                {"TMUX_PANE": "%0"},
                ("ccgram:@0", "@0", "project", "/dev/ttys012"),
                None,
                SelfIdentity(
                    "tmux", "ccgram:@0", "@0", "project", pane_tty="/dev/ttys012"
                ),
            ),
            # herdr: an exact workspace/pane lookup returns one opaque target.
            (
                {
                    "HERDR_WORKSPACE_ID": "w2",
                    "HERDR_PANE_ID": "w2:p1",
                    "HERDR_SOCKET_PATH": "/tmp/herdr.sock",
                },
                None,
                lambda _workspace, _pane: "herdr-session-v1-target",
                SelfIdentity(
                    "herdr",
                    "herdr:herdr-session-v1-target",
                    "herdr-session-v1-target",
                    "",
                ),
            ),
            # herdr: no herdr_query → probe unavailable → None (symmetric with tmux)
            (
                {"HERDR_PANE_ID": "w2:p1", "HERDR_SOCKET_PATH": "/tmp/herdr.sock"},
                None,
                None,
                None,
            ),
            # herdr: herdr_query returns None (probe failure) → None (skip session_map write)
            (
                {
                    "HERDR_WORKSPACE_ID": "w2",
                    "HERDR_PANE_ID": "w2:p1",
                    "HERDR_SOCKET_PATH": "/tmp/herdr.sock",
                },
                None,
                lambda _workspace, _pane: None,
                None,
            ),
            ({}, None, None, None),
            ({"TMUX_PANE": "%0"}, None, None, None),
        ],
        ids=[
            "tmux",
            "herdr-with-query",
            "herdr-no-query-fallback",
            "herdr-query-fail-fallback",
            "neither",
            "tmux-query-fail",
        ],
    )
    def test_resolution_table(self, env, tmux_result, herdr_query, expected) -> None:
        ident = resolve_self_identity(
            env,
            tmux_query=lambda _pane: tmux_result,
            herdr_query=herdr_query,
        )
        assert ident == expected

    def test_herdr_without_workspace_fails_closed(self) -> None:
        assert (
            resolve_self_identity(
                {"HERDR_PANE_ID": "w0:p0"},
                tmux_query=_fail_query,
                herdr_query=lambda _workspace, _pane: "herdr-session-v1-target",
            )
            is None
        )

    def test_tmux_wins_when_both_present(self) -> None:
        # A herdr pane nested inside a tmux pane reports the outer tmux identity.
        env = {"TMUX_PANE": "%1", "HERDR_PANE_ID": "w2:p1"}
        ident = resolve_self_identity(
            env, tmux_query=lambda _pane: ("s:@1", "@1", "win", "/dev/ttys1")
        )
        assert ident is not None and ident.mux == "tmux"

    def test_neither_env_does_not_probe_tmux(self) -> None:
        assert resolve_self_identity({}, tmux_query=_fail_query) is None

    def test_agterm_session_id_resolves_without_a_probe(self) -> None:
        # The session UUID is the identity: agterm persists and restores it, so
        # unlike herdr there is no locator to resolve and nothing to fail on.
        env = {"AGTERM_SESSION_ID": "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"}
        ident = resolve_self_identity(env, tmux_query=_fail_query)
        assert ident is not None
        assert ident.mux == "agterm"
        assert ident.window_id == "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"
        assert ident.session_window_key == "agterm:157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"
        assert ident.pane_tty == ""

    def test_tmux_inside_agterm_reports_tmux(self) -> None:
        # Every shell agterm spawns inherits AGTERM_SESSION_ID, including one
        # running a nested tmux, so agterm must be the last branch checked or it
        # would claim panes belonging to the inner multiplexer.
        env = {"TMUX_PANE": "%1", "AGTERM_SESSION_ID": "157B4C8C"}
        ident = resolve_self_identity(
            env, tmux_query=lambda _pane: ("s:@1", "@1", "win", "/dev/ttys1")
        )
        assert ident is not None and ident.mux == "tmux"

    def test_herdr_inside_agterm_reports_herdr(self) -> None:
        env = {
            "HERDR_PANE_ID": "w2:p1",
            "HERDR_WORKSPACE_ID": "w2",
            "AGTERM_SESSION_ID": "157B4C8C",
        }
        ident = resolve_self_identity(
            env,
            tmux_query=_fail_query,
            herdr_query=lambda _workspace, _pane: "herdr-session-v1-target",
        )
        assert ident is not None and ident.mux == "herdr"

    def test_failed_herdr_probe_inside_agterm_does_not_fall_through(self) -> None:
        # A herdr pane whose probe fails must skip the session_map write, not
        # silently record the surrounding agterm session as its identity.
        env = {
            "HERDR_PANE_ID": "w2:p1",
            "HERDR_WORKSPACE_ID": "w2",
            "AGTERM_SESSION_ID": "157B4C8C",
        }
        assert (
            resolve_self_identity(
                env,
                tmux_query=_fail_query,
                herdr_query=lambda _workspace, _pane: None,
            )
            is None
        )


class TestResolveHerdrTarget:
    @staticmethod
    def _record(*, workspace_id: str = "w2", pane_id: str = "w2:p1") -> dict:
        return {
            "workspace_id": workspace_id,
            "pane_id": pane_id,
            "terminal_id": "term-1",
            "tab_id": "w2:t1",
            "agent_session": {
                "source": "claude",
                "agent": "claude",
                "kind": "session",
                "value": "session-1",
            },
        }

    def _resolve(
        self,
        monkeypatch,
        records: list[dict],
        provider_name: str | None = None,
    ) -> str | None:
        import ccgram.hook as hook

        monkeypatch.setattr(
            hook.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"result": {"agents": records}}),
                stderr="",
            ),
        )
        monkeypatch.setattr(
            hook,
            "get_multiplexer",
            lambda _name: SimpleNamespace(
                target_id_for_live_record=lambda _record: "herdr-session-v1-target"
            ),
        )
        return hook._resolve_herdr_target_id(
            "w2",
            "w2:p1",
            provider_name,  # type: ignore[arg-type]
        )

    def test_unique_workspace_pane_locator_returns_opaque_target(
        self, monkeypatch
    ) -> None:
        assert self._resolve(monkeypatch, [self._record()]) == "herdr-session-v1-target"

    def test_zero_workspace_pane_locator_matches_fail_closed(self, monkeypatch) -> None:
        assert self._resolve(monkeypatch, [self._record(pane_id="w2:p2")]) is None

    def test_duplicate_workspace_pane_locator_matches_fail_closed(
        self, monkeypatch
    ) -> None:
        assert self._resolve(monkeypatch, [self._record(), self._record()]) is None

    def test_nested_provider_in_live_pane_fails_closed(self, monkeypatch) -> None:
        assert self._resolve(monkeypatch, [self._record()], provider_name="pi") is None

    def test_live_provider_hook_resolves(self, monkeypatch) -> None:
        assert (
            self._resolve(monkeypatch, [self._record()], provider_name="claude")
            == "herdr-session-v1-target"
        )


class TestLocatePrimaryWindowThroughResolver:
    """`_locate_primary_window` routes through the resolver and keeps the
    tmux nested-session guard intact."""

    def test_primary_tmux_claude_accepted(self, monkeypatch) -> None:
        monkeypatch.setenv("TMUX_PANE", "%0")
        monkeypatch.setattr(
            "ccgram.hook._resolve_window_id",
            lambda _pane: ("ccgram:@0", "@0", "project", "/dev/ttys012"),
        )
        monkeypatch.setattr("ccgram.hook._is_nested_session", lambda _tty: False)
        from ccgram.hook import _locate_primary_window

        assert _locate_primary_window("sid", "Stop", "claude") == (
            "ccgram:@0",
            "@0",
            "project",
        )

    def test_nested_tmux_claude_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("TMUX_PANE", "%0")
        monkeypatch.setattr(
            "ccgram.hook._resolve_window_id",
            lambda _pane: ("ccgram:@0", "@0", "project", "/dev/ttys012"),
        )
        monkeypatch.setattr("ccgram.hook._is_nested_session", lambda _tty: True)
        from ccgram.hook import _locate_primary_window

        assert _locate_primary_window("sid", "Stop", "claude") is None

    def test_no_env_returns_none(self, monkeypatch) -> None:
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("HERDR_PANE_ID", raising=False)
        # Every shell agterm spawns exports this, so a suite run from inside
        # agterm would otherwise resolve an identity here.
        monkeypatch.delenv("AGTERM_SESSION_ID", raising=False)
        from ccgram.hook import _locate_primary_window

        assert _locate_primary_window("sid", "Stop", "claude") is None

    def test_agterm_session_resolves_through_the_hook(self, monkeypatch) -> None:
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("HERDR_PANE_ID", raising=False)
        monkeypatch.setenv("AGTERM_SESSION_ID", "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4")
        from ccgram.hook import _locate_primary_window

        located = _locate_primary_window("sid", "Stop", "claude")
        assert located is not None
        assert located[0] == "agterm:157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"

    def test_herdr_pane_resolves_to_session_target(self, monkeypatch) -> None:
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setenv("HERDR_WORKSPACE_ID", "w2")
        monkeypatch.setenv("HERDR_PANE_ID", "w2:p1")
        monkeypatch.setattr(
            "ccgram.hook._resolve_herdr_target_id",
            lambda _workspace, _pane, _provider: "herdr-session-v1-target",
        )
        from ccgram.hook import _locate_primary_window

        assert _locate_primary_window("sid", "Stop", "claude") == (
            "herdr:herdr-session-v1-target",
            "herdr-session-v1-target",
            "",
        )

    def test_herdr_pane_probe_failure_returns_none(self, monkeypatch) -> None:
        # probe returns None → resolve_self_identity returns None → hook skips write.
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setenv("HERDR_WORKSPACE_ID", "w2")
        monkeypatch.setenv("HERDR_PANE_ID", "w2:p1")
        monkeypatch.setattr(
            "ccgram.hook._resolve_herdr_target_id",
            lambda _workspace, _pane, _provider: None,
        )
        from ccgram.hook import _locate_primary_window

        assert _locate_primary_window("sid", "Stop", "claude") is None
