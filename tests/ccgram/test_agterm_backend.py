"""agterm adapter tests.

The backend shells out to ``agtermctl``; every test here feeds canned JSON
through the injected runner, so nothing needs a live agterm or its socket.

Two behaviours get the most attention because they are the ones that would
silently corrupt a session instead of failing loudly: an explicit ``--pane`` on
every read and write (agterm's read and write defaults address different
surfaces), and the tmux key-name translation callers rely on for ``literal=False``
sends.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import pytest

from ccgram.multiplexer.agterm import AgtermManager, _key_to_bytes
from ccgram.multiplexer.base import AgentStatus, Multiplexer


class FakeAgtermctl:
    """Prefix-matching canned ``agtermctl`` runner."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []
        self.responses: dict[tuple[str, ...], tuple[int, str, str]] = {}
        self.default = (1, "", "no canned response")

    def on(
        self, *prefix: str, rc: int = 0, out: str = "", err: str = ""
    ) -> FakeAgtermctl:
        self.responses[prefix] = (rc, out, err)
        return self

    def ok(self, *prefix: str, result: dict | None = None) -> FakeAgtermctl:
        return self.on(*prefix, out=json.dumps({"ok": True, "result": result or {}}))

    async def __call__(
        self, args: Sequence[str], stdin_text: str | None = None
    ) -> tuple[int, str, str]:
        call = list(args)
        self.calls.append(call)
        self.stdins.append(stdin_text)
        matching = [key for key in self.responses if call[: len(key)] == list(key)]
        return self.responses[max(matching, key=len)] if matching else self.default

    def call_with(self, *prefix: str) -> list[str] | None:
        """Return the first recorded call starting with *prefix*."""
        for call in self.calls:
            if call[: len(prefix)] == list(prefix):
                return call
        return None


SESSION_A = "157B4C8C-EFAE-40C2-BA54-9A5D7FD8B5E4"
SESSION_B = "47B39C37-222B-452A-B6D4-4AE4E920E2FB"
WORKSPACE = "6BE961A6-8F8E-40B2-AF6E-821711318162"


WINDOW_A = "C24BC79B-E218-496F-A4D8-1B0A3E513FC3"
WINDOW_B = "7F4089F9-4AAB-4DC8-8415-88CA8D9A3E07"


def _windows(*ids: str, closed: tuple[str, ...] = ()) -> str:
    """A ``window list`` envelope; ``closed`` entries carry ``open: false``."""
    entries = [{"id": wid, "name": wid[:8], "open": True} for wid in ids or (WINDOW_A,)]
    entries += [{"id": wid, "name": wid[:8], "open": False} for wid in closed]
    return json.dumps({"ok": True, "result": {"windows": entries}})


def _tree(*sessions: dict, workspace: str = "code") -> str:
    return _multi_tree({workspace: list(sessions)})


def _multi_tree(by_workspace: dict[str, list[dict]]) -> str:
    """A tree envelope with one workspace per key."""
    return json.dumps(
        {
            "ok": True,
            "result": {
                "tree": {
                    "workspaces": [
                        {
                            "id": WORKSPACE
                            if index == 0
                            else f"{WORKSPACE[:-1]}{index}",
                            "name": name,
                            "active": index == 0,
                            "sessions": sessions,
                        }
                        for index, (name, sessions) in enumerate(by_workspace.items())
                    ]
                }
            },
        }
    )


def _session(
    session_id: str = SESSION_A,
    *,
    name: str = "ccgram",
    cwd: str = "/Users/dmitry/code/ccgram",
    foreground: Sequence[str] | None = ("claude",),
    title: str = "",
    status: str | None = None,
    realized: bool = True,
) -> dict:
    node: dict = {"id": session_id, "name": name, "cwd": cwd, "realized": realized}
    if foreground is not None:
        node["foreground"] = list(foreground)
    if title:
        node["title"] = title
    if status:
        node["status"] = status
    return node


def _manager(
    fake: FakeAgtermctl,
    *,
    own: str = "",
    workspaces: tuple[str, ...] | None = None,
) -> AgtermManager:
    # own="" by default so an ambient AGTERM_SESSION_ID (this suite may well run
    # inside agterm) cannot silently exclude a fixture session; workspaces=None
    # so the fixture workspace name is not a hidden precondition of every test.
    return AgtermManager(
        socket_path="/tmp/agterm.sock",
        runner=fake,
        own_session_id=own,
        workspaces=workspaces,
    )


# ── capabilities and protocol ──────────────────────────────────────────


def test_satisfies_the_multiplexer_protocol() -> None:
    assert isinstance(_manager(FakeAgtermctl()), Multiplexer)


def test_capability_values() -> None:
    caps = _manager(FakeAgtermctl()).capabilities
    assert caps.name == "agterm"
    # agterm persists and restores session UUIDs, so no alias reconciliation.
    assert caps.ids_stable_across_restart is True
    assert caps.exposes_pane_tty is False
    assert caps.read_max_lines is None
    assert caps.self_identify_env == "AGTERM_SESSION_ID"
    assert caps.supports_event_stream is False
    assert caps.native_worktrees is False
    assert caps.supports_workspace_selection is True
    assert caps.native_topic_targets is False
    assert caps.native_agent_status is True


def test_agterm_native_status_is_reported() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(status="active")))
    )

    status = asyncio.run(_manager(fake).agent_status(SESSION_A))

    assert status == AgentStatus(state="active")


def test_agterm_native_status_is_none_when_unreported() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
    )

    assert asyncio.run(_manager(fake).agent_status(SESSION_A)) is None


# ── key translation ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("Enter", "\r"),
        ("Escape", "\x1b"),
        ("Tab", "\t"),
        ("BSpace", "\x7f"),
        ("Up", "\x1b[A"),
        ("Down", "\x1b[B"),
        ("Left", "\x1b[D"),
        ("Right", "\x1b[C"),
        ("C-c", "\x03"),
        ("C-d", "\x04"),
        ("C-u", "\x15"),
        ("M-Enter", "\x1b\r"),
        ("C-[", "\x1b"),
    ],
)
def test_key_names_translate_to_pty_bytes(token: str, expected: str) -> None:
    assert _key_to_bytes(token) == expected


def test_meta_plus_a_printable_character_is_esc_then_that_character() -> None:
    """The shipped Think toolbar button sends ``M-t``; no key table covers it."""
    assert _key_to_bytes("M-t") == "\x1bt"
    assert _key_to_bytes("M-b") == "\x1bb"
    assert _key_to_bytes("M-.") == "\x1b."


async def test_one_unknown_token_fails_the_whole_multi_key_send() -> None:
    """All or nothing: a half-applied key sequence is worse than a refusal."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "Up Bogus Down", literal=False) is False
    assert fake.call_with("session", "type") is None


async def test_a_multi_key_send_of_known_tokens_goes_out_as_one_call() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "Down Down", enter=True, literal=False)
    assert fake.stdins[-1] == "\x1b[B\x1b[B\r"
    assert len([c for c in fake.calls if c[:2] == ["session", "type"]]) == 1


def test_unknown_key_name_is_rejected_not_typed_literally() -> None:
    assert _key_to_bytes("Nonsense") is None
    assert _key_to_bytes("C-toolong") is None


async def test_send_keys_refuses_an_unknown_name_instead_of_typing_it() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "Bogus", literal=False) is False
    assert fake.call_with("session", "type") is None


async def test_send_keys_sends_translated_bytes_on_stdin() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "Escape", enter=False, literal=False)
    assert fake.stdins[-1] == "\x1b"


async def test_literal_send_types_then_submits_as_two_injections() -> None:
    """The tmux backend types, pauses, then submits; batching them races the TUI."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "hello")
    types = [c for c in fake.calls if c[:2] == ["session", "type"]]
    assert len(types) == 2
    assert fake.stdins[-2] == "hello"
    assert fake.stdins[-1] == "\r"


async def test_literal_send_without_enter_does_not_submit() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "hello", enter=False)
    assert fake.stdins == ["hello"]


async def test_multi_line_text_is_bracketed_so_newlines_are_not_submits() -> None:
    """Without markers each interior newline submits, splitting one message."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    assert await _manager(fake).send(SESSION_A, "line one\nline two")
    assert fake.stdins[-2] == "\x1b[200~line one\nline two\x1b[201~"
    assert fake.stdins[-1] == "\r"


async def test_single_line_text_is_not_bracketed() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    await _manager(fake).send(SESSION_A, "one line")
    assert fake.stdins[-2] == "one line"


async def test_a_failed_type_does_not_submit() -> None:
    fake = FakeAgtermctl().on("session", "type", rc=1, err="gone")
    assert await _manager(fake).send(SESSION_A, "hello") is False
    assert len([c for c in fake.calls if c[:2] == ["session", "type"]]) == 1


# ── the explicit-pane contract ─────────────────────────────────────────


async def test_every_injection_names_its_pane() -> None:
    """agterm types into the main pane by default but READS the focused one."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    await _manager(fake).send(SESSION_A, "hi")
    call = fake.call_with("session", "type")
    assert call is not None
    assert "--pane" in call
    assert call[call.index("--pane") + 1] == "left"
    # stdin, never argv: control bytes must not pass through argument parsing.
    assert "--stdin" in call


async def test_every_capture_names_its_pane() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "text", result={"text": "output"})
    )
    await _manager(fake).capture_pane(SESSION_A)
    call = fake.call_with("session", "text")
    assert call is not None
    assert call[call.index("--pane") + 1] == "left"


async def test_capture_scrollback_passes_the_line_budget() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "text", result={"text": "a\nb"})
    )
    captured = await _manager(fake).capture_scrollback(SESSION_A, lines=500)
    assert captured is not None
    # agterm has no line cap, so a capture is never reported as truncated.
    assert captured.truncated is False
    call = fake.call_with("session", "text")
    assert call is not None
    assert call[call.index("--lines") + 1] == "500"


# ── window listing and identity ────────────────────────────────────────


async def test_list_windows_maps_sessions_to_neutral_refs() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_tree(
                _session(SESSION_A, name="ccgram", foreground=["claude", "--resume"]),
                _session(SESSION_B, name="blog", foreground=None),
            ),
        )
    )
    windows = await _manager(fake).list_windows()
    assert [w.window_id for w in windows] == [SESSION_A, SESSION_B]  # noqa: E501
    assert windows[0].window_name == "ccgram"
    assert windows[0].pane_current_command == "claude"
    # A session sitting at a shell prompt reports no foreground argv at all.
    assert windows[1].pane_current_command == ""
    # agterm exposes neither a tty nor pane geometry over the control channel.
    assert windows[0].pane_tty == ""
    assert (windows[0].pane_width, windows[0].pane_height) == (0, 0)


async def test_find_window_by_id_is_case_insensitive() -> None:
    """A caller may round-trip the UUID lowercased; agterm reports it uppercase."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A)))
    )
    found = await _manager(fake).find_window_by_id(SESSION_A.lower())
    assert found is not None
    assert found.window_id == SESSION_A


async def test_find_window_by_id_returns_none_when_gone() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_B)))
    )
    assert await _manager(fake).find_window_by_id(SESSION_A) is None


async def test_list_workspaces_reports_agterm_workspaces() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
    )
    workspaces = await _manager(fake).list_workspaces()
    assert [w.workspace_id for w in workspaces] == [WORKSPACE]
    assert workspaces[0].label == "code"
    assert workspaces[0].cwd == "/Users/dmitry/code/ccgram"


# ── failure handling ───────────────────────────────────────────────────


async def test_unreachable_socket_reports_empty_not_an_exception() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", rc=1, err="is agterm running?")
    )
    manager = _manager(fake)
    assert await manager.list_windows() == []
    assert await manager.list_workspaces() == []
    assert await manager.find_window_by_id(SESSION_A) is None


async def test_ensure_session_raises_when_agterm_is_not_running() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", rc=1, err="is agterm running?")
    )
    with pytest.raises(RuntimeError, match="agterm is not reachable"):
        await _manager(fake).ensure_session()


async def test_ensure_session_passes_against_a_live_socket() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
    )
    await _manager(fake).ensure_session()


async def test_an_error_envelope_is_a_failure_even_on_a_zero_exit() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=json.dumps({"ok": False, "error": "no such session"}))
    )
    assert await _manager(fake).list_windows() == []


async def test_non_json_output_is_a_failure() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out="not json at all")
    )
    assert await _manager(fake).list_windows() == []


async def test_send_to_a_vanished_session_reports_failure() -> None:
    """agterm rejects an unknown --target itself, so there is no preflight.

    One call, not two: a probe would double every send and still leave a gap
    between the check and the injection.
    """
    fake = FakeAgtermctl().on("session", "type", rc=1, err="no such session")
    assert await _manager(fake).send(SESSION_A, "hi") is False
    assert [c for c in fake.calls if c[0] == "tree"] == []


async def test_capture_of_a_vanished_session_reports_failure() -> None:
    fake = FakeAgtermctl().on("session", "text", rc=1, err="no such session")
    assert await _manager(fake).capture_pane(SESSION_A) is None
    assert [c for c in fake.calls if c[0] == "tree"] == []


# ── panes: no durable sibling handle ───────────────────────────────────


async def test_split_window_returns_none() -> None:
    """agterm promotes the split survivor into the main pane on primary exit.

    That makes the ``right`` role an unstable identity, and the Protocol says a
    backend without a safe sibling handle returns None instead of a raw locator.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
    )
    assert await _manager(fake).split_window(SESSION_A) is None


async def test_list_panes_returns_no_handles() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
    )
    assert await _manager(fake).list_panes(SESSION_A) == []


async def test_raw_pane_operations_are_rejected() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    manager = _manager(fake)
    assert (
        await manager.send_to_pane(
            f"surface:{SESSION_A}:right", "hi", window_id=SESSION_A
        )
        is False
    )
    assert (
        await manager.capture_pane_by_id(
            f"surface:{SESSION_A}:right", window_id=SESSION_A
        )
        is None
    )
    assert fake.call_with("session", "type") is None


async def test_pane_operations_authorised_by_the_session_id_go_through() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
        .ok("session", "text", result={"text": "out"})
    )
    manager = _manager(fake)
    assert await manager.send_to_pane(SESSION_A, "hi", window_id=SESSION_A) is True
    assert await manager.capture_pane_by_id(SESSION_A, window_id=SESSION_A) == "out"


# ── creation, renaming, teardown ───────────────────────────────────────


async def test_create_window_expands_tilde_before_calling_agterm(
    tmp_path, monkeypatch
) -> None:
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, name="project")))
    )

    ok, _message, _name, _window_id = await _manager(fake).create_window(
        "~/project", start_agent=False
    )

    assert ok is True
    call = fake.call_with("session", "new")
    assert call is not None
    assert call[call.index("--cwd") + 1] == str(work_dir)


async def test_create_window_makes_a_shell_and_types_the_command(tmp_path) -> None:
    """Never agterm's --command: it uses the GUI PATH and dies with the agent.

    A ``~/.local/bin`` agent exits 127 under it, and because --command replaces
    the login shell the session then closes, so creation reports success and
    the window is gone a moment later.
    """
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, name="ccgram")))
        .ok("session", "type")
    )
    ok, message, name, window_id = await _manager(fake).create_window(
        str(tmp_path), window_name="ccgram", launch_command="claude"
    )
    assert (ok, message, name, window_id) == (True, "", "ccgram", SESSION_A)
    call = fake.call_with("session", "new")
    assert call is not None
    assert "--command" not in call
    # An automated creation must not move the selection away from the user.
    assert "--no-select" in call
    assert call[call.index("--cwd") + 1] == str(tmp_path)
    typed = [
        stdin
        for c, stdin in zip(fake.calls, fake.stdins, strict=True)
        if c[:2] == ["session", "type"]
    ]
    assert typed == ["claude", "\r"]


async def test_create_window_appends_agent_args_to_the_typed_command(tmp_path) -> None:
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A)))
        .ok("session", "type")
    )
    await _manager(fake).create_window(
        str(tmp_path), launch_command="claude", agent_args="--resume"
    )
    typed = [
        stdin
        for call, stdin in zip(fake.calls, fake.stdins, strict=True)
        if call[:2] == ["session", "type"]
    ]
    assert typed == ["claude --resume", "\r"]


async def test_create_window_fails_when_the_terminal_never_comes_up(tmp_path) -> None:
    """agterm answers `ok` before the surface exists; sending too early is lost."""
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, realized=False)))
    )
    ok, message, _name, window_id = await _manager(fake).create_window(str(tmp_path))
    assert ok is False
    assert "never came up" in message
    assert window_id == ""


async def test_create_window_reports_a_failed_agent_launch(tmp_path) -> None:
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A)))
        .on("session", "type", rc=1, err="gone")
    )
    ok, message, _name, _window_id = await _manager(fake).create_window(
        str(tmp_path), launch_command="claude"
    )
    assert ok is False
    assert "claude" in message


async def test_create_window_pins_an_explicit_workspace() -> None:
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A)))
    )
    await _manager(fake).create_window("/tmp", workspace_id=WORKSPACE)
    call = fake.call_with("session", "new")
    assert call is not None
    assert call[call.index("--workspace") + 1] == WORKSPACE


async def test_create_window_reports_failure_when_agterm_refuses() -> None:
    fake = FakeAgtermctl().on("session", "new", rc=1, err="display asleep")
    ok, message, _name, window_id = await _manager(fake).create_window("/tmp")
    assert ok is False
    assert window_id == ""
    assert message


async def test_create_topic_target_uses_the_session_uuid_as_the_target(
    tmp_path,
) -> None:
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, name="ccgram")))
        .ok("session", "type")
    )
    target = await _manager(fake).create_topic_target(
        str(tmp_path), launch_command="claude", workspace_id=None
    )
    assert target.target_id == SESSION_A
    assert target.window_id == SESSION_A
    assert target.label == "ccgram"


async def test_create_topic_target_raises_on_failure() -> None:
    fake = FakeAgtermctl().on("session", "new", rc=1, err="nope")
    with pytest.raises(RuntimeError):
        await _manager(fake).create_topic_target(
            "/tmp", launch_command=None, workspace_id=None
        )


async def test_create_worktree_window_is_unsupported() -> None:
    ok, message, _name, _window_id = await _manager(
        FakeAgtermctl()
    ).create_worktree_window("/repo", "/wt", "branch")
    assert ok is False
    assert message


async def test_kill_and_rename_target_the_session() -> None:
    fake = FakeAgtermctl().ok("session", "close").ok("session", "rename")
    manager = _manager(fake)
    assert await manager.kill_window(SESSION_A) is True
    assert await manager.rename_window(SESSION_A, "renamed") is True
    rename = fake.call_with("session", "rename")
    assert rename is not None
    assert rename[2] == "renamed"
    assert rename[rename.index("--target") + 1] == SESSION_A


# ── foreground, title, events ──────────────────────────────────────────


async def test_foreground_reports_argv_with_no_pid_or_tty() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(foreground=["claude", "--resume", "abc"])))
    )
    fg = await _manager(fake).foreground(SESSION_A)
    assert fg is not None
    assert fg.argv == ["claude", "--resume", "abc"]
    # Provider detection classifies on argv; agterm exposes no pid, pgid or tty.
    assert (fg.pid, fg.pgid, fg.tty) == (0, 0, "")


async def test_foreground_is_none_at_a_shell_prompt() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(foreground=None)))
    )
    assert await _manager(fake).foreground(SESSION_A) is None


async def test_pane_dims_are_unavailable() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
    )
    assert await _manager(fake).pane_dims(SESSION_A) is None


async def test_get_pane_title_reads_the_osc_title() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(title="ssh prod")))
    )
    assert await _manager(fake).get_pane_title(SESSION_A) == "ssh prod"


async def test_stamp_pane_title_is_a_no_op() -> None:
    fake = FakeAgtermctl()
    await _manager(fake).stamp_pane_title(SESSION_A, "claude")
    assert fake.calls == []


async def test_watch_events_yields_nothing() -> None:
    fake = FakeAgtermctl()
    assert [event async for event in _manager(fake).watch_events([SESSION_A])] == []


# ── reconciliation listing ─────────────────────────────────────────────


async def test_reconciliation_listing_returns_none_when_unreachable() -> None:
    """None means unknown. Returning [] here would prune live state as dead."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", rc=1, err="is agterm running?")
    )
    assert await _manager(fake).list_windows_for_reconciliation() is None


async def test_reconciliation_listing_returns_none_on_a_bad_envelope() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=json.dumps({"ok": False, "error": "boom"}))
    )
    assert await _manager(fake).list_windows_for_reconciliation() is None


async def test_reconciliation_listing_returns_none_on_non_json() -> None:
    fake = (
        FakeAgtermctl().on("window", "list", out=_windows()).on("tree", out="not json")
    )
    assert await _manager(fake).list_windows_for_reconciliation() is None


async def test_reconciliation_listing_returns_empty_for_a_confirmed_empty_tree() -> (
    None
):
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=json.dumps({"ok": True, "result": {"tree": {"workspaces": []}}}),
        )
    )
    assert await _manager(fake).list_windows_for_reconciliation() == []


async def test_list_windows_flattens_an_unavailable_listing_to_empty() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", rc=1, err="down")
    )
    assert await _manager(fake).list_windows() == []


# ── create_window directory validation ─────────────────────────────────


async def test_create_window_rejects_a_missing_directory_without_calling_agterm() -> (
    None
):
    """agterm accepts any cwd and can answer ok before the surface fails."""
    fake = FakeAgtermctl()
    ok, message, _name, window_id = await _manager(fake).create_window(
        "/definitely/not/here"
    )
    assert ok is False
    assert "does not exist" in message
    assert window_id == ""
    assert fake.calls == []


async def test_create_window_rejects_a_file_path(tmp_path) -> None:
    target = tmp_path / "a-file"
    target.write_text("")
    fake = FakeAgtermctl()
    ok, message, _name, _window_id = await _manager(fake).create_window(str(target))
    assert ok is False
    assert "Not a directory" in message


async def test_create_window_skips_the_name_read_when_given_one(tmp_path) -> None:
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, name="auto-basename")))
    )
    ok, _message, name, _window_id = await _manager(fake).create_window(
        str(tmp_path), window_name="explicit"
    )
    # An explicit name cannot differ from what agterm stored, so the name is
    # taken as given even though the realization poll reads the tree.
    assert (ok, name) == (True, "explicit")


async def test_create_window_falls_back_to_the_cwd_basename(tmp_path) -> None:
    """An unnamed session must not yield an empty label."""
    fake = (
        FakeAgtermctl()
        .ok("session", "new", result={"id": SESSION_A})
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, name="")))
    )
    ok, _message, name, _window_id = await _manager(fake).create_window(str(tmp_path))
    assert ok is True
    assert name == tmp_path.name


# ── discovery exclusions ───────────────────────────────────────────────


async def test_listings_exclude_the_session_ccgram_itself_runs_in() -> None:
    """ccgram's own session is never offered, in either listing.

    The tmux backend skips ``config.own_window_id`` for this reason; without
    the equivalent the bot binds a topic to its own terminal. This exclusion is
    not the ``topic_eligible`` verdict, which governs unattended adoption only:
    a picker selection is an explicit bind and may name a window this listing
    would not auto-adopt. Binding the bot's own session is wrong either way.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A), _session(SESSION_B, name="blog")))
    )
    manager = _manager(fake, own=SESSION_A)

    # Gone from the selection listing: the picker must never offer it.
    assert [w.window_id for w in await manager.list_windows()] == [SESSION_B]

    # Present in the complete listing, because it exists — but refused for
    # adoption there, which is what keeps discovery off it.
    complete = await manager.list_windows_for_reconciliation()
    assert complete is not None
    assert {w.window_id: w.topic_eligible for w in complete} == {
        SESSION_A: False,
        SESSION_B: True,
    }


async def test_own_session_exclusion_is_case_insensitive() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A)))
    )
    assert await _manager(fake, own=SESSION_A.lower()).list_windows() == []


async def test_listings_exclude_underscore_prefixed_names() -> None:
    """tmux's hidden-window convention, which also covers herdr's __…__ form."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_tree(
                _session(SESSION_A, name="_scratch"),
                _session(SESSION_B, name="__main__"),
            ),
        )
    )
    assert await _manager(fake).list_windows() == []


async def test_reconciliation_listing_keeps_what_discovery_would_exclude() -> None:
    """Cleanup asks what exists; discovery asks what may be adopted.

    Excluding here would let a bound session read as gone the moment it was
    renamed with a leading underscore or moved to another workspace.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A), _session(SESSION_B, name="_hidden")))
    )
    windows = await _manager(fake, own=SESSION_A).list_windows_for_reconciliation()
    assert windows is not None
    assert sorted(w.window_id for w in windows) == sorted([SESSION_A, SESSION_B])
    # ...while discovery still refuses both.
    assert await _manager(fake, own=SESSION_A).list_windows() == []


async def test_an_excluded_session_is_still_addressable_by_id() -> None:
    """Exclusion governs discovery, not operations on a known id."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A)))
    )
    found = await _manager(fake, own=SESSION_A).find_window_by_id(SESSION_A)
    assert found is not None


# ── multi-window tree ──────────────────────────────────────────────────


class WindowScopedFake(FakeAgtermctl):
    """Answers ``tree --window <id>`` per window, like agterm does."""

    def __init__(self, per_window: dict[str, str]) -> None:
        super().__init__()
        self.per_window = per_window

    async def __call__(
        self, args: Sequence[str], stdin_text: str | None = None
    ) -> tuple[int, str, str]:
        call = list(args)
        if call[:1] == ["tree"] and "--window" in call:
            self.calls.append(call)
            self.stdins.append(stdin_text)
            wid = call[call.index("--window") + 1]
            canned = self.per_window.get(wid)
            return (0, canned, "") if canned else (1, "", "window not open")
        return await super().__call__(args, stdin_text)


async def test_tree_merges_every_open_window() -> None:
    """`tree` alone answers for the frontmost window only.

    Trusting it would hide every session in any other window while looking
    like a complete listing, and reconciliation would prune them as dead.
    """
    fake = WindowScopedFake(
        {
            WINDOW_A: _tree(_session(SESSION_A, name="in-window-a")),
            WINDOW_B: _tree(_session(SESSION_B, name="in-window-b")),
        }
    ).on("window", "list", out=_windows(WINDOW_A, WINDOW_B))
    windows = await _manager(fake).list_windows()
    assert sorted(w.window_name for w in windows) == ["in-window-a", "in-window-b"]
    # never the bare, frontmost-scoped form
    assert all("--window" in c for c in fake.calls if c[0] == "tree")


async def test_a_session_in_another_window_is_findable() -> None:
    fake = WindowScopedFake(
        {WINDOW_A: _tree(_session(SESSION_A)), WINDOW_B: _tree(_session(SESSION_B))}
    ).on("window", "list", out=_windows(WINDOW_A, WINDOW_B))
    assert await _manager(fake).find_window_by_id(SESSION_B) is not None


async def test_closed_windows_are_not_queried() -> None:
    """agterm answers `window not open` for a closed window's tree."""
    fake = WindowScopedFake({WINDOW_A: _tree(_session(SESSION_A))}).on(
        "window", "list", out=_windows(WINDOW_A, closed=(WINDOW_B,))
    )
    assert len(await _manager(fake).list_windows()) == 1
    assert not [c for c in fake.calls if c[0] == "tree" and WINDOW_B in c]


async def test_one_unreadable_window_makes_the_whole_listing_unavailable() -> None:
    """A partial tree is indistinguishable from sessions having gone away."""
    fake = WindowScopedFake({WINDOW_A: _tree(_session(SESSION_A))}).on(
        "window", "list", out=_windows(WINDOW_A, WINDOW_B)
    )
    assert await _manager(fake).list_windows_for_reconciliation() is None


async def test_a_failed_window_list_makes_the_listing_unavailable() -> None:
    fake = FakeAgtermctl().on("window", "list", rc=1, err="down")
    assert await _manager(fake).list_windows_for_reconciliation() is None


# ── created sessions are never abandoned ───────────────────────────────


async def test_an_unrealised_session_is_closed_not_left_behind(tmp_path) -> None:
    """The caller never learns the id, so only this method can clean it up."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .ok("session", "new", result={"id": SESSION_A})
        .on("tree", out=_tree(_session(SESSION_A, realized=False)))
        .ok("session", "close")
    )
    ok, _message, _name, _window_id = await _manager(fake).create_window(str(tmp_path))
    assert ok is False
    close = fake.call_with("session", "close")
    assert close is not None
    assert close[close.index("--target") + 1] == SESSION_A


async def test_a_session_whose_agent_would_not_start_is_closed(tmp_path) -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .ok("session", "new", result={"id": SESSION_A})
        .on("tree", out=_tree(_session(SESSION_A)))
        .on("session", "type", rc=1, err="gone")
        .ok("session", "close")
    )
    ok, _message, _name, _window_id = await _manager(fake).create_window(
        str(tmp_path), launch_command="claude"
    )
    assert ok is False
    assert fake.call_with("session", "close") is not None


# ── raw sends bypass the TUI shaping ───────────────────────────────────


async def test_a_raw_multi_line_send_is_not_bracketed() -> None:
    """A shell without DEC 2004 would echo the markers as text.

    `!`-prefixed shell commands reach a pane through the raw path.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    await _manager(fake).send(SESSION_A, "one\ntwo", raw=True)
    assert fake.stdins[-2] == "one\ntwo"


async def test_a_non_raw_multi_line_send_is_still_bracketed() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session()))
        .ok("session", "type")
    )
    await _manager(fake).send(SESSION_A, "one\ntwo")
    assert fake.stdins[-2] == "\x1b[200~one\ntwo\x1b[201~"


async def test_a_failed_cleanup_close_is_reported_not_swallowed(tmp_path) -> None:
    """The outage that fails creation usually fails the close as well.

    The caller never learns the id, so a silent failure leaves a session
    nobody can reach and discovery later adopts.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .ok("session", "new", result={"id": SESSION_A})
        .on("tree", out=_tree(_session(SESSION_A, realized=False)))
        .on("session", "close", rc=1, err="socket down")
    )
    ok, message, _name, _window_id = await _manager(fake).create_window(str(tmp_path))
    assert ok is False
    assert "may still be open" in message


# ── workspace scope ────────────────────────────────────────────────────


async def test_discovery_is_scoped_to_the_configured_workspaces() -> None:
    """agterm has no per-app container, so without a scope every session
    the user has open would surface as a Telegram topic."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_multi_tree(
                {
                    "ccgram": [_session(SESSION_A, name="agent")],
                    "personal": [_session(SESSION_B, name="my-own-shell")],
                }
            ),
        )
    )
    windows = await _manager(fake, workspaces=("ccgram",)).list_windows()
    assert [w.window_name for w in windows] == ["agent"]


async def test_workspace_matching_is_case_insensitive() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_multi_tree({"CCGram": [_session(SESSION_A, name="agent")]}))
    )
    assert len(await _manager(fake, workspaces=("ccgram",)).list_windows()) == 1


async def test_several_workspaces_can_be_scoped() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_multi_tree(
                {
                    "ccgram": [_session(SESSION_A, name="a")],
                    "work": [_session(SESSION_B, name="b")],
                    "personal": [_session(WORKSPACE, name="c")],
                }
            ),
        )
    )
    windows = await _manager(fake, workspaces=("ccgram", "work")).list_windows()
    assert sorted(w.window_name for w in windows) == ["a", "b"]


async def test_a_none_scope_adopts_from_every_workspace() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_multi_tree(
                {"ccgram": [_session(SESSION_A)], "personal": [_session(SESSION_B)]}
            ),
        )
    )
    assert len(await _manager(fake, workspaces=None).list_windows()) == 2


async def test_reconciliation_ignores_the_workspace_scope() -> None:
    """Cleanup asks what exists, not what may be adopted.

    A bound session the user dragged to another workspace must not read as
    gone and be pruned.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_multi_tree(
                {
                    "ccgram": [_session(SESSION_A)],
                    "personal": [_session(SESSION_B, name="_hidden")],
                }
            ),
        )
    )
    windows = await _manager(
        fake, own=SESSION_A, workspaces=("ccgram",)
    ).list_windows_for_reconciliation()
    assert windows is not None
    assert sorted(w.window_id for w in windows) == sorted([SESSION_A, SESSION_B])


# ── adoption travels on the window ─────────────────────────────────────


async def test_reconciliation_listing_marks_what_discovery_may_not_adopt() -> None:
    """The listing stays complete, but each window carries its verdict.

    session_monitor hands the reconciliation listing straight to discovery, so
    a filter that lives only in list_windows never runs for discovery at all.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_multi_tree(
                {
                    "code": [_session(SESSION_A, name="agent")],
                    "personal": [_session(SESSION_B, name="elsewhere")],
                }
            ),
        )
    )
    windows = await _manager(
        fake, workspaces=("code",)
    ).list_windows_for_reconciliation()
    assert windows is not None
    by_name = {w.window_name: w.topic_eligible for w in windows}
    # both present, so cleanup still sees everything that exists
    assert set(by_name) == {"agent", "elsewhere"}
    # but only the in-scope one may be adopted
    assert by_name == {"agent": True, "elsewhere": False}


async def test_own_session_is_marked_ineligible_in_the_reconciliation_listing() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A), _session(SESSION_B, name="other")))
    )
    windows = await _manager(
        fake, own=SESSION_A, workspaces=None
    ).list_windows_for_reconciliation()
    assert windows is not None
    assert {w.window_id: w.topic_eligible for w in windows} == {
        SESSION_A: False,
        SESSION_B: True,
    }


async def test_underscore_named_sessions_are_marked_ineligible() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, name="_scratch")))
    )
    windows = await _manager(fake, workspaces=None).list_windows_for_reconciliation()
    assert windows is not None
    assert windows[0].topic_eligible is False


async def test_find_window_by_id_is_not_gated_by_eligibility() -> None:
    """Addressing a known id is not discovery: an out-of-scope window stays drivable."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_multi_tree({"personal": [_session(SESSION_A)]}))
    )
    found = await _manager(fake, workspaces=("code",)).find_window_by_id(SESSION_A)
    assert found is not None
    assert found.window_id == SESSION_A


async def test_a_non_agent_session_in_scope_is_present_but_ineligible() -> None:
    """Present for cleanup, refused for adoption.

    agterm reports whatever holds the pane, so an in-scope session running an
    editor or a build must not be adopted, while still existing as far as
    reconciliation is concerned.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_tree(
                _session(SESSION_A, name="editing", foreground=["vim", "notes.md"]),
                _session(SESSION_B, name="agent", foreground=["claude"]),
            ),
        )
    )
    windows = await _manager(fake).list_windows_for_reconciliation()
    assert windows is not None
    assert {w.window_name: w.topic_eligible for w in windows} == {
        "editing": False,
        "agent": True,
    }


async def test_a_non_agent_session_in_scope_is_still_offered_for_explicit_binding() -> (
    None
):
    """The selection listing narrows by visibility, never by adoptability.

    This is the case the three-question contract rests on: `list_windows`
    keeps an in-scope session running an editor, carrying
    ``topic_eligible=False``, so a user can bind it from the picker while
    unattended discovery — which reads the reconciliation listing and this
    verdict — leaves it alone. Filtering it out here would silently remove a
    session the user can see in agterm from the picker.
    """
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_tree(
                _session(SESSION_A, name="editing", foreground=["vim", "notes.md"]),
                _session(SESSION_B, name="agent", foreground=["claude"]),
            ),
        )
    )

    windows = await _manager(fake).list_windows()

    assert {w.window_name: w.topic_eligible for w in windows} == {
        "editing": False,
        "agent": True,
    }


async def test_a_wrapped_agent_is_eligible() -> None:
    """codex runs as ``node .../codex``; argv[0] alone would reject it."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_tree(
                _session(
                    SESSION_A,
                    name="codex-session",
                    foreground=["node", "/opt/homebrew/bin/codex", "--yolo"],
                )
            ),
        )
    )
    windows = await _manager(fake).list_windows_for_reconciliation()
    assert windows is not None
    assert windows[0].topic_eligible is True


async def test_a_build_running_under_node_is_not_an_agent() -> None:
    """The wrapper skip must not turn every node process into an agent."""
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on(
            "tree",
            out=_tree(_session(SESSION_A, foreground=["node", "server.js"])),
        )
    )
    windows = await _manager(fake).list_windows_for_reconciliation()
    assert windows is not None
    assert windows[0].topic_eligible is False


async def test_an_idle_shell_session_is_ineligible() -> None:
    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree(_session(SESSION_A, foreground=None)))
    )
    windows = await _manager(fake).list_windows_for_reconciliation()
    assert windows is not None
    assert windows[0].topic_eligible is False


# ── targeted existence, because the merged tree cannot be a snapshot ───


async def test_a_known_session_is_reported_present() -> None:
    fake = FakeAgtermctl().ok("session", "text", result={"text": "hi"})

    assert await _manager(fake).window_exists(SESSION_A) is True


async def test_no_such_session_is_proof_of_absence() -> None:
    """The one answer that licenses a destructive repair.

    agterm answers an unknown target with a well-formed refusal, which is why
    this backend can prove absence at all: its listing is assembled from
    per-window RPCs with no isolation, so a session staying ahead of the sweep
    is read in no window, and two identical passes can both omit it. An atomic
    snapshot cannot be built out of non-atomic reads.
    """
    fake = FakeAgtermctl().on(
        "session",
        "text",
        out=json.dumps({"ok": False, "error": f"no such session: {SESSION_A}"}),
    )

    assert await _manager(fake).window_exists(SESSION_A) is False


async def test_an_unreachable_socket_is_not_absence() -> None:
    """No envelope means the question was never answered."""
    fake = FakeAgtermctl().on(
        "session", "text", rc=1, err="connect(/tmp/nope.sock) failed"
    )

    assert await _manager(fake).window_exists(SESSION_A) is None


async def test_an_unrelated_refusal_is_not_absence() -> None:
    """A refusal for some other reason says nothing about existence."""
    fake = FakeAgtermctl().on(
        "session", "text", out=json.dumps({"ok": False, "error": "pane is busy"})
    )

    assert await _manager(fake).window_exists(SESSION_A) is None


async def test_presence_through_the_seam_uses_the_targeted_probe() -> None:
    """The seam must prefer the authoritative answer over the merged listing.

    The listing here omits the session entirely, which is exactly the torn read
    the probe exists to survive: absence in the aggregate is not evidence.
    """
    from ccgram.multiplexer.reconciliation import window_presence

    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        .on("tree", out=_tree())
        .ok("session", "text", result={"text": "hi"})
    )

    assert await window_presence(SESSION_B, _manager(fake)) is True


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        pytest.param(
            {"message": f"no such session: {SESSION_A}"},
            "the reason is not under the key the contract names",
            id="wrong-key",
        ),
        pytest.param(
            {"ok": False, "error": {"text": "no such session"}},
            "the reason is not a string",
            id="non-string-error",
        ),
        pytest.param(
            {"ok": False, "error": "no such session: some-other-id"},
            "the refusal names a different session",
            id="other-session",
        ),
        pytest.param(
            {"ok": "false", "error": f"no such session: {SESSION_A}"},
            "ok is a string, so the envelope is not the documented one",
            id="stringly-ok",
        ),
    ],
)
async def test_a_malformed_refusal_is_never_proof_of_absence(
    payload: dict, why: str
) -> None:
    """Absence licenses deletion, so it needs the whole documented shape.

    Anything looser lets a payload from a changed or broken agterm authorise
    closing a live session's topic, which is the failure this probe exists to
    prevent rather than introduce.
    """
    fake = FakeAgtermctl().on("session", "text", out=json.dumps(payload))

    assert await _manager(fake).window_exists(SESSION_A) is None, why


async def test_a_backend_without_a_probe_uses_the_listing() -> None:
    """tmux and herdr have no targeted answer, so the seam still asks them."""
    from ccgram.multiplexer.base import WindowRef
    from ccgram.multiplexer.reconciliation import window_presence

    class _ListingOnlyBackend:
        async def list_windows_for_reconciliation(self) -> list[WindowRef]:
            return [WindowRef(window_id="@5", window_name="p", cwd="/p")]

    assert await window_presence("@5", _ListingOnlyBackend()) is True
    assert await window_presence("@9", _ListingOnlyBackend()) is False


async def test_a_probe_that_answers_nonsense_is_unknown_not_a_fallback() -> None:
    """Once a backend claims an authoritative probe, that answer is the answer.

    Falling back would reach for the aggregate listing this probe exists to
    avoid trusting.
    """
    from ccgram.multiplexer.base import WindowRef
    from ccgram.multiplexer.reconciliation import window_presence

    class _BrokenProbeBackend:
        async def window_exists(self, window_id: str):
            return "yes"

        async def list_windows_for_reconciliation(self) -> list[WindowRef]:
            return [WindowRef(window_id="@5", window_name="p", cwd="/p")]

    assert await window_presence("@5", _BrokenProbeBackend()) is None


async def test_presence_through_the_module_facade_reaches_the_probe() -> None:
    """Production callers hold the facade, not the backend.

    The facade's type defines only __getattr__, so a static lookup on it finds
    no methods at all and the seam would conclude this backend has no targeted
    probe, falling back to the very aggregate the probe exists to distrust. A
    test that passes a concrete AgtermManager cannot see that.
    """
    from ccgram.multiplexer import (
        _reset_multiplexer_for_testing,
        install_multiplexer,
        multiplexer,
    )
    from ccgram.multiplexer.reconciliation import window_presence

    fake = (
        FakeAgtermctl()
        .on("window", "list", out=_windows())
        # The aggregate omits SESSION_B: the torn read this must survive.
        .on("tree", out=_tree())
        .ok("session", "text", result={"text": "hi"})
    )
    install_multiplexer(_manager(fake))
    try:
        assert await window_presence(SESSION_B, multiplexer) is True
        assert await window_presence(SESSION_B) is True
    finally:
        _reset_multiplexer_for_testing()
