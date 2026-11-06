"""hc (Herdr desktop / herdr-core) adapter tests, driven by recorded fixtures.

The fixtures under ``fixtures/hc`` are herdr-core 0.1.18 responses recorded by
the step-1 probe, scrubbed of personal paths. No test touches a socket or runs
``herdr-app``: the transport and the app runner are injected.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from ccgram.multiplexer import hc as hc_module
from ccgram.multiplexer.base import AgentStatus, CaptureResult, ForegroundInfo, PaneDims
from ccgram.multiplexer.hc import (
    HcError,
    HcManager,
    address_for_window_id,
    translate_key,
    window_id_for_address,
)
from ccgram.multiplexer.hc_socket import HcRpcError

FIXTURES = Path(__file__).parent / "fixtures" / "hc"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _terminals() -> list[dict]:
    return copy.deepcopy(_fixture("terminal_list.json")["result"]["terminals"])


def _terminal(address: str) -> dict:
    return next(t for t in _terminals() if t["address"] == address)


Handler = Callable[[Mapping[str, object]], object]


class FakeHc:
    """Canned herdr-core transport: one handler or result per method."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, float]] = []
        self.handlers: dict[str, Handler] = {}

    def on(self, method: str, result: object) -> FakeHc:
        self.handlers[method] = result if callable(result) else (lambda _p: result)
        return self

    def calls_to(self, method: str) -> list[dict]:
        return [params for name, params, _t in self.calls if name == method]

    async def __call__(
        self, method: str, params: Mapping[str, object], timeout: float
    ) -> dict:
        self.calls.append((method, dict(params), timeout))
        handler = self.handlers.get(method)
        if handler is None:
            raise HcRpcError(f"no canned {method}", kind="transport")
        result = handler(params)
        if isinstance(result, BaseException):
            raise result
        if asyncio.iscoroutine(result):
            result = await result
        assert isinstance(result, dict)
        return result


def _not_found(address: object) -> HcRpcError:
    return HcRpcError(f'no terminal "{address}"', kind="not_found", code=-32000)


def _get_from(terminals: Sequence[dict]) -> Handler:
    by_address = {t["address"]: t for t in terminals}

    def handler(params: Mapping[str, object]) -> object:
        found = by_address.get(str(params["terminal"]))
        return copy.deepcopy(found) if found else _not_found(params["terminal"])

    return handler


class FakeApp:
    """Canned ``herdr-app`` runner keyed by the first two arguments."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: dict[tuple[str, ...], tuple[int, str, str]] = {}
        self.side_effects: dict[tuple[str, ...], Callable[[list[str]], None]] = {}

    def on(
        self,
        *prefix: str,
        rc: int = 0,
        out: str = "",
        err: str = "",
        effect: Callable[[list[str]], None] | None = None,
    ) -> FakeApp:
        self.responses[prefix] = (rc, out, err)
        if effect is not None:
            self.side_effects[prefix] = effect
        return self

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        call = list(args)
        self.calls.append(call)
        for prefix in sorted(self.responses, key=len, reverse=True):
            if tuple(call[: len(prefix)]) == prefix:
                effect = self.side_effects.get(prefix)
                if effect is not None:
                    effect(call)
                return self.responses[prefix]
        return (127, "", "herdr-app: command not found")


# Keep listings hermetic: never read this machine's ~/.claude/sessions.
_NO_SESSIONS = "/nonexistent/claude-sessions"


def _manager(
    fake: FakeHc,
    app: FakeApp | None = None,
    *,
    own: str = "",
    stream_opener=None,
) -> HcManager:
    return HcManager(
        socket_path="/tmp/hc-test.sock",
        transport=fake,
        app_runner=app or FakeApp(),
        own_terminal_id=own,
        stream_opener=stream_opener,
        claude_sessions_dir=_NO_SESSIONS,
    )


def _listing_fake(terminals: Sequence[dict] | None = None) -> FakeHc:
    terminals = list(terminals if terminals is not None else _terminals())
    return (
        FakeHc()
        .on(
            "terminal.list",
            {"terminals": terminals, "hidden_finished": 0, "machines": []},
        )
        .on("terminal.get", _get_from(terminals))
    )


# ── Construction and capabilities ───────────────────────────────────────


def test_constructor_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_a, **_k):
        raise AssertionError("constructor touched I/O")

    class _Explode:
        def __getattr__(self, _name):
            return explode

    # Replace only the hc module's references, not the global modules.
    monkeypatch.setattr(hc_module, "shutil", _Explode())
    monkeypatch.setattr(hc_module, "subprocess", _Explode())
    monkeypatch.setattr(hc_module, "request", explode)
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/tmp/from-env.sock")
    manager = HcManager()
    assert manager._socket_path == "/tmp/from-env.sock"


def test_default_socket_path_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HC_TERMINAL_SOCKET", raising=False)
    path = HcManager()._socket_path
    assert path.endswith(".local/state/herdr-core/default/default.sock")


def test_capability_values() -> None:
    caps = HcManager(transport=FakeHc()).capabilities
    assert caps.name == "hc"
    assert caps.ids_stable_across_restart is True
    assert caps.exposes_pane_tty is False
    assert caps.native_agent_status is True
    assert caps.read_max_lines == 1000
    assert caps.self_identify_env == "HC_TERMINAL_ID"
    assert caps.supports_event_stream is True
    assert caps.native_worktrees is False
    assert caps.supports_workspace_selection is True
    assert caps.native_topic_targets is False
    assert caps.supports_shell_prompt_markers is False


# ── Identity ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "window_id,address",
    [
        ("hc-t5", "t5"),
        ("hc-t1y", "t1y"),
        ("HC-T1Y", "t1y"),
        ("hc-rohan/t7", "rohan/t7"),
        ("hc-Rohan/T7", "Rohan/t7"),
    ],
)
def test_window_id_round_trip(window_id: str, address: str) -> None:
    assert address_for_window_id(window_id) == address


@pytest.mark.parametrize(
    "window_id",
    [
        "t5",
        "@5",
        "hc-",
        "hc-name",
        "hc-t5 ",
        "hc-../t5",
        "hc-a/b/t5",
        "hc-ro han/t7",
        "hc-t5|x",
        "hc-t5:x",
        "herdr-session-v1-" + "a" * 64,
    ],
)
def test_window_id_rejects_non_addresses(window_id: str) -> None:
    assert address_for_window_id(window_id) is None


def test_window_id_for_address_keeps_the_slash() -> None:
    assert window_id_for_address("rohan/t7") == "hc-rohan/t7"


async def test_unknown_ids_never_reach_herdr_core() -> None:
    fake = FakeHc()
    manager = _manager(fake)
    assert await manager.find_window_by_id("@1") is None
    assert await manager.capture_pane("hc-name") is None
    assert await manager.send("w1:p1", "hi") is False
    assert await manager.kill_window("hc-../t1") is False
    assert await manager.window_exists("@1") is None
    assert fake.calls == []


# ── Listing and eligibility ─────────────────────────────────────────────


async def test_reconciliation_lists_every_live_terminal() -> None:
    windows = await _manager(
        _listing_fake(), own="t3"
    ).list_windows_for_reconciliation()
    assert windows is not None
    assert [w.window_id for w in windows] == [
        "hc-t3",
        "hc-t5",
        "hc-t16",
        "hc-t1y",
        "hc-t9",
        "hc-t2b",
        "hc-t2e",
        "hc-rohan/t7",
    ]


async def test_topic_eligibility() -> None:
    windows = await _manager(
        _listing_fake(), own="t3"
    ).list_windows_for_reconciliation()
    assert windows is not None
    eligible = {w.window_id for w in windows if w.topic_eligible}
    # Agents reporting status are eligible; ccgram's own terminal (t3), a bare
    # shell (t9), probe terminals with or without status (t2b, t2e) and the
    # remote terminal are not.
    assert eligible == {"hc-t5", "hc-t16", "hc-t1y"}


async def test_list_windows_hides_own_probe_and_remote_but_keeps_shells() -> None:
    windows = await _manager(_listing_fake(), own="t3").list_windows()
    by_id = {w.window_id: w for w in windows}
    assert set(by_id) == {"hc-t5", "hc-t16", "hc-t1y", "hc-t9"}
    assert by_id["hc-t9"].topic_eligible is False


async def test_window_projection_fields() -> None:
    windows = await _manager(_listing_fake()).list_windows_for_reconciliation()
    assert windows is not None
    by_id = {w.window_id: w for w in windows}
    chore = by_id["hc-t5"]
    # Just the tab: the topic layer adds any provider label.
    assert chore.window_name == "chore"
    assert chore.cwd == "/Users/alice/code/work"
    assert chore.pane_current_command == "claude"
    assert (chore.pane_width, chore.pane_height) == (130, 33)
    assert chore.pane_tty == ""
    # The codex app word wins over the node wrapper in the foreground.
    assert by_id["hc-t1y"].window_name == "review"
    assert by_id["hc-t1y"].pane_current_command == "codex"
    # A bare shell keeps the shell name for shell detection.
    assert by_id["hc-t9"].pane_current_command == "zsh"
    assert by_id["hc-t9"].window_name == "notes"
    # No `where` label: fall back to the terminal title, name or address.
    assert by_id["hc-t2b"].window_name
    assert "▸" not in by_id["hc-t2b"].window_name


async def test_own_terminal_defaults_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HC_TERMINAL_ID", "t5")
    manager = HcManager(
        transport=_listing_fake(),
        app_runner=FakeApp(),
        claude_sessions_dir=_NO_SESSIONS,
    )
    windows = await manager.list_windows()
    assert "hc-t5" not in {w.window_id for w in windows}


async def test_listing_failure_is_unknown_not_empty() -> None:
    fake = FakeHc().on("terminal.list", HcRpcError("down", kind="transport"))
    manager = _manager(fake)
    assert await manager.list_windows_for_reconciliation() is None
    assert await manager.list_windows() == []


async def test_finished_terminals_are_not_listed() -> None:
    terminals = _terminals()
    terminals[1]["state"] = "finished"
    windows = await _manager(_listing_fake(terminals)).list_windows_for_reconciliation()
    assert windows is not None
    assert "hc-t5" not in {w.window_id for w in windows}


# ── Presence ────────────────────────────────────────────────────────────


async def test_window_exists_is_authoritative() -> None:
    terminals = _terminals()
    finished = copy.deepcopy(terminals[0])
    finished.update(address="t4", state="finished")
    manager = _manager(_listing_fake([*terminals, finished]))
    assert await manager.window_exists("hc-t5") is True
    assert await manager.window_exists("hc-t4") is False
    assert await manager.window_exists("hc-t999") is False


async def test_window_exists_is_unknown_when_unreachable() -> None:
    fake = FakeHc().on("terminal.get", HcRpcError("down", kind="transport"))
    assert await _manager(fake).window_exists("hc-t5") is None
    fake = FakeHc().on("terminal.get", HcRpcError("gone", kind="machine_unknown"))
    assert await _manager(fake).window_exists("hc-rohan/t7") is None


async def test_reconciliation_presence_through_the_seam() -> None:
    from ccgram.multiplexer.reconciliation import window_presence, window_snapshot

    manager = _manager(_listing_fake())
    assert await window_presence("hc-t5", manager) is True
    assert await window_presence("hc-t999", manager) is False
    confirmed, window = await window_snapshot("hc-t16", manager)
    assert confirmed and window is not None and window.window_id == "hc-t16"


@pytest.mark.parametrize(
    "window_id", ["@42", "herdr-session-v1-" + "a" * 64, "hc-name"]
)
async def test_foreign_ids_are_never_proven_absent(window_id: str) -> None:
    from ccgram.multiplexer.reconciliation import window_presence

    fake = FakeHc()
    assert await window_presence(window_id, _manager(fake)) is None
    assert fake.calls == []


# ── Reads ───────────────────────────────────────────────────────────────


async def test_capture_pane_uses_a_screen_only_read() -> None:
    fake = FakeHc().on("terminal.read", _fixture("terminal_read_screen.json")["result"])
    text = await _manager(fake).capture_pane("hc-t5", with_ansi=True)
    assert text is not None and text.endswith("auto mode on (shift+tab to cycle)")
    assert fake.calls_to("terminal.read") == [{"terminal": "t5", "lines": 0}]


async def test_capture_pane_none_when_gone() -> None:
    fake = FakeHc().on("terminal.read", _not_found("t5"))
    assert await _manager(fake).capture_pane("hc-t5") is None


async def test_capture_pane_by_id_rejects_raw_pane_ids() -> None:
    fake = FakeHc().on("terminal.read", _fixture("terminal_read_screen.json")["result"])
    manager = _manager(fake)
    assert await manager.capture_pane_by_id("p1", window_id="hc-t5") is None
    assert await manager.capture_pane_by_id("hc-t5", window_id="hc-t5") is not None


async def test_scrollback_clamps_and_flags_truncation() -> None:
    fake = FakeHc().on(
        "terminal.read", _fixture("terminal_read_history.json")["result"]
    )
    manager = _manager(fake)
    result = await manager.capture_scrollback("hc-t5", lines=5000)
    assert isinstance(result, CaptureResult)
    assert result.truncated is True
    assert fake.calls_to("terminal.read")[-1] == {"terminal": "t5", "lines": 1000}
    result = await manager.capture_scrollback("hc-t5", lines=200)
    assert result is not None and result.truncated is False
    assert "history line 1" in result.text
    assert await manager.capture_pane_scrollback("hc-t5", history=50) == result.text


async def test_scrollback_incomplete_read_is_truncated() -> None:
    history = _fixture("terminal_read_history.json")["result"]
    history["complete"] = False
    fake = FakeHc().on("terminal.read", history)
    result = await _manager(fake).capture_scrollback("hc-t5", lines=100)
    assert result is not None and result.truncated is True


async def test_slow_history_read_times_out_to_the_screen_and_backs_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _fixture("terminal_read_screen.json")["result"]

    def read(params: Mapping[str, object]) -> object:
        if params["lines"]:
            return HcRpcError("hc terminal.read timed out", kind="timeout")
        return screen

    fake = FakeHc().on("terminal.read", read)
    manager = _manager(fake)
    result = await manager.capture_scrollback("hc-t5", lines=200)
    assert result == CaptureResult(text=screen["text"], truncated=True)
    history_reads = [c for c in fake.calls if c[0] == "terminal.read" and c[1]["lines"]]
    assert len(history_reads) == 1
    # The history read is time-boxed well below herdr-core's 10 s worst case.
    assert history_reads[0][2] == hc_module._HISTORY_READ_TIMEOUT < 5
    # Backing off: the next capture serves the screen without another slow read.
    again = await manager.capture_scrollback("hc-t5", lines=200)
    assert again is not None and again.truncated is True
    assert (
        len([c for c in fake.calls if c[0] == "terminal.read" and c[1]["lines"]]) == 1
    )
    # After the backoff, history is tried again.
    monkeypatch.setattr(hc_module.time, "monotonic", lambda: 10_000_000.0)
    await manager.capture_scrollback("hc-t5", lines=200)
    assert (
        len([c for c in fake.calls if c[0] == "terminal.read" and c[1]["lines"]]) == 2
    )


async def test_slow_history_read_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real slow transport is cut off by the budget, not awaited to the end."""
    monkeypatch.setattr(hc_module, "_HISTORY_READ_TIMEOUT", 0.5)
    screen = _fixture("terminal_read_screen.json")["result"]

    async def slow_transport(method, params, timeout):
        if method == "terminal.read" and params["lines"]:
            try:
                async with asyncio.timeout(timeout):
                    await asyncio.sleep(60)
            except TimeoutError as exc:
                raise HcRpcError("timed out", kind="timeout") from exc
        return screen

    manager = HcManager(
        transport=slow_transport, app_runner=FakeApp(), own_terminal_id=""
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        result = await manager.capture_scrollback("hc-t5", lines=200)
    finally:
        task.cancel()
    assert result is not None and result.truncated
    assert loop.time() - started < hc_module._HISTORY_READ_TIMEOUT + 1
    assert ticks >= 5


async def test_pane_dims_foreground_status_title() -> None:
    manager = _manager(_listing_fake())
    assert await manager.pane_dims("hc-t5") == PaneDims(width=130, height=33)
    assert await manager.foreground("hc-t5") == ForegroundInfo(
        pid=53443,
        pgid=0,
        argv=["claude"],
        cwd="/Users/alice/code/sessions",
        tty="",
    )
    assert await manager.agent_status("hc-t5") == AgentStatus(
        state="working", agent="claude", custom_status="Reading the plan"
    )
    assert await manager.agent_status("hc-t9") is None
    assert await manager.get_pane_title("hc-t5") == "chore"
    assert await manager.get_pane_title("hc-t999") == ""
    assert await manager.foreground("hc-t999") is None


async def test_finished_terminal_reads_as_gone() -> None:
    terminal = _terminal("t5")
    terminal.update(state="finished", foreground=None, status=None)
    manager = _manager(_listing_fake([terminal]))
    assert await manager.find_window_by_id("hc-t5") is None
    assert await manager.foreground("hc-t5") is None
    assert await manager.pane_dims("hc-t5") is None


async def test_single_pane_model() -> None:
    manager = _manager(FakeHc())
    assert await manager.list_panes("hc-t5") == []
    assert await manager.split_window("hc-t5") is None
    ok, msg, name, wid = await manager.create_worktree_window("/r", "/r.wt/x", "ccg/x")
    assert (ok, name, wid) == (False, "", "")
    assert "hc" in msg


# ── Session handshake ───────────────────────────────────────────────────


async def test_ensure_session_accepts_generation_one() -> None:
    fake = FakeHc().on("hello", _fixture("hello.json")["result"])
    await _manager(fake).ensure_session()
    assert fake.calls_to("hello") == [{}]


async def test_ensure_session_warns_on_other_generations(monkeypatch) -> None:
    hello = _fixture("hello.json")["result"]
    hello["generation"] = 2
    warnings: list[tuple[str, dict]] = []

    class _Logger:
        def warning(self, message, **kwargs):
            warnings.append((message, kwargs))

    monkeypatch.setattr(hc_module, "logger", _Logger())
    await _manager(FakeHc().on("hello", hello)).ensure_session()
    assert warnings and warnings[0][1]["generation"] == 2


async def test_ensure_session_raises_when_unreachable() -> None:
    fake = FakeHc().on("hello", HcRpcError("refused", kind="transport"))
    with pytest.raises(HcError, match="not reachable"):
        await _manager(fake).ensure_session()


# ── Key translation and sending ─────────────────────────────────────────


@pytest.mark.parametrize(
    "token,expected",
    [
        ("Enter", "enter"),
        ("Escape", "esc"),
        ("Tab", "tab"),
        ("BTab", "shift+tab"),
        ("S-Tab", "shift+tab"),
        ("Space", "space"),
        ("BSpace", "backspace"),
        ("Up", "up"),
        ("Down", "down"),
        ("Left", "left"),
        ("Right", "right"),
        ("Home", "home"),
        ("End", "end"),
        ("PageUp", "pageup"),
        ("PPage", "pageup"),
        ("NPage", "pagedown"),
        ("DC", "delete"),
        ("F5", "f5"),
        ("C-c", "ctrl+c"),
        ("C-d", "ctrl+d"),
        ("C-z", "ctrl+z"),
        ("C-y", "ctrl+y"),
        ("C-u", "ctrl+u"),
        ("C-A", "ctrl+a"),
        ("C-\\", "ctrl+\\"),
        ("M-t", "alt+t"),
        ("M-Enter", "alt+enter"),
        ("M-Up", "alt+up"),
        ("M-Down", "alt+down"),
        ("S-Enter", "shift+enter"),
        ("C-M-x", "ctrl+alt+x"),
        ("C-Left", "ctrl+left"),
        ("1", "1"),
        ("y", "y"),
        ("Y", "shift+y"),
        ("/", "/"),
    ],
)
def test_translate_key(token: str, expected: str) -> None:
    assert translate_key(token) == expected


@pytest.mark.parametrize("token", ["NoSuchKey", "?", "中", "C-", "Ctrl+c", ""])
def test_translate_key_refuses_unknown_names(token: str) -> None:
    assert translate_key(token) is None


def test_every_key_ccgram_sends_translates() -> None:
    """Every tmux key name the shipped handlers and menus send has a mapping."""
    from ccgram.handlers.interactive.interactive_callbacks import INTERACTIVE_KEY_MAP
    from ccgram.handlers.live.screenshot_callbacks import KEYS_SEND_MAP
    from ccgram.toolbar_config import BUILTIN_ACTIONS

    tokens = {key for key, _refresh in INTERACTIVE_KEY_MAP.values()}
    tokens |= {key for key, _enter, literal in KEYS_SEND_MAP.values() if not literal}
    tokens |= {
        action.payload
        for action in BUILTIN_ACTIONS.values()
        if action.action_type == "key" and not action.literal
    }
    # Fixed keys in session_protocol and the provider TUI menus.
    tokens |= {"C-c", "Escape", "M-Enter", "C-u", "BSpace", "C-s", "C-a", "C-x"}
    tokens |= {"C-p", "M-Up", "M-Down", "S-Enter", "Down", "Enter", "Space", "Tab"}
    missing = sorted(t for t in tokens if translate_key(t) is None)
    assert missing == []


def test_translated_names_are_ones_herdr_core_accepted() -> None:
    """Names the step-1 probe saw herdr-core accept stay in the vocabulary."""
    accepted = {
        name
        for name, verdict in _fixture("probe_input_results.json")["keys"].items()
        if verdict == "ok"
    }
    for token in (
        "Enter",
        "Escape",
        "Tab",
        "BTab",
        "BSpace",
        "Up",
        "C-c",
        "C-d",
        "C-u",
    ):
        assert translate_key(token) in accepted


async def test_literal_text_with_enter_uses_terminal_send() -> None:
    fake = FakeHc().on("terminal.send", {"started": True, "status": "idle", "turn": 4})
    manager = _manager(fake)
    assert await manager.send("hc-t5", "fix the tests\nplease", enter=True) is True
    assert fake.calls_to("terminal.send") == [
        {"terminal": "t5", "text": "fix the tests\nplease", "force": True}
    ]
    timeout = fake.calls[-1][2]
    assert timeout >= 10  # the quiet-keyboard wait is ~3 s


async def test_send_failure_returns_false() -> None:
    fake = FakeHc().on("terminal.send", HcRpcError("finished", kind="finished"))
    assert await _manager(fake).send("hc-t5", "hello") is False


async def test_key_names_go_to_terminal_keys() -> None:
    fake = FakeHc().on("terminal.keys", {"status": None, "turn": 0})
    manager = _manager(fake)
    assert await manager.send("hc-t5", "Escape", enter=False, literal=False) is True
    assert await manager.send("hc-t5", "C-c", enter=False, literal=False) is True
    assert await manager.send("hc-t5", "Down Enter", enter=False, literal=False)
    assert await manager.send_keys("hc-t5", "Up", enter=True, literal=False)
    assert fake.calls_to("terminal.keys") == [
        {"terminal": "t5", "keys": ["esc"]},
        {"terminal": "t5", "keys": ["ctrl+c"]},
        {"terminal": "t5", "keys": ["down", "enter"]},
        {"terminal": "t5", "keys": ["up", "enter"]},
    ]


async def test_unknown_key_name_refuses_the_whole_send() -> None:
    fake = FakeHc().on("terminal.keys", {"turn": 0})
    assert await _manager(fake).send("hc-t5", "Up Bogus", literal=False) is False
    assert fake.calls == []


async def test_literal_without_enter_is_typed_key_by_key() -> None:
    fake = FakeHc().on("terminal.keys", {"turn": 0})
    manager = _manager(fake)
    # An approval answer, a menu reply, and the toolbar Mode button's Shift-Tab.
    assert await manager.send("hc-t5", "2", enter=False) is True
    assert await manager.send("hc-t5", "Use v2", enter=False) is True
    assert await manager.send("hc-t5", "\x1b[Z", enter=False) is True
    assert fake.calls_to("terminal.keys") == [
        {"terminal": "t5", "keys": ["2"]},
        {"terminal": "t5", "keys": ["shift+u", "s", "e", "space", "v", "2"]},
        {"terminal": "t5", "keys": ["shift+tab"]},
    ]
    assert fake.calls_to("terminal.send") == []


async def test_literal_without_enter_refuses_untypeable_text() -> None:
    fake = FakeHc().on("terminal.keys", {"turn": 0})
    assert await _manager(fake).send("hc-t5", "why?", enter=False) is False
    assert await _manager(fake).send("hc-t5", "中文", enter=False) is False
    assert fake.calls == []


async def test_empty_literal_with_enter_presses_enter() -> None:
    fake = FakeHc().on("terminal.keys", {"turn": 0})
    assert await _manager(fake).send("hc-t5", "", enter=True) is True
    assert fake.calls_to("terminal.keys") == [{"terminal": "t5", "keys": ["enter"]}]


async def test_send_to_pane_rejects_raw_pane_ids() -> None:
    fake = FakeHc().on("terminal.send", {"turn": 1})
    manager = _manager(fake)
    assert await manager.send_to_pane("p1", "x", window_id="hc-t5") is False
    assert await manager.send_keys_to_pane("hc-t5", "x", window_id="hc-t5") is True


# ── Kill and rename ─────────────────────────────────────────────────────


async def test_kill_window_calls_terminal_kill() -> None:
    fake = FakeHc().on("terminal.kill", _terminal("t5"))
    assert await _manager(fake).kill_window("hc-t5") is True
    assert fake.calls_to("terminal.kill") == [{"terminal": "t5"}]
    fake = FakeHc().on("terminal.kill", _not_found("t5"))
    assert await _manager(fake).kill_window("hc-t5") is False


async def test_rename_window_renames_the_herdr_tab() -> None:
    app = FakeApp().on("do", "tab.rename", out="{}")
    manager = _manager(_listing_fake(), app)
    assert await manager.rename_window("hc-t5", "New name") is True
    assert app.calls == [
        [
            "do",
            "tab.rename",
            json.dumps(
                {"id": "tbdf4448221061c406863872cc3eb0b084", "name": "New name"}
            ),
        ]
    ]


async def test_rename_window_without_a_tab_label_fails() -> None:
    app = FakeApp().on("do", "tab.rename")
    manager = _manager(_listing_fake(), app)
    assert await manager.rename_window("hc-t2b", "x") is False
    assert app.calls == []


# ── Workspaces ──────────────────────────────────────────────────────────

_APP_LS = {
    "spaces": [
        {
            "id": "spwork",
            "name": "Work",
            "folder": "/Users/alice/code/work",
            "machine": None,
            "tabs": [],
        },
        {
            "id": "spsess",
            "name": "Sessions",
            "folder": "/Users/alice/code",
            "machine": None,
            "tabs": [],
        },
        {
            "id": "spremote",
            "name": "Rohan",
            "folder": "/home/alice",
            "machine": "rohan",
            "tabs": [],
        },
    ]
}


async def test_list_workspaces_from_herdr_app_ls() -> None:
    app = FakeApp().on("ls", out=json.dumps(_APP_LS))
    workspaces = await _manager(FakeHc(), app).list_workspaces()
    assert [(w.workspace_id, w.label, w.cwd) for w in workspaces] == [
        ("spwork", "Work", "/Users/alice/code/work"),
        ("spsess", "Sessions", "/Users/alice/code"),
    ]


async def test_list_workspaces_empty_when_app_unreachable() -> None:
    assert await _manager(FakeHc(), FakeApp()).list_workspaces() == []


# ── Creation and terminal discovery ─────────────────────────────────────


class _CreationWorld:
    """terminal.list that gains a terminal once ``tab.new`` (or run) happens."""

    def __init__(self, new_terminal: dict | None, *, delay_polls: int = 0) -> None:
        self.terminals = [t for t in _terminals() if t["address"] != "t2b"]
        self.new_terminal = new_terminal
        self.delay_polls = delay_polls
        self.created = False

    def create(self, _call: list[str] | None = None) -> None:
        self.created = True

    def listing(self, _params: Mapping[str, object]) -> dict:
        terminals = list(self.terminals)
        if self.created and self.new_terminal is not None:
            if self.delay_polls > 0:
                self.delay_polls -= 1
            else:
                terminals.append(self.new_terminal)
        return {"terminals": terminals, "hidden_finished": 0, "machines": []}


def _new_agent_terminal(work_dir: Path, address: str = "t2g") -> dict:
    terminal = _terminal("t16")
    terminal.update(address=address, cwd=str(work_dir), status=None, turn=0)
    terminal["argv"] = [
        "/bin/zsh",
        "-lic",
        '"$@"; exec "$0" -li',
        "/bin/zsh",
        "claude",
        "--resume",
    ]
    terminal["labels"] = {"herdr.tab": "tbnew", "where": "Work / new"}
    return terminal


@pytest.fixture
def fast_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc_module, "_CREATE_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(hc_module, "_CREATE_DISCOVERY_TIMEOUT", 1.0)


@pytest.mark.usefixtures("fast_discovery")
async def test_create_window_opens_a_tab_and_finds_its_terminal(tmp_path: Path) -> None:
    world = _CreationWorld(_new_agent_terminal(tmp_path), delay_polls=2)
    fake = FakeHc().on("terminal.list", world.listing)
    app = (
        FakeApp()
        .on("do", "tab.new", out="{}", effect=world.create)
        .on("ls", out=json.dumps(_APP_LS))
    )
    ok, message, name, window_id = await _manager(fake, app).create_window(
        str(tmp_path),
        window_name="feature",
        launch_command="claude",
        agent_args="--resume",
        workspace_id="spwork",
    )
    assert (ok, message, name, window_id) == (True, "", "feature", "hc-t2g")
    tab_new = next(c for c in app.calls if c[:2] == ["do", "tab.new"])
    assert json.loads(tab_new[2]) == {
        "folder": str(tmp_path.resolve()),
        "argv": ["claude", "--resume"],
        "name": "feature",
        "space": "spwork",
    }
    assert fake.calls_to("terminal.run") == []


@pytest.mark.usefixtures("fast_discovery")
async def test_create_window_ignores_unrelated_new_terminals(tmp_path: Path) -> None:
    world = _CreationWorld(_new_agent_terminal(tmp_path, "t2h"))
    stray = _new_agent_terminal(tmp_path / "..", "t2g")
    world.terminals.append(stray)  # existed before: never a candidate
    other_dir = _new_agent_terminal(Path("/tmp"), "t2i")

    def listing(params):
        result = world.listing(params)
        if world.created:
            result["terminals"].append(other_dir)  # new, but elsewhere
        return result

    fake = FakeHc().on("terminal.list", listing)
    app = FakeApp().on("do", "tab.new", effect=world.create)
    ok, _msg, name, window_id = await _manager(fake, app).create_window(
        str(tmp_path), launch_command="claude --resume", workspace_id="spwork"
    )
    assert (ok, window_id) == (True, "hc-t2h")
    assert name == tmp_path.name


@pytest.mark.usefixtures("fast_discovery")
async def test_create_window_uses_an_address_the_app_reports(tmp_path: Path) -> None:
    world = _CreationWorld(_new_agent_terminal(tmp_path))
    fake = FakeHc().on("terminal.list", world.listing)
    app = FakeApp().on(
        "do",
        "tab.new",
        out=json.dumps({"tab": {"id": "tbnew", "terminals": [{"terminal": "t2g"}]}}),
    )
    ok, _msg, _name, window_id = await _manager(fake, app).create_window(
        str(tmp_path), launch_command="claude", workspace_id="spwork"
    )
    assert (ok, window_id) == (True, "hc-t2g")
    assert len(fake.calls_to("terminal.list")) == 1  # no discovery polling needed


@pytest.mark.usefixtures("fast_discovery")
async def test_create_window_fails_when_the_terminal_never_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hc_module, "_CREATE_DISCOVERY_TIMEOUT", 0.0)
    world = _CreationWorld(None)
    fake = FakeHc().on("terminal.list", world.listing)
    app = FakeApp().on("do", "tab.new", effect=world.create)
    ok, message, _name, window_id = await _manager(fake, app).create_window(
        str(tmp_path), launch_command="claude", workspace_id="spwork"
    )
    assert (ok, window_id) == (False, "")
    assert "could not be identified" in message
    assert fake.calls_to("terminal.run") == []


@pytest.mark.usefixtures("fast_discovery")
async def test_create_window_refuses_to_guess_between_two_matches(
    tmp_path: Path,
) -> None:
    world = _CreationWorld(_new_agent_terminal(tmp_path, "t2g"))
    twin = _new_agent_terminal(tmp_path, "t2h")

    def listing(params):
        result = world.listing(params)
        if world.created:
            result["terminals"].append(twin)
        return result

    fake = FakeHc().on("terminal.list", listing)
    app = FakeApp().on("do", "tab.new", effect=world.create)
    ok, _message, _name, window_id = await _manager(fake, app).create_window(
        str(tmp_path), launch_command="claude --resume", workspace_id="spwork"
    )
    assert (ok, window_id) == (False, "")


@pytest.mark.usefixtures("fast_discovery")
async def test_create_window_falls_back_to_terminal_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "/bin/zsh")
    run_result = _fixture("terminal_run.json")["result"] | {"address": "t2j"}
    fake = (
        FakeHc()
        .on("terminal.list", _CreationWorld(None).listing)
        .on("terminal.run", run_result)
    )
    app = FakeApp()  # herdr-app missing: every call fails with 127
    ok, message, name, window_id = await _manager(fake, app).create_window(
        str(tmp_path), launch_command="codex", agent_args="--full-auto"
    )
    assert (ok, message, name, window_id) == (True, "", tmp_path.name, "hc-t2j")
    (run_params,) = fake.calls_to("terminal.run")
    assert run_params == {
        "cwd": str(tmp_path.resolve()),
        "no_parent": True,
        "no_notify": True,
        "keep": True,
        "argv": [
            "/bin/zsh",
            "-lic",
            '"$@"; exec "$0" -li',
            "/bin/zsh",
            "codex",
            "--full-auto",
        ],
    }


@pytest.mark.usefixtures("fast_discovery")
async def test_failed_app_call_that_still_opened_a_tab_is_not_duplicated(
    tmp_path: Path,
) -> None:
    world = _CreationWorld(_new_agent_terminal(tmp_path))
    fake = (
        FakeHc()
        .on("terminal.list", world.listing)
        .on("terminal.run", {"address": "t9z"})
    )
    app = FakeApp().on("do", "tab.new", rc=124, err="timed out", effect=world.create)
    ok, _msg, _name, window_id = await _manager(fake, app).create_window(
        str(tmp_path), launch_command="claude --resume", workspace_id="spwork"
    )
    assert (ok, window_id) == (True, "hc-t2g")
    assert fake.calls_to("terminal.run") == []


async def test_create_window_picks_the_most_specific_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hc_module, "_CREATE_POLL_INTERVAL", 0.0)
    nested = tmp_path / "work" / "repo"
    nested.mkdir(parents=True)
    spaces = {
        "spaces": [
            {"id": "sproot", "name": "Root", "folder": str(tmp_path), "machine": None},
            {
                "id": "spwork",
                "name": "Work",
                "folder": str(tmp_path / "work"),
                "machine": None,
            },
        ]
    }
    world = _CreationWorld(_new_agent_terminal(nested.resolve()))
    world.new_terminal["argv"] = ["/bin/zsh", "-l"]  # type: ignore[index]
    fake = FakeHc().on("terminal.list", world.listing)
    app = (
        FakeApp()
        .on("ls", out=json.dumps(spaces))
        .on("do", "tab.new", effect=world.create)
    )
    ok, _msg, _name, _wid = await _manager(fake, app).create_window(
        str(nested), start_agent=False
    )
    assert ok
    payload = json.loads(next(c for c in app.calls if c[:2] == ["do", "tab.new"])[2])
    assert payload == {"folder": str(nested.resolve()), "space": "spwork"}


async def test_create_window_rejects_a_missing_directory(tmp_path: Path) -> None:
    fake = FakeHc()
    ok, message, _n, _w = await _manager(fake).create_window(str(tmp_path / "nope"))
    assert ok is False and "does not exist" in message
    assert fake.calls == []


async def test_create_window_fails_when_hc_is_unreachable(tmp_path: Path) -> None:
    fake = FakeHc().on("terminal.list", HcRpcError("down", kind="transport"))
    app = FakeApp().on("do", "tab.new")
    ok, message, _n, _w = await _manager(fake, app).create_window(str(tmp_path))
    assert ok is False and "not reachable" in message
    assert app.calls == []


@pytest.mark.usefixtures("fast_discovery")
async def test_create_topic_target_returns_the_window_id(tmp_path: Path) -> None:
    world = _CreationWorld(_new_agent_terminal(tmp_path))
    fake = FakeHc().on("terminal.list", world.listing)
    app = FakeApp().on("do", "tab.new", effect=world.create)
    result = await _manager(fake, app).create_topic_target(
        str(tmp_path),
        launch_command="claude --resume",
        workspace_id="spwork",
        window_name="topic",
    )
    assert (result.target_id, result.window_id, result.label) == (
        "hc-t2g",
        "hc-t2g",
        "topic",
    )


async def test_create_topic_target_raises_on_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        await _manager(FakeHc()).create_topic_target(
            str(tmp_path / "missing"), launch_command="claude", workspace_id=None
        )


async def test_shared_tab_names_add_the_space() -> None:
    """Two terminals in tabs of the same name are told apart by their space."""
    manager = _manager(_listing_fake())
    first = {"address": "t1", "labels": {"where": "Sessions / research"}}
    second = {"address": "t2", "labels": {"where": "Tabula / research"}}
    alone = {"address": "t3", "labels": {"where": "Digients / cua"}}
    shared = manager._shared_tabs([first, second, alone])
    assert shared == frozenset({"research"})
    assert manager._to_window(first, shared).window_name == "research (Sessions)"
    assert manager._to_window(second, shared).window_name == "research (Tabula)"
    assert manager._to_window(alone, shared).window_name == "cua"


async def test_claude_session_name_names_the_topic(tmp_path) -> None:
    """A named Claude Code session gives the topic its name; a rename follows."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    manager = HcManager(
        transport=_listing_fake(),
        app_runner=FakeApp(),
        own_terminal_id="",
        claude_sessions_dir=str(sessions),
    )
    terminal = {
        "address": "t20",
        "labels": {"where": "Digients / cua"},
        "foreground": {"pid": 4242, "argv": ["claude"]},
    }
    assert manager._to_window(terminal).window_name == "cua"
    session_file = sessions / "4242.json"
    session_file.write_text(json.dumps({"pid": 4242, "name": "dig-cua"}))
    assert manager._to_window(terminal).window_name == "dig-cua"
    session_file.write_text(json.dumps({"pid": 4242, "name": "dig-cua-2"}))
    os.utime(session_file, (time.time() + 5, time.time() + 5))
    assert manager._to_window(terminal).window_name == "dig-cua-2"
    # An unnamed session falls back to the tab.
    session_file.write_text(json.dumps({"pid": 4242, "name": None}))
    os.utime(session_file, (time.time() + 10, time.time() + 10))
    assert manager._to_window(terminal).window_name == "cua"
