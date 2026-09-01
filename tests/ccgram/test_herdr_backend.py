"""Herdr adapter tests for the guarded session-target contract.

Every target-facing test supplies ``agent list`` records.  Tab and pane IDs are
intentionally asserted only as one-shot herdr dispatch locators; no test binds
a topic to either locator.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.multiplexer import herdr as herdr_module
from ccgram.multiplexer.base import AgentStatus, ForegroundInfo, PaneDims
from ccgram.multiplexer.herdr_events import translate_event
from ccgram.multiplexer.herdr import (
    HERDR_PROTOCOL_VERSION,
    HERDR_SUPPORTED_PROTOCOLS,
    HerdrAgentListError,
    HerdrError,
    HerdrManager,
    HerdrSessionComposite,
    HerdrUnresolvedTargetError,
    _workspace_cwd_from_panes,
    canonical_session_bytes,
    herdr_session_target_id,
)


class FakeHerdr:
    """Prefix-matching canned Herdr command runner."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: dict[tuple[str, ...], tuple[int, str, str]] = {}
        self.default = (1, "", "no canned response")

    def on(self, *prefix: str, rc: int = 0, out: str = "", err: str = "") -> FakeHerdr:
        self.responses[prefix] = (rc, out, err)
        return self

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        call = list(args)
        self.calls.append(call)
        matching = [key for key in self.responses if call[: len(key)] == list(key)]
        return self.responses[max(matching, key=len)] if matching else self.default


@pytest.fixture
def expired_discovery_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the created-session discovery deadline to a single poll.

    The loop always polls once before checking the deadline, so the tests that
    assert post-deadline behaviour keep their meaning without waiting out the
    real 5s window.
    """
    monkeypatch.setattr(herdr_module, "_CREATED_SESSION_DISCOVERY_TIMEOUT_SECONDS", 0.0)


def _manager(fake: FakeHerdr) -> HerdrManager:
    return HerdrManager(socket_path="/tmp/herdr.sock", runner=fake)


def _result(**result: object) -> str:
    return json.dumps({"result": result})


def _agent(
    *,
    pane_id: str = "w2:p1",
    tab_id: str = "w2:t1",
    workspace_id: str = "w2",
    value: str = "session-a",
    agent: str = "claude",
    **extra: object,
) -> dict[str, object]:
    return {
        "terminal_id": "term-a",
        "pane_id": pane_id,
        "tab_id": tab_id,
        "workspace_id": workspace_id,
        "agent_session": {
            "source": "herdr",
            "agent": agent,
            "kind": "id",
            "value": value,
        },
        **extra,
    }


def _agents(*records: Mapping[str, object]) -> str:
    return _result(agents=list(records))


def _target(value: str = "session-a", agent: str = "claude") -> str:
    return herdr_session_target_id(HerdrSessionComposite("herdr", agent, "id", value))


def _sessionless_target(terminal_id: str, agent: str = "claude") -> str:
    return herdr_session_target_id(
        HerdrSessionComposite("herdr", agent, "terminal", terminal_id)
    )


class _SnapshotSequence:
    """Serve one canned Herdr runner per snapshot, advanced by the test.

    The lineage seam is the only per-manager state that spans snapshots, so a
    test of it needs one manager reading two different ``agent list`` replies.
    """

    def __init__(self, *fakes: FakeHerdr) -> None:
        self._fakes = list(fakes)
        self._index = 0

    def advance(self) -> None:
        self._index += 1

    async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
        return await self._fakes[min(self._index, len(self._fakes) - 1)](args)


def _sessionless(terminal_id: str = "term-a") -> dict[str, object]:
    return {
        "terminal_id": terminal_id,
        "pane_id": "w2:p1",
        "tab_id": "w2:t1",
        "workspace_id": "w2",
        "agent": "claude",
    }


def _live_fake(*records: Mapping[str, object]) -> FakeHerdr:
    workspaces = {
        str(record.get("workspace_id", "w2")): {
            "workspace_id": record.get("workspace_id", "w2"),
            "label": "workspace",
        }
        for record in records
    }
    tabs = {
        str(record.get("tab_id", "w2:t1")): {
            "tab_id": record.get("tab_id", "w2:t1"),
            "label": "tab",
        }
        for record in records
    }
    return (
        FakeHerdr()
        .on("agent", "list", out=_agents(*records))
        .on("workspace", "list", out=_result(workspaces=list(workspaces.values())))
        .on("tab", "list", out=_result(tabs=list(tabs.values())))
    )


# ── session identity and discovery ─────────────────────────────────────


def test_capabilities_are_pinned() -> None:
    caps = HerdrManager().capabilities
    assert caps.name == "herdr"
    assert caps.ids_stable_across_restart is False
    assert caps.exposes_pane_tty is False
    assert caps.native_agent_status is True
    assert caps.read_max_lines == 1000
    assert caps.self_identify_env == "HERDR_PANE_ID"
    assert caps.supports_event_stream is True
    assert caps.native_worktrees is True


def test_session_target_digest_is_deterministic_and_private() -> None:
    composite = HerdrSessionComposite("herdr", "claude", "id", "opaque-value")
    target = herdr_session_target_id(composite)
    assert target == herdr_session_target_id(composite)
    assert target == (
        "herdr-session-v1-"
        "d3b1d621b2aae61dd3bfa3355f9b513cc7864f5c0e1974a58c39fced97d20e17"
    )
    assert "opaque-value" not in target
    assert canonical_session_bytes(composite).startswith(b'{"source":"herdr"')


async def test_list_windows_exposes_all_detected_agent_targets() -> None:
    live = _agent(pane_id="w2:p1", tab_id="w2:t9", value="one")
    bare_shell = {
        "terminal_id": "term-c",
        "pane_id": "w2:p3",
        "tab_id": "w2:t9",
        "workspace_id": "w2",
    }
    windows = await _manager(_live_fake(live, bare_shell)).list_windows()
    assert [
        (win.window_id, win.window_name, win.pane_current_command) for win in windows
    ] == [(_target("one"), "workspace ▸ tab ▸ p1", "claude")]
    assert all("w2:" not in win.window_id for win in windows)


async def test_legacy_locator_aliases_are_migration_only_and_adapter_attested() -> None:
    """The adapter exposes old tab/pane keys only through the typed seam."""
    window = (await _manager(_live_fake(_agent())).list_windows())[0]

    assert window.window_id == _target("session-a")
    assert window.legacy_alias_window_ids == ()
    assert window.alias_window_ids == ()


async def test_multiple_agents_in_one_tab_get_pane_topics_and_no_shared_tab_alias() -> (
    None
):
    first = _agent(pane_id="w2:p1", tab_id="w2:t9", terminal_id="term-a", value="one")
    second = _agent(pane_id="w2:p2", tab_id="w2:t9", terminal_id="term-b", value="two")

    windows = await _manager(_live_fake(first, second)).list_windows()

    assert [window.window_name for window in windows] == [
        "workspace ▸ tab ▸ p1",
        "workspace ▸ tab ▸ p2",
    ]
    assert [window.legacy_alias_window_ids for window in windows] == [(), ()]
    for pane, target in [("p1", _target("one")), ("p2", _target("two"))]:
        found = await _manager(_live_fake(first, second)).find_window_by_id(target)
        assert found is not None
        assert found.window_name == f"workspace ▸ tab ▸ {pane}"
        assert found.legacy_alias_window_ids == ()


async def test_sessionless_snapshot_is_not_reconcilable() -> None:
    """A pane becomes reconcilable only after Herdr publishes its session."""
    at_hook_time = {
        "terminal_id": "term-a",
        "pane_id": "w2:p1",
        "tab_id": "w2:t1",
        "workspace_id": "w2",
        "agent": "claude",
    }
    once_published = _agent(value="session-a")

    assert await _manager(_live_fake(at_hook_time)).list_windows() == []
    live_window = (await _manager(_live_fake(once_published)).list_windows())[0]

    assert live_window.window_id == _target("session-a")

    # It is not addressable, but it is a live agent pane ccgram may already be
    # bound to, so it is a gap in the account of what exists: reconciliation
    # says unknown instead of returning an empty listing that would confirm
    # every persisted target dead.
    assert (
        await _manager(_live_fake(at_hook_time)).list_windows_for_reconciliation()
        is None
    )
    assert await _manager(
        _live_fake(once_published)
    ).list_windows_for_reconciliation() == [live_window]
    assert live_window.alias_window_ids == ()


async def test_a_bare_pane_with_no_agent_evidence_leaves_the_listing_complete() -> None:
    """The distinction the completeness rule turns on.

    A pane naming no agent was never a topic, so skipping it costs nothing and
    must not make liveness unknown — otherwise an ordinary open shell would
    disable cleanup permanently.
    """
    bare = {
        "terminal_id": "term-a",
        "pane_id": "w9:p1",
        "tab_id": "w9:t1",
        "workspace_id": "w9",
    }
    live = _agent(pane_id="w9:p2", tab_id="w9:t2", value="two")

    windows = await _manager(_live_fake(bare, live)).list_windows_for_reconciliation()

    assert windows is not None
    assert [window.window_id for window in windows] == [_target("two")]


async def test_terminal_derived_target_never_resolves_to_a_live_session() -> None:
    live = _agent(value="session-a")
    found = await _manager(_live_fake(live)).find_window_by_id(
        _sessionless_target("term-a")
    )

    assert found is None


async def test_session_rekey_does_not_implicitly_rebind_the_previous_target() -> None:
    """A new Herdr session target remains distinct from its predecessor."""
    runner = _SnapshotSequence(
        _live_fake(_agent(value="session-a")),
        _live_fake(_agent(value="session-b")),
    )
    manager = HerdrManager(socket_path="/tmp/herdr.sock", runner=runner)

    before = (await manager.list_windows())[0]
    runner.advance()
    after = (await manager.list_windows())[0]

    assert before.window_id == _target("session-a")
    assert after.window_id == _target("session-b")
    assert after.alias_window_ids == ()
    assert before.alias_window_ids == ()


async def test_direct_lookup_and_reconciliation_share_the_same_projection() -> None:
    """A direct lookup does not invent aliases or alternate display names."""
    runner = _SnapshotSequence(
        _live_fake(_agent(value="session-a")),
        _live_fake(_agent(value="session-b")),
    )
    manager = HerdrManager(socket_path="/tmp/herdr.sock", runner=runner)

    await manager.list_windows()
    runner.advance()
    # An unrelated guarded lookup reads the first post-re-key snapshot.
    assert await manager.find_window_by_id(_target("session-b")) is not None

    after = (await manager.list_windows())[0]
    assert after.alias_window_ids == ()


async def test_sessionless_gap_does_not_create_or_inherit_identity() -> None:
    """A missing session never inherits another session's identity."""
    runner = _SnapshotSequence(
        _live_fake(_agent(value="session-a")),
        _live_fake(_sessionless()),
        _live_fake(
            _agent(value="session-b"),
            _agent(pane_id="w3:p1", value="other", terminal_id="term-b"),
        ),
    )
    manager = HerdrManager(socket_path="/tmp/herdr.sock", runner=runner)

    await manager.list_windows()
    runner.advance()
    await manager.list_windows()
    runner.advance()
    windows = {window.window_id: window for window in await manager.list_windows()}

    assert windows[_target("session-b")].alias_window_ids == ()
    assert windows[_target("other")].alias_window_ids == ()


async def test_sessionless_agent_target_is_not_actionable() -> None:
    sessionless = {
        "terminal_id": "term-b",
        "pane_id": "w2:p2",
        "tab_id": "w2:t9",
        "workspace_id": "w2",
        "agent": "claude",
    }
    found = await _manager(_live_fake(sessionless)).find_window_by_id(
        _sessionless_target("term-b")
    )
    assert found is None


async def test_sessionless_agent_is_not_preserved_by_pane_compaction() -> None:
    before = {
        "terminal_id": "term-b",
        "pane_id": "w2:p8",
        "tab_id": "w2:t9",
        "workspace_id": "w2",
        "agent": "claude",
    }
    after = {**before, "pane_id": "w1:p2", "tab_id": "w1:t3", "workspace_id": "w1"}

    assert await _manager(_live_fake(before)).list_windows() == []
    assert await _manager(_live_fake(after)).list_windows() == []


async def test_find_window_requires_a_fresh_matching_session_target() -> None:
    fake = _live_fake(_agent(value="one"))
    found = await _manager(fake).find_window_by_id(_target("one"))
    assert found is not None
    assert found.window_id == _target("one")
    assert found.window_name == "workspace ▸ tab ▸ p1"
    assert await _manager(fake).find_window_by_id("w2:t1") is None
    assert fake.calls == [
        ["agent", "list"],
        ["workspace", "list"],
        ["tab", "list"],
    ]


async def test_reconciliation_distinguishes_empty_snapshot_from_agent_list_failure() -> (
    None
):
    assert await _manager(_live_fake()).list_windows_for_reconciliation() == []
    assert await _manager(FakeHerdr()).list_windows_for_reconciliation() is None


@pytest.mark.parametrize(
    "record",
    [
        {"pane_id": "w2:p1", "agent_session": {}},
        _agent(terminal_id="", pane_id="w2:p1"),
        _agent(pane_id="", tab_id="w2:t1"),
    ],
)
async def test_guard_rejects_malformed_live_records(record: dict[str, object]) -> None:
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(_live_fake(record)).guard_session_target(_target())


async def test_malformed_record_does_not_hide_an_unrelated_session() -> None:
    malformed = {"pane_id": "w2:p9", "agent_session": {}}
    valid = _agent(value="valid")

    windows = await _manager(_live_fake(malformed, valid)).list_windows()

    assert [window.window_id for window in windows] == [_target("valid")]


async def test_guard_reports_unresolved_ambiguous_and_transport_failures() -> None:
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(_live_fake(_agent(value="other"))).guard_session_target(
            _target()
        )
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(_live_fake(_agent(), _agent())).guard_session_target(_target())
    with pytest.raises(HerdrAgentListError):
        await _manager(FakeHerdr()).guard_session_target(_target())


# ── target actions: fresh guard then locator dispatch ───────────────────


async def test_capture_and_scrollback_guard_then_read_the_matched_pane() -> None:
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "read", rc=0, out="screen text"
    )
    mux = _manager(fake)
    assert await mux.capture_pane(_target()) == "screen text"
    scrollback = await mux.capture_scrollback(_target(), lines=1200)
    assert scrollback is not None and scrollback.text == "screen text"
    assert scrollback.truncated is True
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "read", "w7:p4", "--source", "visible", "--format", "text"],
        ["agent", "list"],
        [
            "pane",
            "read",
            "w7:p4",
            "--source",
            "recent",
            "--lines",
            "1000",
            "--format",
            "text",
        ],
    ]


async def test_send_variants_guard_then_dispatch_to_live_pane() -> None:
    """Literal+enter is split into send-text then a separate Enter key."""
    fake = (
        _live_fake(_agent(pane_id="w7:p4"))
        .on("pane", "send-text", out=_result(type="ok"))
        .on("pane", "send-keys", out=_result(type="ok"))
    )
    mux = _manager(fake)
    with patch("ccgram.multiplexer.herdr.asyncio.sleep"):
        assert await mux.send(_target(), "hello")
    assert await mux.send(_target(), "partial", enter=False)
    assert await mux.send(_target(), "C-c Up", literal=False)
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "send-text", "w7:p4", "hello"],
        ["pane", "send-keys", "w7:p4", "Enter"],
        ["agent", "list"],
        ["pane", "send-text", "w7:p4", "partial"],
        ["agent", "list"],
        ["pane", "send-keys", "w7:p4", "C-c", "Up", "Enter"],
    ]


async def test_send_literal_enter_is_split_with_delay() -> None:
    """#177 regression: text and Enter must never arrive as one ``pane run`` batch."""
    import ccgram.multiplexer.herdr as herdr_mod

    fake = (
        _live_fake(_agent(pane_id="w7:p5"))
        .on("pane", "send-text", out=_result(type="ok"))
        .on("pane", "send-keys", out=_result(type="ok"))
    )
    sleep_mock = AsyncMock()
    with patch.object(herdr_mod.asyncio, "sleep", sleep_mock):
        assert await _manager(fake).send(_target(), "hello world")

    # ``pane run`` must never appear — that is the batched-Enter bug.
    assert ["pane", "run", "w7:p5", "hello world"] not in fake.calls
    # Text arrives before Enter.
    text_idx = fake.calls.index(["pane", "send-text", "w7:p5", "hello world"])
    enter_idx = fake.calls.index(["pane", "send-keys", "w7:p5", "Enter"])
    assert text_idx < enter_idx
    # The delay was awaited between them.
    sleep_mock.assert_awaited_once_with(herdr_mod._SEND_ENTER_DELAY_SECONDS)


async def test_every_mutating_tab_action_uses_live_record_locator() -> None:
    fake = (
        _live_fake(_agent(pane_id="w7:p4", tab_id="w7:t3"))
        .on("pane", "close", out=_result(type="ok"))
        .on("tab", "rename", out=_result(type="ok"))
        .on("pane", "report-metadata", out=_result(type="ok"))
    )
    mux = _manager(fake)
    assert await mux.kill_window(_target())
    assert await mux.rename_window(_target(), "renamed")
    await mux.stamp_pane_title(_target(), "codex")
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "close", "w7:p4"],
        ["agent", "list"],
        ["tab", "rename", "w7:t3", "renamed"],
        ["agent", "list"],
        [
            "pane",
            "report-metadata",
            "w7:p4",
            "--source",
            "ccgram",
            "--title",
            "ccgram:codex",
        ],
    ]


async def test_kill_window_closes_only_target_session_pane_in_shared_tab() -> None:
    first = _agent(pane_id="w7:p4", tab_id="w7:t3", value="session-a")
    sibling = _agent(pane_id="w7:p5", tab_id="w7:t3", value="session-b")
    fake = _live_fake(first, sibling).on("pane", "close", out=_result(type="ok"))

    assert await _manager(fake).kill_window(_target("session-a"))

    assert fake.calls == [
        ["agent", "list"],
        ["pane", "close", "w7:p4"],
    ]


async def test_rename_window_refuses_shared_tab_without_renaming_siblings() -> None:
    first = _agent(pane_id="w7:p4", tab_id="w7:t3", value="session-a")
    sibling = _agent(pane_id="w7:p5", tab_id="w7:t3", value="session-b")
    fake = _live_fake(first, sibling).on("tab", "rename", out=_result(type="ok"))

    assert not await _manager(fake).rename_window(_target("session-a"), "renamed")
    assert fake.calls == [["agent", "list"]]


async def test_status_panes_dims_foreground_and_title_are_guarded() -> None:
    pane = {
        "pane_id": "w7:p4",
        "agent_status": "working",
        "custom_status": "doing",
        "title": "ccgram:claude",
    }
    layout = {
        "layout": {"panes": [{"pane_id": "w7:p4", "rect": {"width": 99, "height": 42}}]}
    }
    process = {
        "process_info": {
            "foreground_process_group_id": 12,
            "foreground_processes": [
                {"pid": 12, "argv": ["claude"], "cwd": "/project"}
            ],
        }
    }
    fake = (
        _live_fake(_agent(pane_id="w7:p4"))
        .on("pane", "get", out=_result(pane=pane))
        .on("pane", "layout", out=_result(**layout))
        .on("pane", "process-info", out=_result(**process))
    )
    mux = _manager(fake)
    assert await mux.agent_status(_target()) == AgentStatus(
        "working", "claude", "doing"
    )
    assert await mux.list_panes(_target()) == []
    assert await mux.pane_dims(_target()) == PaneDims(99, 42)
    assert await mux.foreground(_target()) == ForegroundInfo(
        12, 12, ["claude"], "/project", ""
    )
    assert await mux.get_pane_title(_target()) == "ccgram:claude"
    assert [call for call in fake.calls if call == ["agent", "list"]] == [
        ["agent", "list"]
    ] * 4


async def test_herdr_split_is_unsupported_without_any_raw_pane_side_effect() -> None:
    fake = _live_fake(_agent(pane_id="w7:p4"))
    mux = _manager(fake)
    assert await mux.split_window(_target()) is None
    assert fake.calls == []
    assert await mux._resolve_panes([_target(), "w7:p4"]) == {"w7:p4": _target()}
    assert fake.calls == [["agent", "list"]]


async def test_event_targets_share_one_agent_snapshot() -> None:
    first = _agent(pane_id="w7:p4", tab_id="w7:t3", value="one")
    second = _agent(pane_id="w7:p5", tab_id="w7:t3", value="two")
    fake = _live_fake(first, second)

    panes, tabs = await _manager(fake)._resolve_event_targets(
        [_target("one"), _target("two")]
    )

    assert panes == {"w7:p4": _target("one"), "w7:p5": _target("two")}
    assert tabs == {"w7:t3": (_target("one"), _target("two"))}
    assert fake.calls == [["agent", "list"]]


async def test_nested_dims_and_foreground_payloads_fail_closed() -> None:
    fake = (
        _live_fake(_agent(pane_id="w7:p4"))
        .on("pane", "layout", out=_result(layout={"panes": {"bad": "shape"}}))
        .on(
            "pane",
            "process-info",
            out=_result(process_info={"foreground_processes": {}}),
        )
    )
    mux = _manager(fake)
    assert await mux._dims_for_pane("w7:p4") is None
    assert await mux._foreground_for_pane("w7:p4") is None


def test_translate_event_maps_shared_tab_closure_to_each_opaque_target() -> None:
    first, second = _target("first"), _target("second")
    events = translate_event(
        {"event": "tab.closed", "data": {"tab": {"tab_id": "w7:t3"}}},
        {"w7:p4": first, "w7:p5": second},
        {"w7:t3": (first, second)},
    )
    assert [(event.kind, event.window_id) for event in events] == [
        ("window_died", first),
        ("window_died", second),
    ]


def test_translate_event_uses_refreshed_locator_after_a_target_move() -> None:
    target = _target()
    event = {"event": "pane.agent_status_changed", "data": {"pane_id": "w7:p9"}}
    assert translate_event(event, {"w7:p4": target}, {}) == ()
    translated = translate_event(event, {"w7:p9": target}, {})
    assert translated and translated[0].window_id == target
    assert translated[0].pane_id == "w7:p9"


def test_translate_event_maps_target_pane_exit_without_killing_siblings() -> None:
    first, second = _target("first"), _target("second")
    translated = translate_event(
        {"event": "pane.exited", "data": {"pane_id": "w7:p4"}},
        {"w7:p4": first, "w7:p5": second},
        {"w7:t3": (first, second)},
    )
    assert [(event.kind, event.window_id) for event in translated] == [
        ("window_died", first)
    ]


@pytest.mark.parametrize(
    "event_name",
    ["pane.agent_status_changed", "pane_agent_status_changed"],
)
def test_translate_event_maps_the_agent_status_payload(event_name: str) -> None:
    """herdr spells event names with a dot or an underscore; both must map."""
    target = _target()
    (event,) = translate_event(
        {
            "event": event_name,
            "data": {
                "pane_id": "w7:p4",
                "agent_status": "working",
                "agent": "claude",
                "custom_status": "running tests",
            },
        },
        {"w7:p4": target},
        {},
    )
    assert event.kind == "agent_status"
    assert event.window_id == target
    assert event.status == AgentStatus("working", "claude", "running tests")


def test_translate_event_defaults_an_absent_agent_status_to_unknown() -> None:
    (event,) = translate_event(
        {"event": "pane.agent_status_changed", "data": {"pane_id": "w7:p4"}},
        {"w7:p4": _target()},
        {},
    )
    assert event.status == AgentStatus("unknown", "", "")


def test_translate_event_ignores_unknown_future_event_and_fields() -> None:
    assert (
        translate_event(
            {
                "event": "pane.protocol_21_event",
                "data": {"pane_id": "w7:p4", "future_field": {"nested": True}},
                "future_envelope_field": [1, 2, 3],
            },
            {"w7:p4": _target()},
            {},
        )
        == ()
    )


@pytest.mark.parametrize(
    "event_name", ["pane.exited", "pane_exited", "pane.closed", "pane_closed"]
)
def test_translate_event_maps_every_pane_death_spelling(event_name: str) -> None:
    translated = translate_event(
        {"event": event_name, "data": {"pane_id": "w7:p4"}}, {"w7:p4": _target()}, {}
    )
    assert [(event.kind, event.window_id) for event in translated] == [
        ("window_died", _target())
    ]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"pane_id": "w7:p4"}, id="flat-locator"),
        pytest.param({"pane": {"pane_id": "w7:p4"}}, id="nested-locator"),
    ],
)
def test_translate_event_reads_both_locator_shapes(data: dict) -> None:
    translated = translate_event(
        {"event": "pane.exited", "data": data}, {"w7:p4": _target()}, {}
    )
    assert translated and translated[0].window_id == _target()


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(
            {"event": "pane.focused", "data": {"pane_id": "w7:p4"}}, id="unwatched-kind"
        ),
        pytest.param({"event": "pane.exited"}, id="no-data"),
        pytest.param({"event": "pane.exited", "data": "not-a-mapping"}, id="bad-data"),
        pytest.param({"data": {"pane_id": "w7:p4"}}, id="no-event-name"),
    ],
)
def test_translate_event_ignores_what_it_cannot_map(obj: dict) -> None:
    assert translate_event(obj, {"w7:p4": _target()}, {"w7:t3": (_target(),)}) == ()


async def test_watch_events_emits_each_guarded_target_for_shared_tab_close() -> None:
    first = _agent(pane_id="w7:p4", tab_id="w7:t3", value="first")
    second = _agent(pane_id="w7:p5", tab_id="w7:t3", value="second")
    seen_subscriptions: list[Mapping[str, object]] = []

    async def stream(subscriptions: Sequence[Mapping[str, object]]):
        seen_subscriptions.extend(subscriptions)
        yield {"__subscribed__": True}
        yield {"event": "tab_closed", "data": {"tab_id": "w7:t3"}}

    fake = _live_fake(first, second).on("pane", "get", out=_result(pane={}))
    events = _manager(fake)
    events._open_stream = stream
    watcher = events.watch_events([_target("first"), _target("second")])
    try:
        assert (await anext(watcher)).window_id == _target("first")
        assert (await anext(watcher)).window_id == _target("second")
        assert {subscription["type"] for subscription in seen_subscriptions} >= {
            "pane.exited",
            "pane.closed",
        }
    finally:
        await watcher.aclose()


async def test_watch_events_keeps_one_quiet_stream_when_mapping_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(herdr_module, "_STREAM_REPRIME_INTERVAL", 0.01)
    opens = 0
    blocked = asyncio.Event()

    async def stream(_subscriptions: Sequence[Mapping[str, object]]):
        nonlocal opens
        opens += 1
        yield {"__subscribed__": True}
        await blocked.wait()

    fake = _live_fake(_agent()).on(
        "pane", "get", out=_result(pane={"agent_status": "working"})
    )
    mux = _manager(fake)
    mux._open_stream = stream
    watcher = mux.watch_events([_target()])
    assert (await anext(watcher)).kind == "agent_status"

    pending = asyncio.create_task(anext(watcher))
    await asyncio.sleep(0.035)
    assert opens == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await watcher.aclose()


async def test_watch_events_reconnects_only_after_guarded_mapping_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(herdr_module, "_STREAM_REPRIME_INTERVAL", 0.01)

    class MovingRunner(FakeHerdr):
        def __init__(self) -> None:
            super().__init__()
            self.agent_reads = 0

        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            if args == ["agent", "list"]:
                self.agent_reads += 1
                pane_id = "w2:p1" if self.agent_reads < 3 else "w2:p2"
                return 0, _agents(_agent(pane_id=pane_id)), ""
            if args[:2] == ["pane", "get"]:
                return 0, _result(pane={"agent_status": "working"}), ""
            return await super().__call__(args)

    opens = 0
    blocked = asyncio.Event()

    async def stream(_subscriptions: Sequence[Mapping[str, object]]):
        nonlocal opens
        opens += 1
        yield {"__subscribed__": True}
        await blocked.wait()

    mux = _manager(MovingRunner())
    mux._open_stream = stream
    watcher = mux.watch_events([_target()])
    try:
        assert (await anext(watcher)).pane_id == "w2:p1"
        assert (await anext(watcher)).pane_id == "w2:p2"
        assert opens == 2
    finally:
        await watcher.aclose()


async def test_watch_events_keeps_pre_refresh_mapping_for_terminal_event() -> None:
    record = _agent(pane_id="w7:p4", tab_id="w7:t3")

    class ClosingRunner(FakeHerdr):
        def __init__(self) -> None:
            super().__init__()
            self.agent_reads = 0

        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            if args == ["agent", "list"]:
                self.agent_reads += 1
                # Initial subscription and reprime guard see the target. A
                # post-event refresh would no longer see the closed pane.
                return 0, _agents(record) if self.agent_reads < 3 else _agents(), ""
            if args[:2] == ["pane", "get"]:
                return 0, _result(pane={}), ""
            return await super().__call__(args)

    async def stream(_subscriptions: Sequence[Mapping[str, object]]):
        yield {"__subscribed__": True}
        yield {"event": "pane.exited", "data": {"pane_id": "w7:p4"}}

    mux = _manager(ClosingRunner())
    mux._open_stream = stream
    watcher = mux.watch_events([_target()])
    try:
        event = await anext(watcher)
        assert (event.kind, event.window_id) == ("window_died", _target())
    finally:
        await watcher.aclose()


async def test_raw_pane_helpers_cannot_bypass_target_guard() -> None:
    fake = _live_fake(_agent())
    mux = _manager(fake)
    assert not await mux.send_to_pane("w2:p1", "unsafe", window_id=_target())
    assert await mux.capture_pane_by_id("w2:p1", window_id=_target()) is None
    assert not await mux.send_keys_to_pane("w2:p1", "unsafe", window_id=_target())
    assert fake.calls == []


async def test_send_to_a_replaced_target_fails_without_dispatching() -> None:
    """The bound target re-keyed its session; nothing may be typed into the
    pane now carrying a different one."""
    fake = _live_fake(_agent(value="replacement"))

    assert not await _manager(fake).send(_target("session-a"), "must not dispatch")

    assert fake.calls == [["agent", "list"]]


async def test_action_error_refreshes_guard_without_retargeting() -> None:
    fake = _live_fake(_agent(pane_id="w2:p1")).on(
        "pane", "send-text", rc=1, err="closed"
    )
    assert not await _manager(fake).send(_target(), "hello")
    assert fake.calls == [
        ["agent", "list"],
        ["pane", "send-text", "w2:p1", "hello"],
        ["agent", "list"],
    ]


async def test_post_guard_dispatch_race_never_retargets_another_pane() -> None:
    class RacingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.agent_reads = 0

        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            self.calls.append(list(args))
            if args == ["agent", "list"]:
                self.agent_reads += 1
                # First read authorizes p1; refresh observes only replacement p2.
                record = (
                    _agent(pane_id="w2:p1")
                    if self.agent_reads == 1
                    else _agent(pane_id="w2:p2", value="replacement")
                )
                return 0, _agents(record), ""
            return 1, "", "pane disappeared"

    runner = RacingRunner()
    assert not await HerdrManager(runner=runner).send(_target(), "hello")
    assert runner.calls == [
        ["agent", "list"],
        ["pane", "send-text", "w2:p1", "hello"],
        ["agent", "list"],
    ]
    assert not any(
        "w2:p2" in call for call in runner.calls if call[:2] != ["agent", "list"]
    )


# ── selected-workspace creation and rollback ───────────────────────────


def _workspace(workspace_id: str, cwd: Path) -> str:
    return _result(
        workspaces=[
            {"workspace_id": workspace_id, "label": "selected", "cwd": str(cwd)}
        ]
    )


def _created(tab_id: str = "w9:t1", pane_id: str = "w9:p1") -> str:
    return _result(
        tab={"tab_id": tab_id, "label": "new"}, root_pane={"pane_id": pane_id}
    )


def _tabs(
    tab_id: str = "w9:t1", workspace_id: str = "selected", label: str = "new"
) -> str:
    return _result(
        tabs=[{"tab_id": tab_id, "workspace_id": workspace_id, "label": label}]
    )


async def test_create_topic_target_uses_selected_workspace_and_returns_session_target(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("tab", "list", out=_tabs())
        .on("pane", "run", out=_result(type="ok"))
        .on(
            "agent",
            "list",
            out=_agents(
                _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected")
            ),
        )
    )
    target = await _manager(fake).create_topic_target(
        str(tmp_path),
        launch_command="claude",
        workspace_id="selected",
        agent_args="--dangerously-skip-permissions",
    )
    assert target.target_id == _target()
    assert target.label == "selected ▸ new ▸ p1"
    assert target.window_id == "w9:t1"
    assert target.pane_id == "w9:p1"
    assert fake.calls == [
        ["workspace", "list"],
        [
            "tab",
            "create",
            "--cwd",
            str(tmp_path),
            "--no-focus",
            "--workspace",
            "selected",
        ],
        ["pane", "run", "w9:p1", "claude --dangerously-skip-permissions"],
        ["agent", "list"],
        ["workspace", "list"],
        ["tab", "list"],
    ]


def test_workspace_cwd_prefers_stable_pane_cwd_and_accepts_matching_split_panes() -> (
    None
):
    workspace = {"workspace_id": "w2", "active_tab_id": "w2:t1"}
    panes = [
        {
            "workspace_id": "w2",
            "tab_id": "w2:t1",
            "cwd": "/repo",
            "foreground_cwd": "/repo/.venv/bin",
        },
        {
            "workspace_id": "w2",
            "tab_id": "w2:t1",
            "cwd": "/repo",
            "foreground_cwd": "/repo/.venv/bin",
        },
    ]
    assert _workspace_cwd_from_panes(workspace, panes) == "/repo"
    panes[1].pop("cwd")
    assert _workspace_cwd_from_panes(workspace, panes) is None


async def test_list_workspaces_resolves_cwdless_workspaces_from_panes_once() -> None:
    fake = (
        FakeHerdr()
        .on(
            "workspace",
            "list",
            out=_result(
                workspaces=[
                    {"workspace_id": "w1", "label": "one", "active_tab_id": "w1:t1"},
                    {"workspace_id": "w2", "label": "two", "active_tab_id": "w2:t1"},
                ]
            ),
        )
        .on(
            "pane",
            "list",
            out=_result(
                panes=[
                    {"workspace_id": "w1", "tab_id": "w1:t1", "cwd": "/one"},
                    {"workspace_id": "w2", "tab_id": "w2:t1", "cwd": "/two"},
                ]
            ),
        )
    )

    workspaces = await _manager(fake).list_workspaces()

    assert [(item.workspace_id, item.label, item.cwd) for item in workspaces] == [
        ("w1", "one", "/one"),
        ("w2", "two", "/two"),
    ]
    assert fake.calls == [["workspace", "list"], ["pane", "list"]]


async def test_created_session_discovery_waits_for_delayed_pi_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pi can publish agent_session after several polling intervals (~2.7s live)."""
    calls = 0

    async def delayed_agents(_: float) -> None:
        return None

    class DelayedRunner(FakeHerdr):
        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            nonlocal calls
            if args == ["agent", "list"]:
                calls += 1
                return (
                    0,
                    _agents()
                    if calls < 29
                    else _agents(
                        _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected")
                    ),
                    "",
                )
            return await super().__call__(args)

    monkeypatch.setattr(asyncio, "sleep", delayed_agents)
    runner = (
        DelayedRunner()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("tab", "list", out=_tabs())
    )
    target = await _manager(runner).create_topic_target(
        str(tmp_path), launch_command=None, workspace_id="selected"
    )
    assert target.target_id == _target()
    assert calls == 29


async def test_create_topic_target_without_selection_creates_workspace_at_cwd(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on(
            "workspace",
            "create",
            out=_result(workspace={"workspace_id": "created"}),
        )
        .on("workspace", "list", out=_workspace("created", tmp_path))
        .on("tab", "create", out=_created())
        .on("tab", "list", out=_tabs(workspace_id="created"))
        .on(
            "agent",
            "list",
            out=_agents(
                _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="created")
            ),
        )
    )
    target = await _manager(fake).create_topic_target(
        str(tmp_path), launch_command=None, workspace_id=None
    )
    assert target.target_id == _target()
    assert fake.calls == [
        ["workspace", "create", "--cwd", str(tmp_path), "--no-focus"],
        [
            "tab",
            "create",
            "--cwd",
            str(tmp_path),
            "--no-focus",
            "--workspace",
            "created",
        ],
        ["agent", "list"],
        ["workspace", "list"],
        ["tab", "list"],
    ]


@pytest.mark.parametrize(
    "tab_response, agent_response, expected_error",
    [
        (_result(), None, "no tab id"),
        (_created(), _agents(), "did not report a session"),
        (
            _created(),
            _agents(
                _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="owned"),
                _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="owned"),
            ),
            "did not report a session",
        ),
    ],
)
async def test_implicit_workspace_is_closed_for_tab_and_session_failures(
    tmp_path: Path,
    tab_response: str,
    agent_response: str | None,
    expected_error: str,
    expired_discovery_window: None,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "create", out=_result(workspace={"workspace_id": "owned"}))
        .on("tab", "create", out=tab_response)
        .on("workspace", "close", out=_result(type="ok"))
    )
    if agent_response is not None:
        fake.on("agent", "list", out=agent_response).on(
            "tab", "close", out=_result(type="ok")
        )
    with pytest.raises(HerdrError, match=expected_error):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id=None
        )
    assert fake.calls[-1] == ["workspace", "close", "owned"]


async def test_duplicate_live_targets_are_quarantined_without_hiding_others() -> None:
    """Quarantined records stay unaddressable, and liveness becomes unknown.

    The duplicate panes are live and ccgram may hold bindings to them, so a
    listing that silently omits them is not an account of what exists. Handing
    that partial subset back as complete makes those bindings look like ghosts,
    which the audit, the polling loop and the destructive guards act on.
    Addressing is a different question and still refuses the ambiguous target.
    """
    duplicate = _agent(pane_id="w9:p1", tab_id="w9:t1", value="same")
    unrelated = _agent(pane_id="w9:p2", tab_id="w9:t2", value="other")
    fake = _live_fake(duplicate, duplicate, unrelated)

    manager = _manager(fake)

    assert await manager.list_windows_for_reconciliation() is None
    assert await manager.find_window_by_id(_target("same")) is None
    assert await manager.find_window_by_id(_target("other")) is not None


async def test_distinct_targets_on_one_pane_are_quarantined() -> None:
    first = _agent(pane_id="w9:p1", tab_id="w9:t1", value="first")
    second = _agent(pane_id="w9:p1", tab_id="w9:t1", value="second")
    unrelated = _agent(pane_id="w9:p2", tab_id="w9:t1", value="other")
    fake = _live_fake(first, second, unrelated)

    assert await _manager(fake).list_windows_for_reconciliation() is None
    assert await _manager(fake).find_window_by_id(_target("first")) is None
    assert await _manager(fake).find_window_by_id(_target("second")) is None
    assert await _manager(fake).capture_pane(_target("first")) is None
    panes, tabs = await _manager(fake)._resolve_event_targets(
        [_target("first"), _target("second"), _target("other")]
    )
    assert panes == {"w9:p2": _target("other")}
    assert tabs == {"w9:t1": (_target("other"),)}


async def test_implicit_workspace_is_closed_when_agent_launch_fails(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "create", out=_result(workspace={"workspace_id": "owned"}))
        .on("tab", "create", out=_created())
        .on("pane", "run", rc=1, err="launch failed")
        .on("tab", "close", out=_result(type="ok"))
        .on("workspace", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrError, match="Failed to start"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id=None
        )
    assert fake.calls[-2:] == [
        ["tab", "close", "w9:t1"],
        ["workspace", "close", "owned"],
    ]


async def test_implicit_workspace_is_closed_when_creation_is_cancelled(
    tmp_path: Path,
) -> None:
    class CancellingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            self.calls.append(list(args))
            if args[:2] == ["workspace", "create"]:
                return 0, _result(workspace={"workspace_id": "owned"}), ""
            if args[:2] == ["tab", "create"]:
                return 0, _created(), ""
            if args == ["agent", "list"]:
                raise asyncio.CancelledError()
            if args in (["tab", "close", "w9:t1"], ["workspace", "close", "owned"]):
                return 0, _result(type="ok"), ""
            return 1, "", "unexpected call"

    runner = CancellingRunner()
    with pytest.raises(asyncio.CancelledError):
        await HerdrManager(runner=runner).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id=None
        )
    assert runner.calls[-2:] == [
        ["tab", "close", "w9:t1"],
        ["workspace", "close", "owned"],
    ]


async def test_create_topic_target_rejects_missing_selected_workspace(
    tmp_path: Path,
) -> None:
    fake = FakeHerdr().on("workspace", "list", out=_workspace("other", tmp_path))
    with pytest.raises(HerdrError, match="Selected Herdr workspace"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )
    assert fake.calls == [["workspace", "list"]]


async def test_create_topic_target_rejects_non_string_creation_label(
    tmp_path: Path,
) -> None:
    malformed = _result(
        tab={"tab_id": "w9:t1", "label": None},
        root_pane={"pane_id": "w9:p1"},
    )
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=malformed)
        .on("tab", "close", out=_result(type="ok"))
    )

    with pytest.raises(HerdrError, match="no valid label"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id="selected"
        )

    assert fake.calls[-1] == ["tab", "close", "w9:t1"]


async def test_create_topic_target_rolls_back_malformed_root_pane_response(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_result(tab={"tab_id": "w9:t1", "label": "new"}))
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrError, match="no root pane"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )
    assert fake.calls[-1] == ["tab", "close", "w9:t1"]


async def test_create_topic_target_rolls_back_only_its_new_tab_when_launch_fails(
    tmp_path: Path,
) -> None:
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("pane", "run", rc=1, err="launch failed")
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrError, match="Failed to start"):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )
    assert ["tab", "close", "w9:t1"] in fake.calls
    assert not any(call[:2] == ["workspace", "close"] for call in fake.calls)


async def test_create_topic_target_rolls_back_on_duplicate_or_missing_session(
    tmp_path: Path, expired_discovery_window: None
) -> None:
    duplicate = _agents(
        _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected"),
        _agent(pane_id="w9:p1", tab_id="w9:t1", workspace_id="selected"),
    )
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("agent", "list", out=duplicate)
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(fake).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id="selected"
        )
    assert ["tab", "close", "w9:t1"] in fake.calls

    missing = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("agent", "list", out=_agents())
        .on("tab", "close", out=_result(type="ok"))
    )
    with pytest.raises(HerdrUnresolvedTargetError):
        await _manager(missing).create_topic_target(
            str(tmp_path), launch_command=None, workspace_id="selected"
        )
    assert ["tab", "close", "w9:t1"] in missing.calls


async def test_native_worktree_returns_session_target_or_fails_unbound(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    created = _result(
        tab={"tab_id": "w10:t1", "label": "worktree"},
        root_pane={"pane_id": "w10:p1"},
        workspace={"workspace_id": "worktree-ws"},
    )
    fake = (
        FakeHerdr()
        .on("worktree", "create", out=created)
        .on("workspace", "list", out=_workspace("worktree-ws", repo))
        .on(
            "tab",
            "list",
            out=_tabs("w10:t1", "worktree-ws", "worktree"),
        )
        .on("pane", "run", out=_result(type="ok"))
        .on(
            "agent",
            "list",
            out=_agents(
                _agent(pane_id="w10:p1", tab_id="w10:t1", workspace_id="worktree-ws")
            ),
        )
    )
    ok, _message, label, target = await _manager(fake).create_worktree_window(
        str(repo), str(worktree), "ccg/topic", launch_command="claude"
    )
    assert ok and target == _target()
    assert label == "selected ▸ worktree ▸ p1"
    assert ["pane", "run", "w10:p1", "claude"] in fake.calls

    malformed = (
        FakeHerdr()
        .on("worktree", "create", out=_result(tab={"tab_id": "w10:t1"}))
        .on("tab", "close", out=_result(type="ok"))
    )
    ok, message, _label, target = await _manager(malformed).create_worktree_window(
        str(repo), str(worktree), "ccg/topic", launch_command="claude"
    )
    assert not ok and target == "" and "root pane" in message
    assert malformed.calls[-1] == ["tab", "close", "w10:t1"]


async def test_worktree_cancellation_closes_allocated_tab_and_reraises(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    class CancellingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def __call__(self, args: Sequence[str]) -> tuple[int, str, str]:
            self.calls.append(list(args))
            if args[:2] == ["worktree", "create"]:
                return (
                    0,
                    _result(
                        tab={"tab_id": "w10:t1", "label": "worktree"},
                        root_pane={"pane_id": "w10:p1"},
                        workspace={"workspace_id": "worktree-ws"},
                    ),
                    "",
                )
            if args == ["agent", "list"]:
                raise asyncio.CancelledError()
            if args == ["tab", "close", "w10:t1"]:
                return 0, _result(type="ok"), ""
            return 1, "", "unexpected call"

    runner = CancellingRunner()
    with pytest.raises(asyncio.CancelledError):
        await HerdrManager(runner=runner).create_worktree_window(
            str(repo), str(tmp_path / "worktree"), "ccg/topic"
        )
    assert runner.calls[-1] == ["tab", "close", "w10:t1"]


async def test_agent_status_and_workspace_list_fail_closed_on_malformed_fields() -> (
    None
):
    target = _target()
    malformed_status = _live_fake(_agent()).on(
        "pane", "get", out=_result(pane={"agent_status": ["working"]})
    )
    assert await _manager(malformed_status).agent_status(target) is None

    malformed_workspaces = FakeHerdr().on(
        "workspace",
        "list",
        out=_result(workspaces=[{"workspace_id": "w", "label": 1, "cwd": "/x"}]),
    )
    assert await _manager(malformed_workspaces).list_workspaces() == []


async def test_reconciliation_keeps_internal_records_but_marks_them_unadoptable() -> (
    None
):
    visible = _agent(value="visible")
    internal_workspace = _agent(
        value="workspace-internal", workspace_id="internal", pane_id="w2:p2"
    )
    internal_tab = _agent(value="tab-internal", tab_id="internal-tab", pane_id="w2:p3")
    fake = (
        FakeHerdr()
        .on("agent", "list", out=_agents(visible, internal_workspace, internal_tab))
        .on(
            "workspace",
            "list",
            out=_result(
                workspaces=[
                    {"workspace_id": "w2", "label": "workspace"},
                    {"workspace_id": "internal", "label": "__main__"},
                ]
            ),
        )
        .on(
            "tab",
            "list",
            out=_result(
                tabs=[
                    {"tab_id": "w2:t1", "label": "tab"},
                    {"tab_id": "internal-tab", "label": "__worker__"},
                ]
            ),
        )
    )
    manager = _manager(fake)

    # Reconciliation answers "what exists". Dropping internal records here made
    # a bound session renamed into an internal workspace look dead, so the
    # audit called its binding a fixable ghost and Fix closed the topic.
    windows = await manager.list_windows_for_reconciliation()
    assert windows is not None
    assert {w.window_id: w.topic_eligible for w in windows} == {
        _target("visible"): True,
        _target("workspace-internal"): False,
        _target("tab-internal"): False,
    }

    # The UI listing still hides ccgram's own panes.
    assert [(w.window_id, w.window_name) for w in await manager.list_windows()] == [
        (_target("visible"), "workspace ▸ tab ▸ p1")
    ]


async def test_missing_label_uses_fallback_without_hiding_other_sessions() -> None:
    visible = _agent(
        pane_id="w2:p1", tab_id="w2:t1", workspace_id="w2", value="visible"
    )
    missing = _agent(
        pane_id="w3:p1", tab_id="w3:t1", workspace_id="w3", value="missing"
    )
    fake = (
        FakeHerdr()
        .on("agent", "list", out=_agents(visible, missing))
        .on(
            "workspace",
            "list",
            out=_result(
                workspaces=[
                    {"workspace_id": "w2", "label": "workspace"},
                    {"workspace_id": "w3", "label": "other"},
                ]
            ),
        )
        .on("tab", "list", out=_result(tabs=[{"tab_id": "w2:t1", "label": "tab"}]))
    )

    windows = await _manager(fake).list_windows_for_reconciliation()

    assert windows is not None
    assert [window.window_id for window in windows] == [
        _target("visible"),
        _target("missing"),
    ]
    assert windows[0].window_name == "workspace ▸ tab ▸ p1"
    assert windows[1].window_name.startswith("Herdr ▸ ")
    found = await _manager(fake).find_window_by_id(_target("missing"))
    assert found is not None
    assert found.window_name.startswith("Herdr ▸ ")

    # An unlabeled record cannot be shown to be non-internal, so it is kept for
    # liveness and addressing but never offered for adoption.
    by_id = {w.window_id: w for w in windows}
    assert by_id[_target("missing")].topic_eligible is False
    assert by_id[_target("visible")].topic_eligible is True
    assert _target("missing") not in {
        w.window_id for w in await _manager(fake).list_windows()
    }


async def test_malformed_prefixed_target_never_reads_agent_list() -> None:
    fake = _live_fake(_agent())
    assert (
        await _manager(fake).find_window_by_id("herdr-session-v1-not-a-digest") is None
    )
    assert fake.calls == []


# ── non-target transport/protocol behavior ─────────────────────────────


async def test_subprocess_run_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["herdr", "status"], timeout=5)

    monkeypatch.setattr("ccgram.multiplexer.herdr.asyncio.to_thread", fail)
    assert await HerdrManager()._subprocess_run(["status"]) == (
        124,
        "",
        "herdr call timed out",
    )


@pytest.mark.parametrize("protocol", range(14, 21))
async def test_ensure_session_accepts_supported_protocol_without_warning(
    protocol: int,
) -> None:
    assert frozenset(range(14, 21)) == HERDR_SUPPORTED_PROTOCOLS
    status = json.dumps(
        {
            "server": {
                "running": True,
                "protocol": protocol,
                "compatible": True,
            }
        }
    )
    fake = FakeHerdr().on("status", out=status)

    with patch.object(herdr_module.logger, "warning") as warning:
        await _manager(fake).ensure_session()

    warning.assert_not_called()
    assert fake.calls == [["status", "--json"]]


@pytest.mark.parametrize(
    ("protocol", "compatible"),
    [(13, True), (HERDR_PROTOCOL_VERSION + 1, True), (99, False)],
)
async def test_ensure_session_attempts_unknown_protocol_best_effort(
    protocol: int, compatible: bool
) -> None:
    status = json.dumps(
        {
            "server": {
                "running": True,
                "protocol": protocol,
                "compatible": compatible,
            }
        }
    )
    fake = FakeHerdr().on("status", out=status)

    with patch.object(herdr_module.logger, "warning") as warning:
        await _manager(fake).ensure_session()

    warning.assert_called_once()
    assert fake.calls == [["status", "--json"]]


async def test_ensure_session_still_rejects_unavailable_server() -> None:
    with pytest.raises(HerdrError):
        await _manager(FakeHerdr().on("status", out="not json")).ensure_session()


async def test_creation_closes_pane_when_agent_never_reports_a_session(
    tmp_path: Path, expired_discovery_window: None
) -> None:
    """A pane without a real session never receives a persistent topic target."""
    fake = (
        FakeHerdr()
        .on("workspace", "list", out=_workspace("selected", tmp_path))
        .on("tab", "create", out=_created())
        .on("pane", "run", out=_result(type="ok"))
        .on("agent", "list", out=_agents())
        .on(
            "pane",
            "list",
            out=_result(
                panes=[
                    {
                        "terminal_id": "term-new",
                        "pane_id": "w9:p1",
                        "tab_id": "w9:t1",
                        "workspace_id": "selected",
                    }
                ]
            ),
        )
    )
    manager = _manager(fake)

    with pytest.raises(HerdrUnresolvedTargetError):
        await manager.create_topic_target(
            str(tmp_path), launch_command="claude", workspace_id="selected"
        )

    assert ["tab", "close", "w9:t1"] in fake.calls


async def test_two_sessions_on_one_terminal_are_siblings_not_a_re_key() -> None:
    """A terminal carrying two live sessions has superseded nothing.

    Herdr's shared-tab shape puts sibling agents in one tab, and a snapshot can
    report both against the same terminal. Reading the second as the successor
    of the first would publish the live sibling's target as an alias, so a
    guarded action on it would resolve to two records and fail as ambiguous —
    which is how ``kill_window`` on a shared tab stopped working.
    """
    first = _agent(pane_id="w7:p4", tab_id="w7:t3", value="session-a")
    sibling = _agent(pane_id="w7:p5", tab_id="w7:t3", value="session-b")
    manager = _manager(_live_fake(first, sibling))

    windows = {w.window_id: w for w in await manager.list_windows()}

    assert set(windows) == {_target("session-a"), _target("session-b")}
    for window in windows.values():
        assert _target("session-a") not in window.alias_window_ids
        assert _target("session-b") not in window.alias_window_ids
    assert await manager.find_window_by_id(_target("session-a")) is not None


async def test_a_still_live_target_is_never_published_as_superseded() -> None:
    """The lineage stops publishing an alias the moment Herdr reports it again.

    An agent that re-keys and then reappears under its old session — a resume
    onto the previous id — is not one identity superseding another; both are
    live, and an alias would make the old target ambiguous.
    """
    runner = _SnapshotSequence(
        _live_fake(_agent(value="session-a")),
        _live_fake(_agent(value="session-b")),
        _live_fake(
            _agent(value="session-b"),
            _agent(pane_id="w2:p2", tab_id="w2:t2", value="session-a"),
        ),
    )
    manager = HerdrManager(socket_path="/tmp/herdr.sock", runner=runner)

    await manager.list_windows()
    runner.advance()
    after_rekey = {w.window_id: w for w in await manager.list_windows()}
    assert after_rekey[_target("session-b")].alias_window_ids == ()

    runner.advance()
    windows = {w.window_id: w for w in await manager.list_windows()}
    assert _target("session-a") not in windows[_target("session-b")].alias_window_ids
    assert await manager.find_window_by_id(_target("session-a")) is not None


# ── Fix 1: _foreground_for_pane argv0 fallback ─────────────────────────────


async def test_foreground_falls_back_to_argv0_when_leader_has_no_argv() -> None:
    """Pi rewrites its process title; herdr publishes argv0 but no argv.

    The leader (pid == foreground_process_group_id) carries argv0="pi" and
    name="node". A non-leader carries argv so we confirm the leader's argv0
    wins over both the non-leader's argv AND the leader's name field.
    """
    process = {
        "process_info": {
            "foreground_process_group_id": 12,
            "foreground_processes": [
                # Non-leader has argv — must not be selected.
                {"pid": 7, "argv": ["node", "/app/index.js"], "cwd": "/other"},
                # Leader: no argv, argv0="pi", misleading name="node".
                {"pid": 12, "argv0": "pi", "name": "node", "cwd": "/project"},
            ],
        }
    }
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "process-info", out=_result(**process)
    )
    result = await _manager(fake).foreground(_target())
    # Must be ["pi"] from argv0, never ["node"] from name.
    assert result == ForegroundInfo(12, 12, ["pi"], "/project", "")


@pytest.mark.parametrize(
    "leader",
    [
        pytest.param(
            {"pid": 12, "argv0": "/bin/bash", "name": "bash", "cwd": "/project"},
            id="name-matches-argv0",
        ),
        pytest.param(
            {"pid": 12, "argv0": "/bin/bash", "cwd": "/project"},
            id="no-name-published",
        ),
    ],
)
async def test_foreground_does_not_synthesize_argv_for_an_unrenamed_shell(
    leader,
) -> None:
    """A shell whose args are unreadable must not look like an idle prompt.

    ``shell_infra._is_interactive_shell`` reads a known-shell basename plus
    ``len(argv) == 1`` as "sitting at a prompt, safe to interrupt", and
    ``setup_shell_prompt`` sends C-c on that. Synthesizing ["bash"] for a pane
    running ``bash ./deploy.sh`` would interrupt the script, so the fallback is
    limited to an observed self-rename. A missing ``name`` is not evidence of
    one — both shapes here fall through to the fail-safe None.
    """
    process = {
        "process_info": {
            "foreground_process_group_id": 12,
            "foreground_processes": [leader],
        }
    }
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "process-info", out=_result(**process)
    )
    assert await _manager(fake)._foreground_for_pane("w7:p4") is None


async def test_foreground_returns_none_when_leader_has_neither_argv_nor_argv0() -> None:
    process = {
        "process_info": {
            "foreground_process_group_id": 12,
            "foreground_processes": [
                {"pid": 12, "name": "node", "cwd": "/project"},
            ],
        }
    }
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "process-info", out=_result(**process)
    )
    # Call _foreground_for_pane directly to avoid the second agent-list refresh
    # that foreground() issues when the result is None.
    assert await _manager(fake)._foreground_for_pane("w7:p4") is None


async def test_foreground_uses_argv_verbatim_when_present() -> None:
    process = {
        "process_info": {
            "foreground_process_group_id": 12,
            "foreground_processes": [
                {
                    "pid": 12,
                    "argv": ["node", "/app/pi/dist/index.js"],
                    "cwd": "/project",
                }
            ],
        }
    }
    fake = _live_fake(_agent(pane_id="w7:p4")).on(
        "pane", "process-info", out=_result(**process)
    )
    result = await _manager(fake).foreground(_target())
    assert result == ForegroundInfo(
        12, 12, ["node", "/app/pi/dist/index.js"], "/project", ""
    )


# ── Fix 2: list_windows WindowRef.cwd uses agent cwd not foreground_cwd ────


async def test_list_windows_uses_agent_cwd_not_foreground_cwd() -> None:
    """Pi's agent cwd is the project root; foreground_cwd follows shell chdir."""
    agent = _agent(cwd="/project", foreground_cwd="/project/worktrees/review")
    windows = await _manager(_live_fake(agent)).list_windows()
    assert len(windows) == 1
    assert windows[0].cwd == "/project"


async def test_list_windows_cwd_empty_when_agent_record_has_no_cwd_key() -> None:
    agent = _agent()  # _agent() never sets cwd
    windows = await _manager(_live_fake(agent)).list_windows()
    assert len(windows) == 1
    assert windows[0].cwd == ""


async def test_live_records_are_stamped_adoptable() -> None:
    """The herdr topic rules live here now, not in ``is_agent_topic_window``.

    Keying that gate on a capability flag broke twice: once on
    ``native_agent_status`` when agterm began reporting status natively, once
    on ``native_topic_targets``, whose documented meaning is which creation
    flow to use. Whether a record carries a guarded target and an agent label
    is herdr's own knowledge, so herdr answers it.
    """
    fake = _live_fake(_agent(pane_id="w9:p1", tab_id="w9:t1", value="one"))

    windows = await _manager(fake).list_windows_for_reconciliation()

    assert windows is not None
    assert [w.window_id for w in windows] == [_target("one")]
    assert windows[0].topic_eligible is True
    assert windows[0].window_id.startswith("herdr-session-v1-")


async def test_a_pane_without_an_agent_is_not_listed_or_adoptable() -> None:
    """A bare shell pane was never a herdr topic; that must not have changed."""
    fake = _live_fake(_agent(pane_id="w9:p1", tab_id="w9:t1", value="one", agent=""))

    # An agent_session present but incomplete is a malformed record, not a
    # bare pane: herdr published something that cannot be parsed, so this
    # snapshot cannot account for what is live and says so.
    assert await _manager(fake).list_windows_for_reconciliation() is None


async def test_a_pane_with_no_agent_session_keeps_the_listing_complete() -> None:
    """A genuinely agent-less pane is not a gap in the account.

    herdr publishes no agent_session for a plain shell pane. That pane was
    never a topic and never will be, so it is skipped without making liveness
    unknown. Treating it as a gap would make the listing unknown whenever the
    user has an ordinary pane open, which is to say always, and nothing would
    ever be cleaned up again.
    """
    bare = {
        "terminal_id": "term-a",
        "pane_id": "w9:p1",
        "tab_id": "w9:t1",
        "workspace_id": "w9",
    }
    live = _agent(pane_id="w9:p2", tab_id="w9:t2", value="two")

    windows = await _manager(_live_fake(bare, live)).list_windows_for_reconciliation()

    assert windows is not None
    assert [window.window_id for window in windows] == [_target("two")]


async def test_ui_listing_is_best_effort_when_agent_list_fails() -> None:
    """The picker empties rather than raising through the handler.

    Only ``list_windows_for_reconciliation`` distinguishes this from an empty
    herdr; the selection listing has always degraded, and it must keep doing so
    now that it no longer routes through the guarded reconciliation path.
    """
    fake = FakeHerdr().on("agent", "list", rc=1)

    assert await _manager(fake).list_windows() == []
    assert await _manager(fake).list_windows_for_reconciliation() is None


async def test_ui_listing_is_best_effort_when_label_lookup_fails() -> None:
    """Same for the workspace and tab label listings the projection needs."""
    fake = (
        FakeHerdr()
        .on("agent", "list", out=_agents(_agent(value="one")))
        .on("workspace", "list", rc=1)
    )

    assert await _manager(fake).list_windows() == []
