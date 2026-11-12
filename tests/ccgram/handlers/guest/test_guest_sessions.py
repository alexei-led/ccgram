"""Guests starting sessions with /sessions new, at most N per guest chat."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccgram.config import config
from ccgram.handlers import callback_registry, sessions_dashboard
from ccgram.handlers.callback_data import (
    CB_SESSIONS_KILL,
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
)
from ccgram.handlers.callback_helpers import callback_user_id
from ccgram.handlers.guest import access, scope
from ccgram.handlers.guest import sessions as guest_sessions
from ccgram.multiplexer.base import WorkspaceRef

OWNER = 1644026884
GUEST = 555
GUEST_CHAT = -1009990001
OWN_CHAT = -1004477516122
LAUNCH = "/admin/start-friend.sh {label}"
CODEX_LAUNCH = "/admin/start-friend-codex.sh {label}"
# Captured before the fixture replaces it.
_REAL_SEED_TRUST = guest_sessions._seed_trust
WORKSPACE = WorkspaceRef(workspace_id="sp-xy", label="XY", cwd="/friends/XY")


class World:
    """Bound windows per chat and live terminals, as the module sees them."""

    def __init__(self) -> None:
        self.bound: dict[int, set[str]] = {GUEST_CHAT: set(), OWN_CHAT: {"hc-t5"}}
        self.live: set[str] = {"hc-t5"}
        self.created: list[dict] = []
        self.next_id = 40

    async def listing(self, _mux=None):
        return [SimpleNamespace(window_id=w) for w in sorted(self.live)]

    async def create(self, data, **kwargs):
        window_id = f"hc-t{self.next_id}"
        self.next_id += 1
        self.created.append({**data, **kwargs})
        self.live.add(window_id)
        return SimpleNamespace(target_id=window_id, title="")

    def adopt(self, window_id: str) -> None:
        self.bound[GUEST_CHAT].add(window_id)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[World]:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(config, "guest_chats", {GUEST_CHAT: "XY"})
    monkeypatch.setattr(config, "allowed_users", {OWNER})
    monkeypatch.setattr(config, "group_id", None)
    monkeypatch.setattr(config, "guest_launch_command", LAUNCH)
    monkeypatch.setattr(config, "guest_codex_launch_command", CODEX_LAUNCH)
    monkeypatch.setattr(config, "guest_max_sessions", 3)
    w = World()
    monkeypatch.setattr(
        guest_sessions, "windows_in_chat", lambda chat: set(w.bound.get(chat, ()))
    )
    monkeypatch.setattr(guest_sessions, "list_windows_for_reconciliation", w.listing)
    monkeypatch.setattr(
        guest_sessions, "routed_workspace", AsyncMock(return_value=WORKSPACE)
    )
    monkeypatch.setattr(guest_sessions, "_seed_trust", AsyncMock())
    access._reset_guest_access_for_testing()
    guest_sessions._reset_guest_sessions_for_testing()
    yield w
    access._reset_guest_access_for_testing()
    guest_sessions._reset_guest_sessions_for_testing()


async def test_start_runs_the_guest_launcher_in_the_chats_workspace(world) -> None:
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world)
    assert reply == guest_sessions.STARTED.format(provider="Claude", count=1, max=3)
    (call,) = world.created
    assert call["directory"] == "/friends/XY"
    assert call["launch_command"] == "/admin/start-friend.sh XY"
    assert call["workspaceId"] == "sp-xy"
    assert call["provider"] == "claude"


async def test_label_is_shell_quoted(world, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "guest_chats", {GUEST_CHAT: "A B;rm"})
    await guest_sessions.start_guest_session(GUEST_CHAT, world)
    assert world.created[0]["launch_command"] == "/admin/start-friend.sh 'A B;rm'"


async def test_limit_counts_live_bound_sessions(world) -> None:
    for wid in ("hc-t1", "hc-t2", "hc-t3"):
        world.bound[GUEST_CHAT].add(wid)
        world.live.add(wid)
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world)
    assert reply == guest_sessions.LIMIT.format(count=3, max=3)
    assert world.created == []


async def test_dead_sessions_do_not_count(world) -> None:
    world.bound[GUEST_CHAT] |= {"hc-t1", "hc-t2", "hc-t3"}  # none live
    assert await guest_sessions.count_sessions(GUEST_CHAT) == 0


async def test_sessions_still_starting_hold_a_slot(world) -> None:
    for _ in range(3):
        await guest_sessions.start_guest_session(GUEST_CHAT, world)
    assert len(world.created) == 3
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world)
    assert reply.startswith("This chat already has 3 sessions")
    # Adoption moves a slot from "starting" to "bound" without double counting.
    world.adopt("hc-t40")
    assert await guest_sessions.count_sessions(GUEST_CHAT) == 3


async def test_a_start_that_is_never_adopted_frees_its_slot(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(guest_sessions, "_clock", lambda: clock[0])
    await guest_sessions.start_guest_session(GUEST_CHAT, world)
    assert await guest_sessions.count_sessions(GUEST_CHAT) == 1
    clock[0] += guest_sessions.PENDING_TTL_S + 1
    assert await guest_sessions.count_sessions(GUEST_CHAT) == 0


async def test_disabled_without_a_launch_command(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "guest_launch_command", "")
    monkeypatch.setattr(config, "guest_codex_launch_command", "")
    assert (
        await guest_sessions.start_guest_session(GUEST_CHAT, world)
        == guest_sessions.DISABLED
    )
    assert world.created == []


async def test_codex_runs_the_codex_launcher(world) -> None:
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world, "codex")
    assert reply == guest_sessions.STARTED.format(provider="Codex", count=1, max=3)
    (call,) = world.created
    assert call["provider"] == "codex"
    assert call["launch_command"] == "/admin/start-friend-codex.sh XY"
    assert call["directory"] == "/friends/XY"


async def test_codex_without_its_launcher_is_refused(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "guest_codex_launch_command", "")
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world, "codex")
    assert reply == guest_sessions.DISABLED_PROVIDER.format(provider="Codex")
    assert world.created == []


async def test_unknown_agent_is_refused(world) -> None:
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world, "gemini")
    assert reply.startswith("Starting gemini sessions")
    assert world.created == []


async def test_the_limit_counts_both_agents_together(world) -> None:
    await guest_sessions.start_guest_session(GUEST_CHAT, world, "claude")
    await guest_sessions.start_guest_session(GUEST_CHAT, world, "codex")
    await guest_sessions.start_guest_session(GUEST_CHAT, world, "codex")
    reply = await guest_sessions.start_guest_session(GUEST_CHAT, world, "claude")
    assert reply.startswith("This chat already has 3 sessions")
    assert [c["provider"] for c in world.created] == ["claude", "codex", "codex"]


async def test_trust_is_seeded_for_claude_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex guests trust their folder in their own CODEX_HOME; seeding would
    # write the owner's ~/.codex/config.toml instead.
    seeded: list[str] = []
    monkeypatch.setattr(
        "ccgram.agent_startup.preseed_trust",
        lambda provider, _folder: seeded.append(provider),
    )
    await _REAL_SEED_TRUST("codex", "/friends/XY")
    await _REAL_SEED_TRUST("claude", "/friends/XY")
    assert seeded == ["claude"]


async def test_not_a_guest_chat_is_disabled(world) -> None:
    assert (
        await guest_sessions.start_guest_session(OWN_CHAT, world)
        == guest_sessions.DISABLED
    )


async def test_unreachable_terminals(world, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        guest_sessions, "list_windows_for_reconciliation", AsyncMock(return_value=None)
    )
    assert (
        await guest_sessions.start_guest_session(GUEST_CHAT, world)
        == guest_sessions.UNREACHABLE
    )


# --- /sessions new from a guest ------------------------------------------------


def _command(user_id: int, chat_id: int, *args: str):
    update = MagicMock()
    update.effective_user = SimpleNamespace(id=user_id)
    update.effective_chat = SimpleNamespace(id=chat_id)
    update.message = MagicMock()
    context = MagicMock()
    context.args = list(args)
    return update, context


@pytest.fixture
def replies(monkeypatch: pytest.MonkeyPatch, world) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(sessions_dashboard, "safe_reply", mock)
    monkeypatch.setattr(sessions_dashboard, "adapter_for", lambda _data: world)
    return mock


async def test_guest_sessions_new_starts_one(world, replies) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, context = _command(GUEST, GUEST_CHAT, "new")
    await sessions_dashboard.sessions_command(update, context)
    assert len(world.created) == 1
    assert replies.await_args is not None
    assert replies.await_args.args[1].startswith("Starting a new Claude session")


async def test_guest_sessions_new_codex_starts_codex(world, replies) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, context = _command(GUEST, GUEST_CHAT, "new", "codex")
    await sessions_dashboard.sessions_command(update, context)
    assert [c["provider"] for c in world.created] == ["codex"]


async def test_stranger_cannot_start_one(world, replies) -> None:
    update, context = _command(777, GUEST_CHAT, "new")
    await sessions_dashboard.sessions_command(update, context)
    assert world.created == []
    assert replies.await_args is not None
    assert "not authorized" in replies.await_args.args[1]


async def test_owner_sessions_new_outside_guest_chats_just_lists(
    world, replies, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = AsyncMock(return_value=("Sessions", MagicMock()))
    monkeypatch.setattr(sessions_dashboard, "_build_dashboard", build)
    update, context = _command(OWNER, OWN_CHAT, "new")
    await sessions_dashboard.sessions_command(update, context)
    assert world.created == []
    build.assert_awaited_once_with(OWNER, OWN_CHAT)


async def test_guest_listing_acts_as_owner_in_the_chat(
    world, replies, monkeypatch: pytest.MonkeyPatch
) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    build = AsyncMock(return_value=("Sessions", MagicMock()))
    monkeypatch.setattr(sessions_dashboard, "_build_dashboard", build)
    update, context = _command(GUEST, GUEST_CHAT)
    await sessions_dashboard.sessions_command(update, context)
    build.assert_awaited_once_with(OWNER, GUEST_CHAT)


# --- dashboard buttons from a guest ---------------------------------------------


def _callback(user_id: int, data: str, chat_id: int = GUEST_CHAT):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.message = SimpleNamespace(chat=SimpleNamespace(id=chat_id, type="supergroup"))
    update = MagicMock()
    update.callback_query = query
    update.effective_user = SimpleNamespace(id=user_id)
    update.effective_chat = SimpleNamespace(id=chat_id)
    return update, query


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch, world) -> list[int | None]:
    acted: list[int | None] = []

    async def handler(update, _context) -> None:
        acted.append(callback_user_id(update))

    monkeypatch.setattr(callback_registry, "_find_handler", lambda _d: handler)
    monkeypatch.setattr(callback_registry, "thread_router", MagicMock())
    monkeypatch.setattr(
        scope.session_query,
        "iter_bound_topics",
        lambda: [(OWNER, GUEST_CHAT, 2, "hc-t9"), (OWNER, OWN_CHAT, 4, "hc-t5")],
    )
    return acted


@pytest.mark.parametrize(
    "data",
    [
        CB_SESSIONS_NEW,
        f"{CB_SESSIONS_NEW}:claude",
        f"{CB_SESSIONS_NEW}:codex",
        CB_SESSIONS_REFRESH,
    ],
)
async def test_guest_may_press_chat_buttons(seen, data) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, _query = _callback(GUEST, data)
    await callback_registry.dispatch(update, MagicMock())
    assert seen == [OWNER]


async def test_guest_may_end_a_session_of_this_chat(seen) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, _query = _callback(GUEST, f"{CB_SESSIONS_KILL}hc-t9")
    await callback_registry.dispatch(update, MagicMock())
    assert seen == [OWNER]


async def test_guest_cannot_end_the_owners_other_sessions(seen) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, query = _callback(GUEST, f"{CB_SESSIONS_KILL}hc-t5")
    await callback_registry.dispatch(update, MagicMock())
    assert seen == []
    query.answer.assert_awaited_once_with("Only the owner can use these buttons.")


async def test_guest_cannot_press_a_forged_new_button(seen) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, query = _callback(GUEST, f"{CB_SESSIONS_NEW}:gemini")
    await callback_registry.dispatch(update, MagicMock())
    assert seen == []
    query.answer.assert_awaited_once_with("Only the owner can use these buttons.")


# --- the guest dashboard ----------------------------------------------------------


def _button_rows(keyboard) -> list[list[tuple[str, str]]]:
    return [
        [(b.text, b.callback_data) for b in row] for row in keyboard.inline_keyboard
    ]


async def test_guest_dashboard_offers_one_start_button_per_agent(world) -> None:
    text, keyboard = await sessions_dashboard._build_dashboard(OWNER, GUEST_CHAT)
    assert "Start one with a button below" in text
    assert _button_rows(keyboard) == [
        [("\U0001f504 Refresh", CB_SESSIONS_REFRESH)],
        [
            ("\u2795 Claude", f"{CB_SESSIONS_NEW}:claude"),
            ("\u2795 Codex", f"{CB_SESSIONS_NEW}:codex"),
        ],
    ]


async def test_guest_dashboard_hides_agents_without_a_launcher(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "guest_codex_launch_command", "")
    _text, keyboard = await sessions_dashboard._build_dashboard(OWNER, GUEST_CHAT)
    assert _button_rows(keyboard)[-1] == [
        ("\u2795 Claude", f"{CB_SESSIONS_NEW}:claude")
    ]


async def test_codex_button_starts_codex(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = MagicMock()
    query.answer = AsyncMock()
    query.message = SimpleNamespace(chat=SimpleNamespace(id=GUEST_CHAT))
    monkeypatch.setattr(sessions_dashboard, "safe_edit", AsyncMock())
    await sessions_dashboard.handle_sessions_new(query, OWNER, world, "codex")
    assert [c["provider"] for c in world.created] == ["codex"]
