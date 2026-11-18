"""Native primary-session ownership is independent of conversation identity."""

import pytest

from test_agterm_topic_marks import SESSION, FlagSocket, manager
from _topic_mark_helpers import RecordingTopicMarkGuard
from test_agterm_topic_marks import _native_marker_endpoint as _native_marker_endpoint

OTHER = "47B39C37-222B-452A-B6D4-4AE4E920E2FB"


async def test_unmarked_primary_has_witness_but_split_does_not() -> None:
    mux = manager(FlagSocket())
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None
    primary, split = windows
    assert primary.topic_mark_owner == f"agterm:{SESSION.lower()}"
    assert primary.topic_marked is False
    assert split.topic_mark_owner is None


async def test_expected_owner_survives_shell_and_stale_routing_id() -> None:
    socket = FlagSocket()
    mux = manager(socket)
    window = await mux.find_window_by_id(SESSION)
    assert window is not None and window.topic_mark_owner
    owner = window.topic_mark_owner
    assert await mux.set_topic_mark(
        SESSION, True, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    socket.sessions[0]["foreground"] = []
    assert await mux.set_topic_mark(
        OTHER, False, owner=owner, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.sessions[0]["flagged"] is False
    assert socket.writes[-1][socket.writes[-1].index("--target") + 1] == SESSION


async def test_owner_absence_clear_never_touches_a_replacement() -> None:
    socket = FlagSocket()
    mux = manager(socket)
    old = await mux.find_window_by_id(SESSION)
    assert old is not None and old.topic_mark_owner
    socket.sessions[0].update(id=OTHER, flagged=True)
    assert await mux.set_topic_mark(
        SESSION,
        False,
        owner=old.topic_mark_owner,
        write_guard=RecordingTopicMarkGuard(),
    )
    assert not await mux.set_topic_mark(
        SESSION, True, owner=old.topic_mark_owner, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.sessions[0]["flagged"] is True
    assert socket.writes == []


@pytest.mark.parametrize("bad_owner", ["tmux:abc", "agterm:invalid"])
async def test_foreign_or_malformed_owner_does_not_mutate(bad_owner: str) -> None:
    socket = FlagSocket()
    assert not await manager(socket).set_topic_mark(
        SESSION, True, owner=bad_owner, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.writes == []


async def test_missing_owner_can_mark_a_valid_fresh_target() -> None:
    socket = FlagSocket()
    assert await manager(socket).set_topic_mark(
        SESSION, True, owner=None, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.sessions[0]["flagged"] is True
    assert len(socket.writes) == 1


async def test_invalid_target_never_mutates_a_valid_owner() -> None:
    socket = FlagSocket()
    assert not await manager(socket).set_topic_mark(
        "@1",
        True,
        owner=f"agterm:{SESSION.lower()}",
        write_guard=RecordingTopicMarkGuard(),
    )
    assert socket.writes == []


async def test_duplicate_native_owners_are_quarantined() -> None:
    socket = FlagSocket()
    socket.sessions.append(dict(socket.sessions[0]))
    mux = manager(socket)
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None
    assert all(window.topic_mark_owner is None for window in windows)
    assert not await mux.set_topic_mark(
        SESSION,
        True,
        owner=f"agterm:{SESSION.lower()}",
        write_guard=RecordingTopicMarkGuard(),
    )
    assert socket.writes == []


async def test_unconfirmed_absence_does_not_finish_clear() -> None:
    socket = FlagSocket()
    socket.failure = "read"
    assert not await manager(socket).set_topic_mark(
        OTHER,
        False,
        owner=f"agterm:{SESSION.lower()}",
        write_guard=RecordingTopicMarkGuard(),
    )


async def test_flag_write_ack_loss_retries_same_owner() -> None:
    socket = FlagSocket()
    mux = manager(socket)
    row = await mux.find_window_by_id(SESSION)
    assert row is not None and row.topic_mark_owner
    normal = socket.__call__

    async def lost_ack(args, stdin=None):
        result = await normal(args, stdin)
        return (
            (1, "", "reply lost") if list(args)[:2] == ["session", "flag"] else result
        )

    mux = type(mux)(runner=lost_ack, own_session_id="", workspaces=None)
    assert not await mux.set_topic_mark(
        SESSION, True, owner=row.topic_mark_owner, write_guard=RecordingTopicMarkGuard()
    )
    assert socket.sessions[0]["flagged"] is True
    assert await mux.set_topic_mark(
        SESSION, True, owner=row.topic_mark_owner, write_guard=RecordingTopicMarkGuard()
    )
    assert len(socket.writes) == 1
