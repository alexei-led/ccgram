"""Native boundary races and completeness failures for Herdr topic marks."""

from __future__ import annotations

from collections.abc import Sequence

from ccgram.multiplexer.herdr import HerdrManager

from test_herdr_topic_marks import _target
from _topic_mark_helpers import RecordingTopicMarkGuard
from test_herdr_topic_marks import native_factory as native_factory


async def test_herdr_mark_owner_absent_never_clears_replacement(native_factory) -> None:
    native = await native_factory()
    old_owner = (await native.window()).topic_mark_owner
    native.panes[0]["terminal_id"] = "replacement"
    native.agents = [native.agent("replacement", "w1:p1", "two")]
    current = await native.window()
    assert await native.manager.set_topic_mark(
        _target("two"),
        True,
        owner=current.topic_mark_owner,
        write_guard=RecordingTopicMarkGuard(),
    )
    replacement_label = native.panes[0]["label"]
    assert await native.manager.set_topic_mark(
        _target(), False, owner=old_owner, write_guard=RecordingTopicMarkGuard()
    )
    assert native.panes[0]["label"] == replacement_label
    assert (await native.window()).topic_marked is True
    assert len(native.writes) == 1


async def test_herdr_marker_detects_label_read_race(native_factory) -> None:
    native = await native_factory("original")

    async def label_race(args: Sequence[str]) -> tuple[int, str, str]:
        if tuple(args[:2]) == ("pane", "get"):
            native.panes[0]["label"] = "concurrent user rename"
        return await native(args)

    manager = HerdrManager(socket_path="/tmp/herdr-test.sock", runner=label_race)
    assert not await manager.set_topic_mark(
        _target(), True, write_guard=RecordingTopicMarkGuard()
    )
    assert native.panes[0]["label"] == "concurrent user rename"
    assert native.writes == []


async def test_herdr_mark_owner_requires_complete_native_inventory(
    native_factory,
) -> None:
    native = await native_factory()
    native.panes.append({"terminal_id": "broken-record"})
    window = await native.window()
    assert window.topic_marked is None
    assert window.topic_mark_owner is None
    assert not await native.manager.set_topic_mark(
        _target(), False, owner="a" * 32, write_guard=RecordingTopicMarkGuard()
    )
    assert native.writes == []
