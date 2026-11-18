"""Cancellation preserves durable ownership instead of waiting on a remote flag."""

import asyncio

import pytest

from ccgram.multiplexer.agterm import AgtermManager
from test_agterm_topic_marks import SESSION, FlagSocket
from _topic_mark_helpers import RecordingTopicMarkGuard
from test_agterm_topic_marks import _native_marker_endpoint as _native_marker_endpoint

OWNER = f"agterm:{SESSION.lower()}"


class DelayedFlag(FlagSocket):
    def __init__(self, original: bool, outcome: str):
        super().__init__()
        self.sessions[0]["flagged"] = original
        self.outcome = outcome
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.remote: asyncio.Task | None = None

    async def __call__(self, args, stdin=None):
        if list(args)[:2] != ["session", "flag"]:
            return await super().__call__(args, stdin)
        apply = super().__call__

        async def remote():
            self.entered.set()
            await self.release.wait()
            try:
                if self.outcome == "worker-error":
                    raise RuntimeError("native worker failed")
                if self.outcome != "success":
                    return 1, "", self.outcome
                return await apply(args, stdin)
            finally:
                self.finished.set()

        self.remote = asyncio.create_task(remote())
        return await asyncio.shield(self.remote)


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize(
    "outcome", ["success", "api-error", "os-error", "timeout", "worker-error"]
)
async def test_cancelled_flag_retains_ambiguity_without_waiting_for_app(
    marked, cancellations, outcome
):
    native = DelayedFlag(not marked, outcome)
    mux = AgtermManager(runner=native, own_session_id="", workspaces=None)
    write_guard = RecordingTopicMarkGuard()
    task = asyncio.create_task(
        mux.set_topic_mark(SESSION, marked, owner=OWNER, write_guard=write_guard)
    )
    try:
        await native.entered.wait()
        assert len(write_guard.armed) == 1
        task.cancel("original cancellation")
        await asyncio.sleep(0)
        for _ in range(cancellations - 1):
            task.cancel("repeated cancellation")
            await asyncio.sleep(0)
        async with asyncio.timeout(1):
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await task
        assert not native.finished.is_set()
        assert write_guard.terminal_ids == []
        native.release.set()
        await native.finished.wait()
        assert write_guard.terminal_ids == []
    finally:
        native.release.set()
        await asyncio.gather(
            *(item for item in (task, native.remote) if item is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize(
    "outcome", ["success", "api-error", "os-error", "timeout", "worker-error"]
)
async def test_flag_terminal_callback_requires_valid_native_completion(marked, outcome):
    native = DelayedFlag(not marked, outcome)
    mux = AgtermManager(runner=native, own_session_id="", workspaces=None)
    write_guard = RecordingTopicMarkGuard()
    native.release.set()
    if outcome == "worker-error":
        with pytest.raises(RuntimeError, match="native worker failed"):
            await mux.set_topic_mark(
                SESSION, marked, owner=OWNER, write_guard=write_guard
            )
    else:
        assert await mux.set_topic_mark(
            SESSION, marked, owner=OWNER, write_guard=write_guard
        ) is (outcome == "success")
    assert native.finished.is_set()
    assert len(write_guard.armed) == 1
    expected = [write_guard.armed[0].dispatch_id] if outcome == "success" else []
    assert write_guard.terminal_ids == expected
