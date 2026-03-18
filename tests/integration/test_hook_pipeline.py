"""Integration tests for hook event offset persistence across monitor instances.

Tests that byte offsets survive monitor restarts and that multiple event types
are dispatched correctly through the full pipeline. Basic read/skip/truncate
behaviors are covered by unit tests in test_session_monitor_events.py.
"""

import pytest

from ccgram.handlers.hook_events import HookEvent
from ccgram.session_monitor import SessionMonitor

pytestmark = pytest.mark.integration


async def test_event_offset_persists_across_monitor_instances(
    state_dir, append_event
) -> None:
    received: list[HookEvent] = []

    async def on_event(event: HookEvent) -> None:
        received.append(event)

    append_event("Stop")

    monitor1 = SessionMonitor(
        projects_path=state_dir / "projects",
        poll_interval=0.1,
        state_file=state_dir / "monitor_state.json",
    )
    monitor1.set_hook_event_callback(on_event)
    await monitor1._read_hook_events()
    monitor1.state.save()
    assert len(received) == 1

    monitor2 = SessionMonitor(
        projects_path=state_dir / "projects",
        poll_interval=0.1,
        state_file=state_dir / "monitor_state.json",
    )
    monitor2.set_hook_event_callback(on_event)
    await monitor2._read_hook_events()
    assert len(received) == 1


async def test_multiple_event_types_dispatched(state_dir, append_event) -> None:
    received: list[HookEvent] = []

    async def on_event(event: HookEvent) -> None:
        received.append(event)

    monitor = SessionMonitor(
        projects_path=state_dir / "projects",
        poll_interval=0.1,
        state_file=state_dir / "monitor_state.json",
    )
    monitor.set_hook_event_callback(on_event)

    append_event("SubagentStart", data={"subagent_id": "a1", "name": "researcher"})
    append_event("TeammateIdle", data={"teammate_name": "coder"})
    append_event(
        "TaskCompleted",
        data={"task_subject": "Write tests", "teammate_name": "coder"},
    )
    append_event("SubagentStop", data={"subagent_id": "a1"})

    await monitor._read_hook_events()

    assert len(received) == 4
    types = [e.event_type for e in received]
    assert types == ["SubagentStart", "TeammateIdle", "TaskCompleted", "SubagentStop"]
    assert received[0].data["name"] == "researcher"
    assert received[1].data["teammate_name"] == "coder"
    assert received[2].data["task_subject"] == "Write tests"
