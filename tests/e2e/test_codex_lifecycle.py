"""E2E tests for Codex lifecycle — binding, messaging, recovery."""

import asyncio
import shutil

import pytest

from ._helpers import (
    TEST_THREAD_ID,
    TEST_USER_ID,
    find_message_id_for,
    make_callback_update,
    make_text_update,
    setup_bound_topic,
    wait_for_pane,
    wait_for_send,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("codex") is None, reason="codex CLI not installed"),
]


async def test_basic_lifecycle(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app

    window_id, _ = await setup_bound_topic(app, calls, work_dir, provider="codex")

    # Verify agent launched
    await wait_for_pane(tmux, window_id, timeout=30)

    # Wait for agent response delivered to topic
    await wait_for_send(
        calls,
        predicate=lambda d: (
            d.get("message_thread_id") == TEST_THREAD_ID
            and len(d.get("text", "")) > 5
            and "Bound" not in d.get("text", "")
            and "Select" not in d.get("text", "")
        ),
        timeout=120,
    )


async def test_command_forwarding(e2e_app, work_dir):
    app, calls, tmux, _session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir, provider="codex")

    await wait_for_pane(tmux, window_id, timeout=30)
    calls.clear()

    u = make_text_update("/status", bot=app.bot)
    await app.process_update(u)

    await wait_for_pane(tmux, window_id, pattern="status", timeout=15)


async def test_recovery_fresh(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir, provider="codex")

    await wait_for_pane(tmux, window_id, timeout=30)

    await tmux.kill_window(window_id)
    await asyncio.sleep(1)

    calls.clear()
    u = make_text_update("are you there?", bot=app.bot)
    await app.process_update(u)

    await wait_for_send(
        calls,
        predicate=lambda d: "no longer running" in d.get("text", ""),
        timeout=10,
    )

    recovery_msg_id = find_message_id_for(
        calls,
        predicate=lambda d: "no longer running" in d.get("text", ""),
    )
    assert recovery_msg_id is not None

    u_fresh = make_callback_update(
        f"rec:f:{window_id}",
        recovery_msg_id,
        bot=app.bot,
    )
    await app.process_update(u_fresh)
    await asyncio.sleep(5)

    new_window_id = session_mgr.get_window_for_thread(TEST_USER_ID, TEST_THREAD_ID)
    assert new_window_id is not None
    new_pane = await tmux.capture_pane(new_window_id)
    assert new_pane is not None
