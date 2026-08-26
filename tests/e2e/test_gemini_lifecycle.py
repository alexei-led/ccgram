"""E2E tests for Gemini CLI lifecycle — binding, messaging, recovery."""

import asyncio
import shutil

import pytest

from ._helpers import (
    make_callback_update,
    make_text_update,
    setup_bound_topic,
    wait_for_agent_reply,
    wait_for_pane,
    wait_for_rebind,
    wait_for_recovery_prompt,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        shutil.which("gemini") is None, reason="gemini CLI not installed"
    ),
]


async def test_basic_lifecycle(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app

    window_id, _ = await setup_bound_topic(app, calls, work_dir, provider="gemini")

    # Verify agent launched
    await wait_for_pane(tmux, window_id, timeout=30)

    # Wait for agent response delivered to topic (Gemini has no hooks, slower)
    await wait_for_agent_reply(calls, timeout=180)


async def test_command_forwarding(e2e_app, work_dir):
    app, calls, tmux, _session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir, provider="gemini")

    await wait_for_pane(tmux, window_id, timeout=30)
    calls.clear()

    u = make_text_update("/help", bot=app.bot)
    await app.process_update(u)

    await wait_for_pane(tmux, window_id, pattern="help", timeout=15)


@pytest.mark.xfail(
    reason=(
        "Pre-existing flaky test: same root cause as Claude's recovery_fresh "
        "— recovery-created window vanishes before wait_for_pane succeeds. "
        "Failing since 2026-04-13 across multiple refactor cycles."
    ),
    strict=False,
)
async def test_recovery_fresh(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir, provider="gemini")

    await wait_for_pane(tmux, window_id, timeout=30)

    await tmux.kill_window(window_id)
    await asyncio.sleep(1)

    calls.clear()
    u = make_text_update("are you there?", bot=app.bot)
    await app.process_update(u)

    recovery_msg_id = await wait_for_recovery_prompt(calls)

    u_fresh = make_callback_update(
        f"rec:f:{window_id}",
        recovery_msg_id,
        bot=app.bot,
    )
    await app.process_update(u_fresh)

    new_window_id = await wait_for_rebind()
    assert await wait_for_pane(tmux, new_window_id, timeout=30) is not None
