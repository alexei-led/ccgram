"""E2E tests for Claude Code lifecycle — binding, messaging, commands, recovery."""

import asyncio
import shutil

import pytest

from ._helpers import (
    TEST_THREAD_ID,
    TEST_USER_ID,
    _bump_message_id,
    find_message_id_for,
    make_callback_update,
    make_text_update,
    setup_bound_topic,
    wait_for_pane,
    wait_for_send,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        shutil.which("claude") is None, reason="claude CLI not installed"
    ),
]


# ---------------------------------------------------------------------------
# Test 1: Basic lifecycle — bind topic, forward message, get response
# ---------------------------------------------------------------------------


async def test_basic_lifecycle(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app

    # Phase 1-4: Bind topic via directory browser flow
    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Phase 5: Verify agent launched and pending message forwarded
    await wait_for_pane(tmux, window_id, pattern="hello agent|╭|>", timeout=30)

    # Phase 6: Wait for agent response delivered to topic
    await wait_for_send(
        calls,
        predicate=lambda d: (
            d.get("message_thread_id") == TEST_THREAD_ID
            and len(d.get("text", "")) > 10
            and "Bound" not in d.get("text", "")
            and "Select" not in d.get("text", "")
        ),
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Test 5: CCGram commands — /sessions, /screenshot
# ---------------------------------------------------------------------------


async def test_sessions_command(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    await setup_bound_topic(app, calls, work_dir)

    # Clear call log so we can find the /sessions response easily
    calls.clear()

    u = make_text_update("/sessions", bot=app.bot)
    await app.process_update(u)

    # Wait for the sendMessage from /sessions response
    await wait_for_send(calls, timeout=10)


async def test_screenshot_command(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    await setup_bound_topic(app, calls, work_dir)

    # Wait for agent to initialize
    await asyncio.sleep(3)
    calls.clear()

    u = make_text_update("/screenshot", bot=app.bot)
    await app.process_update(u)

    await wait_for_send(
        calls,
        method="sendDocument",
        timeout=15,
    )


# ---------------------------------------------------------------------------
# Test 4: Command forwarding — /help
# ---------------------------------------------------------------------------


async def test_help_command_forwarded(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Wait for agent to start
    await wait_for_pane(tmux, window_id, pattern="╭|>|\\$", timeout=30)
    calls.clear()

    u = make_text_update("/help", bot=app.bot)
    await app.process_update(u)

    # Verify /help was forwarded to the pane
    await wait_for_pane(tmux, window_id, pattern="help", timeout=15)


# ---------------------------------------------------------------------------
# Test 6: Dead window recovery — fresh start
# ---------------------------------------------------------------------------


async def test_recovery_fresh_start(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Wait for agent to start
    await wait_for_pane(tmux, window_id, pattern="╭|>|\\$", timeout=30)

    # Kill the window to simulate a dead session
    await tmux.kill_window(window_id)
    await asyncio.sleep(1)

    # Send a message to the dead topic → recovery UI
    calls.clear()
    u = make_text_update("are you there?", bot=app.bot)
    await app.process_update(u)

    # Should get recovery keyboard
    recovery_data = await wait_for_send(
        calls,
        predicate=lambda d: "no longer running" in d.get("text", ""),
        timeout=10,
    )
    assert recovery_data is not None

    recovery_msg_id = find_message_id_for(
        calls,
        predicate=lambda d: "no longer running" in d.get("text", ""),
    )
    assert recovery_msg_id is not None

    # Click "Fresh" recovery button
    u_fresh = make_callback_update(
        f"rec:f:{window_id}",
        recovery_msg_id,
        bot=app.bot,
    )
    await app.process_update(u_fresh)

    # Wait for new window to be created
    await asyncio.sleep(5)

    # Verify topic is now rebound
    new_window_id = session_mgr.get_window_for_thread(TEST_USER_ID, TEST_THREAD_ID)
    assert new_window_id is not None
    new_pane = await tmux.capture_pane(new_window_id)
    assert new_pane is not None


# ---------------------------------------------------------------------------
# Test 7: Dead window recovery — continue
# ---------------------------------------------------------------------------


async def test_recovery_continue(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Wait for agent to start
    await wait_for_pane(tmux, window_id, pattern="╭|>|\\$", timeout=30)

    # Kill the window
    await tmux.kill_window(window_id)
    await asyncio.sleep(1)

    # Trigger recovery UI
    calls.clear()
    u = make_text_update("continue please", bot=app.bot)
    await app.process_update(u)

    recovery_data = await wait_for_send(
        calls,
        predicate=lambda d: "no longer running" in d.get("text", ""),
        timeout=10,
    )
    assert recovery_data is not None

    recovery_msg_id = find_message_id_for(
        calls,
        predicate=lambda d: "no longer running" in d.get("text", ""),
    )

    # Click "Continue" recovery button
    u_cont = make_callback_update(
        f"rec:c:{window_id}",
        recovery_msg_id,
        bot=app.bot,
    )
    await app.process_update(u_cont)
    await asyncio.sleep(5)

    # Verify topic is rebound and window exists
    new_window_id = session_mgr.get_window_for_thread(TEST_USER_ID, TEST_THREAD_ID)
    assert new_window_id is not None
    new_pane = await tmux.capture_pane(new_window_id)
    assert new_pane is not None


# ---------------------------------------------------------------------------
# Test 8: Status transitions
# ---------------------------------------------------------------------------


async def test_status_transitions(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Wait for agent to start
    await wait_for_pane(tmux, window_id, pattern="╭|>|\\$", timeout=30)

    # After sending a message, status polling should send typing action
    calls.clear()
    u = make_text_update("say hello", bot=app.bot)
    await app.process_update(u)

    # Verify sendChatAction was called
    await wait_for_send(
        calls,
        method="sendChatAction",
        timeout=10,
    )


# ---------------------------------------------------------------------------
# Test 9: Multi-topic isolation
# ---------------------------------------------------------------------------


async def test_multi_topic_isolation(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app

    thread_a = 42
    thread_b = 99

    # Setup topic A
    window_a, _ = await setup_bound_topic(
        app, calls, work_dir, thread_id=thread_a, initial_text="you are agent A"
    )

    # Setup topic B (different thread_id)
    window_b, _ = await setup_bound_topic(
        app, calls, work_dir, thread_id=thread_b, initial_text="you are agent B"
    )

    # Verify different windows
    assert window_a != window_b

    # Verify each topic is bound to its own window
    assert session_mgr.get_window_for_thread(TEST_USER_ID, thread_a) == window_a
    assert session_mgr.get_window_for_thread(TEST_USER_ID, thread_b) == window_b

    # Send message to topic A and verify it arrives in window A
    calls.clear()
    u_a = make_text_update("say A", bot=app.bot, thread_id=thread_a)
    await app.process_update(u_a)

    # Verify typing action for thread_a appears (background polling may also
    # produce actions for thread_b, so filter specifically)
    await wait_for_send(
        calls,
        method="sendChatAction",
        predicate=lambda d: d.get("message_thread_id") == thread_a,
        timeout=10,
    )


# ---------------------------------------------------------------------------
# Test 10: Notification mode cycling
# ---------------------------------------------------------------------------


async def test_notification_mode_cycling(e2e_app, work_dir):
    app, calls, tmux, session_mgr = e2e_app
    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Wait for status message with notify button
    await asyncio.sleep(5)

    # Cycle notification mode: all → errors_only
    calls.clear()

    status_msg_id = _bump_message_id()
    u = make_callback_update(
        f"st:nfy:{window_id}",
        status_msg_id,
        bot=app.bot,
    )
    await app.process_update(u)

    notify_data = await wait_for_send(
        calls,
        method="answerCallbackQuery",
        timeout=5,
    )
    assert notify_data is not None

    # Cycle again: errors_only → muted
    calls.clear()
    u2 = make_callback_update(
        f"st:nfy:{window_id}",
        status_msg_id,
        bot=app.bot,
    )
    await app.process_update(u2)

    notify_data2 = await wait_for_send(
        calls,
        method="answerCallbackQuery",
        timeout=5,
    )
    assert notify_data2 is not None
