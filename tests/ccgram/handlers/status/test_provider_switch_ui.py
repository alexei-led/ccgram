"""Provider identity should stay visible without duplicating topic chrome."""

from unittest.mock import AsyncMock

import pytest

from ccgram.handlers.agent_command import _picker_text
from ccgram.handlers import hook_events
from ccgram.config import config
from ccgram.providers.base import HookEvent
from ccgram.handlers.status import topic_emoji, provider_switch
from telegram.error import RetryAfter, NetworkError
from ccgram.thread_router import ThreadRouter, get_thread_router, install_thread_router
from ccgram.window_state_store import (
    WindowState,
    WindowStateStore,
    get_window_store,
    install_window_store,
)

WINDOW = "@871"
CHAT = -871
THREAD = 871


@pytest.fixture
def bound_topic():
    old_store, old_router = get_window_store(), get_thread_router()
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
    )
    router = ThreadRouter(
        schedule_save=lambda: None,
        has_window_state=store.has_window,
        default_group_id=CHAT,
    )
    install_window_store(store)
    install_thread_router(router)
    store.window_states[WINDOW] = WindowState(
        window_name="router", provider_name="pi", initial_provider_name="pi"
    )
    router.bind_thread(1, THREAD, WINDOW, window_name="router", chat_id=CHAT)
    topic_emoji.reset_all_state()
    provider_switch.clear_provider_switch_state(CHAT, THREAD)
    yield store
    topic_emoji.clear_topic_emoji_state(CHAT, THREAD)
    provider_switch.clear_provider_switch_state(CHAT, THREAD)
    topic_emoji.reset_all_state()
    install_window_store(old_store)
    install_thread_router(old_router)


@pytest.mark.parametrize("name", ["router", "Pi · router", "Pi ▸ router"])
async def test_topic_has_one_provider_prefix_and_keeps_existing_status(
    bound_topic, name
):
    client = AsyncMock()
    topic_emoji.mark_awaiting_first_paint(CHAT, THREAD)
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", name)
    assert (
        client.edit_forum_topic.call_args.kwargs["name"]
        == f"{topic_emoji.EMOJI_IDLE} Pi · router"
    )


async def test_provider_change_refreshes_title_without_lifecycle_change(bound_topic):
    client = AsyncMock()
    topic_emoji.mark_awaiting_first_paint(CHAT, THREAD)
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    bound_topic.window_states[WINDOW].provider_name = "claude"
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    assert client.edit_forum_topic.await_count == 2
    assert (
        client.edit_forum_topic.call_args.kwargs["name"]
        == f"{topic_emoji.EMOJI_IDLE} Claude · router"
    )


async def test_long_title_keeps_provider_within_telegram_limit(bound_topic):
    client = AsyncMock()
    topic_emoji.mark_awaiting_first_paint(CHAT, THREAD)
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "project" * 40)
    title = client.edit_forum_topic.call_args.kwargs["name"]
    assert title.startswith(f"{topic_emoji.EMOJI_IDLE} Pi · ")
    assert len(title) <= 128


async def test_provider_title_change_retries_after_flood_pause(
    bound_topic, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(topic_emoji.time, "monotonic", lambda: clock[0])
    client = AsyncMock()
    topic_emoji.mark_awaiting_first_paint(CHAT, THREAD)
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    bound_topic.window_states[WINDOW].provider_name = "claude"
    client.edit_forum_topic.side_effect = [RetryAfter(1), None]
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    clock[0] = 1
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    assert client.edit_forum_topic.await_count == 2
    clock[0] = topic_emoji.FLOOD_COOLDOWN_SECONDS + 1
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    assert client.edit_forum_topic.await_count == 3
    assert "Claude · router" in client.edit_forum_topic.call_args.kwargs["name"]


async def test_forced_sync_failure_keeps_provider_name_dirty_for_next_poll(bound_topic):
    client = AsyncMock()
    topic_emoji.mark_awaiting_first_paint(CHAT, THREAD)
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    bound_topic.window_states[WINDOW].provider_name = "claude"
    client.edit_forum_topic.side_effect = [NetworkError("offline"), None]
    await topic_emoji.sync_topic_name(client, CHAT, THREAD, "router")
    await topic_emoji.update_topic_emoji(client, CHAT, THREAD, "idle", "router")
    assert client.edit_forum_topic.await_count == 3


def test_picker_uses_name_not_opaque_identity(bound_topic):
    text = _picker_text(WINDOW)
    assert "router" in text and "Pi" in text and "Auto" in text
    assert WINDOW not in text


async def test_manual_change_during_stop_summary_does_not_update_new_provider(
    bound_topic, monkeypatch
):
    state = bound_topic.window_states[WINDOW]
    state.provider_name = "claude"
    state.provider_manual_override = True
    state.transcript_path = "/transcript.jsonl"
    monkeypatch.setattr(config, "multiplexer_name", "tmux")

    async def delayed_summary(_path):
        state.provider_name = "shell"
        return "old summary"

    monkeypatch.setattr(hook_events, "_get_llm_summary", delayed_summary)
    enqueue = AsyncMock()
    monkeypatch.setattr(hook_events, "enqueue_status_update", enqueue)
    event = HookEvent(
        event_type="Stop",
        session_id="old",
        timestamp=0.0,
        window_key=f"{config.tmux_session_name}:{WINDOW}",
        data={"provider_name": "claude"},
    )
    await hook_events._handle_stop(event, AsyncMock())
    enqueue.assert_not_called()


async def test_manual_change_during_notification_wait_clears_old_interactive_mode(
    bound_topic, monkeypatch
):
    state = bound_topic.window_states[WINDOW]
    state.provider_name = "claude"
    state.provider_manual_override = True
    monkeypatch.setattr(config, "multiplexer_name", "tmux")

    async def render_wait(_delay):
        state.provider_name = "shell"

    monkeypatch.setattr(hook_events.asyncio, "sleep", render_wait)
    event = HookEvent(
        event_type="Notification",
        session_id="old",
        timestamp=0.0,
        window_key=f"{config.tmux_session_name}:{WINDOW}",
        data={"provider_name": "claude"},
    )
    await hook_events._handle_notification(event, AsyncMock())
    assert hook_events.get_interactive_window(1, THREAD, chat_id=CHAT) is None


@pytest.fixture
def notices(bound_topic, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(provider_switch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(provider_switch, "rate_limit_send", AsyncMock())
    return bound_topic, clock, AsyncMock()


async def observe(notices, provider: str, now: float):
    store, clock, client = notices
    store.window_states[WINDOW].provider_name = provider
    clock[0] = now
    await provider_switch.observe_provider_switch(
        client, CHAT, THREAD, WINDOW, "router"
    )
    return client


async def test_one_quiet_notice_after_stable_change_not_startup_or_each_poll(notices):
    client = await observe(notices, "pi", 0)
    await observe(notices, "pi", 4)
    client.send_message.assert_not_called()
    await observe(notices, "claude", 5)
    client.send_message.assert_not_called()
    await observe(notices, "claude", 9)
    await observe(notices, "claude", 20)
    client.send_message.assert_awaited_once_with(
        chat_id=CHAT,
        message_thread_id=THREAD,
        text="Pi → Claude.",
        disable_notification=True,
    )
    assert "Claude · router" in client.edit_forum_topic.call_args.kwargs["name"]


async def test_transient_shell_does_not_emit_notice_but_real_terminal_change_explains_routing(
    notices,
):
    client = await observe(notices, "pi", 0)
    await observe(notices, "shell", 1)
    await observe(notices, "pi", 2)
    await observe(notices, "pi", 8)
    client.send_message.assert_not_called()
    await observe(notices, "shell", 10)
    await observe(notices, "shell", 14)
    assert client.send_message.await_count == 1
    assert "Terminal mode" in client.send_message.call_args.kwargs["text"]


async def test_manual_reply_suppresses_duplicate_switch_notice(notices):
    client = await observe(notices, "pi", 0)
    await observe(notices, "claude", 1)
    provider_switch.remember_provider_selection(CHAT, THREAD, WINDOW, "claude")
    await observe(notices, "claude", 10)
    client.send_message.assert_not_called()


async def test_notice_failure_backs_off_and_retries_once(notices):
    client = await observe(notices, "pi", 0)
    client.send_message.side_effect = [RetryAfter(2), object()]
    await observe(notices, "claude", 1)
    await observe(notices, "claude", 5)
    await observe(notices, "claude", 6)
    assert client.send_message.await_count == 1
    await observe(notices, "claude", 36)
    await observe(notices, "claude", 40)
    assert client.send_message.await_count == 2


async def test_superseded_notice_is_not_sent_after_rate_limit_wait(
    notices, monkeypatch
):
    store, _clock, client = notices
    await observe(notices, "pi", 0)
    await observe(notices, "claude", 1)

    async def switch_while_waiting(_chat):
        store.window_states[WINDOW].provider_name = "shell"
        provider_switch.remember_provider_selection(CHAT, THREAD, WINDOW, "shell")

    monkeypatch.setattr(provider_switch, "rate_limit_send", switch_while_waiting)
    await observe(notices, "claude", 5)
    client.send_message.assert_not_called()


async def test_topic_cleanup_starts_a_new_silent_baseline(notices):
    client = await observe(notices, "pi", 0)
    provider_switch.clear_provider_switch_state(CHAT, THREAD)
    await observe(notices, "claude", 10)
    await observe(notices, "claude", 20)
    client.send_message.assert_not_called()
