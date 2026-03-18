"""Shared E2E test helpers — update factories, polling, topic setup."""

import asyncio
import re

from telegram import (
    CallbackQuery,
    Chat,
    Message,
    MessageEntity,
    Update,
    User,
)

TEST_USER_ID = 12345
TEST_CHAT_ID = -100999
TEST_THREAD_ID = 42

# Monotonically increasing IDs for factory functions
_next_update_id = 1000
_next_message_id = 5000


def _bump_update_id():
    global _next_update_id
    _next_update_id += 1
    return _next_update_id


def _bump_message_id():
    global _next_message_id
    _next_message_id += 1
    return _next_message_id


# ---------------------------------------------------------------------------
# Update factories
# ---------------------------------------------------------------------------


def make_text_update(
    text,
    *,
    bot=None,
    thread_id=TEST_THREAD_ID,
    user_id=TEST_USER_ID,
    chat_id=TEST_CHAT_ID,
):
    """Build a text Update with optional bot_command entity."""
    update_id = _bump_update_id()
    user = User(id=user_id, first_name="TestUser", is_bot=False)
    chat = Chat(id=chat_id, type="supergroup")
    entities = None
    if text and text.startswith("/"):
        cmd_end = text.index(" ") if " " in text else len(text)
        entities = [
            MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=cmd_end)
        ]
    message = Message(
        message_id=update_id,
        date=None,
        chat=chat,
        from_user=user,
        text=text,
        entities=entities,
        message_thread_id=thread_id,
    )
    update = Update(update_id=update_id, message=message)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
        for entity in entities or []:
            entity.set_bot(bot)
    return update


def make_callback_update(
    data,
    message_id,
    *,
    bot=None,
    thread_id=TEST_THREAD_ID,
    user_id=TEST_USER_ID,
    chat_id=TEST_CHAT_ID,
):
    """Build a CallbackQuery Update."""
    update_id = _bump_update_id()
    user = User(id=user_id, first_name="TestUser", is_bot=False)
    chat = Chat(id=chat_id, type="supergroup")
    message = Message(
        message_id=message_id,
        date=None,
        chat=chat,
        from_user=user,
        text="(callback source)",
        message_thread_id=thread_id,
    )
    callback_query = CallbackQuery(
        id=str(update_id),
        chat_instance="test",
        from_user=user,
        data=data,
        message=message,
    )
    update = Update(update_id=update_id, callback_query=callback_query)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
        callback_query.set_bot(bot)
    return update


# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------


async def wait_for_send(
    calls,
    *,
    method="sendMessage",
    predicate=None,
    timeout=120.0,
    poll_interval=0.3,
):
    """Poll intercepted calls until a matching entry appears.

    Returns the data dict of the first match. Raises TimeoutError on expiry.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for endpoint, data in calls:
            if endpoint == method and (predicate is None or predicate(data)):
                return data
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"No {method} call matching predicate within {timeout}s "
        f"(total calls: {len(calls)})"
    )


async def wait_for_pane(
    tmux_manager,
    window_id,
    *,
    pattern=None,
    timeout=60.0,
    poll_interval=1.0,
):
    """Poll capture_pane until content matches pattern.

    pattern can be a regex string or plain substring.
    Returns the captured pane text on match. Raises TimeoutError on expiry.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        content = await tmux_manager.capture_pane(window_id)
        if content is not None:
            if pattern is None:
                return content
            if isinstance(pattern, str) and re.search(pattern, content):
                return content
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"Pane {window_id} did not match pattern {pattern!r} within {timeout}s"
    )


def find_sends(calls, *, method="sendMessage", predicate=None):
    """Return all intercepted calls matching criteria."""
    results = []
    for endpoint, data in calls:
        if endpoint == method and (predicate is None or predicate(data)):
            results.append(data)
    return results


def find_message_id_for(calls, *, method="sendMessage", predicate=None):
    """Find the message_id from the interceptor's response for a matching call.

    Since our router returns incrementing message_ids, we track them by position.
    Returns the message_id that was assigned to the matching call.
    """
    msg_id_counter = 5000  # matches _next_message_id starting value
    for endpoint, data in calls:
        if endpoint in ("sendMessage", "sendPhoto", "sendDocument"):
            msg_id_counter += 1
            if endpoint == method and (predicate is None or predicate(data)):
                return msg_id_counter
    return None


# ---------------------------------------------------------------------------
# Bound topic helper
# ---------------------------------------------------------------------------


async def setup_bound_topic(
    app,
    calls,
    work_dir,
    *,
    provider="claude",
    thread_id=TEST_THREAD_ID,
    user_id=TEST_USER_ID,
    chat_id=TEST_CHAT_ID,
    initial_text="hello agent",
):
    """Drive through dir browser → provider → mode → bound topic.

    Returns (window_id, browser_msg_id).
    """
    bot = app.bot

    # Phase 1: Send text to trigger directory browser
    u1 = make_text_update(
        initial_text,
        bot=bot,
        thread_id=thread_id,
        user_id=user_id,
        chat_id=chat_id,
    )
    await app.process_update(u1)

    # Wait for the directory browser or window picker to appear
    await wait_for_send(
        calls,
        predicate=lambda d: "Select" in d.get("text", ""),
        timeout=10.0,
    )

    # Find the message_id for the browser message
    browser_msg_id = find_message_id_for(
        calls,
        predicate=lambda d: "Select" in d.get("text", ""),
    )
    assert browser_msg_id is not None, "Could not find browser message_id"

    # Set the browse_path in user_data to our work_dir
    # Access the mutable internal defaultdict (app.user_data is a read-only proxy)
    from ccgram.handlers.directory_browser import BROWSE_PATH_KEY

    user_data = app._user_data[user_id]  # defaultdict auto-creates entry
    user_data[BROWSE_PATH_KEY] = str(work_dir)

    # Phase 2: Confirm directory → provider picker
    u2 = make_callback_update(
        "db:confirm",
        browser_msg_id,
        bot=bot,
        thread_id=thread_id,
        user_id=user_id,
        chat_id=chat_id,
    )
    await app.process_update(u2)
    await asyncio.sleep(0.5)

    # Phase 3: Select provider → mode picker
    u3 = make_callback_update(
        f"prov:{provider}",
        browser_msg_id,
        bot=bot,
        thread_id=thread_id,
        user_id=user_id,
        chat_id=chat_id,
    )
    await app.process_update(u3)
    await asyncio.sleep(0.5)

    # Phase 4: Select mode → window created
    u4 = make_callback_update(
        f"mode:{provider}:normal",
        browser_msg_id,
        bot=bot,
        thread_id=thread_id,
        user_id=user_id,
        chat_id=chat_id,
    )
    await app.process_update(u4)

    # Wait for the "Bound to this topic" confirmation
    await wait_for_send(
        calls,
        method="editMessageText",
        predicate=lambda d: "Bound" in d.get("text", ""),
        timeout=30.0,
    )

    # Resolve the window_id from the session manager
    from ccgram.session import session_manager

    window_id = session_manager.get_window_for_thread(user_id, thread_id)
    assert window_id is not None, "Topic not bound after setup flow"

    return window_id, browser_msg_id
