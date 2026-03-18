"""E2E tests for voice message transcription lifecycle.

Tests the full flow: receive voice message → transcribe (mocked Whisper API)
→ confirm keyboard → send to agent / discard.

The Whisper API call is always mocked — no real audio API calls are made.
Real PTB Application, real tmux, real session binding (via setup_bound_topic).
"""

import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Message, Update, User

from ._helpers import (
    TEST_THREAD_ID,
    TEST_USER_ID,
    _bump_message_id,
    _bump_update_id,
    make_callback_update,
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
# Voice update factory
# ---------------------------------------------------------------------------


def make_voice_update(
    *,
    bot=None,
    file_id: str = "voice_file_id_abc",
    file_size: int = 32_000,
    duration: int = 5,
    thread_id: int = TEST_THREAD_ID,
    user_id: int = TEST_USER_ID,
    chat_id: int = -100999,
):
    """Build an Update with a Voice message."""
    from telegram import Voice

    update_id = _bump_update_id()
    user = User(id=user_id, first_name="TestUser", is_bot=False)
    chat = Chat(id=chat_id, type="supergroup")
    voice = Voice(
        file_id=file_id,
        file_unique_id=f"unique_{file_id}",
        duration=duration,
        file_size=file_size,
        mime_type="audio/ogg",
    )
    message = Message(
        message_id=_bump_message_id(),
        date=None,
        chat=chat,
        from_user=user,
        voice=voice,
        message_thread_id=thread_id,
    )
    update = Update(update_id=update_id, message=message)
    if bot:
        update.set_bot(bot)
        message.set_bot(bot)
    return update


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_FAKE_AUDIO = b"\x00" * 1024  # minimal fake OGG bytes


def _mock_get_file():
    """Return a mock Telegram File that yields fake audio bytes."""
    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(_FAKE_AUDIO))
    return mock_file


def _mock_transcriber(text: str = "hello from voice"):
    """Return a mock WhisperTranscriber that immediately returns a result."""
    from ccgram.whisper.base import TranscriptionResult

    mock = MagicMock()
    mock.transcribe = AsyncMock(
        return_value=TranscriptionResult(text=text, language="en")
    )
    return mock


# ---------------------------------------------------------------------------
# Test 1: Voice message when transcription not configured
# ---------------------------------------------------------------------------


async def test_voice_not_configured(e2e_app, work_dir):
    """Voice message without CCGRAM_WHISPER_PROVIDER → friendly setup hint."""
    app, calls, tmux, session_mgr = e2e_app

    await setup_bound_topic(app, calls, work_dir)
    calls.clear()

    import ccgram.handlers.voice_handler as vh

    with patch.object(vh, "get_transcriber", return_value=None):
        u = make_voice_update(bot=app.bot)
        await app.process_update(u)

    await wait_for_send(
        calls,
        predicate=lambda d: (
            "not configured" in d.get("text", "").lower()
            and "supported providers" in d.get("text", "").lower()
        ),
        timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Test 2: Voice message → transcription shown with confirm keyboard
# ---------------------------------------------------------------------------


async def test_voice_transcription_shows_confirm_keyboard(e2e_app, work_dir):
    """Successful transcription renders text + ✓/✗ inline keyboard."""
    app, calls, tmux, session_mgr = e2e_app

    await setup_bound_topic(app, calls, work_dir)
    calls.clear()

    import ccgram.handlers.voice_handler as vh

    mock_transcriber = _mock_transcriber("please add logging to main.py")

    with (
        patch.object(vh, "get_transcriber", return_value=mock_transcriber),
        patch.object(type(app.bot), "get_file", new_callable=AsyncMock) as mock_gf,
    ):
        mock_gf.return_value = _mock_get_file()

        u = make_voice_update(bot=app.bot)
        await app.process_update(u)

    # Initial sendMessage with transcription text
    sent = await wait_for_send(
        calls,
        predicate=lambda d: "please add logging" in d.get("text", ""),
        timeout=5.0,
    )
    assert sent is not None

    # editMessageReplyMarkup to attach vc:send / vc:drop keyboard
    await wait_for_send(
        calls,
        method="editMessageReplyMarkup",
        timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Test 3: Confirm (vc:send) → text forwarded to agent window
# ---------------------------------------------------------------------------


async def test_voice_confirm_sends_to_agent(e2e_app, work_dir):
    """Pressing ✓ Send to agent forwards transcription text to tmux window."""
    app, calls, tmux, session_mgr = e2e_app

    window_id, _ = await setup_bound_topic(app, calls, work_dir)

    # Wait for claude to start so pane is ready
    await wait_for_pane(tmux, window_id, pattern="╭|>|\\$", timeout=30)
    calls.clear()

    transcribed = "what is the purpose of config.py"

    import ccgram.handlers.voice_handler as vh

    mock_transcriber = _mock_transcriber(transcribed)

    with (
        patch.object(vh, "get_transcriber", return_value=mock_transcriber),
        patch.object(type(app.bot), "get_file", new_callable=AsyncMock) as mock_gf,
    ):
        mock_gf.return_value = _mock_get_file()

        u = make_voice_update(bot=app.bot)
        await app.process_update(u)

    # Wait for the transcription message to arrive
    await wait_for_send(
        calls,
        predicate=lambda d: transcribed in d.get("text", ""),
        timeout=5.0,
    )
    # Extract the confirm message_id from the reply keyboard edit
    await wait_for_send(calls, method="editMessageReplyMarkup", timeout=5.0)

    # Find the confirm msg_id from user_data
    user_data = app._user_data[TEST_USER_ID]
    from ccgram.handlers.user_state import VOICE_PENDING

    pending = user_data.get(VOICE_PENDING, {})
    assert pending, "Expected a pending voice entry in user_data"
    pending_key = next(iter(pending))
    confirm_msg_id = pending_key[1]

    calls.clear()

    # Press ✓ Send
    cb = make_callback_update(
        f"vc:send:{confirm_msg_id}",
        confirm_msg_id,
        bot=app.bot,
    )
    await app.process_update(cb)

    # Agent window should receive the transcribed text
    await wait_for_pane(tmux, window_id, pattern=transcribed[:20], timeout=15)

    # Pending state must be cleared
    assert pending_key not in user_data.get(VOICE_PENDING, {})


# ---------------------------------------------------------------------------
# Test 4: Discard (vc:drop) → message deleted, pending cleared
# ---------------------------------------------------------------------------


async def test_voice_discard_clears_pending(e2e_app, work_dir):
    """Pressing ✗ Discard deletes the transcription message and clears state."""
    app, calls, tmux, session_mgr = e2e_app

    await setup_bound_topic(app, calls, work_dir)
    calls.clear()

    import ccgram.handlers.voice_handler as vh

    mock_transcriber = _mock_transcriber("discard this message")

    with (
        patch.object(vh, "get_transcriber", return_value=mock_transcriber),
        patch.object(type(app.bot), "get_file", new_callable=AsyncMock) as mock_gf,
    ):
        mock_gf.return_value = _mock_get_file()

        u = make_voice_update(bot=app.bot)
        await app.process_update(u)

    await wait_for_send(
        calls,
        predicate=lambda d: "discard this message" in d.get("text", ""),
        timeout=5.0,
    )
    await wait_for_send(calls, method="editMessageReplyMarkup", timeout=5.0)

    user_data = app._user_data[TEST_USER_ID]
    from ccgram.handlers.user_state import VOICE_PENDING

    pending = user_data.get(VOICE_PENDING, {})
    assert pending
    pending_key = next(iter(pending))
    confirm_msg_id = pending_key[1]

    calls.clear()

    # Press ✗ Discard
    cb = make_callback_update(
        f"vc:drop:{confirm_msg_id}",
        confirm_msg_id,
        bot=app.bot,
    )
    await app.process_update(cb)

    # deleteMessage must be called for the transcription message
    await wait_for_send(calls, method="deleteMessage", timeout=5.0)

    # Pending state must be cleared
    assert pending_key not in user_data.get(VOICE_PENDING, {})


# ---------------------------------------------------------------------------
# Test 5: Voice message too large → size error, no API call
# ---------------------------------------------------------------------------


async def test_voice_too_large(e2e_app, work_dir):
    """Voice file over 25 MB is rejected before download."""
    app, calls, tmux, session_mgr = e2e_app

    await setup_bound_topic(app, calls, work_dir)
    calls.clear()

    import ccgram.handlers.voice_handler as vh

    mock_transcriber = _mock_transcriber()

    with patch.object(vh, "get_transcriber", return_value=mock_transcriber):
        u = make_voice_update(
            bot=app.bot,
            file_size=26 * 1024 * 1024,  # 26 MB — over limit
        )
        await app.process_update(u)

    sent = await wait_for_send(
        calls,
        predicate=lambda d: "too large" in d.get("text", "").lower(),
        timeout=5.0,
    )
    assert sent is not None
    # transcriber.transcribe must never have been called
    mock_transcriber.transcribe.assert_not_called()
