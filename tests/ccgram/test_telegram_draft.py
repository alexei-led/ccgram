from unittest.mock import AsyncMock, MagicMock

import asyncio
from datetime import timedelta

import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, RetryAfter, TelegramError

from ccgram import telegram_draft
from ccgram.telegram_draft import (
    DRAFT_LEGACY,
    DRAFT_STREAMING,
    DraftStream,
    is_draft_unavailable,
    is_peer_draft_unsupported,
    mark_draft_unavailable,
    reset_draft_state,
)


@pytest.fixture(autouse=True)
def _reset_draft_state(monkeypatch):
    monkeypatch.setattr(telegram_draft, "_MIN_DRAFT_INTERVAL", 0.0)
    reset_draft_state()
    yield
    reset_draft_state()


def _make_bot(*, send_id=42):
    bot = MagicMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    bot.do_api_request = AsyncMock(return_value=True)
    sent_msg = MagicMock(message_id=send_id)
    bot.send_message = AsyncMock(return_value=sent_msg)
    bot.edit_message_text = AsyncMock(return_value=None)
    bot.delete_message = AsyncMock(return_value=None)
    return bot


def _draft_calls(bot):
    return bot.send_message_draft.call_args_list


class TestDraftStreamHappyPath:
    async def test_start_uses_nonzero_draft_id_and_returns_no_message_id(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        message_id = await stream.start("hello")

        assert message_id is None
        assert stream.message_id is None
        assert stream.mode == DRAFT_STREAMING
        call = _draft_calls(bot)[0]
        payload = call.kwargs
        assert payload["chat_id"] == 100
        assert payload["text"] == "hello"
        assert payload["draft_id"] > 0
        assert "reply_to_message_id" not in payload
        assert "reply_markup" not in payload
        bot.send_message.assert_not_awaited()

    async def test_updates_reuse_draft_id_and_finalize_with_send_message(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100, message_thread_id=5)

        await stream.start("a")
        await stream.append("b")
        await stream.replace("abc")
        await stream.finalize("final")

        calls = _draft_calls(bot)
        assert [call.kwargs["text"] for call in calls] == [
            "a",
            "ab",
            "abc",
        ]
        draft_ids = {call.kwargs["draft_id"] for call in calls}
        assert len(draft_ids) == 1
        bot.send_message.assert_awaited_once_with(
            chat_id=100,
            text="final",
            message_thread_id=5,
        )
        assert stream.message_id == 42
        assert stream.closed is True

    async def test_confirmed_replace_propagates_stream_failure(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)
        await stream.start("draft")
        bot.send_message_draft.side_effect = TelegramError("update failed")

        with pytest.raises(TelegramError, match="update failed"):
            await stream.replace_confirmed("updated")

    async def test_finalize_uses_final_text_without_extra_draft_call(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        await stream.start("draft")
        await stream.finalize("final")

        assert len(_draft_calls(bot)) == 1
        bot.send_message.assert_awaited_once_with(chat_id=100, text="final")

    async def test_throttled_updates_flush_latest_snapshot(self, monkeypatch) -> None:
        monkeypatch.setattr(telegram_draft, "_MIN_DRAFT_INTERVAL", 0.01)
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        await stream.start("a")
        await stream.append("b")
        await stream.append("c")
        await asyncio.sleep(0.02)

        assert [call.kwargs["text"] for call in _draft_calls(bot)] == ["a", "abc"]

    async def test_retry_after_defers_and_retries_latest_snapshot(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(telegram_draft, "_MIN_DRAFT_INTERVAL", 0.0)
        bot = _make_bot()
        bot.send_message_draft.side_effect = [
            True,
            RetryAfter(timedelta(milliseconds=1)),
            True,
        ]
        stream = DraftStream(bot, chat_id=100)

        await stream.start("a")
        await stream.append("b")
        assert bot.send_message_draft.await_count == 2

        await asyncio.sleep(0.02)
        assert bot.send_message_draft.await_count == 3
        assert bot.send_message_draft.call_args.kwargs["text"] == "ab"

    async def test_final_send_failure_keeps_stream_open_for_retry(self) -> None:
        bot = _make_bot()
        bot.send_message.side_effect = TelegramError("temporary")
        stream = DraftStream(bot, chat_id=100)

        await stream.start("draft")
        with pytest.raises(TelegramError, match="temporary"):
            await stream.finalize("final")

        assert stream.closed is False


class TestDraftStreamFallback:
    async def test_method_not_found_falls_back_to_legacy(self) -> None:
        bot = _make_bot()
        bot.send_message_draft.side_effect = BadRequest("method not found")

        stream = DraftStream(bot, chat_id=100)
        message_id = await stream.start("hi")

        assert message_id == 42
        assert stream.mode == DRAFT_LEGACY
        assert is_draft_unavailable() is True
        bot.send_message.assert_awaited_once_with(chat_id=100, text="hi")

    async def test_peer_invalid_is_cached_without_disabling_other_peers(self) -> None:
        bot = _make_bot()
        bot.send_message_draft.side_effect = BadRequest("draft_peer_invalid")

        stream = DraftStream(bot, chat_id=100, message_thread_id=5)
        await stream.start("hi")

        assert stream.mode == DRAFT_LEGACY
        assert is_draft_unavailable() is False
        assert is_peer_draft_unsupported(100, 5) is True

        other_bot = _make_bot()
        other = DraftStream(other_bot, chat_id=200, message_thread_id=5)
        await other.start("hi")
        assert other.mode == DRAFT_STREAMING
        assert _draft_calls(other_bot)

    async def test_legacy_updates_use_edit_message_text(self) -> None:
        mark_draft_unavailable("test")
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        await stream.start("a")
        await stream.append("b")
        await stream.finalize()

        assert bot.send_message_draft.await_count == 0
        assert bot.edit_message_text.await_count == 2
        assert bot.edit_message_text.call_args_list[-1].kwargs["text"] == "ab"

    async def test_legacy_final_edit_failure_keeps_stream_open(self) -> None:
        mark_draft_unavailable("test")
        bot = _make_bot()
        bot.edit_message_text.side_effect = TelegramError("temporary")
        stream = DraftStream(bot, chat_id=100)

        await stream.start("draft")
        with pytest.raises(TelegramError, match="temporary"):
            await stream.finalize("final")

        assert stream.closed is False

    async def test_streaming_failures_keep_native_draft_without_duplicate_message(
        self,
    ) -> None:
        bot = _make_bot()
        bot.send_message_draft.side_effect = [True, TelegramError("transient")]
        stream = DraftStream(bot, chat_id=100)

        await stream.start("a")
        await stream.append("b")
        assert stream.mode == DRAFT_STREAMING

        bot.send_message_draft.side_effect = TelegramError("transient")
        await stream.append("c")
        await stream.append("d")
        assert stream.mode == DRAFT_STREAMING
        assert bot.send_message.await_count == 0


class TestDraftStreamLifecycle:
    async def test_abort_does_not_delete_ephemeral_draft(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        await stream.start("hi")
        await stream.abort()

        assert stream.closed is True
        bot.delete_message.assert_not_awaited()

    async def test_abort_deletes_legacy_message(self) -> None:
        mark_draft_unavailable("test")
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        await stream.start("hi")
        await stream.abort()

        bot.delete_message.assert_awaited_once_with(chat_id=100, message_id=42)

    async def test_empty_start_does_not_call_api(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        assert await stream.start("") is None
        bot.do_api_request.assert_not_awaited()
        bot.send_message.assert_not_awaited()

    async def test_text_is_truncated_to_4096_for_draft_and_final(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        await stream.start("a" * 5000)
        await stream.finalize()

        assert len(_draft_calls(bot)[0].kwargs["text"]) == 4096
        assert len(bot.send_message.call_args.kwargs["text"]) == 4096

    async def test_guards(self) -> None:
        bot = _make_bot()
        stream = DraftStream(bot, chat_id=100)

        with pytest.raises(RuntimeError, match="not started"):
            await stream.append("x")
        await stream.start("hi")
        with pytest.raises(RuntimeError, match="start called twice"):
            await stream.start("again")
        await stream.finalize()
        with pytest.raises(RuntimeError, match="already closed"):
            await stream.append("x")


class TestDraftStreamMarkup:
    @staticmethod
    def _markup() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Esc", callback_data="esc")]]
        )

    async def test_markup_is_only_sent_with_persisted_message(self) -> None:
        mark_draft_unavailable("test")
        bot = _make_bot()
        markup = self._markup()
        stream = DraftStream(bot, chat_id=100, reply_markup=markup)

        await stream.start("hi")
        await stream.finalize("done")

        assert bot.send_message.call_args.kwargs["reply_markup"] is markup
        assert not _draft_calls(bot)


class TestModuleStateHelpers:
    def test_mark_unavailable_is_idempotent(self) -> None:
        mark_draft_unavailable("first")
        mark_draft_unavailable("second")
        assert telegram_draft.draft_unavailable_reason() == "first"

    def test_reset_clears_state(self) -> None:
        mark_draft_unavailable("x")
        reset_draft_state()
        assert is_draft_unavailable() is False
        assert telegram_draft.draft_unavailable_reason() == ""
