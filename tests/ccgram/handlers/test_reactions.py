from unittest.mock import AsyncMock

import pytest
from telegram import ReactionTypeEmoji
from telegram.error import BadRequest, TelegramError

from ccgram.handlers.messaging_pipeline import message_sender
from ccgram.handlers.reactions import (
    ALLOWED_REACTIONS,
    REACT_DONE,
    REACT_FAIL,
    REACT_INBOX,
    REACT_RUNNING,
    REACT_SEEN,
    REACT_THINKING,
    _last_reaction,
    clear_reaction,
    react,
)


@pytest.fixture(autouse=True)
def _clear_dedupe_cache():
    _last_reaction.clear()
    yield
    _last_reaction.clear()


class TestAllowedReactions:
    @pytest.mark.parametrize(
        ("emoji", "allowed"),
        [
            pytest.param("👀", True, id="eyes"),
            pytest.param("🤔", True, id="thinking"),
            pytest.param("🔥", True, id="fire"),
            pytest.param("💔", True, id="broken-heart"),
            pytest.param("✅", False, id="check-mark"),
            pytest.param("❌", False, id="cross-mark"),
            pytest.param("📬", False, id="mailbox"),
            pytest.param("⚙", False, id="gear"),
        ],
    )
    def test_telegram_reaction_allowlist(self, emoji: str, allowed: bool) -> None:
        assert (emoji in ALLOWED_REACTIONS) is allowed

    def test_semantic_constants_are_valid(self) -> None:
        for emoji in (
            REACT_SEEN,
            REACT_THINKING,
            REACT_DONE,
            REACT_FAIL,
            REACT_INBOX,
            REACT_RUNNING,
        ):
            assert emoji in ALLOWED_REACTIONS, (
                f"semantic constant {emoji!r} must be in ALLOWED_REACTIONS"
            )


class TestReactSuccess:
    async def test_sets_single_emoji_reaction(self) -> None:
        bot = AsyncMock()
        ok = await react(bot, 100, 7, REACT_SEEN)
        assert ok is True
        bot.set_message_reaction.assert_awaited_once()
        kwargs = bot.set_message_reaction.call_args.kwargs
        assert kwargs["chat_id"] == 100
        assert kwargs["message_id"] == 7
        reactions = kwargs["reaction"]
        assert len(reactions) == 1
        assert isinstance(reactions[0], ReactionTypeEmoji)
        assert reactions[0].emoji == REACT_SEEN

    async def test_records_in_dedupe_cache(self) -> None:
        bot = AsyncMock()
        await react(bot, 100, 7, REACT_SEEN)
        assert _last_reaction[(100, 7)] == REACT_SEEN


class TestReactDisallowed:
    async def test_disallowed_emoji_returns_false(self) -> None:
        bot = AsyncMock()
        ok = await react(bot, 100, 7, "✅")
        assert ok is False
        bot.set_message_reaction.assert_not_awaited()

    async def test_disallowed_emoji_falls_back_to_toast(self) -> None:
        bot = AsyncMock()
        query = AsyncMock()
        ok = await react(
            bot, 100, 7, "✅", fallback_query=query, fallback_toast="✓ Sent"
        )
        assert ok is False
        query.answer.assert_awaited_once_with("✓ Sent", show_alert=False)

    async def test_disallowed_emoji_no_toast_when_query_missing(self) -> None:
        bot = AsyncMock()
        ok = await react(bot, 100, 7, "✅", fallback_toast="✓ Sent")
        assert ok is False


class TestReactFailure:
    async def test_bad_request_returns_false(self) -> None:
        bot = AsyncMock()
        bot.set_message_reaction.side_effect = BadRequest("message not found")
        ok = await react(bot, 100, 7, REACT_SEEN)
        assert ok is False
        assert (100, 7) not in _last_reaction

    async def test_bad_request_falls_back_to_toast(self) -> None:
        bot = AsyncMock()
        bot.set_message_reaction.side_effect = BadRequest("message not found")
        query = AsyncMock()
        ok = await react(
            bot, 100, 7, REACT_SEEN, fallback_query=query, fallback_toast="✓ Sent"
        )
        assert ok is False
        query.answer.assert_awaited_once_with("✓ Sent", show_alert=False)

    async def test_generic_telegram_error_returns_false(self) -> None:
        bot = AsyncMock()
        bot.set_message_reaction.side_effect = TelegramError("network down")
        ok = await react(bot, 100, 7, REACT_SEEN)
        assert ok is False

    async def test_toast_failure_does_not_propagate(self) -> None:
        bot = AsyncMock()
        bot.set_message_reaction.side_effect = BadRequest("message not found")
        query = AsyncMock()
        query.answer.side_effect = TelegramError("answer failed")
        ok = await react(
            bot, 100, 7, REACT_SEEN, fallback_query=query, fallback_toast="oops"
        )
        assert ok is False


class TestReactDedupe:
    async def test_repeat_same_emoji_skips_api_call(self) -> None:
        bot = AsyncMock()
        await react(bot, 100, 7, REACT_SEEN)
        await react(bot, 100, 7, REACT_SEEN)
        bot.set_message_reaction.assert_awaited_once()

    async def test_change_emoji_calls_api_again(self) -> None:
        bot = AsyncMock()
        await react(bot, 100, 7, REACT_SEEN)
        await react(bot, 100, 7, REACT_DONE)
        assert bot.set_message_reaction.await_count == 2
        assert _last_reaction[(100, 7)] == REACT_DONE

    async def test_dedupe_scoped_per_message(self) -> None:
        bot = AsyncMock()
        await react(bot, 100, 7, REACT_SEEN)
        await react(bot, 100, 8, REACT_SEEN)
        await react(bot, 200, 7, REACT_SEEN)
        assert bot.set_message_reaction.await_count == 3

    async def test_failed_call_not_cached(self) -> None:
        bot = AsyncMock()
        bot.set_message_reaction.side_effect = [BadRequest("nope"), None]
        first = await react(bot, 100, 7, REACT_SEEN)
        second = await react(bot, 100, 7, REACT_SEEN)
        assert first is False
        assert second is True
        assert bot.set_message_reaction.await_count == 2


class TestClearReaction:
    async def test_clear_sends_empty_reaction_list(self) -> None:
        bot = AsyncMock()
        ok = await clear_reaction(bot, 100, 7)
        assert ok is True
        kwargs = bot.set_message_reaction.call_args.kwargs
        assert kwargs["reaction"] == []

    async def test_clear_evicts_dedupe_entry(self) -> None:
        bot = AsyncMock()
        await react(bot, 100, 7, REACT_SEEN)
        assert (100, 7) in _last_reaction
        await clear_reaction(bot, 100, 7)
        assert (100, 7) not in _last_reaction

    async def test_clear_failure_returns_false(self) -> None:
        bot = AsyncMock()
        bot.set_message_reaction.side_effect = TelegramError("nope")
        ok = await clear_reaction(bot, 100, 7)
        assert ok is False


class TestMessageSenderReExports:
    def test_react_re_exported(self) -> None:
        assert message_sender.react is react
        assert message_sender.clear_reaction is clear_reaction
        assert message_sender.ALLOWED_REACTIONS is ALLOWED_REACTIONS
        assert message_sender.REACT_SEEN == REACT_SEEN
        assert message_sender.REACT_DONE == REACT_DONE
