"""Telegram replies reach the agent as a quote of the replied message."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Chat, ForumTopicCreated, Message, TextQuote, User

from ccgram.handlers.text import reply_quote
from ccgram.handlers.text.reply_quote import (
    LABEL_AGENT,
    LABEL_OTHER,
    LABEL_SELF,
    QUOTE_BUDGET,
    compose_with_quote,
    quoted_context,
    truncate_weighted,
    weighted_length,
)
from ccgram.handlers.text.text_handler import _forward_message

_TH = "ccgram.handlers.text.text_handler"
CHAT = Chat(id=-100, type="supergroup", is_forum=True)
ME = User(id=7, first_name="Jason", is_bot=False)
BOT = User(id=1, first_name="CcGramBot", is_bot=True)
STRANGER = User(id=9, first_name="Someone", is_bot=False)
THREAD = 42
NOW = datetime(2026, 11, 8, tzinfo=UTC)


def _msg(message_id: int, text: str | None = None, **kwargs) -> Message:
    return Message(
        message_id=message_id,
        date=NOW,
        chat=CHAT,
        text=text,
        message_thread_id=kwargs.pop("message_thread_id", THREAD),
        **kwargs,
    )


def _reply(to: Message, text: str = "why?", **kwargs) -> Message:
    return _msg(200, text, from_user=ME, reply_to_message=to, **kwargs)


def test_reply_to_the_agent_is_quoted() -> None:
    target = _msg(100, "Step 3 uses X.\nThen Y.", from_user=BOT)
    assert compose_with_quote("why?", _reply(target)) == (
        f"{LABEL_AGENT}\n> Step 3 uses X.\n> Then Y.\n\nwhy?"
    )


def test_partial_quote_wins_over_the_whole_message() -> None:
    target = _msg(100, "First sentence. Second sentence.", from_user=BOT)
    message = _reply(target, quote=TextQuote("Second sentence.", position=16))
    assert quoted_context(message) == f"{LABEL_AGENT}\n> Second sentence."


def test_labels_follow_the_author() -> None:
    mine = _msg(100, "my earlier ask", from_user=ME)
    theirs = _msg(101, "their note", from_user=STRANGER)
    anonymous = _msg(102, "no author")
    assert quoted_context(_reply(mine)).startswith(LABEL_SELF)  # type: ignore[union-attr]
    assert quoted_context(_reply(theirs)).startswith(LABEL_OTHER)  # type: ignore[union-attr]
    assert quoted_context(_reply(anonymous)).startswith(LABEL_OTHER)  # type: ignore[union-attr]


def test_the_topic_root_is_not_a_reply() -> None:
    # In a forum topic every message points at the topic's creation message.
    root = _msg(
        THREAD,
        forum_topic_created=ForumTopicCreated(name="aivideo", icon_color=0),
    )
    plain = _msg(200, "hello", from_user=ME, reply_to_message=root)
    assert compose_with_quote("hello", plain) == "hello"

    # Even without the service payload, the topic id itself is the root.
    bare_root = _msg(THREAD, "x", from_user=BOT)
    assert quoted_context(_reply(bare_root)) is None


def test_caption_is_quoted_and_media_without_text_is_not() -> None:
    photo = _msg(100, caption="the chart", from_user=BOT)
    sticker = _msg(101, from_user=BOT)
    assert quoted_context(_reply(photo)) == f"{LABEL_AGENT}\n> the chart"
    assert compose_with_quote("look", _reply(sticker)) == "look"


def test_no_reply_leaves_text_alone() -> None:
    assert compose_with_quote("hello", _msg(200, "hello", from_user=ME)) == "hello"


def test_blank_lines_keep_the_quote_marker() -> None:
    target = _msg(100, "a\n\nb", from_user=BOT)
    assert quoted_context(_reply(target)) == f"{LABEL_AGENT}\n> a\n>\n> b"


def test_weighted_length_counts_cjk_three_times() -> None:
    assert weighted_length("abc") == 3
    assert weighted_length("引用") == 6
    assert weighted_length("Ａ") == 3  # full-width Latin


def test_truncation_keeps_head_and_tail_within_budget() -> None:
    english = "head " + "x" * 5000 + " tail"
    chinese = "开头" + "中" * 2000 + "结尾"
    for text in (english, chinese):
        cut = truncate_weighted(text)
        assert weighted_length(cut) <= QUOTE_BUDGET
        assert "…" in cut
        assert cut.startswith(text[:2])
        assert cut.endswith(text[-2:])
    # About 1000 Chinese characters or 3000 English ones fit.
    assert truncate_weighted("中" * 1000) == "中" * 1000
    assert truncate_weighted("x" * 3000) == "x" * 3000


async def test_forwarding_sends_the_quote_and_history_keeps_the_typed_text() -> None:
    target = _msg(100, "Step 3 uses X.", from_user=BOT)
    message = MagicMock(wraps=_reply(target, text="why?"))
    message.chat = MagicMock(id=CHAT.id, send_action=AsyncMock())
    message.message_id = 200
    message.reply_to_message = target
    message.quote = None
    message.message_thread_id = THREAD
    message.from_user = ME
    with (
        patch(f"{_TH}.get_interactive_window", return_value=None),
        patch(f"{_TH}.enqueue_status_update", new=AsyncMock()),
        patch(f"{_TH}.ack_reaction", new=AsyncMock()),
        patch(
            f"{_TH}.send_telegram_to_window",
            new=AsyncMock(return_value=(True, "ok")),
        ) as send,
        patch("ccgram.handlers.command_history.record_command") as record,
    ):
        await _forward_message(
            "@0", 7, THREAD, "why?", AsyncMock(), message, adapter=MagicMock()
        )
    assert send.call_args.args[3] == f"{LABEL_AGENT}\n> Step 3 uses X.\n\nwhy?"
    record.assert_called_once_with(7, THREAD, "why?")


async def test_bash_commands_are_never_quoted() -> None:
    with patch.object(reply_quote, "quoted_context", return_value="Q") as quote:
        message = MagicMock()
        message.chat = MagicMock(id=CHAT.id, send_action=AsyncMock())
        with (
            patch(f"{_TH}.get_interactive_window", return_value=None),
            patch(f"{_TH}.enqueue_status_update", new=AsyncMock()),
            patch(f"{_TH}.ack_reaction", new=AsyncMock()),
            patch(f"{_TH}.asyncio.create_task"),
            patch(
                f"{_TH}.send_telegram_to_window",
                new=AsyncMock(return_value=(True, "ok")),
            ) as send,
        ):
            await _forward_message(
                "@0", 7, THREAD, "!ls", AsyncMock(), message, adapter=MagicMock()
            )
    assert send.call_args.args[3] == "!ls"
    quote.assert_not_called()
