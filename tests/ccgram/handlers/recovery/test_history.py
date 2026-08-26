"""Tests for history helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.expandable_quote import EXPANDABLE_QUOTE_END, EXPANDABLE_QUOTE_START
from ccgram.handlers.callback_data import CB_HISTORY_NEXT, CB_HISTORY_PREV
from ccgram.handlers.recovery.history import (
    _build_history_keyboard,
    _format_timestamp,
    history_command,
    send_history,
)
from ccgram.telegram_client import FakeTelegramClient

_H = "ccgram.handlers.recovery.history"


def _message(text: str, **overrides) -> dict:
    return {"text": text, "role": "assistant", **overrides}


def _recent_messages(messages: list[dict], total: int | None = None):
    return patch(
        f"{_H}.session_query.get_recent_messages",
        new_callable=AsyncMock,
        return_value=(messages, len(messages) if total is None else total),
    )


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("ts", "expected"),
        [
            ("2024-01-15T14:32:00.000Z", "14:32"),
            ("2024-01-15T14:32:00Z", "14:32"),
            ("2024-01-15T14:32:00+05:30", "14:32"),
            ("2024-01-15T14:32:59", "14:32"),
            ("2024-01-15 14:32:00", "14:32"),
            ("not-a-timestamp", ""),
            ("", ""),
            (None, ""),
        ],
        ids=[
            "standard-iso-with-Z",
            "no-millis-with-Z",
            "timezone-offset",
            "no-timezone",
            "space-separator",
            "invalid-string",
            "empty-string",
            "none",
        ],
    )
    def test_format_timestamp(self, ts: str | None, expected: str) -> None:
        assert _format_timestamp(ts) == expected


class TestBuildHistoryKeyboard:
    def test_single_page_has_no_keyboard(self) -> None:
        assert _build_history_keyboard("@7", page_index=0, total_pages=1) is None

    @pytest.mark.parametrize(
        ("page_index", "total_pages", "expected_labels"),
        [
            (0, 3, ["1/3", "Newer ▶"]),
            (1, 3, ["◀ Older", "2/3", "Newer ▶"]),
            (2, 3, ["◀ Older", "3/3"]),
        ],
        ids=["first-page", "middle-page", "last-page"],
    )
    def test_navigation_buttons(
        self, page_index: int, total_pages: int, expected_labels: list[str]
    ) -> None:
        kb = _build_history_keyboard("@7", page_index, total_pages)

        assert kb is not None
        assert [b.text for b in kb.inline_keyboard[0]] == expected_labels

    def test_byte_range_round_trips_through_callback_data(self) -> None:
        kb = _build_history_keyboard(
            "@7", page_index=1, total_pages=3, start_byte=100, end_byte=250
        )

        assert kb is not None
        older, _counter, newer = kb.inline_keyboard[0]
        assert older.callback_data == f"{CB_HISTORY_PREV}0:@7:100:250"
        assert newer.callback_data == f"{CB_HISTORY_NEXT}2:@7:100:250"


class TestSendHistoryContent:
    """Rendering of the transcript page itself."""

    @pytest.fixture(autouse=True)
    def _router(self):
        with patch(f"{_H}.thread_router") as router:
            router.get_display_name.return_value = "win-name"
            router.resolve_chat_id.return_value = -100
            yield router

    async def _render(self, messages: list[dict], **kwargs) -> str:
        target = MagicMock()
        target.reply_text = AsyncMock()
        with (
            _recent_messages(messages),
            patch(f"{_H}.safe_reply", new_callable=AsyncMock) as reply,
        ):
            await send_history(target, "@7", **kwargs)
        return reply.call_args.args[1]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, "📋 [win-name] No messages yet."),
            ({"start_byte": 100}, "📬 [win-name] No unread messages."),
        ],
        ids=["full-history", "unread-range"],
    )
    async def test_empty_transcript_message(self, kwargs: dict, expected: str) -> None:
        assert await self._render([], **kwargs) == expected

    async def test_header_counts_messages(self) -> None:
        text = await self._render([_message("hi")])

        assert text.startswith("📋 [win-name] Messages (1 total)")

    async def test_unread_header(self) -> None:
        text = await self._render([_message("hi")], start_byte=10, end_byte=20)

        assert text.startswith("📬 [win-name] 1 unread messages")

    @pytest.mark.parametrize(
        ("message", "expected_line"),
        [
            (_message("hello", role="user"), "👤 hello"),
            (_message("assistant reply"), "assistant reply"),
            (
                _message("pondering", content_type="thinking"),
                "\U0001f9e0 Thinking…\npondering",
            ),
        ],
        ids=["user", "assistant", "thinking"],
    )
    async def test_message_prefixes(self, message: dict, expected_line: str) -> None:
        text = await self._render([message])

        assert expected_line in text

    async def test_expandable_quote_sentinels_are_stripped(self) -> None:
        quoted = f"{EXPANDABLE_QUOTE_START}quoted body{EXPANDABLE_QUOTE_END}"

        text = await self._render([_message(quoted)])

        assert "quoted body" in text
        assert EXPANDABLE_QUOTE_START not in text
        assert EXPANDABLE_QUOTE_END not in text

    async def test_timestamped_separator(self) -> None:
        text = await self._render([_message("hi", timestamp="2024-01-15T14:32:00Z")])

        assert "───── 14:32 ─────" in text

    async def test_separator_without_a_usable_timestamp(self) -> None:
        text = await self._render([_message("hi", timestamp="nonsense")])

        assert "─────────────" in text


class TestSendHistoryPagination:
    @pytest.fixture(autouse=True)
    def _router(self):
        with patch(f"{_H}.thread_router") as router:
            router.get_display_name.return_value = "win-name"
            router.resolve_chat_id.return_value = -100
            yield router

    def _long_transcript(self) -> list[dict]:
        return [_message(f"{i:04d} " + "x" * 2000) for i in range(5)]

    async def _render(self, **kwargs):
        target = MagicMock()
        with (
            _recent_messages(self._long_transcript()),
            patch(f"{_H}.safe_reply", new_callable=AsyncMock) as reply,
        ):
            await send_history(target, "@7", **kwargs)
        return reply.call_args

    async def test_negative_offset_serves_the_newest_page(self) -> None:
        call = await self._render(offset=-1)

        keyboard = call.kwargs["reply_markup"]
        labels = [b.text for b in keyboard.inline_keyboard[0]]
        assert "◀ Older" in labels
        assert "Newer ▶" not in labels

    async def test_offset_zero_serves_the_oldest_page(self) -> None:
        call = await self._render(offset=0)

        keyboard = call.kwargs["reply_markup"]
        labels = [b.text for b in keyboard.inline_keyboard[0]]
        assert "◀ Older" not in labels
        assert "Newer ▶" in labels

    async def test_offset_past_the_end_clamps_to_the_last_page(self) -> None:
        last = await self._render(offset=-1)
        clamped = await self._render(offset=999)

        assert clamped.args[1] == last.args[1]


class TestSendHistoryDelivery:
    """Where the rendered page is sent: edit, direct send, or reply."""

    @pytest.fixture(autouse=True)
    def _router(self):
        with patch(f"{_H}.thread_router") as router:
            router.get_display_name.return_value = "win-name"
            router.resolve_chat_id.return_value = -100
            yield router

    async def test_direct_send_uses_client_protocol(self) -> None:
        client = FakeTelegramClient()

        with _recent_messages([]):
            await send_history(
                MagicMock(),
                "@7",
                edit=False,
                user_id=42,
                client=client,
                message_thread_id=99,
            )

        assert client.call_count("send_message") == 1
        sent = client.last_call("send_message")
        assert sent is not None
        assert sent.kwargs["chat_id"] == -100
        assert sent.kwargs["message_thread_id"] == 99

    async def test_no_client_falls_back_to_safe_reply(self) -> None:
        target = MagicMock()
        target.reply_text = AsyncMock()

        with _recent_messages([]):
            await send_history(target, "@7", edit=False)

        target.reply_text.assert_awaited()

    async def test_edit_mode_edits_the_existing_message(self) -> None:
        target = MagicMock()

        with (
            _recent_messages([]),
            patch(f"{_H}.safe_edit", new_callable=AsyncMock) as edit,
        ):
            await send_history(target, "@7", edit=True)

        edit.assert_awaited_once()
        assert edit.call_args.args[0] is target

    async def test_viewing_an_unread_range_advances_the_read_offset(self) -> None:
        with (
            _recent_messages([_message("hi")]),
            patch(f"{_H}.safe_reply", new_callable=AsyncMock),
            patch(f"{_H}.user_preferences") as prefs,
        ):
            await send_history(
                MagicMock(), "@7", start_byte=100, end_byte=250, user_id=42
            )

        prefs.update_user_window_offset.assert_called_once_with(42, "@7", 250)

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"start_byte": 0, "end_byte": 0, "user_id": 42}, "full-history"),
            ({"start_byte": 100, "end_byte": 0, "user_id": 42}, "open-ended-range"),
            ({"start_byte": 100, "end_byte": 250}, "no-user"),
        ],
    )
    async def test_read_offset_untouched_outside_a_closed_unread_range(
        self, kwargs: dict, reason: str
    ) -> None:
        with (
            _recent_messages([_message("hi")]),
            patch(f"{_H}.safe_reply", new_callable=AsyncMock),
            patch(f"{_H}.user_preferences") as prefs,
        ):
            await send_history(MagicMock(), "@7", **kwargs)

        prefs.update_user_window_offset.assert_not_called()


class TestHistoryCommand:
    @pytest.fixture()
    def env(self):
        with (
            patch(f"{_H}.config") as config,
            patch(f"{_H}.thread_router") as router,
            patch(f"{_H}.window_query") as wq,
            patch(f"{_H}.get_provider_for_window") as provider,
            patch(f"{_H}.safe_reply", new_callable=AsyncMock) as reply,
            patch(f"{_H}.send_history", new_callable=AsyncMock) as send,
        ):
            config.is_user_allowed.return_value = True
            router.resolve_window_for_thread.return_value = "@7"
            wq.get_window_provider.return_value = "claude"
            provider.return_value.capabilities.supports_structured_transcript = True
            yield MagicMock(
                config=config,
                router=router,
                wq=wq,
                provider=provider,
                reply=reply,
                send=send,
            )

    def _update(self, *, user=True, message=True) -> MagicMock:
        update = MagicMock()
        update.effective_user = MagicMock(id=100) if user else None
        update.message = MagicMock() if message else None
        if message:
            update.message.message_thread_id = 42
        update.callback_query = None
        return update

    async def test_shows_history_for_the_bound_window(self, env) -> None:
        await history_command(self._update(), MagicMock())

        env.send.assert_awaited_once()
        assert env.send.call_args.args[1] == "@7"
        env.reply.assert_not_called()

    async def test_unbound_topic_reports_no_session(self, env) -> None:
        env.router.resolve_window_for_thread.return_value = None

        await history_command(self._update(), MagicMock())

        assert "No session bound" in env.reply.call_args.args[1]
        env.send.assert_not_awaited()

    async def test_provider_without_structured_transcript_rejected(self, env) -> None:
        env.provider.return_value.capabilities.supports_structured_transcript = False

        await history_command(self._update(), MagicMock())

        assert "No transcript available" in env.reply.call_args.args[1]
        env.send.assert_not_awaited()

    @pytest.mark.parametrize(
        "update_kwargs",
        [{"user": False}, {"message": False}],
        ids=["no-user", "no-message"],
    )
    async def test_incomplete_update_is_ignored(self, env, update_kwargs: dict) -> None:
        await history_command(self._update(**update_kwargs), MagicMock())

        env.reply.assert_not_called()
        env.send.assert_not_awaited()

    async def test_unauthorized_user_is_ignored(self, env) -> None:
        env.config.is_user_allowed.return_value = False

        await history_command(self._update(), MagicMock())

        env.reply.assert_not_called()
        env.send.assert_not_awaited()
