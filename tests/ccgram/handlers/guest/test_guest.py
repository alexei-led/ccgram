"""Guest channel: access list, speaker labels, inbound text, /guest, desk exemptions."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.config import config, parse_guest_chats
from ccgram.handlers.guest import access, command, framing, inbound
from ccgram.handlers.presence import desk_brief, digest, presence

OWNER = 1644026884
GUEST = 555
STRANGER = 777
GUEST_CHAT = -1009990001
OTHER_CHAT = -1004477516122
THREAD = 12


@pytest.fixture(autouse=True)
def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
    monkeypatch.setattr(config, "guest_chats", {GUEST_CHAT: "QFY"})
    monkeypatch.setattr(config, "allowed_users", {OWNER})
    access._reset_guest_access_for_testing()
    yield
    access._reset_guest_access_for_testing()


def _args(mock: AsyncMock) -> tuple:
    call = mock.await_args
    assert call is not None
    return call.args


def _allow_guest(name: str = "QFY") -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, name)


# --- config ------------------------------------------------------------------


def test_parse_guest_chats() -> None:
    assert parse_guest_chats("QFY=-1009990001, bad, X=5,=-1") == {-1009990001: "QFY"}


# --- access list -------------------------------------------------------------


def test_guest_acts_for_the_owner_only_in_their_chat() -> None:
    assert access.owner_for(GUEST, GUEST_CHAT) is None
    _allow_guest()
    assert access.owner_for(GUEST, GUEST_CHAT) == OWNER
    assert access.owner_for(GUEST, OTHER_CHAT) is None
    assert access.owner_for(GUEST, GUEST) is None  # private chat with the bot
    assert access.owner_for(STRANGER, GUEST_CHAT) is None


def test_chat_must_still_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_guest()
    monkeypatch.setattr(config, "guest_chats", {})
    assert access.owner_for(GUEST, GUEST_CHAT) is None


def test_off_switch_survives_a_restart() -> None:
    _allow_guest()
    access.set_enabled(False)
    access._reset_guest_access_for_testing()  # as after a restart
    assert access.owner_for(GUEST, GUEST_CHAT) is None
    access.set_enabled(True)
    assert access.owner_for(GUEST, GUEST_CHAT) == OWNER


def test_unreadable_state_fails_closed(tmp_path) -> None:
    (tmp_path / "guests.json").write_text("{not json")
    assert access.is_enabled() is False
    assert access.owner_for(GUEST, GUEST_CHAT) is None


def test_remove() -> None:
    _allow_guest()
    assert access.remove(GUEST, GUEST_CHAT) == 1
    assert access.owner_for(GUEST, GUEST_CHAT) is None
    assert access.remove(GUEST) == 0


# --- speaker labels ----------------------------------------------------------


def test_guest_chat_messages_are_labelled() -> None:
    _allow_guest()
    owner_text = framing.frame_for_agent(
        "hi", chat_id=GUEST_CHAT, user_id=OWNER, first_name="Jason", is_guest=False
    )
    assert owner_text == "Jason: hi"
    guest_text = framing.frame_for_agent(
        "help me", chat_id=GUEST_CHAT, user_id=GUEST, first_name="Q", is_guest=True
    )
    assert guest_text.startswith("QFY: help me\n\n[Guest message from QFY.")
    assert "needs approval goes to the owner" in guest_text


def test_other_chats_are_not_labelled() -> None:
    text = framing.frame_for_agent(
        "hi", chat_id=OTHER_CHAT, user_id=OWNER, first_name="Jason", is_guest=False
    )
    assert text == "hi"


async def test_forwarded_text_carries_the_label() -> None:
    from ccgram.handlers.text.text_handler import _forward_message

    _allow_guest()
    th = "ccgram.handlers.text.text_handler"
    message = MagicMock()
    message.chat = MagicMock(id=GUEST_CHAT, send_action=AsyncMock())
    message.message_id = 5
    message.reply_to_message = None
    message.quote = None
    message.from_user = SimpleNamespace(id=GUEST, first_name="Q")
    with (
        patch(f"{th}.get_interactive_window", return_value=None),
        patch(f"{th}.enqueue_status_update", new=AsyncMock()),
        patch(f"{th}.ack_reaction", new=AsyncMock()),
        patch(
            f"{th}.send_telegram_to_window", new=AsyncMock(return_value=(True, "ok"))
        ) as send,
        patch("ccgram.handlers.command_history.record_command"),
    ):
        delivered = await _forward_message(
            "hc-t9",
            OWNER,
            THREAD,
            "hello",
            AsyncMock(),
            message,
            MagicMock(),
            is_guest=True,
        )
    assert delivered is True
    assert send.call_args.args[0] == OWNER
    assert send.call_args.args[3].startswith("QFY: hello\n\n[Guest message")


# --- inbound text ------------------------------------------------------------


def _update(text: str, *, user_id: int = GUEST, thread: int | None = THREAD):
    message = MagicMock()
    message.text = text
    message.chat = SimpleNamespace(id=GUEST_CHAT)
    message.date = datetime.now(UTC)
    message.message_thread_id = thread
    update = MagicMock()
    update.message = message
    update.effective_user = SimpleNamespace(id=user_id, full_name="Q", first_name="Q")
    return update, message


@pytest.fixture
def reply(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(inbound, "safe_reply", mock)
    return mock


@pytest.fixture
def forward(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr("ccgram.handlers.text.text_handler.forward_guest_text", mock)
    return mock


async def test_guest_shell_commands_are_refused(reply, forward) -> None:
    update, message = _update("!rm -rf ~")
    await inbound.handle_guest_text(update, MagicMock(), OWNER)
    reply.assert_awaited_once_with(message, inbound.NO_SHELL)
    forward.assert_not_awaited()


async def test_guest_needs_a_bound_topic(
    monkeypatch: pytest.MonkeyPatch, reply, forward
) -> None:
    monkeypatch.setattr(inbound, "get_thread_id", lambda _u: THREAD)
    monkeypatch.setattr(inbound, "bound_window", lambda *_a: None)
    update, message = _update("hi")
    await inbound.handle_guest_text(update, MagicMock(), OWNER)
    reply.assert_awaited_once_with(message, inbound.NOT_BOUND)
    forward.assert_not_awaited()


async def test_guest_text_goes_to_the_owners_session(
    monkeypatch: pytest.MonkeyPatch, reply, forward
) -> None:
    monkeypatch.setattr(inbound, "get_thread_id", lambda _u: THREAD)
    monkeypatch.setattr(inbound, "bound_window", lambda *_a: "hc-t9")
    update, message = _update("hi")
    await inbound.handle_guest_text(update, MagicMock(), OWNER)
    forward.assert_awaited_once()
    assert _args(forward)[:4] == ("hc-t9", OWNER, THREAD, "hi")
    reply.assert_not_awaited()


async def test_late_guest_message_gets_a_delay_note(
    monkeypatch: pytest.MonkeyPatch, reply, forward
) -> None:
    monkeypatch.setattr(inbound, "get_thread_id", lambda _u: THREAD)
    monkeypatch.setattr(inbound, "bound_window", lambda *_a: "hc-t9")
    update, message = _update("are you there?")
    message.date = datetime.now(UTC) - timedelta(hours=3)
    await inbound.handle_guest_text(update, MagicMock(), OWNER)
    reply.assert_awaited_once()
    assert _args(reply)[1].startswith("⏰ Sent ")


def test_late_notice_threshold() -> None:
    now = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
    assert inbound.late_notice(now - timedelta(seconds=30), now) is None
    assert "offline" in (inbound.late_notice(now - timedelta(hours=1), now) or "")


# --- text_handler dispatch ---------------------------------------------------


async def test_text_handler_routes_guests_and_ignores_strangers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ccgram.handlers.text import text_handler as th

    guest_branch = AsyncMock()
    reply = AsyncMock()
    monkeypatch.setattr(th, "handle_guest_text", guest_branch)
    monkeypatch.setattr(th, "safe_reply", reply)
    _allow_guest()

    update, _message = _update("hi", user_id=GUEST)
    await th.text_handler(update, MagicMock())
    guest_branch.assert_awaited_once()

    guest_branch.reset_mock()
    update, _message = _update("hi", user_id=STRANGER)
    await th.text_handler(update, MagicMock())
    guest_branch.assert_not_awaited()
    reply.assert_not_awaited()  # silence for strangers in a guest chat

    access.set_enabled(False)
    update, _message = _update("hi", user_id=GUEST)
    await th.text_handler(update, MagicMock())
    guest_branch.assert_not_awaited()


# --- /guest ------------------------------------------------------------------


def _command(args: list[str], *, user_id: int = OWNER, reply_to=None):
    message = MagicMock()
    message.chat = SimpleNamespace(id=GUEST_CHAT)
    update = MagicMock()
    update.message = message
    update.effective_user = SimpleNamespace(id=user_id)
    context = MagicMock()
    context.args = args
    return update, message, context


@pytest.fixture
def command_reply(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(command, "safe_reply", mock)
    return mock


async def test_allow_by_reply(monkeypatch: pytest.MonkeyPatch, command_reply) -> None:
    target = SimpleNamespace(
        from_user=SimpleNamespace(id=GUEST, is_bot=False, first_name="Joy")
    )
    monkeypatch.setattr(command, "explicit_reply", lambda _m: target)
    update, _message, context = _command(["allow", "QFY"])
    await command.guest_command(update, context)
    assert access.owner_for(GUEST, GUEST_CHAT) == OWNER
    assert access.guests()[0].name == "QFY"
    assert "QFY can now talk" in _args(command_reply)[1]


async def test_off_and_on(command_reply) -> None:
    _allow_guest()
    update, _message, context = _command(["off"])
    await command.guest_command(update, context)
    assert access.owner_for(GUEST, GUEST_CHAT) is None
    update, _message, context = _command(["on"])
    await command.guest_command(update, context)
    assert access.owner_for(GUEST, GUEST_CHAT) == OWNER


async def test_guests_cannot_run_guest_command(command_reply) -> None:
    _allow_guest()
    update, _message, context = _command(["off"], user_id=GUEST)
    await command.guest_command(update, context)
    assert access.is_enabled() is True
    command_reply.assert_not_awaited()


# --- desk mode ---------------------------------------------------------------


def test_guest_chat_is_never_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "desk_mode", True)
    presence._reset_presence_for_testing(idle_reader=AsyncMock(return_value=0.0))
    monkeypatch.setattr(
        presence, "_users", {OWNER: SimpleNamespace(mode=presence.DESK)}
    )
    assert presence.is_quiet(OWNER) is True
    assert presence.is_quiet(OWNER, OTHER_CHAT) is True
    assert presence.is_quiet(OWNER, GUEST_CHAT) is False


async def test_desk_brief_mentions_guest_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "desk_brief", "dm")
    monkeypatch.setattr(config, "desk_digest_minutes", 30.0)
    monkeypatch.setattr(digest, "iter_bound_topics", lambda: [])
    monkeypatch.setattr(digest, "_wall_time", lambda: "14:32")
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(digest, "_clock", lambda: clock.now)
    dm = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(desk_brief, "rate_limit_send_message", dm)
    digest._reset_digest_for_testing()
    for role in ("user", "assistant", "assistant"):
        digest.record_guest_traffic(
            OWNER, GUEST_CHAT, THREAD, role=role, content_type="text"
        )
    clock.now += 30 * 60
    await digest.flush_due(MagicMock())
    text = _args(dm)[2]
    assert "👥 **QFY** · 3 messages (guest chat)" in text
    assert "https://t.me/c/9990001/12" in text
    digest._reset_digest_for_testing()


async def test_guest_chat_never_starts_a_session_from_telegram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ccgram.handlers.text import text_handler as th

    reply = AsyncMock()
    start = AsyncMock(return_value=True)
    monkeypatch.setattr(th, "safe_reply", reply)
    monkeypatch.setattr(th, "start_from_text", start)
    router = MagicMock()
    router.get_window_for_thread.return_value = None
    monkeypatch.setattr(th, "thread_router", router)
    message = MagicMock()
    message.chat = SimpleNamespace(id=GUEST_CHAT, type="supergroup")
    handled = await th._handle_unbound_topic(
        OWNER, THREAD, "claude sessions do X", {}, message, MagicMock()
    )
    assert handled is True
    reply.assert_awaited_once_with(message, th.GUEST_CHAT_NO_NEW_SESSION)
    start.assert_not_awaited()
