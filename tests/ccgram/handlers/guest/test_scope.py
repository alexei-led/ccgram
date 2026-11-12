"""Guest-chat scope: a guest group only sees the sessions bound in it."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.config import config
from ccgram.handlers import callback_registry
from ccgram.handlers.callback_data import (
    CB_SESSIONS_NEW,
    CB_STATUS_ESC,
    CB_STATUS_LAST_REPLY,
    CB_STATUS_SCREENSHOT,
)
from ccgram.handlers.callback_helpers import callback_user_id
from ccgram.handlers.guest import access, scope
from ccgram.handlers.sessions_dashboard import _build_dashboard
from ccgram.session import WindowState

OWNER = 1644026884
GUEST = 555
GUEST_CHAT = -1009990001
OWN_CHAT = -1004477516122
GUEST_WINDOW = "hc-t9"
OWN_WINDOW = "hc-t5"

BINDINGS = [
    (OWNER, GUEST_CHAT, 2, GUEST_WINDOW),
    (OWNER, OWN_CHAT, 7, OWN_WINDOW),
]


@pytest.fixture(autouse=True)
def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(config, "guest_chats", {GUEST_CHAT: "XY"})
    monkeypatch.setattr(config, "allowed_users", {OWNER})
    monkeypatch.setattr(config, "group_id", None)
    monkeypatch.setattr(scope.session_query, "iter_bound_topics", lambda: BINDINGS)
    access._reset_guest_access_for_testing()
    yield
    access._reset_guest_access_for_testing()


def test_windows_in_chat_only_lists_that_chat() -> None:
    assert scope.windows_in_chat(GUEST_CHAT) == {GUEST_WINDOW}
    assert scope.windows_in_chat(OWN_CHAT) == {OWN_WINDOW}


def test_unknown_chat_is_resolved_through_the_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scope.session_query,
        "iter_bound_topics",
        lambda: [(OWNER, None, 2, GUEST_WINDOW)],
    )
    monkeypatch.setattr(
        scope.session_query, "resolve_chat_id", lambda _u, _t: GUEST_CHAT
    )
    assert scope.windows_in_chat(GUEST_CHAT) == {GUEST_WINDOW}


# --- guest button presses ----------------------------------------------------


def _callback(user_id: int, data: str, chat_id: int = GUEST_CHAT):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.message = SimpleNamespace(chat=SimpleNamespace(id=chat_id, type="supergroup"))
    update = MagicMock()
    update.callback_query = query
    update.effective_user = SimpleNamespace(id=user_id)
    update.effective_chat = SimpleNamespace(id=chat_id)
    return update, query


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    acted: list[int | None] = []

    async def handler(update, _context) -> None:
        acted.append(callback_user_id(update))

    monkeypatch.setattr(callback_registry, "_find_handler", lambda _d: handler)
    monkeypatch.setattr(callback_registry, "thread_router", MagicMock())
    return acted


@pytest.mark.parametrize(
    "prefix", [CB_STATUS_ESC, CB_STATUS_SCREENSHOT, CB_STATUS_LAST_REPLY]
)
async def test_guest_button_for_this_chats_session_works(seen, prefix) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, _query = _callback(GUEST, f"{prefix}{GUEST_WINDOW}")
    await callback_registry.dispatch(update, MagicMock())
    assert seen == [OWNER]


@pytest.mark.parametrize(
    "prefix", [CB_STATUS_ESC, CB_STATUS_SCREENSHOT, CB_STATUS_LAST_REPLY]
)
async def test_guest_button_for_owners_other_session_is_refused(seen, prefix) -> None:
    access.allow(OWNER, GUEST, GUEST_CHAT, "XY")
    update, query = _callback(GUEST, f"{prefix}{OWN_WINDOW}")
    await callback_registry.dispatch(update, MagicMock())
    assert seen == []
    query.answer.assert_awaited_once_with("Only the owner can use these buttons.")


async def test_owner_buttons_are_not_scoped_by_this_check(seen) -> None:
    update, _query = _callback(OWNER, f"{CB_STATUS_SCREENSHOT}{OWN_WINDOW}")
    await callback_registry.dispatch(update, MagicMock())
    assert seen == [OWNER]


def test_unparseable_or_expired_data_is_refused() -> None:
    prefixes = callback_registry.GUEST_BUTTON_PREFIXES
    assert not scope.guest_target_allowed("st:ss:", prefixes, OWNER, GUEST_CHAT)
    assert not scope.guest_target_allowed(
        "st:ss:~AAAAAAAAAAAA", prefixes, OWNER, GUEST_CHAT
    )
    assert not scope.guest_target_allowed(
        f"st:ss:{GUEST_WINDOW}", prefixes, OWNER, None
    )


# --- /sessions ---------------------------------------------------------------


@pytest.fixture
def dashboard() -> Iterator[None]:
    with (
        patch("ccgram.handlers.sessions_dashboard.view_window") as view,
        patch("ccgram.handlers.sessions_dashboard.thread_router") as router,
        patch(
            "ccgram.handlers.sessions_dashboard.list_windows_for_reconciliation",
            new_callable=AsyncMock,
        ) as listing,
    ):
        router.get_all_thread_windows.return_value = {2: GUEST_WINDOW, 7: OWN_WINDOW}
        router.get_display_name.side_effect = lambda wid: wid
        view.side_effect = lambda wid: WindowState(cwd=f"/work/{wid}")
        listing.return_value = [
            MagicMock(window_id=GUEST_WINDOW),
            MagicMock(window_id=OWN_WINDOW),
        ]
        yield


def _buttons(keyboard) -> list[str]:
    return [b.callback_data for row in keyboard.inline_keyboard for b in row]


async def test_sessions_in_a_guest_chat_lists_only_its_sessions(dashboard) -> None:
    text, keyboard = await _build_dashboard(OWNER, GUEST_CHAT)
    assert GUEST_WINDOW in text
    assert OWN_WINDOW not in text and f"/work/{OWN_WINDOW}" not in text
    assert not any(OWN_WINDOW in str(data) for data in _buttons(keyboard))
    assert CB_SESSIONS_NEW not in _buttons(keyboard)


async def test_sessions_in_own_chat_lists_everything(dashboard) -> None:
    text, keyboard = await _build_dashboard(OWNER, OWN_CHAT)
    assert GUEST_WINDOW in text and OWN_WINDOW in text
    assert CB_SESSIONS_NEW in _buttons(keyboard)


# --- refused commands --------------------------------------------------------


def _command(user_id: int, chat_id: int):
    update = MagicMock()
    update.effective_chat = SimpleNamespace(id=chat_id)
    update.effective_user = SimpleNamespace(id=user_id)
    update.message = MagicMock()
    return update


async def test_listing_command_refused_in_guest_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = AsyncMock()
    replies = AsyncMock()
    monkeypatch.setattr(scope, "safe_reply", replies)
    wrapped = scope.refuse_in_guest_chats(inner)

    await wrapped(_command(OWNER, GUEST_CHAT), MagicMock())
    inner.assert_not_awaited()
    assert replies.await_args is not None
    assert replies.await_args.args[1] == scope.GUEST_CHAT_REFUSED

    replies.reset_mock()
    await wrapped(_command(GUEST, GUEST_CHAT), MagicMock())
    inner.assert_not_awaited()
    replies.assert_not_awaited()

    await wrapped(_command(OWNER, OWN_CHAT), MagicMock())
    inner.assert_awaited_once()
