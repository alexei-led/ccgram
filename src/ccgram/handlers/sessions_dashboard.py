"""Sessions dashboard — /sessions command showing all bound sessions.

Displays a summary of all thread-bound sessions for the current user
with alive/dead status indicators, per-session action buttons (Esc,
Screenshot, Kill with two-step confirmation), cwd details, and
refresh/new-session actions.

Key functions:
  - sessions_command(): /sessions command handler
  - handle_sessions_refresh(): refresh button callback
  - handle_sessions_kill(): first Kill tap — show confirmation
  - handle_sessions_kill_confirm(): second tap — kill and unbind
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import structlog

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from ..config import config
from ..session import session_manager
from ..telegram_client import PTBTelegramClient, TelegramClient
from ..thread_router import thread_router
from ..multiplexer import multiplexer as tmux_manager
from ..multiplexer.base import canonical_window_id
from ..multiplexer.reconciliation import (
    list_windows_for_reconciliation,
    window_presence,
)
from ..window_query import view_window
from ..session_protocol.telegram import TelegramSessionAdapter, adapter_for
from .callback_data import (
    CB_SESSIONS_KILL,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_SCREENSHOT,
)
from .callback_helpers import callback_user_id, user_owns_window
from .callback_tokens import compact_callback_data, resolve_callback_data
from .callback_registry import register
from .cleanup import clear_topic_state
from .guest import access as guest_access
from .guest import scope as guest_scope
from .guest import sessions as guest_sessions
from .topics.topic_deletion import cleanup_retired_topics, retire_topic_binding
from .messaging_pipeline.message_sender import safe_edit, safe_reply

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_REFRESH_BTN = InlineKeyboardButton(
    "\U0001f504 Refresh", callback_data=CB_SESSIONS_REFRESH
)
_NEW_BTN = InlineKeyboardButton("\u2795 New Session", callback_data=CB_SESSIONS_NEW)

_PROVIDER_LABELS = {"claude": "Claude", "codex": "Codex"}


def _guest_footer() -> list[list[InlineKeyboardButton]]:
    """Refresh, then one ➕ per agent a guest chat can start."""
    starts = [
        InlineKeyboardButton(
            f"\u2795 {_PROVIDER_LABELS.get(p, p)}",
            callback_data=f"{CB_SESSIONS_NEW}:{p}",
        )
        for p in guest_sessions.enabled_providers()
    ]
    return [[_REFRESH_BTN], starts] if starts else [[_REFRESH_BTN]]


# Green running, black stopped, white when the multiplexer could not be asked.
_LIVENESS_MARKER = {True: "\U0001f7e2", False: "\u26ab", None: "\u26aa"}


def _chat_of(query: CallbackQuery) -> int | None:
    return query.message.chat.id if query.message else None


async def _build_dashboard(
    user_id: int, chat_id: int | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    """Build dashboard text and keyboard for a user's sessions.

    In a guest chat every member sees the dashboard, so it lists only the
    sessions bound in that chat, with a count against the chat's limit; the
    ➕ Claude / ➕ Codex buttons there start one with the guest launchers
    (those configured).
    """
    bindings = thread_router.get_all_thread_windows(user_id)
    footer: list[list[InlineKeyboardButton]] = [[_REFRESH_BTN, _NEW_BTN]]
    guest_chat = guest_scope.is_guest_chat(chat_id)
    can_start = guest_chat and bool(guest_sessions.enabled_providers())
    if guest_chat:
        assert chat_id is not None
        here = guest_scope.windows_in_chat(chat_id)
        bindings = {tid: wid for tid, wid in bindings.items() if wid in here}
        footer = _guest_footer()

    if not bindings:
        keyboard = InlineKeyboardMarkup(footer)
        empty = (
            "No sessions here yet.\n\nStart one with a button below."
            if can_start
            else "No active sessions.\n\nCreate a new topic to start a session."
        )
        return empty, keyboard

    # The complete listing, not the adoption-filtered one: every id here is
    # already bound, and a live window a backend merely will not auto-adopt
    # would otherwise show as stopped. None means the backend could not be
    # asked, which is not the same as every session being stopped.
    all_windows = await list_windows_for_reconciliation()
    # Folded on both sides: a binding persisted in different case from the
    # live listing is the same window, and showing it stopped also strips
    # its Esc, Screenshot and Kill controls.
    live_ids = (
        None
        if all_windows is None
        else {canonical_window_id(w.window_id) for w in all_windows}
    )

    lines: list[str] = []
    action_rows: list[list[InlineKeyboardButton]] = []
    for _thread_id, window_id in sorted(bindings.items()):
        display_name = thread_router.get_display_name(window_id)
        view = view_window(window_id)
        alive = None if live_ids is None else canonical_window_id(window_id) in live_ids
        status = _LIVENESS_MARKER[alive]

        # Session line with provider + mode tags and cwd detail
        provider_tag = f" [{view.provider_name}]" if view and view.provider_name else ""
        mode_tag = " [YOLO]" if view and view.approval_mode == "yolo" else ""
        line = f"{status} {display_name}{provider_tag}{mode_tag}"
        if view and view.cwd:
            line += f"\n    {view.cwd}"
        lines.append(line)

        if alive:
            row: list[InlineKeyboardButton] = [
                InlineKeyboardButton(
                    "\u238b Esc",
                    callback_data=compact_callback_data(
                        CB_STATUS_ESC, f"{CB_STATUS_ESC}{window_id}", window_id
                    ),
                ),
                InlineKeyboardButton(
                    "\U0001f4f8",
                    callback_data=compact_callback_data(
                        CB_STATUS_SCREENSHOT,
                        f"{CB_STATUS_SCREENSHOT}{window_id}",
                        window_id,
                    ),
                ),
                InlineKeyboardButton(
                    f"\U0001f5d1 Kill {display_name}",
                    callback_data=compact_callback_data(
                        CB_SESSIONS_KILL, f"{CB_SESSIONS_KILL}{window_id}", window_id
                    ),
                ),
            ]
            action_rows.append(row)

    title = "Sessions"
    if guest_chat:
        title += f" ({len(action_rows)}/{config.guest_max_sessions})"
    text = f"{title}\n\n" + "\n".join(lines)
    rows = action_rows + footer
    return text, InlineKeyboardMarkup(rows)


async def sessions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sessions — show dashboard of all bound sessions.

    In a guest chat a guest may use it too (acting as the owner, scoped to the
    chat), and ``/sessions new [claude|codex]`` starts a session there with the
    guest launcher for that agent (Claude when none is named).
    """
    user = update.effective_user
    if not user or not update.message:
        return

    chat = update.effective_chat
    chat_id = chat.id if chat else None
    acting_id = user.id
    if not config.is_user_allowed(user.id):
        owner_id = guest_access.owner_for(user.id, chat_id)
        if owner_id is None:
            await safe_reply(update.message, "You are not authorized to use this bot.")
            return
        acting_id = owner_id

    args = getattr(context, "args", None) or []
    if guest_scope.is_guest_chat(chat_id) and args and args[0].lower() == "new":
        assert chat_id is not None
        provider = args[1].lower() if len(args) > 1 else "claude"
        reply = await guest_sessions.start_guest_session(
            chat_id, adapter_for(context.bot_data), provider
        )
        await safe_reply(update.message, reply)
        return

    text, keyboard = await _build_dashboard(acting_id, chat_id)
    await safe_reply(update.message, text, reply_markup=keyboard)


async def handle_sessions_refresh(query: CallbackQuery, user_id: int) -> None:
    """Handle refresh button — re-render the dashboard in-place."""
    text, keyboard = await _build_dashboard(user_id, _chat_of(query))
    await safe_edit(query, text, reply_markup=keyboard)


async def handle_sessions_new(
    query: CallbackQuery,
    user_id: int,
    adapter: TelegramSessionAdapter,
    provider: str = "claude",
) -> None:
    """➕ button: start a guest session in a guest chat, else explain."""
    chat_id = _chat_of(query)
    if not (guest_scope.is_guest_chat(chat_id) and guest_sessions.enabled_providers()):
        await query.answer("Create a new topic to start a session.")
        return
    assert chat_id is not None
    reply = await guest_sessions.start_guest_session(chat_id, adapter, provider)
    await query.answer(reply, show_alert=True)
    await handle_sessions_refresh(query, user_id)


async def handle_sessions_kill(
    query: CallbackQuery, _user_id: int, window_id: str
) -> None:
    """First Kill tap — show confirmation prompt."""
    display = thread_router.get_display_name(window_id)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"\u26a0 Confirm kill {display}",
                    callback_data=compact_callback_data(
                        CB_SESSIONS_KILL_CONFIRM,
                        f"{CB_SESSIONS_KILL_CONFIRM}{window_id}",
                        window_id,
                    ),
                ),
            ],
            [_REFRESH_BTN],
        ]
    )
    await safe_edit(
        query,
        f"Kill session '{display}'?\n\nThis will terminate the Claude Code process.",
        reply_markup=keyboard,
    )


async def handle_sessions_kill_confirm(
    query: CallbackQuery,
    user_id: int,
    window_id: str,
    client: TelegramClient,
    adapter: TelegramSessionAdapter,
) -> None:
    """Second tap: end the session's program, unbind it, refresh the dashboard.

    Its topics are closed, not deleted, so the conversation stays readable in
    the group; the agent's own transcript is never touched either.
    """
    display = thread_router.get_display_name(window_id)

    # find_window_by_id answers None both for a window that is gone and for a
    # backend that could not be reached, and everything below deletes the
    # routing to this session. Unknown changes nothing: the session would
    # survive the outage with no way left to reach it.
    presence = await window_presence(window_id, tmux_manager)
    if presence is None:
        await safe_edit(
            query,
            f"\u26a0 Could not reach the multiplexer, so '{display}' was left "
            "alone. Nothing was killed or unbound.",
            reply_markup=None,
        )
        return

    if presence and not (await adapter.close(window_id))[0]:
        # The window was there a moment ago and the kill did not succeed, so
        # it may still be running. Clearing the bindings now would strand it.
        await safe_edit(
            query,
            f"\u26a0 Could not kill '{display}'. Nothing was unbound.",
            reply_markup=None,
        )
        return

    for uid, chat_id, tid, bound_wid in list(
        thread_router.iter_thread_bindings_with_chat()
    ):
        if canonical_window_id(bound_wid) == canonical_window_id(window_id):

            async def clear_state_before_delete() -> None:
                await clear_topic_state(
                    uid,
                    tid,
                    client,
                    window_id=bound_wid,
                    chat_id=chat_id,
                )

            await retire_topic_binding(
                client,
                uid,
                tid,
                bound_wid,
                router=thread_router,
                chat_id=chat_id,
                before_delete=clear_state_before_delete,
                retirement_reason="session_killed",
                keep_topic=True,
            )

    session_manager.flush_state()
    await cleanup_retired_topics(client, router=thread_router)

    logger.info(
        "sessions_kill_confirm: killed window %s (%s), user=%d",
        window_id,
        display,
        user_id,
    )

    # Re-render dashboard
    text, keyboard = await _build_dashboard(user_id, _chat_of(query))
    headline = "\U0001f5d1 Killed" if presence else "\U0001f5d1 Was already gone:"
    await safe_edit(
        query,
        f"{headline} '{display}' (its topic is closed and kept)\n\n{text}",
        reply_markup=keyboard,
    )


@register(
    CB_SESSIONS_REFRESH,
    CB_SESSIONS_NEW,
    CB_SESSIONS_KILL_CONFIRM,
    CB_SESSIONS_KILL,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    user = update.effective_user
    if not user:
        return
    # A guest's press in a guest chat acts as the owner (callback_registry).
    uid = callback_user_id(update) or user.id

    data = resolve_callback_data(query.data, uid, user_owns_window)
    if data is None:
        await query.answer("This button has expired", show_alert=True)
        return

    if data == CB_SESSIONS_REFRESH:
        await handle_sessions_refresh(query, uid)
        await query.answer("Refreshed")
    elif data == CB_SESSIONS_NEW or data.startswith(f"{CB_SESSIONS_NEW}:"):
        # "sess:new" (older dashboards) means Claude; "sess:new:<provider>".
        provider = data[len(CB_SESSIONS_NEW) + 1 :] or "claude"
        await handle_sessions_new(query, uid, adapter_for(context.bot_data), provider)
    elif data.startswith(CB_SESSIONS_KILL_CONFIRM):
        window_id = data[len(CB_SESSIONS_KILL_CONFIRM) :]
        if not user_owns_window(uid, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill_confirm(
            query,
            uid,
            window_id,
            PTBTelegramClient(context.bot),
            adapter_for(context.bot_data),
        )
        await query.answer("Killed")
    elif data.startswith(CB_SESSIONS_KILL):
        window_id = data[len(CB_SESSIONS_KILL) :]
        if not user_owns_window(uid, window_id):
            await query.answer("Not your session", show_alert=True)
            return
        await handle_sessions_kill(query, uid, window_id)
        await query.answer()
