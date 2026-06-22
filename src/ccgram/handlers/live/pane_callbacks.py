"""Pane subscription and rename callback handlers (Theme 5).

Handles inline keyboard callbacks for the ``/panes`` keyboard:
  - CB_PANE_SUBSCRIBE: mark a pane subscribed so its output is forwarded
  - CB_PANE_UNSUBSCRIBE: stop forwarding the pane
  - CB_PANE_RENAME: prompt the user to provide a friendly name for the pane

Rename uses a context.user_data flag (``PANE_RENAME_*``); the next text
message in the same thread is captured by ``apply_pane_rename`` and
written to ``PaneInfo.name``. Auto-clear on pane death is handled by
``PaneStatusStrategy.reconcile_dead_panes`` removing the ``PaneInfo``
entry — the subscribed flag goes away with it.
"""

from __future__ import annotations


from typing import TYPE_CHECKING
import structlog
from telegram import (
    CallbackQuery,
    ForceReply,
    InlineKeyboardButton,
    Message,
    Update,
)

from ...config import config
from ...telegram_client import PTBTelegramClient
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...window_state_ports.pane_state import (
    get_pane_lifecycle_notify,
    get_pane_projection,
    set_pane_lifecycle_notify,
    upsert_pane,
)
from ..callback_data import (
    CB_PANE_DELIMITER,
    CB_PANE_LIFECYCLE_TOGGLE,
    CB_PANE_RENAME,
    CB_PANE_SCREENSHOT,
    CB_PANE_SUBSCRIBE,
    CB_PANE_UNSUBSCRIBE,
)
from ..callback_helpers import get_thread_id, user_owns_window
from ..callback_registry import register
from ..messaging_pipeline.message_sender import safe_reply
from ..user_state import (
    PANE_RENAME_PANE_ID,
    PANE_RENAME_THREAD_ID,
    PANE_RENAME_WINDOW_ID,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_MAX_PANE_NAME_LEN = 32
_RENAME_PROMPT = (
    "✏️ Reply with a name for pane {pane_id} "
    f"(max {_MAX_PANE_NAME_LEN} chars). Send '-' to clear."
)


def _parse_target(data: str, prefix: str) -> tuple[str, str] | None:
    """Parse ``<prefix><window_id>|<pane_id>`` callback data.

    Splits on ``CB_PANE_DELIMITER`` (``|``) so both tmux ids (``@12``,
    ``%5``) and herdr ids (``w2:t1``, ``w2:p1``) round-trip without
    colliding on the colons inside herdr ids.
    """
    rest = data[len(prefix) :]
    sep = rest.find(CB_PANE_DELIMITER)
    if sep < 0:
        return None
    return rest[:sep], rest[sep + 1 :]


def build_pane_buttons(
    window_id: str, pane_id: str, subscribed: bool
) -> list[InlineKeyboardButton]:
    """Return the per-pane action row used in the ``/panes`` keyboard."""
    sub_label = "\U0001f515 Unsub" if subscribed else "\U0001f514 Sub"
    sub_data = f"{CB_PANE_UNSUBSCRIBE if subscribed else CB_PANE_SUBSCRIBE}{window_id}{CB_PANE_DELIMITER}{pane_id}"
    return [
        InlineKeyboardButton(
            f"\U0001f4f7 {pane_id}",
            callback_data=f"{CB_PANE_SCREENSHOT}{window_id}{CB_PANE_DELIMITER}{pane_id}"[
                :64
            ],
        ),
        InlineKeyboardButton(sub_label, callback_data=sub_data[:64]),
        InlineKeyboardButton(
            "✏️ Rename",
            callback_data=f"{CB_PANE_RENAME}{window_id}{CB_PANE_DELIMITER}{pane_id}"[
                :64
            ],
        ),
    ]


def build_pane_lifecycle_button(
    window_id: str, *, enabled: bool
) -> InlineKeyboardButton:
    """Return the per-window pane lifecycle notifications toggle button."""
    icon = "\U0001f514" if enabled else "\U0001f515"
    state = "on" if enabled else "off"
    return InlineKeyboardButton(
        f"{icon} Lifecycle: {state}",
        callback_data=f"{CB_PANE_LIFECYCLE_TOGGLE}{window_id}"[:64],
    )


async def _toggle_subscribe(
    query: CallbackQuery, user_id: int, data: str, *, subscribed: bool, prefix: str
) -> None:
    parsed = _parse_target(data, prefix)
    if parsed is None:
        await query.answer("Invalid pane")
        return
    window_id, pane_id = parsed
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    pane = get_pane_projection(window_id, pane_id)
    if pane is None:
        # Pane scanner hasn't seen this pane yet (common right after a split).
        # If tmux still reports the pane, hydrate the entry inline so the
        # subscribe action succeeds; otherwise the keyboard is showing a
        # stale pane and we should fail loudly.
        try:
            live = await tmux_manager.list_panes(window_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pane subscribe lookup failed: %s", exc)
            await query.answer("Pane lookup failed", show_alert=True)
            return
        if not any(p.pane_id == pane_id for p in live):
            await query.answer("Pane not found", show_alert=True)
            return
    upsert_pane(window_id, pane_id, subscribed=subscribed)
    label = "Subscribed" if subscribed else "Unsubscribed"
    await query.answer(f"✓ {label} {pane_id}")


async def _handle_subscribe(query: CallbackQuery, user_id: int, data: str) -> None:
    await _toggle_subscribe(
        query, user_id, data, subscribed=True, prefix=CB_PANE_SUBSCRIBE
    )


async def _handle_unsubscribe(query: CallbackQuery, user_id: int, data: str) -> None:
    await _toggle_subscribe(
        query, user_id, data, subscribed=False, prefix=CB_PANE_UNSUBSCRIBE
    )


async def _handle_rename(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    parsed = _parse_target(data, CB_PANE_RENAME)
    if parsed is None:
        await query.answer("Invalid pane")
        return
    window_id, pane_id = parsed
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return
    if context.user_data is not None:
        context.user_data[PANE_RENAME_WINDOW_ID] = window_id
        context.user_data[PANE_RENAME_PANE_ID] = pane_id
        context.user_data[PANE_RENAME_THREAD_ID] = thread_id

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    client = PTBTelegramClient(query.get_bot())
    try:
        await client.send_message(
            chat_id=chat_id,
            text=_RENAME_PROMPT.format(pane_id=pane_id),
            message_thread_id=thread_id,
            reply_markup=ForceReply(selective=True),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pane rename prompt failed: %s", exc)
        await query.answer("Failed to open rename prompt", show_alert=True)
        return
    await query.answer("✏️ Rename")


async def apply_pane_rename(
    user_data: dict | None,
    thread_id: int | None,
    text: str,
    message: Message,
) -> bool:
    """Consume an in-flight rename reply.

    Returns True when the message was handled (rename pending in this
    thread); the caller must early-return. Returns False otherwise so
    normal text routing continues.
    """
    if not user_data or thread_id is None:
        return False
    pending_thread = user_data.get(PANE_RENAME_THREAD_ID)
    window_id = user_data.get(PANE_RENAME_WINDOW_ID)
    pane_id = user_data.get(PANE_RENAME_PANE_ID)
    if pending_thread != thread_id or not window_id or not pane_id:
        return False

    user_data.pop(PANE_RENAME_WINDOW_ID, None)
    user_data.pop(PANE_RENAME_PANE_ID, None)
    user_data.pop(PANE_RENAME_THREAD_ID, None)

    name = text.strip()
    if name == "-" or name == "":
        upsert_pane(window_id, pane_id, name=None)
        await safe_reply(message, f"✓ Cleared name for {pane_id}")
        return True
    if len(name) > _MAX_PANE_NAME_LEN:
        # Reject loudly so the user doesn't see a different name from what
        # they typed. They can resend a shorter version.
        await safe_reply(
            message,
            f"❌ Name too long ({len(name)} chars, max {_MAX_PANE_NAME_LEN}).",
        )
        return True
    upsert_pane(window_id, pane_id, name=name)
    await safe_reply(message, f"✓ Renamed {pane_id} → {name}")
    return True


async def _handle_lifecycle_toggle(
    query: CallbackQuery, user_id: int, data: str
) -> None:
    window_id = data[len(CB_PANE_LIFECYCLE_TOGGLE) :]
    if not window_id:
        await query.answer("Invalid window")
        return
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your session", show_alert=True)
        return
    current = get_pane_lifecycle_notify(window_id, config.pane_lifecycle_notify)
    new_value = not current
    set_pane_lifecycle_notify(window_id, new_value)
    label = "on" if new_value else "off"
    await query.answer(f"✓ Pane lifecycle notifications {label}")


@register(
    CB_PANE_SUBSCRIBE, CB_PANE_UNSUBSCRIBE, CB_PANE_RENAME, CB_PANE_LIFECYCLE_TOGGLE
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    data = query.data
    if data.startswith(CB_PANE_SUBSCRIBE):
        await _handle_subscribe(query, user.id, data)
    elif data.startswith(CB_PANE_UNSUBSCRIBE):
        await _handle_unsubscribe(query, user.id, data)
    elif data.startswith(CB_PANE_RENAME):
        await _handle_rename(query, user.id, data, update, context)
    elif data.startswith(CB_PANE_LIFECYCLE_TOGGLE):
        await _handle_lifecycle_toggle(query, user.id, data)
