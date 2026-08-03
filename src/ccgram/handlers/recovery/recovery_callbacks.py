"""Recovery callback dispatcher + shared state cleanup.

After the Round 5 split, this module is a thin dispatcher: it routes
prefix-tagged callback data to the banner handlers in
:mod:`recovery_banner` or the picker handler in :mod:`resume_picker`,
and owns ``_clear_recovery_state`` — the only helper both sibling
modules import. The banner-specific ``_validate_recovery_state`` lives
in :mod:`recovery_banner` next to its callers.

The dispatcher has no top-level imports of the siblings, so they can
both import ``_clear_recovery_state`` eagerly while the dispatcher
imports their handlers lazily inside ``handle_recovery_callback``.

Routes handled:
  - CB_RECOVERY_FRESH: create fresh session in same directory
  - CB_RECOVERY_CONTINUE: continue most recent session
  - CB_RECOVERY_RESUME: show session picker
  - CB_RECOVERY_PICK: a session was picked from the list
  - CB_RECOVERY_BACK: return from the picker to the banner
  - CB_RECOVERY_BROWSE: switch to the cross-project picker
  - CB_RECOVERY_CANCEL: dismiss the recovery flow
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import CallbackQuery, Update

from ..callback_data import (
    CB_RECOVERY_BACK,
    CB_RECOVERY_BROWSE,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_PICK,
    CB_RECOVERY_RESUME,
)
from ..callback_helpers import user_owns_window
from ..callback_registry import register
from ..callback_tokens import resolve_callback_data
from ..user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    RECOVERY_SESSIONS,
    RECOVERY_WINDOW_ID,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


def _clear_recovery_state(user_data: dict | None) -> None:
    """Remove all recovery-related keys from user_data."""
    if user_data is None:
        return
    for key in (
        PENDING_THREAD_ID,
        PENDING_THREAD_TEXT,
        RECOVERY_WINDOW_ID,
        RECOVERY_SESSIONS,
    ):
        user_data.pop(key, None)


async def handle_recovery_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle recovery UI callbacks. Always answers ``query`` once."""
    # Lazy: dispatcher → handler-module cycle (siblings register on import)
    from .recovery_banner import (
        _handle_back,
        _handle_browse,
        _handle_cancel,
        _handle_continue,
        _handle_fresh,
        _handle_resume,
    )

    # Lazy: dispatcher → handler-module cycle (siblings register on import)
    from .resume_picker import _handle_resume_pick

    # All recovery prefixes terminate with a non-overlapping character
    # ("rec:b:" vs "rec:br:" diverge at index 5: ":" vs "r"), so order is
    # not load-bearing — but check the longer prefix first as a safety
    # belt against future renames that introduce a real overlap.
    if data.startswith(CB_RECOVERY_BROWSE):
        await _handle_browse(query, user_id, data, update, context)
        return
    if data.startswith(CB_RECOVERY_BACK):
        await _handle_back(query, data, update, context)
        return
    if data.startswith(CB_RECOVERY_FRESH):
        await _handle_fresh(query, user_id, data, update, context)
        return
    if data.startswith(CB_RECOVERY_CONTINUE):
        await _handle_continue(query, user_id, data, update, context)
        return
    if data.startswith(CB_RECOVERY_RESUME):
        await _handle_resume(query, user_id, data, update, context)
        return
    if data.startswith(CB_RECOVERY_PICK):
        await _handle_resume_pick(query, user_id, data, update, context)
        return
    if data == CB_RECOVERY_CANCEL:
        await _handle_cancel(query, update, context)
        return
    logger.warning("Unhandled recovery callback data: %r", data)
    await query.answer()


@register(
    CB_RECOVERY_BACK,
    CB_RECOVERY_BROWSE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_RESUME,
    CB_RECOVERY_PICK,
    CB_RECOVERY_CANCEL,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    data = resolve_callback_data(query.data, user.id, user_owns_window)
    if data is None:
        await query.answer("This button has expired", show_alert=True)
        return
    await handle_recovery_callback(query, user.id, data, update, context)
