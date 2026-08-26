"""Resume picker UX flow + transcript scan.

Implements the picker the user sees after tapping "Resume" on a dead
window's recovery banner: scans Claude Code session JSONL files for the
bound cwd, renders a 6-row inline keyboard, and binds a freshly created
tmux window to the picked session.

Public surface:
  - :class:`_SessionEntry` (internal — entry shape used by the picker)
  - :func:`scan_sessions_for_cwd` (re-exported from
    :mod:`handlers.recovery` for legacy callers)
  - :func:`_build_resume_picker_keyboard`,
    :func:`_build_empty_resume_keyboard`
  - :func:`_handle_resume_pick` (handler dispatched from
    :mod:`recovery_callbacks`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)

from ... import window_query
from ...providers import get_provider_for_window, providers_to_scan
from ..callback_data import (
    CB_RECOVERY_BACK,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_PICK,
    CB_RECOVERY_BROWSE,
)
from ..callback_helpers import get_thread_id
from ..callback_tokens import compact_callback_data
from ..user_state import (
    PENDING_THREAD_ID,
    RECOVERY_SESSIONS,
    RECOVERY_WINDOW_ID,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_MAX_RESUME_SESSIONS = 6


@dataclass
class _SessionEntry:
    """A resumable session discovered from project directories."""

    session_id: str
    summary: str
    mtime: float = 0.0
    provider_name: str = "claude"


def _build_resume_picker_keyboard(
    sessions: list[_SessionEntry],
    window_id: str,
) -> InlineKeyboardMarkup:
    """Build inline keyboard listing recent sessions for resume."""
    # Lazy: sibling cycle — resume_command imports from this package.
    from .resume_command import format_session_entry

    rows: list[list[InlineKeyboardButton]] = []
    for idx, entry in enumerate(sessions[:_MAX_RESUME_SESSIONS]):
        label = format_session_entry(
            summary=entry.summary,
            session_id=entry.session_id,
            mtime=entry.mtime,
        )
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=f"{CB_RECOVERY_PICK}{idx}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                "⬅ Back",
                callback_data=compact_callback_data(
                    CB_RECOVERY_BACK, f"{CB_RECOVERY_BACK}{window_id}", window_id
                ),
            ),
            InlineKeyboardButton("✖ Cancel", callback_data=CB_RECOVERY_CANCEL),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _build_empty_resume_keyboard(window_id: str) -> InlineKeyboardMarkup:
    """Build the inline keyboard shown when no sessions exist for the cwd.

    Offers two paths so the user is never stuck on a dead toast:
      - Browse other projects (cross-project picker via CB_RECOVERY_BROWSE)
      - Start fresh (reuses the recovery fresh handler)
    """

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "\U0001f5c2 Browse other projects",
                    callback_data=compact_callback_data(
                        CB_RECOVERY_BROWSE,
                        f"{CB_RECOVERY_BROWSE}{window_id}",
                        window_id,
                    ),
                ),
            ],
            [
                InlineKeyboardButton(
                    "\U0001f195 Start fresh",
                    callback_data=compact_callback_data(
                        CB_RECOVERY_FRESH, f"{CB_RECOVERY_FRESH}{window_id}", window_id
                    ),
                ),
            ],
            [InlineKeyboardButton("✖ Cancel", callback_data=CB_RECOVERY_CANCEL)],
        ]
    )


def scan_sessions_for_cwd(
    cwd: str,
    provider_name: str | None = "claude",
) -> list[_SessionEntry]:
    """List resumable sessions for an exact workspace.

    Covers one provider when the caller knows which; every picker-capable one
    when it does not — see ``providers_to_scan``.
    """
    try:
        resolved_cwd = str(Path(cwd).expanduser().resolve())
    except OSError, ValueError:
        return []

    entries = [
        _SessionEntry(
            session_id=session.session_id,
            summary=session.summary,
            mtime=session.mtime,
            provider_name=session.provider_name,
        )
        for provider in providers_to_scan(provider_name)
        for session in provider.discover_resumable_sessions(
            cwd=resolved_cwd,
            limit=_MAX_RESUME_SESSIONS,
        )
    ]
    entries.sort(key=lambda e: e.mtime, reverse=True)
    return entries[:_MAX_RESUME_SESSIONS]


async def _handle_resume_pick(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_PICK: user selected a session from resume picker."""
    # Lazy: sibling cycle — recovery_banner imports from this module
    # for scan_sessions_for_cwd; the picker only needs banner's window
    # creation helper at the moment of selection.
    # Lazy: resume_picker ↔ recovery_banner cycle
    from .recovery_banner import _create_and_bind_window

    idx_str = data[len(CB_RECOVERY_PICK) :]
    try:
        idx = int(idx_str)
    except ValueError:
        await query.answer("Couldn't read selection", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return

    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is None or thread_id != pending_tid:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    stored_sessions = (
        context.user_data.get(RECOVERY_SESSIONS) if context.user_data else None
    )
    if not stored_sessions or idx < 0 or idx >= len(stored_sessions):
        await query.answer("Session no longer in list", show_alert=True)
        return

    picked = stored_sessions[idx]
    session_id = picked["session_id"]
    provider_name = picked.get("provider_name", "")

    old_wid = context.user_data.get(RECOVERY_WINDOW_ID) if context.user_data else None
    if not old_wid:
        await query.answer("Recovery menu expired", show_alert=True)
        return

    # Lazy: resume_picker ↔ recovery_banner cycle
    from .recovery_banner import _recovery_cwd_or_report

    cwd = await _recovery_cwd_or_report(query, old_wid, context)
    if cwd is None:
        return
    view = window_query.view_window(old_wid)
    window_provider = view.provider_name if view else ""
    if not provider_name:
        provider_name = window_provider
    if window_provider and provider_name != window_provider:
        await query.answer("Session provider mismatch", show_alert=True)
        return

    launch_args = get_provider_for_window(
        old_wid, provider_name=provider_name
    ).make_launch_args(resume_id=session_id)
    await _create_and_bind_window(
        query,
        user_id,
        thread_id,
        cwd,
        context,
        agent_args=launch_args,
        success_label=f"Resuming session: {picked['summary'][:40]}",
        old_window_id=old_wid,
    )
