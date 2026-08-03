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

import json
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
from ...config import config
from ...providers import get_provider_for_window
from ...utils import read_session_metadata_from_jsonl
from ..callback_data import (
    CB_RECOVERY_BACK,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_PICK,
    CB_RECOVERY_BROWSE,
)
from ..callback_helpers import get_thread_id
from ..callback_tokens import compact_callback_data
from ..messaging_pipeline.message_sender import safe_edit
from ..user_state import (
    PENDING_THREAD_ID,
    RECOVERY_SESSIONS,
    RECOVERY_WINDOW_ID,
)
from .recovery_callbacks import _clear_recovery_state

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


def scan_sessions_for_cwd(cwd: str) -> list[_SessionEntry]:
    """Scan project directories for sessions matching a working directory.

    Supports both legacy sessions-index.json and bare JSONL files
    (Claude Code >= Feb 2026 no longer writes index files).

    Returns up to _MAX_RESUME_SESSIONS entries, most-recent file first.
    """
    if not config.claude_projects_path.exists():
        return []

    try:
        resolved_cwd = str(Path(cwd).resolve())
    except OSError:
        return []

    candidates: list[tuple[float, _SessionEntry]] = []
    seen_ids: set[str] = set()

    for project_dir in config.claude_projects_path.iterdir():
        if not project_dir.is_dir():
            continue

        index_file = project_dir / "sessions-index.json"
        if index_file.exists():
            _scan_index_for_cwd(index_file, resolved_cwd, seen_ids, candidates)

        _scan_bare_jsonl_for_cwd(project_dir, resolved_cwd, seen_ids, candidates)

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates[:_MAX_RESUME_SESSIONS]]


def _scan_index_for_cwd(
    index_file: Path,
    resolved_cwd: str,
    seen_ids: set[str],
    candidates: list[tuple[float, _SessionEntry]],
) -> None:
    """Scan a sessions-index.json for sessions matching a cwd."""
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return

    original_path = index_data.get("originalPath", "")
    for entry in index_data.get("entries", []):
        session_id = entry.get("sessionId", "")
        full_path = entry.get("fullPath", "")
        project_path = entry.get("projectPath", original_path)
        if not session_id or not full_path or session_id in seen_ids:
            continue

        try:
            norm_pp = str(Path(project_path).resolve())
        except OSError:
            norm_pp = project_path

        if norm_pp != resolved_cwd:
            continue

        file_path = Path(full_path)
        if not file_path.exists():
            continue

        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0.0

        summary = (
            entry.get("summary", "") or entry.get("firstPrompt", "") or session_id[:12]
        )
        seen_ids.add(session_id)
        candidates.append((mtime, _SessionEntry(session_id, summary, mtime)))


def _scan_bare_jsonl_for_cwd(
    project_dir: Path,
    resolved_cwd: str,
    seen_ids: set[str],
    candidates: list[tuple[float, _SessionEntry]],
) -> None:
    """Scan bare JSONL files for sessions matching a cwd."""
    try:
        jsonl_iter = project_dir.glob("*.jsonl")
    except OSError:
        return

    for jsonl_file in jsonl_iter:
        session_id = jsonl_file.stem
        if session_id in seen_ids:
            continue

        file_cwd, summary = read_session_metadata_from_jsonl(jsonl_file)
        if not file_cwd:
            continue

        try:
            norm_cwd = str(Path(file_cwd).resolve())
        except OSError:
            norm_cwd = file_cwd

        if norm_cwd != resolved_cwd:
            continue

        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0

        seen_ids.add(session_id)
        candidates.append(
            (mtime, _SessionEntry(session_id, summary or session_id[:12], mtime))
        )


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

    old_wid = context.user_data.get(RECOVERY_WINDOW_ID) if context.user_data else None
    if not old_wid:
        await query.answer("Recovery menu expired", show_alert=True)
        return

    view = window_query.view_window(old_wid)
    if view is None or not view.cwd or not Path(view.cwd).is_dir():
        await safe_edit(query, "❌ Directory no longer exists.")
        _clear_recovery_state(context.user_data)
        await query.answer("Project gone")
        return
    cwd = view.cwd

    launch_args = get_provider_for_window(
        old_wid, provider_name=view.provider_name
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
