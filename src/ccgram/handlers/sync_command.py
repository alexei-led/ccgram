"""On-demand state audit and cleanup — /sync command.

Audits all state maps against live tmux windows and reports issues.
A "Fix" button runs cleanup operations and re-audits in place.
Enforcement: closes ghost topics and adopts orphaned tmux windows.

Key functions:
  - sync_command(): /sync command handler
  - handle_sync_fix(): fix button callback — run cleanup, re-audit, edit in place
  - handle_sync_dismiss(): dismiss button callback — remove keyboard
"""

import re

import structlog
from telegram import (
    Bot,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..config import config
from ..session import AuditIssue, AuditResult, session_manager
from ..tmux_manager import tmux_manager
from .callback_data import CB_SYNC_DISMISS, CB_SYNC_FIX
from .cleanup import clear_topic_state
from .message_sender import safe_edit, safe_reply

logger = structlog.get_logger()

_GHOST_RE = re.compile(r"user:(\d+) thread:(\d+) window:(@\d+)")
_WINDOW_RE = re.compile(r"(@\d+)")

_CATEGORY_LABELS: dict[str, str] = {
    "ghost_binding": "ghost binding (dead window)",
    "orphaned_display_name": "orphaned display name",
    "orphaned_group_chat_id": "orphaned group chat ID",
    "stale_window_state": "stale window state",
    "stale_offset": "stale offset entry",
    "display_name_drift": "display name drift",
    "orphaned_window": "unbound tmux window (no topic)",
}


async def _run_audit() -> AuditResult:
    """Fetch live tmux state and run audit."""
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]
    return session_manager.audit_state(live_ids, live_pairs)


def _issue_summary_lines(audit: AuditResult) -> list[str]:
    """Build category summary lines from audit issues."""
    category_counts: dict[str, int] = {}
    for issue in audit.issues:
        if issue.category == "ghost_binding":
            continue  # already shown in binding summary
        category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

    if category_counts:
        return [
            f"\u26a0 {count} {_CATEGORY_LABELS.get(cat, cat)}"
            for cat, count in category_counts.items()
        ]
    if audit.total_bindings > 0:
        return ["\u2713 No orphaned entries", "\u2713 Display names in sync"]
    return []


def _format_report(
    audit: AuditResult, *, fixed_count: int = 0, closed_topic_count: int = 0
) -> tuple[str, InlineKeyboardMarkup | None]:
    """Build report text and optional keyboard."""
    lines: list[str] = []

    if fixed_count > 0:
        issue_word = "issue" if fixed_count == 1 else "issues"
        lines.append(f"\u2705 Fixed {fixed_count} {issue_word}\n")
    else:
        lines.append("\U0001f50d State audit\n")

    if closed_topic_count > 0:
        topic_word = "topic" if closed_topic_count == 1 else "topics"
        lines.append(
            f"\u2139 Closed {closed_topic_count} stale {topic_word} "
            "(delete manually in Telegram)"
        )

    # Binding summary
    if audit.total_bindings == 0:
        lines.append("\u2139 No topic bindings")
    elif audit.live_binding_count == audit.total_bindings:
        lines.append(f"\u2713 {audit.total_bindings} topics bound, all windows alive")
    else:
        dead = audit.total_bindings - audit.live_binding_count
        lines.append(
            f"\u26a0 {dead} ghost binding(s) "
            f"({audit.live_binding_count}/{audit.total_bindings} alive)"
        )

    lines.extend(_issue_summary_lines(audit))

    text = "\n".join(lines)

    # Build keyboard
    fixable = audit.fixable_count
    if fixable > 0:
        issue_word = "issue" if fixable == 1 else "issues"
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"\U0001f527 Fix {fixable} {issue_word}",
                        callback_data=CB_SYNC_FIX,
                    ),
                    InlineKeyboardButton(
                        "\u2715 Dismiss", callback_data=CB_SYNC_DISMISS
                    ),
                ]
            ]
        )
    else:
        keyboard = None

    return text, keyboard


async def _close_ghost_topics(bot: Bot, issues: list[AuditIssue]) -> int:
    """Close Telegram topics for ghost bindings. Returns count of closed topics."""
    closed_count = 0
    for issue in issues:
        if issue.category != "ghost_binding":
            continue
        match = _GHOST_RE.search(issue.detail)
        if not match:
            continue
        user_id = int(match.group(1))
        thread_id = int(match.group(2))
        window_id = match.group(3)
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        topic_closed = False
        if chat_id == user_id:
            logger.warning(
                "No group chat_id for ghost topic thread=%d, skipping close",
                thread_id,
            )
        else:
            try:
                await bot.close_forum_topic(chat_id, thread_id)
                topic_closed = True
            except TelegramError:
                logger.exception(
                    "Failed to close ghost topic thread=%d window=%s",
                    thread_id,
                    window_id,
                )
        # Only unbind if topic was closed (or no group chat to close)
        if topic_closed or chat_id == user_id:
            try:
                await clear_topic_state(
                    user_id, thread_id, bot=bot, window_id=window_id
                )
                session_manager.unbind_thread(user_id, thread_id)
                if topic_closed:
                    closed_count += 1
            except OSError, TelegramError:
                logger.exception(
                    "Failed to clean up ghost binding thread=%d window=%s",
                    thread_id,
                    window_id,
                )
    return closed_count


async def _adopt_orphaned_windows(bot: Bot, issues: list[AuditIssue]) -> None:
    """Create Telegram topics for unbound tmux windows."""
    from ..bot import _handle_new_window
    from ..session_monitor import NewWindowEvent

    for issue in issues:
        if issue.category != "orphaned_window":
            continue
        match = _WINDOW_RE.search(issue.detail)
        if not match:
            continue
        window_id = match.group(1)
        ws = session_manager.get_window_state(window_id)
        name = ws.window_name or session_manager.get_display_name(window_id)
        event = NewWindowEvent(
            window_id=window_id,
            session_id=ws.session_id,
            window_name=name,
            cwd=ws.cwd,
        )
        try:
            await _handle_new_window(event, bot)
        except TelegramError:
            logger.exception("Failed to adopt orphaned window %s", window_id)


async def sync_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sync — audit state and show report."""
    user = update.effective_user
    if not user or not update.message:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    audit = await _run_audit()
    text, keyboard = _format_report(audit)
    await safe_reply(update.message, text, reply_markup=keyboard)


async def handle_sync_fix(query: CallbackQuery) -> None:
    """Run all fix operations, re-audit, and edit message in place."""
    # Single list_windows call — reused for both audit and fix
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]

    # Audit before fixing to count fixable issues
    pre_audit = session_manager.audit_state(live_ids, live_pairs)

    # Run state cleanup operations
    try:
        session_manager.sync_display_names(live_pairs)
        session_manager.prune_stale_state(live_ids)
        session_manager.prune_session_map(live_ids)
        session_manager.prune_stale_window_states(live_ids)
        bound_ids: set[str] = {
            wid for _, _, wid in session_manager.iter_thread_bindings()
        }
        state_ids = set(session_manager.window_states.keys())
        session_manager.prune_stale_offsets(live_ids | bound_ids | state_ids)
    except OSError:
        logger.exception("Error during sync fix operations")

    # Enforcement: close ghost topics and kill orphaned windows
    bot = query.get_bot()
    closed_count = await _close_ghost_topics(bot, pre_audit.issues)
    await _adopt_orphaned_windows(bot, pre_audit.issues)

    # Re-audit and compute actual fixed count (handles partial failures)
    post_audit = await _run_audit()
    actual_fixed = pre_audit.fixable_count - post_audit.fixable_count
    text, keyboard = _format_report(
        post_audit, fixed_count=actual_fixed, closed_topic_count=closed_count
    )
    await safe_edit(query, text, reply_markup=keyboard)


async def handle_sync_dismiss(query: CallbackQuery) -> None:
    """Remove keyboard from sync message."""
    original_text = getattr(query.message, "text", None) if query.message else None
    await safe_edit(query, original_text or "Dismissed", reply_markup=None)
