"""On-demand state audit and cleanup — /sync command.

Audits all state maps against live multiplexer windows and reports issues.
A "Fix" button runs cleanup operations and re-audits in place.
Enforcement: closes ghost topics, recreates dead topics, and adopts orphaned windows.

Key functions:
  - sync_command(): /sync command handler
  - handle_sync_fix(): fix button callback — run cleanup, re-audit, edit in place
  - handle_sync_dismiss(): dismiss button callback — remove keyboard
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING
import asyncio
import re

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest, TelegramError
from .. import window_query
from ..config import config
from ..session import AuditIssue, AuditResult, session_manager
from ..session_map import session_map_sync
from ..telegram_client import PTBTelegramClient, TelegramClient
from ..thread_router import RetiredTopic, thread_router
from ..multiplexer import multiplexer as tmux_manager
from ..multiplexer.reconciliation import list_windows_for_reconciliation
from ..user_preferences import user_preferences
from .callback_data import CB_SYNC_DISMISS, CB_SYNC_FIX
from .callback_registry import register
from .cleanup import clear_topic_state
from .messaging_pipeline.message_sender import is_thread_gone, safe_edit, safe_reply
from .status.topic_emoji import sync_topic_name
from .topics.topic_probe import probe_topic_exists

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_TELEGRAM_API_CONCURRENCY = 5
_TELEGRAM_PROBE_TIMEOUT_S = 12.0
_GHOST_RE = re.compile(r"user:(\d+)\s+thread:(\d+)\s+window:([^\s(]+)")
_WINDOW_RE = re.compile(r"([^\s(]+)")

_CATEGORY_LABELS: dict[str, str] = {
    "ghost_binding": "ghost binding (dead window)",
    "dead_topic": "dead topic (window alive, topic deleted)",
    "topic_probe_incomplete": "Telegram topic check incomplete",
    "orphaned_display_name": "orphaned display name",
    "orphaned_group_chat_id": "orphaned group chat ID",
    "stale_window_state": "stale window state",
    "stale_offset": "stale offset entry",
    "display_name_drift": "display name drift",
    "orphaned_window": "unbound window (no topic)",
    "legacy_herdr": "legacy Herdr binding (blocked; archive or explicitly rebind)",
}

_RETIRED_OUTCOME_LABELS = {
    "deleted": "Deleted",
    "closed": "Closed",
    "already_gone": "Already gone",
    "failed": "Could not remove",
    "protected_active": "Protected active or rebound",
}


async def _run_audit() -> AuditResult:
    """Fetch live multiplexer state and run audit."""
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]
    return session_manager.audit_state(live_ids, live_pairs)


def _issue_summary_lines(audit: AuditResult) -> list[str]:
    """Build category summary lines from audit issues."""
    category_counts: dict[str, int] = {}
    for issue in audit.issues:
        if issue.category in ("ghost_binding", "dead_topic"):
            continue  # shown in dedicated report lines
        category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

    retired_reasons = Counter(
        issue.detail.removeprefix("reason:")
        for issue in audit.issues
        if issue.category == "retired_topic"
    )
    if retired_reasons:
        category_counts.pop("retired_topic", None)
    lines = [
        f"⚠ {count} {_CATEGORY_LABELS.get(cat, cat)}"
        for cat, count in category_counts.items()
    ]
    lines.extend(
        f"⚠ {count} known retired topic cleanup candidate(s) ({reason})"
        for reason, count in retired_reasons.items()
    )
    if lines:
        return lines
    if audit.total_bindings > 0:
        return ["✓ No orphaned entries", "✓ Tmux display cache in sync"]
    return []


async def _sync_live_topic_names(
    client: TelegramClient, live_ids: set[str] | None = None
) -> None:
    """Best-effort reconciliation of bound live topic titles."""
    if live_ids is None:
        all_windows = await tmux_manager.list_windows()
        live_ids = {w.window_id for w in all_windows}

    bindings: list[tuple[int, int, str]] = []
    for user_id, thread_id, window_id in thread_router.iter_thread_bindings():
        if window_id not in live_ids:
            continue
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        bindings.append((chat_id, thread_id, window_id))

    sem = asyncio.Semaphore(_TELEGRAM_API_CONCURRENCY)

    async def _sync_one(chat_id: int, thread_id: int, window_id: str) -> None:
        async with sem:
            await sync_topic_name(
                client,
                chat_id,
                thread_id,
                thread_router.get_display_name(window_id),
            )

    results = await asyncio.gather(
        *(_sync_one(*binding) for binding in bindings), return_exceptions=True
    )
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            logger.error("Unexpected error syncing topic name", exc_info=result)


def _retired_outcome_lines(retired_outcomes: dict[str, int] | None) -> list[str]:
    """Format explicit outcomes for locally known retired-topic cleanup."""
    return [
        f"{'⚠' if outcome == 'failed' else 'ℹ'} "
        f"{_RETIRED_OUTCOME_LABELS[outcome]} {count} known retired topic(s)"
        for outcome, count in (retired_outcomes or {}).items()
        if count
    ]


def _format_report(
    audit: AuditResult,
    *,
    fixed_count: int = 0,
    closed_topic_count: int = 0,
    recreated_topic_count: int = 0,
    manual_close_count: int = 0,
    retired_outcomes: dict[str, int] | None = None,
) -> tuple[str, InlineKeyboardMarkup | None]:
    """Build report text and optional keyboard."""
    lines: list[str] = []

    if fixed_count > 0:
        issue_word = "issue" if fixed_count == 1 else "issues"
        lines.append(f"✅ Fixed {fixed_count} {issue_word}\n")
    else:
        lines.append("🔍 State audit\n")

    if closed_topic_count > 0:
        topic_word = "topic" if closed_topic_count == 1 else "topics"
        lines.append(f"ℹ Removed {closed_topic_count} stale {topic_word}")

    if recreated_topic_count > 0:
        topic_word = "topic" if recreated_topic_count == 1 else "topics"
        lines.append(f"ℹ Recreated {recreated_topic_count} {topic_word}")

    if manual_close_count > 0:
        topic_word = "topic" if manual_close_count == 1 else "topics"
        lines.append(
            f"⚠ {manual_close_count} {topic_word} could not be closed automatically; "
            "safe to close manually"
        )

    lines.extend(_retired_outcome_lines(retired_outcomes))

    # Binding summary
    if audit.total_bindings == 0:
        lines.append("ℹ No topic bindings")
    elif audit.live_binding_count == audit.total_bindings:
        lines.append(f"✓ {audit.total_bindings} topics bound, all windows alive")
    else:
        dead = audit.total_bindings - audit.live_binding_count
        lines.append(
            f"⚠ {dead} ghost binding(s) "
            f"({audit.live_binding_count}/{audit.total_bindings} alive)"
        )

    # Dead topic summary (window alive, but Telegram topic deleted)
    dead_topic_count = sum(1 for i in audit.issues if i.category == "dead_topic")
    if dead_topic_count > 0:
        topic_word = "topic" if dead_topic_count == 1 else "topics"
        lines.append(f"⚠ {dead_topic_count} dead {topic_word} (deleted in Telegram)")

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
                    InlineKeyboardButton("✕ Dismiss", callback_data=CB_SYNC_DISMISS),
                ]
            ]
        )
    else:
        keyboard = None

    return text, keyboard


def _retired_topic_issues() -> list[AuditIssue]:
    """Return Fix candidates known from local retired-binding state only."""
    return [
        AuditIssue(
            category="retired_topic",
            detail=f"reason:{topic.reason}",
            fixable=True,
        )
        for topic in thread_router.iter_retired_topics()
        if topic.cleanup_eligible
    ]


async def _remove_topic(client: TelegramClient, chat_id: int, thread_id: int) -> bool:
    """Try to delete a topic, fall back to close. Returns True on success.

    Only "topic not found" BadRequest is treated as success; other BadRequest
    errors (e.g. insufficient rights) fall through to the close fallback.
    """
    try:
        await client.delete_forum_topic(chat_id, thread_id)
        return True
    except BadRequest as e:
        if is_thread_gone(e):
            return True
    except TelegramError:
        pass
    try:
        await client.close_forum_topic(chat_id, thread_id)
        return True
    except TelegramError:
        return False


async def _delete_retired_topic(client: TelegramClient, topic: RetiredTopic) -> str:
    """Delete a known topic, returning a terminal result or close fallback."""
    chat_id = topic.chat_id
    thread_id = topic.thread_id
    try:
        deleted = await client.delete_forum_topic(chat_id, thread_id)
    except BadRequest as exc:
        return "already_gone" if is_thread_gone(exc) else "failed"
    except TelegramError:
        return "failed"
    return "deleted" if deleted is not False else "failed"


async def _close_retired_topic(client: TelegramClient, topic: RetiredTopic) -> str:
    """Close a known topic when deletion is unavailable."""
    try:
        closed = await client.close_forum_topic(topic.chat_id, topic.thread_id)
    except TelegramError as exc:
        return "already_gone" if is_thread_gone(exc) else "failed"
    return "closed" if closed is not False else "failed"


async def _cleanup_retired_topics(client: TelegramClient) -> dict[str, int]:
    """Remove only locally known, eligible retired topics.

    A binding is rechecked immediately before the Telegram request. This does
    not discover remote topics: it operates solely on the bounded registry.
    """
    outcomes: Counter[str] = Counter()
    terminal_outcomes = {"deleted", "closed", "already_gone"}
    for topic in tuple(thread_router.iter_retired_topics()):
        if not topic.cleanup_eligible:
            continue
        if thread_router.get_window_for_chat_thread(topic.chat_id, topic.thread_id):
            outcomes["protected_active"] += 1
            continue
        outcome = await _delete_retired_topic(client, topic)
        if outcome == "failed":
            outcome = await _close_retired_topic(client, topic)
        outcomes[outcome] += 1
        if outcome in terminal_outcomes:
            thread_router.discard_retired_topic(topic)
    return dict(outcomes)


async def _close_ghost_topics(
    client: TelegramClient, issues: list[AuditIssue]
) -> tuple[int, int]:
    """Delete (or close) Telegram topics for ghost bindings.

    Tries ``delete_forum_topic`` first to fully remove the dead topic from the
    sidebar.  Falls back to ``close_forum_topic`` if deletion fails (e.g.
    missing ``can_manage_topics`` or General topic).  Returns
    ``(closed_count, manual_close_count)``.
    """
    closed_count = 0
    manual_close_count = 0
    for issue in issues:
        if issue.category != "ghost_binding":
            continue
        match = _GHOST_RE.search(issue.detail)
        if not match:
            continue
        user_id = int(match.group(1))
        thread_id = int(match.group(2))
        window_id = match.group(3)
        current_window_id = thread_router.get_window_for_thread(user_id, thread_id)
        if current_window_id != window_id:
            continue
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        topic_removed = await _remove_topic(client, chat_id, thread_id)
        if not topic_removed:
            logger.warning(
                "Failed to delete/close ghost topic thread=%d window=%s",
                thread_id,
                window_id,
            )
            manual_close_count += 1
            continue
        try:
            await clear_topic_state(
                user_id, thread_id, client=client, window_id=window_id
            )
            thread_router.unbind_thread(
                user_id,
                thread_id,
                retirement_reason="remote_removed",
            )
            closed_count += 1
        except OSError, TelegramError:
            logger.exception(
                "Failed to clean up ghost binding thread=%d window=%s",
                thread_id,
                window_id,
            )
    return closed_count, manual_close_count


async def _adopt_orphaned_windows(
    client: TelegramClient, issues: list[AuditIssue]
) -> None:
    """Create Telegram topics for unbound multiplexer windows."""
    # Lazy: bidirectional cycle — topic_orchestration.adopt_unbound_windows
    # also lazy-imports _adopt_orphaned_windows from this module.  Either
    # side must remain lazy until one is split into a third module.
    # Lazy: session_monitor / topic_orchestration cycle through window-creation flow
    from ..session_monitor import NewWindowEvent

    # Lazy: session_monitor / topic_orchestration cycle through window-creation flow
    from .topics.topic_orchestration import handle_new_window as _handle_new_window

    for issue in issues:
        if issue.category != "orphaned_window":
            continue
        match = _WINDOW_RE.search(issue.detail)
        if not match:
            continue
        window_id = match.group(1)
        view = window_query.view_window(window_id)
        name = (view.window_name if view else "") or thread_router.get_display_name(
            window_id
        )
        event = NewWindowEvent(
            window_id=window_id,
            session_id=view.session_id if view else "",
            window_name=name,
            cwd=view.cwd if view else "",
        )
        try:
            await _handle_new_window(event, client)
        except TelegramError, OSError:
            logger.exception("Failed to adopt orphaned window %s", window_id)


async def _probe_dead_topics(client: TelegramClient) -> list[AuditIssue]:
    """Probe Telegram topics for all live bindings, return dead_topic issues.

    Sends a silent dot message to each thread and deletes it immediately.
    ``send_chat_action`` does NOT validate thread existence —
    only ``send_message`` reliably throws "thread not found" for deleted topics.
    """
    bindings = [
        (uid, tid, wid, thread_router.resolve_chat_id(uid, tid))
        for uid, tid, wid in thread_router.iter_thread_bindings()
    ]
    if not bindings:
        return []

    sem = asyncio.Semaphore(_TELEGRAM_API_CONCURRENCY)

    async def _probe_one(
        user_id: int, thread_id: int, window_id: str, chat_id: int
    ) -> AuditIssue | None:
        async with sem:
            exists = await probe_topic_exists(client, chat_id, thread_id)
            if exists is False:
                display = thread_router.get_display_name(window_id)
                return AuditIssue(
                    category="dead_topic",
                    detail=f"user:{user_id} thread:{thread_id} window:{window_id} ({display})",
                    fixable=True,
                )
        return None

    results = await asyncio.gather(
        *(_probe_one(*b) for b in bindings), return_exceptions=True
    )
    issues: list[AuditIssue] = []
    for r in results:
        if isinstance(r, AuditIssue):
            issues.append(r)
        elif isinstance(r, BaseException):
            logger.error("Unexpected error probing dead topics", exc_info=r)
    return issues


async def _recreate_dead_topics(
    client: TelegramClient, issues: list[AuditIssue]
) -> int:
    """Unbind dead topics and recreate them via _handle_new_window.

    Returns count of successfully recreated topics.
    """
    # Lazy: same sync_command ↔ topic_orchestration cycle as
    # _adopt_orphaned_windows.
    # Lazy: session_monitor / topic_orchestration cycle through window-creation flow
    from ..session_monitor import NewWindowEvent

    # Lazy: session_monitor / topic_orchestration cycle through window-creation flow
    from .topics.topic_orchestration import handle_new_window as _handle_new_window

    recreated = 0
    for issue in issues:
        if issue.category != "dead_topic":
            continue
        match = _GHOST_RE.search(issue.detail)
        if not match:
            continue
        user_id = int(match.group(1))
        thread_id = int(match.group(2))
        window_id = match.group(3)
        current_window_id = thread_router.get_window_for_thread(user_id, thread_id)
        if current_window_id != window_id:
            continue

        view = window_query.view_window(window_id)
        name = (view.window_name if view else "") or thread_router.get_display_name(
            window_id
        )
        event = NewWindowEvent(
            window_id=window_id,
            session_id=view.session_id if view else "",
            window_name=name,
            cwd=view.cwd if view else "",
        )

        # Preserve group_chat_id before unbinding; the targeted repair passes it
        # directly so another user's binding cannot short-circuit recreation.
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)

        thread_router.unbind_thread(
            user_id,
            thread_id,
            retirement_reason="remote_deleted",
        )

        created = False
        try:
            created = await _handle_new_window(
                event,
                client,
                target_user_id=user_id,
                target_chat_id=chat_id,
            )
            if created:
                recreated += 1
            else:
                logger.warning("Could not recreate topic for window %s", window_id)
        except TelegramError, OSError:
            logger.exception("Failed to recreate topic for window %s", window_id)
        finally:
            if not created:
                thread_router.bind_thread(
                    user_id, thread_id, window_id, window_name=name, chat_id=chat_id
                )
                thread_router.set_group_chat_id(user_id, thread_id, chat_id)
    return recreated


async def sync_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sync — audit state and show report."""
    user = update.effective_user
    if not user or not update.message:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    logger.info(
        "State audit command started",
        chat_id=update.message.chat.id,
        thread_id=update.message.message_thread_id,
    )
    status_msg = await safe_reply(update.message, "🔍 State audit…")
    client = PTBTelegramClient(update.get_bot())
    audit = await _run_audit()
    logger.info(
        "Local state audit completed",
        issue_count=len(audit.issues),
    )
    # Probe Telegram topics for live bindings (async, needs client).
    # Topic-name reconciliation is a mutation and belongs to the Fix action.
    try:
        async with asyncio.timeout(_TELEGRAM_PROBE_TIMEOUT_S):
            dead_issues = await _probe_dead_topics(client)
    except TimeoutError:
        audit.issues.append(
            AuditIssue(
                category="topic_probe_incomplete",
                detail="Telegram topic existence check timed out",
                fixable=False,
            )
        )
        logger.warning(
            "Telegram topic probe timed out",
            timeout_s=_TELEGRAM_PROBE_TIMEOUT_S,
        )
    else:
        audit.issues.extend(dead_issues)
        logger.info(
            "Telegram topic probe completed",
            dead_topic_count=len(dead_issues),
        )
    audit.issues.extend(_retired_topic_issues())
    text, keyboard = _format_report(audit)
    if status_msg is not None:
        await safe_edit(status_msg, text, reply_markup=keyboard)
    else:
        await safe_reply(update.message, text, reply_markup=keyboard)
    logger.info(
        "State audit command completed",
        issue_count=len(audit.issues),
    )


async def handle_sync_fix(query: CallbackQuery) -> None:
    """Run all fix operations, re-audit, and edit message in place."""
    await safe_edit(query, "🔧 Fixing…", reply_markup=None)

    # A destructive repair requires a confirmed multiplexer listing.
    all_windows = await list_windows_for_reconciliation(tmux_manager)
    if all_windows is None:
        await safe_edit(
            query,
            "⚠ Multiplexer unavailable. No state changes were made.",
            reply_markup=None,
        )
        return

    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]

    # Audit before fixing to count fixable issues
    client = PTBTelegramClient(query.get_bot())
    pre_audit = session_manager.audit_state(live_ids, live_pairs)
    dead_issues = await _probe_dead_topics(client)
    pre_audit.issues.extend(dead_issues)
    pre_audit.issues.extend(_retired_topic_issues())

    # Run state cleanup operations
    try:
        session_manager.sync_display_names(live_pairs)
        session_manager.prune_stale_state(live_ids)
        session_map_sync.prune_session_map(live_ids)
        session_manager.prune_stale_window_states(live_ids)
        bound_ids: set[str] = {
            wid for _, _, wid in thread_router.iter_thread_bindings()
        }
        state_ids = set(window_query.iter_window_ids())
        user_preferences.prune_stale_offsets(live_ids | bound_ids | state_ids)
    except OSError:
        logger.exception("Error during sync fix operations")

    await _sync_live_topic_names(client, live_ids)

    # Enforcement: adopt orphans first so stale same-name topics can be rebound.
    await _adopt_orphaned_windows(client, pre_audit.issues)
    closed_count, manual_close_count = await _close_ghost_topics(
        client, pre_audit.issues
    )
    recreated_count = await _recreate_dead_topics(client, pre_audit.issues)
    retired_outcomes = await _cleanup_retired_topics(client)

    # Re-audit and compute actual fixed count (handles partial failures).
    # No skip_threads here: successful recreations use a new thread_id (old
    # one is unbound and won't be probed), while failed ones restore the old
    # binding and must be re-probed to avoid inflating actual_fixed.
    post_audit = await _run_audit()
    post_dead = await _probe_dead_topics(client)
    post_audit.issues.extend(post_dead)
    post_audit.issues.extend(_retired_topic_issues())
    actual_fixed = pre_audit.fixable_count - post_audit.fixable_count
    text, keyboard = _format_report(
        post_audit,
        fixed_count=actual_fixed,
        closed_topic_count=closed_count,
        recreated_topic_count=recreated_count,
        manual_close_count=manual_close_count,
        retired_outcomes=retired_outcomes,
    )
    await safe_edit(query, text, reply_markup=keyboard)


async def handle_sync_dismiss(query: CallbackQuery) -> None:
    """Delete the sync dialog message."""
    if query.message is None:
        return
    try:
        await query.delete_message()
        return
    except TelegramError:
        pass
    original_text = getattr(query.message, "text", None)
    await safe_edit(query, original_text or "Dismissed", reply_markup=None)


@register(CB_SYNC_FIX, CB_SYNC_DISMISS)
async def _dispatch(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    if not query or not query.data:
        return

    if query.data == CB_SYNC_FIX:
        if user is None or not config.is_user_allowed(user.id):
            await query.answer("You are not authorized", show_alert=True)
            return
        await query.answer("Running fix...")
        await handle_sync_fix(query)
    elif query.data == CB_SYNC_DISMISS:
        await query.answer("Dismissed")
        await handle_sync_dismiss(query)
