"""Window creation and topic-binding service for the directory-browser flow.

Extracts the launch sequence from directory_callbacks._create_window_and_bind
into a self-contained service module.  Callers build a ``WindowLaunchRequest``
and call ``launch_window``; the result is a ``WindowLaunchResult``.

CRITICAL ordering invariant (MC-2967):
  create_window() → register_pending_creation(window_id)
  must have NO await between them.  See the inline comment for full context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from ...providers import registry as provider_registry
from ...session import session_manager
from ...session_map import session_map_sync
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.window_ops import send_to_window
from ...user_preferences import user_preferences
from ...window_state_store import CCGRAM_CREATED_WINDOW_ORIGIN
from ..messaging_pipeline.message_sender import safe_edit, safe_send
from ..status.topic_emoji import format_topic_name_for_mode
from .directory_browser import clear_worktree_state, clear_workspace_state
from .topic_creation_draft import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKSPACE_ID,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
)
from . import topic_orchestration

if TYPE_CHECKING:
    from telegram import CallbackQuery
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

__all__ = [
    "WindowLaunchRequest",
    "WindowLaunchResult",
    "launch_window",
]


@dataclass
class WindowLaunchRequest:
    """Parameters for the window-creation + topic-binding step."""

    user_id: int
    thread_id: int | None
    provider_name: str
    cwd: str
    mode: str
    pending_text: str | None
    # Worktree metadata is NOT carried in this request. It flows through
    # context.user_data via PENDING_WORKTREE_PATH / PENDING_WORKTREE_BRANCH /
    # PENDING_WORKTREE_REPO keys, read directly by _persist_worktree_state and
    # _create_topic_window.


@dataclass
class WindowLaunchResult:
    """Outcome of ``launch_window``."""

    success: bool
    window_id: str | None = None
    error_message: str | None = None


# ── helpers ──────────────────────────────────────────────────────────────────


def _cwd_within(cwd: str, worktree_path: str) -> bool:
    """True if *cwd* is the worktree root or nested inside it."""
    try:
        c = Path(cwd).resolve()
        w = Path(worktree_path).resolve()
    except OSError:
        return False
    return c == w or c.is_relative_to(w)


def _persist_worktree_state(
    window_id: str, cwd: str, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Persist a pending worktree path/branch onto the new window state.

    Only persists when the window's *cwd* is the pending worktree path
    (or a subdirectory of it — the new topic may be rooted at a subdir
    of the fresh checkout) so a stale path from an earlier aborted
    attempt can't attach to an unrelated window. Always clears the
    worktree flow keys afterwards.
    """
    user_data = context.user_data
    worktree_path = user_data.get(PENDING_WORKTREE_PATH) if user_data else None
    worktree_branch = user_data.get(PENDING_WORKTREE_BRANCH) if user_data else None
    if worktree_path and worktree_branch and _cwd_within(cwd, worktree_path):
        session_manager.set_window_worktree(window_id, worktree_path, worktree_branch)
    clear_worktree_state(user_data)


async def _abort_topic_creation(
    query: CallbackQuery, message: str, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Surface a window-creation failure and drop all pending-topic state.

    The error message carries no keyboard, so the user must restart the
    flow. Clearing the pending worktree state (including the re-entrancy
    flag) keeps a sticky "creating" guard from rejecting every future
    worktree confirm — the worktree, if any, was already created on disk.
    """
    await safe_edit(query, f"❌ {message}")
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
        context.user_data.pop(PENDING_THREAD_TEXT, None)
    clear_worktree_state(context.user_data)
    clear_workspace_state(context.user_data)


async def _create_topic_window(
    selected_path: str,
    launch_command: str | None,
    chosen_workspace_id: str | None,
    context: ContextTypes.DEFAULT_TYPE,
) -> tuple[bool, str, str, str]:
    """Create the topic's window, returning ``(success, message, name, id)``.

    Native worktree delegation (herdr): when the flow carries a pending worktree
    intent, one ``worktree create`` makes the checkout + grouped workspace + the
    window in a single step. Gated on ``native_worktrees``; tmux always takes the
    ``create_window`` branch (its worktree was already created on disk earlier).
    """
    ud = context.user_data
    wt_repo = ud.get(PENDING_WORKTREE_REPO) if ud else None
    wt_branch = ud.get(PENDING_WORKTREE_BRANCH) if ud else None
    wt_path = ud.get(PENDING_WORKTREE_PATH) if ud else None
    if tmux_manager.capabilities.native_worktrees and wt_repo and wt_branch and wt_path:
        return await tmux_manager.create_worktree_window(
            wt_repo,
            wt_path,
            wt_branch,
            window_name=Path(wt_path).name,
            launch_command=launch_command,
        )
    return await tmux_manager.create_window(
        selected_path,
        launch_command=launch_command,
        workspace_id=chosen_workspace_id,
    )


async def _wait_for_shell_ready(window_id: str, *, attempts: int = 5) -> None:
    """Wait for a freshly created tmux window to show a shell prompt."""
    # Lazy: only needed inside the shell-detection branch
    import os

    # Lazy: providers package heavy bootstrap
    from ccgram.providers.shell import KNOWN_SHELLS

    for _ in range(attempts):
        w = await tmux_manager.find_window_by_id(window_id)
        if w and w.pane_current_command:
            cmd = os.path.basename(w.pane_current_command.split()[0]).lstrip("-")
            if cmd in KNOWN_SHELLS:
                return
        await asyncio.sleep(0.2)


async def _accept_yolo_confirmation(window_id: str, *, timeout: float = 8.0) -> bool:
    """Detect and accept Claude Code's bypass permissions confirmation prompt.

    When launched with --dangerously-skip-permissions, Claude Code shows a
    TUI confirmation where "No, exit" is the default selection. Sends
    Down+Enter to select the "Yes" option so the session can start.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        text = await tmux_manager.capture_pane(window_id)
        if text and "bypass permissions" in text.lower():
            await asyncio.sleep(0.3)
            await tmux_manager.send_keys(window_id, "Down", enter=False, literal=False)
            await asyncio.sleep(0.15)
            await tmux_manager.send_keys(window_id, "Enter", enter=False, literal=False)
            logger.info("Accepted bypass permissions prompt for window %s", window_id)
            return True
        await asyncio.sleep(0.5)
    logger.warning(
        "Bypass permissions prompt not detected within %.0fs for window %s",
        timeout,
        window_id,
    )
    return False


# ── main entry point ──────────────────────────────────────────────────────────


async def launch_window(  # noqa: PLR0915, C901
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    request: WindowLaunchRequest,
) -> WindowLaunchResult:
    """Create a tmux window, bind to the pending topic, and forward pending text.

    Shared by _handle_mode_select (after mode picker) and _handle_provider_select
    (when mode picker is skipped for providers without YOLO flags).

    CRITICAL (MC-2967): ``create_window`` → ``register_pending_creation(wid)``
    must have NO await between them.  The provider's SessionStart hook fires
    inside the new pane within seconds; the SessionMonitor's 1 s poll cycle would
    otherwise see an unbound window and auto-create a duplicate Telegram topic
    before ``bind_thread`` runs below.
    """
    # Lazy: providers package heavy bootstrap
    from ccgram.providers import resolve_launch_command

    user_id = request.user_id
    pending_thread_id = request.thread_id
    selected_path = request.cwd
    provider_name = request.provider_name
    approval_mode = request.mode

    launch_command = resolve_launch_command(provider_name, approval_mode=approval_mode)

    chosen_workspace_id: str | None = (
        context.user_data.get(PENDING_WORKSPACE_ID) if context.user_data else None
    ) or None

    success, message, created_wname, created_wid = await _create_topic_window(
        selected_path, launch_command, chosen_workspace_id, context
    )
    if not success:
        await _abort_topic_creation(query, message, context)
        return WindowLaunchResult(success=False, error_message=message)

    # Race-guard: tag this window as "directory flow in progress" BEFORE any
    # subsequent await. The provider's SessionStart hook fires inside the new
    # tmux pane within seconds; the SessionMonitor's 1s poll cycle would
    # otherwise see an unbound window and auto-create a duplicate Telegram
    # topic before bind_thread() runs below. See MC-2967 for full repro.
    topic_orchestration.register_pending_creation(created_wid)

    user_preferences.update_user_mru(user_id, selected_path)
    session_manager.set_window_origin(created_wid, CCGRAM_CREATED_WINDOW_ORIGIN)
    session_manager.set_window_cwd(created_wid, selected_path)
    session_manager.set_window_provider(created_wid, provider_name)
    session_manager.set_window_approval_mode(created_wid, approval_mode)
    _persist_worktree_state(created_wid, selected_path, context)
    logger.info(
        "Window created: %s (id=%s) at %s provider=%s mode=%s (user=%d, thread=%s)",
        created_wname,
        created_wid,
        selected_path,
        provider_name,
        approval_mode,
        user_id,
        pending_thread_id,
    )
    await tmux_manager.stamp_pane_title(created_wid, provider_name)

    provider_caps = provider_registry.get(provider_name).capabilities
    if provider_caps.chat_first_command_path:
        # Lazy: shell ↔ topics cycle via window_callbacks adoption flow.
        from ..shell.shell_prompt_orchestrator import ensure_setup

        await _wait_for_shell_ready(created_wid)
        await ensure_setup(created_wid, "auto")

    if pending_thread_id is not None:
        thread_router.bind_thread(
            user_id, pending_thread_id, created_wid, window_name=created_wname
        )
        query_message = query.message
        chat = query_message.chat if query_message else None
        if chat and chat.type in ("group", "supergroup"):
            thread_router.set_group_chat_id(user_id, pending_thread_id, chat.id)
        # Bind is durable now — handle_new_window's `_is_window_already_bound`
        # check will find the binding, so the pending-creation race-guard can
        # be released. (Late SessionMonitor polls are still safe: they will
        # take the already-bound branch instead.)
        topic_orchestration.clear_pending_creation(created_wid)

    provider = provider_registry.get(provider_name)
    if approval_mode == "yolo" and provider.capabilities.has_yolo_confirmation:
        await _accept_yolo_confirmation(created_wid)

    if provider.capabilities.supports_hook:
        await session_map_sync.wait_for_session_map_entry(created_wid)

    if pending_thread_id is None:
        await safe_edit(query, f"✅ {message}")
        return WindowLaunchResult(success=True, window_id=created_wid)

    try:
        await context.bot.edit_forum_topic(
            chat_id=thread_router.resolve_chat_id(user_id, pending_thread_id),
            message_thread_id=pending_thread_id,
            name=format_topic_name_for_mode(created_wname, approval_mode),
        )
    except TelegramError as e:
        logger.debug("Failed to rename topic: %s", e)

    await safe_edit(
        query,
        f"✅ {message}\n\nBound to this topic. Send messages here.",
    )

    pending_text = request.pending_text
    if pending_text:
        logger.debug(
            "Forwarding pending text to window %s (len=%d)",
            created_wname,
            len(pending_text),
        )
        if context.user_data is not None:
            context.user_data.pop(PENDING_THREAD_TEXT, None)
            context.user_data.pop(PENDING_THREAD_ID, None)

        # Chat-first providers (shell): route through NL→command approval flow
        if provider_caps.chat_first_command_path:
            # Lazy: telegram_client wraps PTB Bot; shell.shell_commands
            # ↔ topics cycle through approval callback wiring.
            from ...telegram_client import PTBTelegramClient

            # Lazy: shell.shell_commands ↔ topics cycle through approval wiring.
            from ..shell.shell_commands import handle_shell_message

            await handle_shell_message(
                PTBTelegramClient(context.bot),
                user_id,
                pending_thread_id,
                created_wid,
                pending_text,
            )
        else:
            send_ok, send_msg = await send_to_window(created_wid, pending_text)
            if not send_ok:
                logger.warning(
                    "Failed to forward pending text to window %s (user %s): %s",
                    created_wid,
                    user_id,
                    send_msg,
                )
                # Lazy: telegram_client wraps PTB Bot.
                from ...telegram_client import PTBTelegramClient

                await safe_send(
                    PTBTelegramClient(context.bot),
                    thread_router.resolve_chat_id(user_id, pending_thread_id),
                    f"❌ Failed to send pending message: {send_msg}",
                    message_thread_id=pending_thread_id,
                )
    elif context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
    return WindowLaunchResult(success=True, window_id=created_wid)
