"""Directory browser callback handlers.

Handles all inline keyboard callbacks for the directory browser UI:
  - CB_DIR_SELECT: Navigate into a subdirectory
  - CB_DIR_UP: Navigate to parent directory
  - CB_DIR_PAGE: Paginate directory listing
  - CB_DIR_CONFIRM: Confirm directory selection, show provider picker
  - CB_PROV_SELECT: Select provider, then show launch mode picker
  - CB_MODE_SELECT: Select launch mode and create tmux window
  - CB_DIR_CANCEL: Cancel directory browsing
  - CB_DIR_FAV: Select a favorite directory
  - CB_DIR_STAR: Star/unstar a directory

Key function: handle_directory_callback (uniform callback handler signature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio

import structlog
from pathlib import Path

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from ...providers import registry as provider_registry
from ...session import session_manager
from ...session_map import session_map_sync
from ...user_preferences import user_preferences
from ...window_state_store import CCGRAM_CREATED_WINDOW_ORIGIN
from ...thread_router import thread_router
from ...tmux_manager import send_to_window, tmux_manager
from ..callback_data import (
    CB_DIR_CANCEL,
    CB_DIR_CONFIRM,
    CB_DIR_FAV,
    CB_DIR_HOME,
    CB_DIR_PAGE,
    CB_DIR_SELECT,
    CB_DIR_STAR,
    CB_DIR_UP,
    CB_MODE_SELECT,
    CB_PROV_SELECT,
    CB_WT_CONFIRM,
    CB_WT_EDIT_NAME,
    CB_WT_NEW,
    CB_WT_USE_CURRENT,
)
from ..callback_helpers import get_thread_id
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    build_directory_browser,
    build_mode_picker,
    build_provider_picker,
    build_worktree_confirm,
    build_worktree_picker,
    clear_browse_state,
    clear_worktree_state,
    get_favorites,
)
from .worktree import (
    WorktreeError,
    check_worktree_eligibility,
    create_worktree,
    slug_for_path,
    suggest_branch_name,
    worktree_path_for,
)
from ..callback_registry import register
from ..messaging_pipeline.message_sender import safe_edit, safe_send
from ..status.topic_emoji import format_topic_name_for_mode
from ..user_state import (
    AWAITING_WORKTREE_BRANCH_NAME,
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_CREATING,
    PENDING_WORKTREE_DIRTY,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
    PENDING_WORKTREE_SUBDIR,
)
from . import topic_orchestration

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


async def handle_directory_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle directory browser callbacks.

    Dispatches to the appropriate sub-handler based on callback data prefix.
    """
    if data.startswith(CB_DIR_FAV):
        await _handle_fav(query, user_id, data, update, context)
    elif data.startswith(CB_DIR_STAR):
        await _handle_star(query, user_id, data, update, context)
    elif data.startswith(CB_DIR_SELECT):
        await _handle_select(query, user_id, data, update, context)
    elif data == CB_DIR_UP:
        await _handle_up(query, user_id, update, context)
    elif data == CB_DIR_HOME:
        await _handle_home(query, user_id, update, context)
    elif data.startswith(CB_DIR_PAGE):
        await _handle_page(query, user_id, data, update, context)
    elif data == CB_DIR_CONFIRM:
        await _handle_confirm(query, user_id, update, context)
    elif data.startswith(CB_PROV_SELECT):
        await _handle_provider_select(query, user_id, data, update, context)
    elif data.startswith(CB_MODE_SELECT):
        await _handle_mode_select(query, user_id, data, update, context)
    elif data in (CB_WT_USE_CURRENT, CB_WT_NEW, CB_WT_CONFIRM, CB_WT_EDIT_NAME):
        await _handle_worktree_callback(query, data, update, context)
    elif data == CB_DIR_CANCEL:
        await _handle_cancel(query, update, context)


def _browser_flow_stale(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """True if the directory-browser flow was reset or the tap is cross-topic.

    A live browser always has ``PENDING_THREAD_ID`` set in the same topic
    (``_handle_unbound_topic`` / ``_handle_dead_window`` set it together
    with the browse state; navigation never clears it). If it is gone
    (``/new`` or Cancel cleared it) or the tap arrived in a different
    topic, every navigation/favorites handler must fail closed: otherwise
    they repopulate ``BROWSE_PATH_KEY`` (falling back to the bot's own
    cwd) *without* setting ``STATE_KEY``, so ``_check_ui_guards`` can't
    catch the residue and a later stale ``db:confirm`` spawns a window in
    that path. ``_handle_star`` would also toggle a persistent favorite
    off a dead browser.
    """
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    return pending_tid is None or get_thread_id(update) != pending_tid


async def _resolve_fav_index(
    query: CallbackQuery,
    user_id: int,
    data: str,
    prefix: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> str | None:
    """Validate pending thread, parse fav index, and return the fav path or None."""
    if _browser_flow_stale(update, context):
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return None
    try:
        idx = int(data[len(prefix) :])
    except ValueError:
        await query.answer("Invalid data")
        return None

    favorites, _starred = get_favorites(user_id)
    if idx < 0 or idx >= len(favorites):
        await query.answer("Favorite not found", show_alert=True)
        return None
    return favorites[idx]


async def _handle_fav(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_FAV: select a favorite directory and navigate into it."""
    fav_path = await _resolve_fav_index(
        query, user_id, data, CB_DIR_FAV, update, context
    )
    if fav_path is None:
        return
    if not Path(fav_path).is_dir():
        await query.answer("Directory no longer exists", show_alert=True)
        return

    if context.user_data is not None:
        context.user_data[BROWSE_PATH_KEY] = fav_path
        context.user_data[BROWSE_PAGE_KEY] = 0

    msg_text, keyboard, subdirs = build_directory_browser(fav_path, user_id=user_id)
    if context.user_data is not None:
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


async def _handle_star(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_STAR: toggle star on a favorite directory."""
    fav_path = await _resolve_fav_index(
        query, user_id, data, CB_DIR_STAR, update, context
    )
    if fav_path is None:
        return
    now_starred = user_preferences.toggle_user_star(user_id, fav_path)

    # Rebuild browser at current path to update star icons
    default_path = str(Path.cwd())
    current_path = (
        context.user_data.get(BROWSE_PATH_KEY, default_path)
        if context.user_data
        else default_path
    )
    current_page = context.user_data.get(BROWSE_PAGE_KEY, 0) if context.user_data else 0
    msg_text, keyboard, subdirs = build_directory_browser(
        current_path, current_page, user_id=user_id
    )
    if context.user_data is not None:
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer("⭐ Starred" if now_starred else "☆ Unstarred")


async def _handle_select(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_SELECT: navigate into a subdirectory."""
    if _browser_flow_stale(update, context):
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return
    try:
        idx = int(data[len(CB_DIR_SELECT) :])
    except ValueError:
        await query.answer("Invalid data")
        return

    cached_dirs: list[str] = (
        context.user_data.get(BROWSE_DIRS_KEY, []) if context.user_data else []
    )
    if idx < 0 or idx >= len(cached_dirs):
        await query.answer("Directory list changed, please refresh", show_alert=True)
        return
    subdir_name = cached_dirs[idx]

    default_path = str(Path.cwd())
    current_path = (
        context.user_data.get(BROWSE_PATH_KEY, default_path)
        if context.user_data
        else default_path
    )
    new_path = (Path(current_path) / subdir_name).resolve()

    if not new_path.exists() or not new_path.is_dir():
        await query.answer("Directory not found", show_alert=True)
        return

    new_path_str = str(new_path)
    if context.user_data is not None:
        context.user_data[BROWSE_PATH_KEY] = new_path_str
        context.user_data[BROWSE_PAGE_KEY] = 0

    msg_text, keyboard, subdirs = build_directory_browser(new_path_str, user_id=user_id)
    if context.user_data is not None:
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


async def _handle_up(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_UP: navigate to parent directory."""
    if _browser_flow_stale(update, context):
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return
    default_path = str(Path.cwd())
    current_path = (
        context.user_data.get(BROWSE_PATH_KEY, default_path)
        if context.user_data
        else default_path
    )
    current = Path(current_path).resolve()
    parent = current.parent

    parent_path = str(parent)
    if context.user_data is not None:
        context.user_data[BROWSE_PATH_KEY] = parent_path
        context.user_data[BROWSE_PAGE_KEY] = 0

    msg_text, keyboard, subdirs = build_directory_browser(parent_path, user_id=user_id)
    if context.user_data is not None:
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


async def _handle_home(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_HOME: jump to home directory."""
    if _browser_flow_stale(update, context):
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return

    home_path = str(Path.home())
    if context.user_data is not None:
        context.user_data[BROWSE_PATH_KEY] = home_path
        context.user_data[BROWSE_PAGE_KEY] = 0

    msg_text, keyboard, subdirs = build_directory_browser(home_path, user_id=user_id)
    if context.user_data is not None:
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


async def _handle_page(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_PAGE: paginate directory listing."""
    if _browser_flow_stale(update, context):
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return
    try:
        pg = int(data[len(CB_DIR_PAGE) :])
    except ValueError:
        await query.answer("Invalid data")
        return
    default_path = str(Path.cwd())
    current_path = (
        context.user_data.get(BROWSE_PATH_KEY, default_path)
        if context.user_data
        else default_path
    )
    if context.user_data is not None:
        context.user_data[BROWSE_PAGE_KEY] = pg

    msg_text, keyboard, subdirs = build_directory_browser(
        current_path, pg, user_id=user_id
    )
    if context.user_data is not None:
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


def _subdir_within_repo(selected_path: str, repo_path: Path) -> str:
    """Path of *selected_path* relative to *repo_path*, or "" if at the root.

    Both sides are resolved first so a symlinked tmp/realpath mismatch
    (common on macOS) doesn't lose the subdirectory. Returns "" when
    *selected_path* is the repo top-level or not inside the repo.
    """
    try:
        rel = Path(selected_path).resolve().relative_to(repo_path.resolve())
    except ValueError, OSError:
        return ""
    return str(rel) if rel.parts else ""


async def _handle_confirm(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_CONFIRM: confirm directory, show provider picker."""
    selected_path = _required_selected_path(context)
    pending_thread_id: int | None = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )

    # A live browser always has both a selected path and a pending thread
    # (set together when it was shown). Either being absent means the flow
    # was reset (e.g. /new) and this is a stale tap — proceeding would
    # confirm the bot's own cwd and spawn an unbound window/worktree there.
    if selected_path is None or pending_thread_id is None:
        clear_browse_state(context.user_data)
        clear_worktree_state(context.user_data)
        if context.user_data is not None:
            context.user_data.pop(PENDING_THREAD_ID, None)
            context.user_data.pop(PENDING_THREAD_TEXT, None)
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return

    confirm_thread_id = get_thread_id(update)
    if pending_thread_id is not None and confirm_thread_id != pending_thread_id:
        clear_browse_state(context.user_data)
        clear_worktree_state(context.user_data)
        if context.user_data is not None:
            context.user_data.pop(PENDING_THREAD_ID, None)
            context.user_data.pop(PENDING_THREAD_TEXT, None)
        await query.answer("Stale browser (topic mismatch)", show_alert=True)
        return

    await query.answer()

    # Guard against double-click: if thread already has a window, skip
    if pending_thread_id is not None:
        existing_wid = thread_router.get_window_for_thread(user_id, pending_thread_id)
        if existing_wid is not None:
            display = thread_router.get_display_name(existing_wid)
            logger.warning(
                "Thread %d already bound to window %s (%s), ignoring duplicate confirm",
                pending_thread_id,
                existing_wid,
                display,
            )
            clear_browse_state(context.user_data)
            await safe_edit(
                query,
                f"✅ Already bound to window {display}.",
            )
            return

    # Eligible git repo → offer the worktree step before provider pick.
    # Ineligible (non-git, bare, detached, mid-rebase) → unchanged flow.
    # Offloaded: check_worktree_eligibility runs blocking git subprocesses.
    eligibility = await asyncio.to_thread(
        check_worktree_eligibility, Path(selected_path)
    )
    if eligibility.eligible and eligibility.repo_path is not None:
        if context.user_data is not None:
            context.user_data[PENDING_WORKTREE_REPO] = str(eligibility.repo_path)
            context.user_data[PENDING_WORKTREE_DIRTY] = eligibility.dirty
            context.user_data[PENDING_WORKTREE_SUBDIR] = _subdir_within_repo(
                selected_path, eligibility.repo_path
            )
        text, keyboard = build_worktree_picker(
            str(eligibility.repo_path), eligibility.current_branch or "HEAD"
        )
        await safe_edit(query, text, reply_markup=keyboard)
        return

    # Show provider selection keyboard (keep browse state for _handle_provider_select)
    await _show_provider_picker(query, selected_path)


async def _show_provider_picker(query: CallbackQuery, selected_path: str) -> None:
    """Edit the message to the provider picker for *selected_path*."""
    text, keyboard = build_provider_picker(selected_path)
    await safe_edit(query, text, reply_markup=keyboard)


def _cancel_only_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)]]
    )


def _required_selected_path(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Selected directory for a window-creating step, or None if the flow
    was reset (e.g. by ``/new``, which clears ``BROWSE_PATH_KEY``).

    Unlike the navigation handlers, the create path must never fall back
    to the bot's cwd: a stale provider/worktree button tapped after a
    reset would otherwise spawn an unbound tmux window running an agent
    CLI in the bot's own working directory.
    """
    if context.user_data is None:
        return None
    path = context.user_data.get(BROWSE_PATH_KEY)
    return path if isinstance(path, str) and path else None


async def _handle_worktree_callback(
    query: CallbackQuery,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Dispatch the four worktree-picker callbacks (shared stale guard)."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    # Same fail-closed invariant as _handle_confirm / window_callbacks
    # _handle_new: a live worktree picker always has PENDING_THREAD_ID.
    # None means the flow was reset (e.g. /new, or Cancel raced the
    # eligibility probe in _handle_confirm) — a stale tap that would
    # otherwise reach a sub-handler whose only remaining guard is a
    # leftover PENDING_WORKTREE_REPO and spawn an unbound window.
    if pending_tid is None:
        await query.answer("Stale browser (flow reset)", show_alert=True)
        return
    if get_thread_id(update) != pending_tid:
        await query.answer("Stale browser (topic mismatch)", show_alert=True)
        return
    if data == CB_WT_USE_CURRENT:
        await _handle_wt_use_current(query, context)
    elif data == CB_WT_NEW:
        await _handle_wt_new(query, context)
    elif data == CB_WT_CONFIRM:
        await _handle_wt_confirm(query, context)
    elif data == CB_WT_EDIT_NAME:
        await _handle_wt_edit_name(query, context)


async def _handle_wt_use_current(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Keep the current branch — clear worktree state, go to provider pick."""
    await query.answer()
    repo = context.user_data.get(PENDING_WORKTREE_REPO) if context.user_data else None
    selected_path = _required_selected_path(context)
    if not repo or not selected_path:
        await safe_edit(query, "❌ Worktree state lost. Tap Cancel and retry.")
        return
    clear_worktree_state(context.user_data)
    await _show_provider_picker(query, selected_path)


async def _handle_wt_new(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Suggest a non-colliding branch name and show the confirm/edit screen."""
    await query.answer()
    repo = context.user_data.get(PENDING_WORKTREE_REPO) if context.user_data else None
    if not repo:
        await safe_edit(query, "❌ Worktree state lost. Tap Cancel and retry.")
        return
    repo_path = Path(repo)
    # Offloaded: suggest_branch_name runs blocking git branch/worktree list.
    branch = await asyncio.to_thread(suggest_branch_name, None, repo_path)
    worktree_path = worktree_path_for(repo_path, slug_for_path(branch))
    dirty = bool(
        context.user_data.get(PENDING_WORKTREE_DIRTY, False)
        if context.user_data
        else False
    )
    if context.user_data is not None:
        context.user_data[PENDING_WORKTREE_BRANCH] = branch
        context.user_data[PENDING_WORKTREE_PATH] = str(worktree_path)
    text, keyboard = build_worktree_confirm(repo, branch, str(worktree_path), dirty)
    await safe_edit(query, text, reply_markup=keyboard)


async def _handle_wt_confirm(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Create the worktree, then continue to provider pick rooted in it."""
    user_data = context.user_data
    # Re-entrancy guard set synchronously *before* the first await: a
    # fast double-tap on "Use this" would otherwise run create_worktree
    # twice and the second call would overwrite the provider picker with
    # a "branch already exists" error even though the first succeeded.
    if user_data is not None and user_data.get(PENDING_WORKTREE_CREATING):
        await query.answer("Creating worktree…")
        return
    if user_data is not None:
        user_data[PENDING_WORKTREE_CREATING] = True
    await query.answer()
    repo = user_data.get(PENDING_WORKTREE_REPO) if user_data else None
    branch = user_data.get(PENDING_WORKTREE_BRANCH) if user_data else None
    worktree_path = user_data.get(PENDING_WORKTREE_PATH) if user_data else None
    if not (repo and branch and worktree_path):
        if user_data is not None:
            user_data.pop(PENDING_WORKTREE_CREATING, None)
        await safe_edit(query, "❌ Worktree state lost. Tap Cancel and retry.")
        return
    try:
        # Offloaded: create_worktree runs a blocking `git worktree add`
        # (up to 30s) that would otherwise freeze the whole event loop.
        await asyncio.to_thread(
            create_worktree, Path(repo), branch, Path(worktree_path)
        )
    except WorktreeError as exc:
        # Clear the guard so a transient failure (e.g. disk full) is
        # retryable from the same screen.
        if user_data is not None:
            user_data.pop(PENDING_WORKTREE_CREATING, None)
        logger.warning("Worktree creation failed: %s", exc)
        await safe_edit(
            query,
            f"❌ Could not create worktree: {str(exc).splitlines()[0]}",
            reply_markup=_cancel_only_keyboard(),
        )
        return
    subdir = user_data.get(PENDING_WORKTREE_SUBDIR, "") if user_data else ""
    target = Path(worktree_path)
    if subdir:
        candidate = target / subdir
        if candidate.is_dir():
            target = candidate
        else:
            logger.info(
                "Worktree subdir %s absent in fresh checkout; rooting at %s",
                subdir,
                worktree_path,
            )
    target_str = str(target)
    if user_data is not None:
        user_data[BROWSE_PATH_KEY] = target_str
    logger.info(
        "Created worktree %s on branch %s (cwd=%s)",
        worktree_path,
        branch,
        target_str,
    )
    await _show_provider_picker(query, target_str)


async def _handle_wt_edit_name(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Prompt for a custom branch name via a text reply."""
    await query.answer()
    # Fail closed like the other worktree handlers: a stale wt:ed tapped
    # after the flow was reset (e.g. by /new clearing PENDING_WORKTREE_REPO)
    # must not arm AWAITING_WORKTREE_BRANCH_NAME — a leaked flag hijacks the
    # next message in a fresh unbound-topic flow with "Worktree state lost".
    repo = context.user_data.get(PENDING_WORKTREE_REPO) if context.user_data else None
    if not repo:
        await safe_edit(query, "❌ Worktree state lost. Tap Cancel and retry.")
        return
    if context.user_data is not None:
        context.user_data[AWAITING_WORKTREE_BRANCH_NAME] = True
    await safe_edit(
        query,
        "✏️ Send the branch name as a message, or tap Cancel.",
        reply_markup=_cancel_only_keyboard(),
    )


async def _validate_provider_select(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    pending_thread_id: int | None,
) -> bool:
    """Validate provider select callback; returns True if request should proceed."""
    confirm_thread_id = get_thread_id(update)
    if pending_thread_id is not None and confirm_thread_id != pending_thread_id:
        # _handle_mode_select clears browse state before calling this, so
        # _check_ui_guards can no longer catch a leftover worktree flow on
        # a later message — clear it here or the CREATING re-entrancy flag
        # sticks and blocks every future worktree confirm.
        clear_worktree_state(context.user_data)
        if context.user_data is not None:
            context.user_data.pop(PENDING_THREAD_ID, None)
            context.user_data.pop(PENDING_THREAD_TEXT, None)
        await query.answer("Stale browser (topic mismatch)", show_alert=True)
        return False

    await query.answer()

    # Guard against double-click: if thread already has a window, skip
    if pending_thread_id is not None:
        existing_wid = thread_router.get_window_for_thread(user_id, pending_thread_id)
        if existing_wid is not None:
            display = thread_router.get_display_name(existing_wid)
            logger.warning(
                "Thread %d already bound to window %s (%s), ignoring duplicate provider select",
                pending_thread_id,
                existing_wid,
                display,
            )
            await safe_edit(query, f"✅ Already bound to window {display}.")
            return False

    return True


async def _handle_provider_select(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_PROV_SELECT: select provider and show mode picker.

    Providers without a YOLO flag (e.g. shell) skip the mode picker
    and go directly to window creation with approval_mode="normal".
    """
    # Lazy: providers package heavy bootstrap
    from ccgram.providers import has_yolo_mode

    provider_name = data[len(CB_PROV_SELECT) :]
    if not provider_registry.is_valid(provider_name):
        await query.answer("Unknown provider", show_alert=True)
        return

    selected_path = _required_selected_path(context)
    if selected_path is None:
        await query.answer()
        await safe_edit(query, "❌ Selection expired. Tap Cancel and retry.")
        return
    pending_thread_id: int | None = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )

    if not await _validate_provider_select(
        query, user_id, update, context, pending_thread_id
    ):
        return

    if not has_yolo_mode(provider_name):
        # No mode picker needed — go directly to window creation
        clear_browse_state(context.user_data)
        await _create_window_and_bind(
            query, user_id, selected_path, provider_name, "normal", context
        )
        return

    text, keyboard = build_mode_picker(selected_path, provider_name)
    await safe_edit(query, text, reply_markup=keyboard)


def _parse_mode_select(data: str) -> tuple[str, str] | None:
    """Parse mode callback data as (provider_name, approval_mode)."""
    raw = data[len(CB_MODE_SELECT) :]
    provider_name, sep, approval_mode = raw.partition(":")
    if not sep:
        return None
    return provider_name, approval_mode.lower()


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


def _try_install_messaging_skill(provider_name: str, cwd: str) -> None:
    """Install the messaging skill for Claude windows (no-op for other providers)."""
    if provider_name != "claude":
        return
    # Lazy: msg_skill is only needed for Claude topics.
    from ...msg_skill import ensure_skill_installed

    try:
        ensure_skill_installed(cwd)
    except Exception:
        logger.exception("Failed to install messaging skill at %s", cwd)


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


async def _create_window_and_bind(  # noqa: PLR0915
    query: CallbackQuery,
    user_id: int,
    selected_path: str,
    provider_name: str,
    approval_mode: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Create a tmux window, bind to the pending topic, and forward pending text.

    Shared by _handle_mode_select (after mode picker) and _handle_provider_select
    (when mode picker is skipped for providers without YOLO flags).
    """
    # Lazy: providers package heavy bootstrap
    from ccgram.providers import resolve_launch_command

    pending_thread_id: int | None = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )

    launch_command = resolve_launch_command(provider_name, approval_mode=approval_mode)

    success, message, created_wname, created_wid = await tmux_manager.create_window(
        selected_path, launch_command=launch_command
    )
    if not success:
        await _abort_topic_creation(query, message, context)
        return

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

    _try_install_messaging_skill(provider_name, selected_path)

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
        return

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

    pending_text = (
        context.user_data.get(PENDING_THREAD_TEXT) if context.user_data else None
    )
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


async def _handle_mode_select(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_MODE_SELECT: select launch mode and create tmux window."""
    parsed = _parse_mode_select(data)
    if parsed is None:
        await query.answer("Invalid mode", show_alert=True)
        return

    provider_name, approval_mode = parsed
    if not provider_registry.is_valid(provider_name):
        await query.answer("Unknown provider", show_alert=True)
        return
    if approval_mode not in ("normal", "yolo"):
        await query.answer("Unknown mode", show_alert=True)
        return

    selected_path = _required_selected_path(context)
    if selected_path is None:
        await query.answer()
        await safe_edit(query, "❌ Selection expired. Tap Cancel and retry.")
        return
    pending_thread_id: int | None = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )

    clear_browse_state(context.user_data)

    if not await _validate_provider_select(
        query, user_id, update, context, pending_thread_id
    ):
        return

    await _create_window_and_bind(
        query, user_id, selected_path, provider_name, approval_mode, context
    )


async def _handle_cancel(
    query: CallbackQuery,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_DIR_CANCEL: cancel directory browsing."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and get_thread_id(update) != pending_tid:
        await query.answer("Stale browser (topic mismatch)", show_alert=True)
        return
    clear_browse_state(context.user_data)
    clear_worktree_state(context.user_data)
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
        context.user_data.pop(PENDING_THREAD_TEXT, None)
    await safe_edit(query, "Cancelled")
    await query.answer("Cancelled")


# --- Registry dispatch entry point ---


@register(
    CB_DIR_FAV,
    CB_DIR_STAR,
    CB_DIR_SELECT,
    CB_DIR_UP,
    CB_DIR_HOME,
    CB_DIR_PAGE,
    CB_DIR_CONFIRM,
    CB_PROV_SELECT,
    CB_MODE_SELECT,
    CB_WT_USE_CURRENT,
    CB_WT_NEW,
    CB_WT_CONFIRM,
    CB_WT_EDIT_NAME,
    CB_DIR_CANCEL,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await handle_directory_callback(query, user.id, query.data, update, context)
