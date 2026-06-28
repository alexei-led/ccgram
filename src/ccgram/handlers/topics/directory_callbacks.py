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

Navigation, worktree-picker, and cancel logic live here.
Provider/mode selection: provider_mode_callbacks.py
Workspace picker: workspace_callbacks.py
Window launch: window_launch_service.py
Flow state: topic_creation_draft.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)

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
    CB_WS_SELECT,
    CB_WS_SKIP,
)
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..messaging_pipeline.message_sender import safe_edit
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
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    build_directory_browser,
    build_worktree_confirm,
    build_worktree_picker,
    clear_browse_state,
    clear_workspace_state,
    clear_worktree_state,
    get_favorites,
)
from .provider_mode_callbacks import (
    _handle_mode_select,
    _handle_provider_select,
    _parse_mode_select,
    _validate_provider_select,
)
from .topic_creation_draft import (
    _browser_flow_stale,
    _required_selected_path,
)

# Intentional test-compat re-exports: tests import these names from this module.
from .window_launch_service import (
    _accept_yolo_confirmation,
    _cwd_within,
    _persist_worktree_state,
    launch_window as _create_window_and_bind,
)
from .worktree import (
    WorktreeError,
    check_worktree_eligibility,
    create_worktree,
    slug_for_path,
    suggest_branch_name,
    worktree_path_for,
)
from .workspace_callbacks import (
    _handle_workspace_callback,
    _show_provider_picker,
    _show_workspace_picker_or_provider,
)
from ...multiplexer import multiplexer as tmux_manager
from ...thread_router import thread_router
from ...user_preferences import user_preferences

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

# Re-export for backward compatibility (tests import these from directory_callbacks)
__all__ = [
    "handle_directory_callback",
    "_browser_flow_stale",
    "_required_selected_path",
    "_handle_confirm",
    "_handle_fav",
    "_handle_star",
    "_handle_up",
    "_handle_home",
    "_handle_page",
    "_handle_select",
    "_handle_worktree_callback",
    "_handle_wt_confirm",
    "_handle_wt_edit_name",
    "_handle_wt_new",
    "_handle_wt_use_current",
    "_handle_workspace_callback",
    "_show_workspace_picker_or_provider",
    "_create_window_and_bind",
    "_validate_provider_select",
    "_handle_provider_select",
    "_parse_mode_select",
    "_handle_mode_select",
    # test-compat re-exports from window_launch_service / thread_router
    "_accept_yolo_confirmation",
    "_cwd_within",
    "_persist_worktree_state",
    "thread_router",
]


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
    elif data.startswith(CB_WS_SELECT) or data == CB_WS_SKIP:
        await _handle_workspace_callback(query, data, update, context)
    elif data == CB_DIR_CANCEL:
        await _handle_cancel(query, update, context)


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
    # was reset (e.g. /start) and this is a stale tap — proceeding would
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
    await _show_workspace_picker_or_provider(query, selected_path, context)


def _cancel_only_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)]]
    )


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
    # None means the flow was reset (e.g. /start, or Cancel raced the
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
    await _show_workspace_picker_or_provider(query, selected_path, context)


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
    # On herdr the worktree is created later (delegated to `worktree create`
    # in _create_window_and_bind), so no slow git op runs here and no
    # re-entrancy guard is needed. On tmux the guard is set synchronously
    # *before* the first await: a fast double-tap on "Use this" would otherwise
    # run `git worktree add` twice and the second call would overwrite the
    # provider picker with a "branch already exists" error even though the
    # first succeeded.

    delegate = tmux_manager.capabilities.native_worktrees
    if not delegate:
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
    if delegate:
        # herdr makes the checkout + grouped workspace itself at creation time.
        # Skip ccgram's `git worktree add` and the workspace picker (herdr
        # assigns the worktree its own workspace); keep the PENDING_WORKTREE_*
        # keys so _create_window_and_bind issues `worktree create` at this path.
        if user_data is not None:
            user_data[BROWSE_PATH_KEY] = worktree_path
        logger.info("Deferring worktree %s (branch %s) to herdr", worktree_path, branch)
        await _show_provider_picker(query, worktree_path)
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
    await _show_workspace_picker_or_provider(query, target_str, context)


async def _handle_wt_edit_name(
    query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Prompt for a custom branch name via a text reply."""
    await query.answer()
    # Fail closed like the other worktree handlers: a stale wt:ed tapped
    # after the flow was reset (e.g. by /start clearing PENDING_WORKTREE_REPO)
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
    clear_workspace_state(context.user_data)
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
    CB_WS_SELECT,
    CB_WS_SKIP,
    CB_DIR_CANCEL,
)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await handle_directory_callback(query, user.id, query.data, update, context)
