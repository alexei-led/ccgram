"""File search, listing and upload utilities for the /send command.

Provides utilities for the /send Telegram command:
  - _is_image: detect image files by extension
  - _find_files: glob/exact/substring file search with security filtering
  - _list_directory: directory listing with security filtering and sorting
  - _format_file_label: human-readable inline keyboard button labels
  - build_file_browser: build paginated inline keyboard for directory browsing
  - build_search_results: build inline keyboard for search result selection
  - upload_file: send a file to Telegram (photo or document)
  - send_command: handle the /send command
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import fnmatch
import os

import structlog
from pathlib import Path

from telegram import Message, Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from ...telegram_client import PTBTelegramClient, TelegramClient

from ...config import config
from ...window_query import view_window
from ...thread_router import thread_router
from ..callback_data import (
    CB_SEND_CANCEL,
    CB_SEND_DIR,
    CB_SEND_FILE,
    CB_SEND_PAGE,
    CB_SEND_UP,
)
from ..callback_helpers import get_thread_id
from ..messaging_pipeline.message_sender import safe_reply, safe_send
from .send_security import is_excluded_dir, validate_sendable
from ..user_state import (
    SEND_CWD_KEY,
    SEND_ITEMS_KEY,
    SEND_PAGE_KEY,
    SEND_PATH_KEY,
    SEND_WINDOW_ID_KEY,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp"}
)
_ITEMS_PER_PAGE = 8
_BUTTONS_PER_ROW = 2
_KB = 1024
_MB = 1024 * 1024


def _safe_mtime(p: Path) -> float:
    """Return mtime or 0.0 if the file disappeared (TOCTOU guard)."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _is_image(path: Path) -> bool:
    """Return True if *path* has an image file extension."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _walk_filtered(cwd: Path, depth_limit: int) -> list[Path]:
    """Walk *cwd* with in-place pruning of excluded dirs, yielding file paths.

    Stops descending once a directory's depth relative to *cwd* reaches
    *depth_limit*. Files at depths up to *depth_limit* are included.
    """
    cwd_resolved = cwd.resolve()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(cwd):
        dirnames[:] = [d for d in dirnames if not is_excluded_dir(d)]
        dir_path = Path(dirpath)
        try:
            rel_depth = len(dir_path.resolve().relative_to(cwd_resolved).parts)
        except ValueError:
            dirnames[:] = []
            continue
        if rel_depth >= depth_limit:
            dirnames[:] = []
        for filename in filenames:
            files.append(dir_path / filename)
    return files


def _find_files(cwd: Path, pattern: str) -> list[Path]:
    """Search for files matching *pattern* under *cwd*.

    Dispatch rules:
    - Pattern contains ``*`` or ``?``: fnmatch filter against filenames.
    - Otherwise: try exact relative path first; if not found, case-insensitive
      substring match on filenames.

    Uses ``os.walk`` with in-place directory pruning so excluded trees
    (``node_modules``, ``.venv``, etc.) are never descended into. Results
    are filtered via ``validate_sendable``, capped at ``config.send_max_results``,
    and sorted by mtime descending.
    """
    is_glob = "*" in pattern or "?" in pattern

    if not is_glob:
        exact = cwd / pattern
        if exact.exists() and validate_sendable(exact, cwd) is None:
            rel = exact.resolve().relative_to(cwd.resolve())
            if not any(is_excluded_dir(part) for part in rel.parts[:-1]):
                return [exact]

    needle = pattern.lower()

    def _name_matches(name: str) -> bool:
        if is_glob:
            return fnmatch.fnmatch(name, pattern)
        return needle in name.lower()

    results = [
        candidate
        for candidate in _walk_filtered(cwd, config.send_search_depth)
        if _name_matches(candidate.name) and validate_sendable(candidate, cwd) is None
    ]
    results.sort(key=_safe_mtime, reverse=True)
    return results[: config.send_max_results]


def _list_directory(path: Path, cwd: Path) -> tuple[list[Path], list[Path]]:
    """List *path* contents, separated into (dirs, files).

    Filtering:
    - Directories: exclude names matching ``is_excluded_dir``.
    - Files: exclude those where ``validate_sendable`` returns non-None.

    Both lists are sorted alphabetically by name (case-insensitive).
    """
    dirs: list[Path] = []
    files: list[Path] = []

    for entry in path.iterdir():
        if entry.is_dir():
            if not is_excluded_dir(entry.name):
                dirs.append(entry)
        elif entry.is_file() and validate_sendable(entry, cwd) is None:
            files.append(entry)

    dirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())
    return dirs, files


def _format_file_label(path: Path, cwd: Path) -> str:
    """Return a button label string ``"{rel_path} ({size})"`` for *path*.

    Size is formatted as B, KB, or MB. The total label is capped at 30 characters:
    when it exceeds that, the path portion is truncated with ``…`` while the size
    suffix is always preserved.
    """
    try:
        rel = str(path.relative_to(cwd))
    except ValueError:
        rel = path.name

    try:
        size_bytes = path.stat().st_size
    except OSError:
        return rel
    if size_bytes < _KB:
        size_str = f"{size_bytes} B"
    elif size_bytes < _MB:
        size_str = f"{size_bytes / _KB:.1f} KB"
    else:
        size_str = f"{size_bytes / _MB:.1f} MB"

    suffix = f" ({size_str})"
    label = rel + suffix

    max_len = 30
    if len(label) > max_len:
        # Truncate path portion, keep size suffix. Guard against a suffix
        # longer than max_len (negative slice would truncate from the end).
        max_path_len = max(0, max_len - len(suffix) - 1)
        rel = rel[:max_path_len] + "…"
        label = rel + suffix

    return label


def _make_item_button(item: Path, idx: int, cwd: Path) -> InlineKeyboardButton:
    """Return a single InlineKeyboardButton for *item* at position *idx*."""
    if item.is_dir():
        return InlineKeyboardButton(
            f"📁 {item.name}", callback_data=f"{CB_SEND_DIR}{idx}"
        )
    label = _format_file_label(item, cwd)
    icon = "🖼️" if _is_image(item) else "📄"
    return InlineKeyboardButton(f"{icon} {label}", callback_data=f"{CB_SEND_FILE}{idx}")


def _pack_into_rows(
    buttons_flat: list[InlineKeyboardButton],
) -> list[list[InlineKeyboardButton]]:
    """Pack a flat list of buttons into rows of _BUTTONS_PER_ROW."""
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for btn in buttons_flat:
        row.append(btn)
        if len(row) == _BUTTONS_PER_ROW:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def build_file_browser(
    current_path: Path,
    cwd: Path,
    page: int,
) -> tuple[str, InlineKeyboardMarkup, list[Path]]:
    """Build a paginated inline keyboard for browsing files under *cwd*.

    Returns (display_text, markup, items) where *items* is the full list of
    Path objects (dirs first, then files) used to resolve button indices.
    """
    dirs, files = _list_directory(current_path, cwd)
    items: list[Path] = dirs + files

    total_pages = max(1, (len(items) + _ITEMS_PER_PAGE - 1) // _ITEMS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    start = page * _ITEMS_PER_PAGE
    page_items = items[start : start + _ITEMS_PER_PAGE]

    flat = [
        _make_item_button(item, start + i, cwd) for i, item in enumerate(page_items)
    ]
    buttons = _pack_into_rows(flat)

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton("◀", callback_data=f"{CB_SEND_PAGE}{page - 1}")
            )
        nav.append(
            InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop")
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton("▶", callback_data=f"{CB_SEND_PAGE}{page + 1}")
            )
        buttons.append(nav)

    parent_row: list[InlineKeyboardButton] = []
    if current_path != cwd:
        parent_row.append(InlineKeyboardButton("📁 ..", callback_data=CB_SEND_UP))
    parent_row.append(InlineKeyboardButton("✖ Cancel", callback_data=CB_SEND_CANCEL))
    buttons.append(parent_row)

    try:
        display_path = (
            str(current_path.relative_to(cwd)) if current_path != cwd else "."
        )
    except ValueError:
        display_path = current_path.name

    return f"📂 {display_path}", InlineKeyboardMarkup(buttons), items


def build_search_results(
    matches: list[Path],
    cwd: Path,
    query: str = "",
) -> tuple[str, InlineKeyboardMarkup, list[Path]]:
    """Build an inline keyboard for selecting a file from search results.

    Shows up to ``_ITEMS_PER_PAGE * 3`` matches with no pagination or parent nav.
    Returns (display_text, markup, matches).
    """
    shown = matches[: _ITEMS_PER_PAGE * 3]

    flat = [_make_item_button(path, idx, cwd) for idx, path in enumerate(shown)]
    buttons = _pack_into_rows(flat)
    buttons.append([InlineKeyboardButton("✖ Cancel", callback_data=CB_SEND_CANCEL)])

    count = len(matches)
    cap = _ITEMS_PER_PAGE * 3
    header = f"🔍 {cap}+ results" if count > cap else f"🔍 {count} result(s)"
    if query:
        header += f" for '{query}'"
    return header, InlineKeyboardMarkup(buttons), shown


async def upload_file(
    client: TelegramClient, chat_id: int, thread_id: int, path: Path
) -> Message | None:
    """Send *path* to the given Telegram chat/thread as photo or document.

    Returns the resulting Telegram ``Message`` so callers can target it for
    follow-up reactions or replies. Re-raises ``TelegramError`` on failure.
    """
    try:
        with path.open("rb") as fh:
            if _is_image(path):
                return await client.send_photo(
                    chat_id=chat_id,
                    photo=fh,
                    filename=path.name,
                    message_thread_id=thread_id,
                    read_timeout=300,
                )
            return await client.send_document(
                chat_id=chat_id,
                document=fh,
                filename=path.name,
                message_thread_id=thread_id,
                read_timeout=300,
            )
    except TelegramError:
        logger.exception("Failed to upload file", path=str(path))
        raise


def _cache_browser_state(
    user_data: dict,
    cwd: Path,
    items: list[Path],
    window_id: str,
) -> None:
    """Persist browser/search state into PTB user_data for callback handlers."""
    user_data[SEND_PATH_KEY] = str(cwd)
    user_data[SEND_CWD_KEY] = str(cwd)
    user_data[SEND_PAGE_KEY] = 0
    user_data[SEND_ITEMS_KEY] = items
    user_data[SEND_WINDOW_ID_KEY] = window_id


async def open_file_browser(
    client: TelegramClient,
    chat_id: int,
    thread_id: int | None,
    user_data: dict,
    window_id: str,
    cwd: Path,
) -> None:
    """Open the /send file browser — public entry point for the toolbar Send button."""
    text, markup, items = build_file_browser(cwd, cwd, 0)
    _cache_browser_state(user_data, cwd, items, window_id)
    await safe_send(
        client,
        chat_id,
        text,
        message_thread_id=thread_id,
        reply_markup=markup,
    )


async def _upload_with_feedback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    thread_id: int,
    path: Path,
) -> None:
    """Upload *path*, replying with a human-readable error on TelegramError."""
    try:
        await upload_file(PTBTelegramClient(context.bot), chat_id, thread_id, path)
    except TelegramError as exc:
        await safe_reply(update.message, f"Upload failed: {exc}")  # type: ignore[arg-type]


async def _dispatch_search(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    cwd: Path,
    pattern: str,
    chat_id: int,
    thread_id: int,
    window_id: str,
) -> None:
    """Handle glob/substring search dispatch for send_command."""
    assert context.user_data is not None

    is_glob = "*" in pattern or "?" in pattern

    if not is_glob:
        exact = cwd / pattern
        if exact.exists() and exact.is_file():
            error = validate_sendable(exact, cwd)
            if error:
                await safe_reply(update.message, f"Cannot send: {error}")  # type: ignore[arg-type]
                return
            try:
                rel = exact.resolve().relative_to(cwd.resolve())
            except ValueError:
                await safe_reply(
                    update.message,  # type: ignore[arg-type]
                    "Cannot send: file is outside project directory",
                )
                return
            if any(is_excluded_dir(part) for part in rel.parts[:-1]):
                await safe_reply(
                    update.message,  # type: ignore[arg-type]
                    "Cannot send: file is in an excluded directory",
                )
                return
            await _upload_with_feedback(update, context, chat_id, thread_id, exact)
            return

    matches = _find_files(cwd, pattern)
    if not matches:
        await safe_reply(update.message, f"No files found matching: {pattern}")  # type: ignore[arg-type]
        return
    if len(matches) == 1:
        error = validate_sendable(matches[0], cwd)
        if error:
            await safe_reply(update.message, f"Cannot send: {error}")  # type: ignore[arg-type]
            return
        await _upload_with_feedback(update, context, chat_id, thread_id, matches[0])
        return

    display_text, markup, shown = build_search_results(matches, cwd, query=pattern)
    _cache_browser_state(context.user_data, cwd, shown, window_id)
    await safe_reply(update.message, display_text, reply_markup=markup)  # type: ignore[arg-type]


async def send_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /send — search for and upload a file from the session's CWD to Telegram.

    Dispatch modes:
    - No args: show paginated file browser at CWD root.
    - Glob pattern (contains * or ?): search with _find_files; upload if single match,
      show search results keyboard if multiple, error if none.
    - Text: try exact path first; if exists and passes security, upload directly.
      Otherwise fall back to _find_files substring search.
    """
    if not update.message:
        return
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        await safe_reply(update.message, "Not authorized.")
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(update.message, "Use this command inside a topic.")
        return

    window_id = thread_router.resolve_window_for_thread(
        user.id, thread_id, update.message.chat.id
    )
    if not window_id:
        await safe_reply(update.message, "No session bound to this topic.")
        return

    view = view_window(window_id)
    cwd = Path(view.cwd) if view and view.cwd else None
    if not cwd or not cwd.is_dir():
        await safe_reply(update.message, "Working directory not available.")
        return

    assert context.user_data is not None

    text = update.message.text or ""
    parts = text.split(maxsplit=1)
    pattern = parts[1].strip() if len(parts) > 1 else ""

    chat_id: int = (
        thread_router.resolve_chat_id(user.id, thread_id) or update.message.chat_id
    )

    if not pattern:
        display_text, markup, items = build_file_browser(cwd, cwd, 0)
        _cache_browser_state(context.user_data, cwd, items, window_id)
        await safe_reply(update.message, display_text, reply_markup=markup)
        return

    await _dispatch_search(update, context, cwd, pattern, chat_id, thread_id, window_id)
