"""Directory browser and window picker UI for session creation.

Provides UIs in Telegram for:
  - Window picker: list unbound tmux windows for quick binding
  - Directory browser: navigate directory hierarchies to create new sessions

Key components:
  - DIRS_PER_PAGE: Number of directories shown per page
  - User state keys for tracking browse/picker session
  - build_window_picker: Build unbound window picker UI
  - build_directory_browser: Build directory browser UI
  - clear_window_picker_state: Clear picker state from user_data
  - clear_browse_state: Clear browsing state from user_data
"""

from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..config import config
from ..session import parse_emdash_provider, session_manager
from ..window_resolver import is_foreign_window
from .callback_data import (
    CB_DIR_CANCEL,
    CB_DIR_CONFIRM,
    CB_DIR_FAV,
    CB_DIR_PAGE,
    CB_DIR_SELECT,
    CB_DIR_STAR,
    CB_DIR_UP,
    CB_PROV_SELECT,
    CB_MODE_SELECT,
    CB_WIN_BIND,
    CB_WIN_CANCEL,
    CB_WIN_NEW,
)

# Max favorites shown in directory browser
_MAX_FAVORITES = 3

# Max characters for a favorite path label before truncating
_MAX_FAV_LABEL_LEN = 26

# Directories per page in directory browser
DIRS_PER_PAGE = 6

# Max characters to show in a button label before truncating with "…"
_MAX_BUTTON_LABEL_LEN = 13

# User state keys
STATE_KEY = "state"
STATE_BROWSING_DIRECTORY = "browsing_directory"
STATE_SELECTING_WINDOW = "selecting_window"
BROWSE_PATH_KEY = "browse_path"
BROWSE_PAGE_KEY = "browse_page"
BROWSE_DIRS_KEY = "browse_dirs"  # Cache of subdirs for current path
UNBOUND_WINDOWS_KEY = "unbound_windows"  # Cache of (name, cwd) tuples


def clear_browse_state(user_data: dict | None) -> None:
    """Clear directory browsing state keys from user_data."""
    if user_data is not None:
        user_data.pop(STATE_KEY, None)
        user_data.pop(BROWSE_PATH_KEY, None)
        user_data.pop(BROWSE_PAGE_KEY, None)
        user_data.pop(BROWSE_DIRS_KEY, None)


def clear_window_picker_state(user_data: dict | None) -> None:
    """Clear window picker state keys from user_data."""
    if user_data is not None:
        user_data.pop(STATE_KEY, None)
        user_data.pop(UNBOUND_WINDOWS_KEY, None)


def _window_label(window_id: str, window_name: str) -> tuple[str, str]:
    """Return (icon, display_name) for a window in the picker.

    Emdash windows get a distinct icon and provider suffix.
    """
    if is_foreign_window(window_id):
        provider = parse_emdash_provider(window_id.rsplit(":", 1)[0])
        suffix = f" ({provider})" if provider else ""
        return "📎", f"{window_name}{suffix}"
    return "🖥", window_name


def build_window_picker(
    windows: list[tuple[str, str, str]],
) -> tuple[str, InlineKeyboardMarkup, list[str]]:
    """Build window picker UI for unbound tmux windows.

    Args:
        windows: List of (window_id, window_name, cwd) tuples.

    Returns: (text, keyboard, window_ids) where window_ids is the ordered list for caching.
    """
    window_ids = [wid for wid, _, _ in windows]

    lines = [
        "*Bind to Existing Window*\n",
        "These windows are running but not bound to any topic.",
        "Pick one to attach it here, or start a new session.\n",
    ]
    for wid, name, cwd in windows:
        display_cwd = cwd.replace(str(Path.home()), "~")
        icon, display_name = _window_label(wid, name)
        lines.append(f"• {icon} `{display_name}` — {display_cwd}")

    buttons: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(windows), 2):
        row = []
        for j in range(min(2, len(windows) - i)):
            wid = windows[i + j][0]
            name = windows[i + j][1]
            icon, display_name = _window_label(wid, name)
            display = (
                display_name[:12] + "…"
                if len(display_name) > _MAX_BUTTON_LABEL_LEN
                else display_name
            )
            row.append(
                InlineKeyboardButton(
                    f"{icon} {display}", callback_data=f"{CB_WIN_BIND}{i + j}"
                )
            )
        buttons.append(row)

    buttons.append(
        [
            InlineKeyboardButton("➕ New Session", callback_data=CB_WIN_NEW),
            InlineKeyboardButton("Cancel", callback_data=CB_WIN_CANCEL),
        ]
    )

    text = "\n".join(lines)
    return text, InlineKeyboardMarkup(buttons), window_ids


def get_favorites(user_id: int | None) -> tuple[list[str], set[str]]:
    """Get deduplicated favorites list and starred set.

    Returns (favorites, starred_set) where favorites is starred-first then MRU,
    filtered to existing dirs, capped at _MAX_FAVORITES.
    """
    if user_id is None:
        return [], set()
    starred = session_manager.get_user_starred(user_id)
    starred_set = set(starred)
    mru = session_manager.get_user_mru(user_id)
    seen: set[str] = set()
    result: list[str] = []
    for d in [*starred, *mru]:
        if d not in seen:
            try:
                exists = Path(d).is_dir()
            except OSError:
                exists = False
            if exists:
                seen.add(d)
                result.append(d)
        if len(result) >= _MAX_FAVORITES:
            break
    return result, starred_set


def _build_favorites_buttons(
    favorites: list[str],
    starred_set: set[str],
) -> list[list[InlineKeyboardButton]]:
    """Build favorite directory buttons (starred + MRU) with star toggles."""
    if not favorites:
        return []
    rows: list[list[InlineKeyboardButton]] = []
    for idx, fav_path in enumerate(favorites):
        display_fav = fav_path.replace(str(Path.home()), "~")
        trunc = _MAX_FAV_LABEL_LEN - 1
        label = (
            display_fav[:trunc] + "…"
            if len(display_fav) > _MAX_FAV_LABEL_LEN
            else display_fav
        )
        star_icon = "⭐" if fav_path in starred_set else "☆"
        rows.append(
            [
                InlineKeyboardButton(f"📌 {label}", callback_data=f"{CB_DIR_FAV}{idx}"),
                InlineKeyboardButton(star_icon, callback_data=f"{CB_DIR_STAR}{idx}"),
            ]
        )
    rows.append([InlineKeyboardButton("── folders ──", callback_data="noop")])
    return rows


def build_directory_browser(
    current_path: str, page: int = 0, user_id: int | None = None
) -> tuple[str, InlineKeyboardMarkup, list[str]]:
    """Build directory browser UI.

    Returns: (text, keyboard, subdirs) where subdirs is the full list for caching.
    """

    path = Path(current_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        path = Path.cwd()

    try:
        subdirs = sorted(
            [
                d.name
                for d in path.iterdir()
                if d.is_dir()
                and (config.show_hidden_dirs or not d.name.startswith("."))
            ]
        )
    except (PermissionError, OSError):  # fmt: skip
        subdirs = []

    favorites, starred_set = get_favorites(user_id)
    buttons: list[list[InlineKeyboardButton]] = _build_favorites_buttons(
        favorites, starred_set
    )

    # Subdirectory listing
    total_pages = max(1, (len(subdirs) + DIRS_PER_PAGE - 1) // DIRS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    start = page * DIRS_PER_PAGE
    page_dirs = subdirs[start : start + DIRS_PER_PAGE]

    for i in range(0, len(page_dirs), 2):
        row = []
        for j, name in enumerate(page_dirs[i : i + 2]):
            display = name[:12] + "…" if len(name) > _MAX_BUTTON_LABEL_LEN else name
            # Use global index (start + i + j) to avoid long dir names in callback_data
            idx = start + i + j
            row.append(
                InlineKeyboardButton(
                    f"📁 {display}", callback_data=f"{CB_DIR_SELECT}{idx}"
                )
            )
        buttons.append(row)

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton("◀", callback_data=f"{CB_DIR_PAGE}{page - 1}")
            )
        nav.append(
            InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop")
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton("▶", callback_data=f"{CB_DIR_PAGE}{page + 1}")
            )
        buttons.append(nav)

    action_row: list[InlineKeyboardButton] = []
    # Allow going up unless at filesystem root
    if path != path.parent:
        action_row.append(InlineKeyboardButton("..", callback_data=CB_DIR_UP))
    action_row.append(InlineKeyboardButton("Select", callback_data=CB_DIR_CONFIRM))
    action_row.append(InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL))
    buttons.append(action_row)

    display_path = str(path).replace(str(Path.home()), "~")
    if not subdirs and not favorites:
        text = f"*Select Working Directory*\n\nCurrent: `{display_path}`\n\n_(No subdirectories)_"
    else:
        text = f"*Select Working Directory*\n\nCurrent: `{display_path}`\n\nTap a folder to enter, or select current directory"

    return text, InlineKeyboardMarkup(buttons), subdirs


# Provider display metadata: (label, icon)
_PROVIDER_META: dict[str, tuple[str, str]] = {
    "claude": ("Claude", "🟠"),
    "codex": ("Codex", "🟢"),
    "gemini": ("Gemini", "🔵"),
}


def build_provider_picker(selected_path: str) -> tuple[str, InlineKeyboardMarkup]:
    """Build provider selection keyboard shown after directory confirmation.

    Returns: (text, keyboard).
    """
    display_path = selected_path.replace(str(Path.home()), "~")
    text = (
        f"*Select Provider*\n\nDirectory: `{display_path}`\n\nWhich agent CLI to use?"
    )
    buttons: list[list[InlineKeyboardButton]] = []
    for name, (label, icon) in _PROVIDER_META.items():
        suffix = " (default)" if name == "claude" else ""
        buttons.append(
            [
                InlineKeyboardButton(
                    f"{icon} {label}{suffix}",
                    callback_data=f"{CB_PROV_SELECT}{name}",
                )
            ]
        )
    buttons.append([InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)])
    return text, InlineKeyboardMarkup(buttons)


def build_mode_picker(
    selected_path: str, provider_name: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Build launch-mode keyboard shown after provider selection.

    Returns: (text, keyboard).
    """
    display_path = selected_path.replace(str(Path.home()), "~")
    provider_label, provider_icon = _PROVIDER_META.get(
        provider_name, (provider_name.title(), "🤖")
    )
    text = (
        "*Select Session Mode*\n\n"
        f"Directory: `{display_path}`\n"
        f"Provider: {provider_icon} {provider_label}\n\n"
        "Choose how many approvals you want for this session."
    )
    buttons = [
        [
            InlineKeyboardButton(
                "✅ Standard",
                callback_data=f"{CB_MODE_SELECT}{provider_name}:normal",
            )
        ],
        [
            InlineKeyboardButton(
                "🚀 YOLO",
                callback_data=f"{CB_MODE_SELECT}{provider_name}:yolo",
            )
        ],
        [InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)],
    ]
    return text, InlineKeyboardMarkup(buttons)
