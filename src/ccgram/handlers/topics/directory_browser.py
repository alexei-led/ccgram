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
  - build_worktree_picker: Offer current-branch vs new-worktree for a git repo
  - build_worktree_confirm: Confirm a suggested worktree branch (or edit it)
"""

from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ...config import config
from ...user_preferences import user_preferences
from ..callback_data import (
    CB_DIR_CANCEL,
    CB_DIR_CONFIRM,
    CB_DIR_FAV,
    CB_DIR_HOME,
    CB_DIR_PAGE,
    CB_DIR_SELECT,
    CB_DIR_STAR,
    CB_DIR_UP,
    CB_PROV_SELECT,
    CB_MODE_SELECT,
    CB_WIN_BIND,
    CB_WIN_CANCEL,
    CB_WIN_NEW,
    CB_WT_CONFIRM,
    CB_WT_EDIT_NAME,
    CB_WT_NEW,
    CB_WT_USE_CURRENT,
)
from ..user_state import (
    AWAITING_WORKTREE_BRANCH_NAME,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_CREATING,
    PENDING_WORKTREE_DIRTY,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
    PENDING_WORKTREE_SUBDIR,
)

# Max favorites shown in directory browser
_MAX_FAVORITES = 5

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

# Project markers: filename → badge icon (checked via os.scandir)
_PROJECT_MARKERS: dict[str, str] = {
    ".git": "\u2699",
    "pyproject.toml": "\U0001f40d",
    "Cargo.toml": "\U0001f980",
    "go.mod": "\U0001f439",
    "package.json": "\U0001f4e6",
    "Makefile": "\U0001f527",
}


def _detect_project_badge(parent: Path, name: str) -> str:
    """Return a project badge icon for a subdirectory, or empty string."""
    subdir = parent / name
    for marker, icon in _PROJECT_MARKERS.items():
        try:
            if (subdir / marker).exists():
                return icon
        except OSError:
            continue
    return ""


def clear_browse_state(user_data: dict | None) -> None:
    """Clear directory browsing state keys from user_data."""
    if user_data is not None:
        user_data.pop(STATE_KEY, None)
        user_data.pop(BROWSE_PATH_KEY, None)
        user_data.pop(BROWSE_PAGE_KEY, None)
        user_data.pop(BROWSE_DIRS_KEY, None)


def clear_worktree_state(user_data: dict | None) -> None:
    """Clear worktree-picker flow state keys from user_data."""
    if user_data is not None:
        for key in (
            PENDING_WORKTREE_REPO,
            PENDING_WORKTREE_BRANCH,
            PENDING_WORKTREE_PATH,
            PENDING_WORKTREE_DIRTY,
            PENDING_WORKTREE_SUBDIR,
            PENDING_WORKTREE_CREATING,
            AWAITING_WORKTREE_BRANCH_NAME,
        ):
            user_data.pop(key, None)


def clear_window_picker_state(user_data: dict | None) -> None:
    """Clear window picker state keys from user_data."""
    if user_data is not None:
        user_data.pop(STATE_KEY, None)
        user_data.pop(UNBOUND_WINDOWS_KEY, None)


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
    for _wid, name, cwd in windows:
        display_cwd = cwd.replace(str(Path.home()), "~")
        lines.append(f"• 🖥 `{name}` — {display_cwd}")

    buttons: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(windows), 2):
        row = []
        for j in range(min(2, len(windows) - i)):
            name = windows[i + j][1]
            display = name[:12] + "…" if len(name) > _MAX_BUTTON_LABEL_LEN else name
            row.append(
                InlineKeyboardButton(
                    f"🖥 {display}", callback_data=f"{CB_WIN_BIND}{i + j}"
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
    starred = user_preferences.get_user_starred(user_id)
    starred_set = set(starred)
    mru = user_preferences.get_user_mru(user_id)
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
            display = (
                name[:12] + "\u2026" if len(name) > _MAX_BUTTON_LABEL_LEN else name
            )
            badge = _detect_project_badge(path, name)
            icon = badge if badge else "\U0001f4c1"
            idx = start + i + j
            row.append(
                InlineKeyboardButton(
                    f"{icon} {display}", callback_data=f"{CB_DIR_SELECT}{idx}"
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
    action_row.append(InlineKeyboardButton("\U0001f3e0", callback_data=CB_DIR_HOME))
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
    "claude": ("Claude", "\U0001f7e0"),
    "codex": ("Codex", "\U0001f9e9"),
    "gemini": ("Gemini", "\u264a"),
    "pi": ("Pi", "\U0001f916"),
    "shell": ("Shell", "\U0001f41a"),
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
                "🎲 YOLO",
                callback_data=f"{CB_MODE_SELECT}{provider_name}:yolo",
            )
        ],
        [InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)],
    ]
    return text, InlineKeyboardMarkup(buttons)


def build_worktree_picker(
    repo_path: str, current_branch: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Build the worktree opt-in keyboard for an eligible git repo.

    Shown between directory-confirm and provider-pick. The user either
    keeps the current branch (today's behaviour) or spins up a new
    worktree on a fresh branch.

    Returns: (text, keyboard).
    """
    display_path = repo_path.replace(str(Path.home()), "~")
    text = (
        "*Git Worktree*\n\n"
        f"Repo: `{display_path}`\n"
        f"Current branch: `{current_branch}`\n\n"
        "Work on the current branch, or create an isolated worktree "
        "on a new branch?"
    )
    buttons = [
        [
            InlineKeyboardButton(
                f"🌿 Use current ({current_branch})",
                callback_data=CB_WT_USE_CURRENT,
            )
        ],
        [InlineKeyboardButton("➕ New worktree", callback_data=CB_WT_NEW)],
        [InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)],
    ]
    return text, InlineKeyboardMarkup(buttons)


def build_worktree_confirm(
    repo_path: str, branch: str, worktree_path: str, dirty: bool
) -> tuple[str, InlineKeyboardMarkup]:
    """Build the worktree confirm/edit keyboard with a suggested branch.

    The suggested branch name is the happy path (one-tap confirm); the
    Edit name button opens the only free-text step in this flow. A
    dirty source repo is allowed but warned about.

    Returns: (text, keyboard).
    """
    display_repo = repo_path.replace(str(Path.home()), "~")
    display_wt = worktree_path.replace(str(Path.home()), "~")
    lines = [
        "*New Worktree*\n",
        f"Repo: `{display_repo}`",
        f"Branch: `{branch}`",
        f"Worktree: `{display_wt}`",
    ]
    if dirty:
        lines.append(
            "\n⚠️ The source repo has uncommitted changes. The worktree "
            "starts from HEAD; uncommitted work stays where it is."
        )
    text = "\n".join(lines)
    buttons = [
        [InlineKeyboardButton("✅ Use this", callback_data=CB_WT_CONFIRM)],
        [InlineKeyboardButton("✏️ Edit name", callback_data=CB_WT_EDIT_NAME)],
        [InlineKeyboardButton("Cancel", callback_data=CB_DIR_CANCEL)],
    ]
    return text, InlineKeyboardMarkup(buttons)
