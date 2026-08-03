"""Toolbar keyboard builder — constructs the /toolbar inline keyboard.

Owns the TOML-backed toolbar config singleton, per-window label overrides
(toggle-button state), and the InlineKeyboardMarkup builder. Separated from
toolbar_callbacks so keyboard construction has no dependency on dispatch,
scraping, or PTB update plumbing.
"""

from __future__ import annotations

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from ...config import config
from ...providers import get_provider_for_window
from ... import window_query
from ...topic_state_registry import topic_state
from ...toolbar_config import (
    ToolbarAction,
    ToolbarConfig,
    load_toolbar_config,
)
from ..callback_data import CB_TOOLBAR
from ..callback_tokens import compact_callback_data

# ──────────────────────────────────────────────────────────────────────
# Loaded config (lazy singleton)
# ──────────────────────────────────────────────────────────────────────

_toolbar_cfg: ToolbarConfig | None = None


def get_toolbar_config() -> ToolbarConfig:
    """Return the loaded ToolbarConfig, lazy-loading on first access."""
    global _toolbar_cfg  # noqa: PLW0603
    if _toolbar_cfg is None:
        _toolbar_cfg = load_toolbar_config(config.toolbar_config_path)
    return _toolbar_cfg


def reload_toolbar_config() -> None:
    """Force-reload of the toolbar config. Used by tests and future /reload."""
    global _toolbar_cfg  # noqa: PLW0603
    _toolbar_cfg = None


# ──────────────────────────────────────────────────────────────────────
# Per-window per-action label overrides
# ──────────────────────────────────────────────────────────────────────

_window_action_labels: dict[str, dict[str, str]] = {}


def _set_action_label(window_id: str, action_name: str, label: str) -> None:
    _window_action_labels.setdefault(window_id, {})[action_name] = label


def _get_action_label(window_id: str, action_name: str) -> str | None:
    return _window_action_labels.get(window_id, {}).get(action_name)


@topic_state.register("window")
def _clear_toolbar_labels(window_id: str) -> None:
    _window_action_labels.pop(window_id, None)


# ──────────────────────────────────────────────────────────────────────
# Keyboard builder
# ──────────────────────────────────────────────────────────────────────


def _make_button(
    action: ToolbarAction, window_id: str, style: str
) -> InlineKeyboardButton:
    """Render one ToolbarAction as a Telegram inline button."""
    override = _get_action_label(window_id, action.name)
    if override is not None:
        label = f"{action.emoji} {override}" if style == "emoji_text" else override
    else:
        label = action.render(style)  # type: ignore[arg-type]
    cb = compact_callback_data(
        CB_TOOLBAR, f"{CB_TOOLBAR}{window_id}:{action.name}", window_id
    )
    return InlineKeyboardButton(label, callback_data=cb)


def build_toolbar_keyboard(
    window_id: str, provider_name: str = "claude"
) -> InlineKeyboardMarkup:
    """Build the inline keyboard for /toolbar from per-provider config."""
    cfg = get_toolbar_config()
    layout = cfg.for_provider(provider_name)
    rows: list[list[InlineKeyboardButton]] = []
    for row_names in layout.buttons:
        cells: list[InlineKeyboardButton] = []
        for name in row_names:
            action = cfg.actions.get(name)
            if action is not None:
                cells.append(_make_button(action, window_id, layout.style))
        if cells:
            rows.append(cells)
    return InlineKeyboardMarkup(rows)


# ──────────────────────────────────────────────────────────────────────
# State seeding
# ──────────────────────────────────────────────────────────────────────


async def seed_button_states(window_id: str) -> None:
    """Populate toggle-button label overrides with the actual current state.

    Called from /toolbar handler BEFORE rendering the keyboard, so the
    initial button text reflects the real mode (Edit/Plan/Full/Def) rather
    than the static "Mode" placeholder.
    """
    cfg = get_toolbar_config()
    mode_action = cfg.actions.get("mode")
    if mode_action is None or not mode_action.read_state:
        return
    provider = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )
    label = await provider.scrape_current_mode(window_id)
    if label:
        _set_action_label(window_id, "mode", label)


async def refresh_button_label(
    action: ToolbarAction,
    query: CallbackQuery,
    window_id: str,
    *,
    delay: float = 0.35,
) -> str:
    """Scrape the pane, update the button label, rebuild the keyboard.

    Returns the short label so the caller can echo it in the toast.
    """
    # Lazy: only needed when a label-readback task is scheduled
    import asyncio

    # Lazy: only needed inside the readback error path
    import structlog

    # Lazy: only used inside the BadRequest branch
    from telegram.error import TelegramError

    logger = structlog.get_logger()

    await asyncio.sleep(delay)
    view = window_query.view_window(window_id)
    provider_name = view.provider_name if view else "claude"
    provider = get_provider_for_window(window_id, provider_name=provider_name)
    short_label = await provider.scrape_current_mode(window_id)
    if not short_label:
        short_label = "Def"
    _set_action_label(window_id, action.name, short_label)
    new_kb = build_toolbar_keyboard(window_id, provider_name)
    try:
        await query.edit_message_reply_markup(reply_markup=new_kb)
    except TelegramError as exc:
        logger.debug("Toolbar reply_markup edit failed: %s", exc)
    return short_label
