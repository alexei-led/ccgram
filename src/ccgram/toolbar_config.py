"""Toolbar layout configuration — per-provider button grids loaded from TOML.

The ``/toolbar`` inline keyboard is configurable per provider. Each provider
has a grid of buttons (rows × cols, no fixed shape) and a rendering style
(emoji-only, text-only, or emoji+text). Users override the built-in defaults
by placing a TOML file at ``~/.ccgram/toolbar.toml`` (auto-detected) or at
``$CCGRAM_TOOLBAR_CONFIG`` (explicit path).

This module is **pure data + loader** — it imports nothing from Telegram,
PTB, or tmux, so it is trivially testable in isolation. The handler layer
(``handlers/toolbar_callbacks.py``) consumes ``ToolbarConfig`` and dispatches
button clicks by ``ToolbarAction.action_type``.

Action types:
  - ``key``: send a tmux key sequence (Ctrl-C, Esc, Shift-Tab, etc.)
  - ``text``: send literal text followed by Enter (slash commands, prompts)
  - ``builtin``: trigger a special handler (screenshot, send-file-browser,
    live-view, dismiss). Built-in actions are reserved — users cannot
    define new builtins (the dispatch table is closed).

TOML schema::

    # Optional: define your own actions or override built-ins by name.
    [actions.clear]
    emoji = "🧹"
    text  = "Clear"
    type  = "text"
    payload = "/clear"

    [actions.deepthink]
    emoji = "🧠"
    text  = "Deep"
    type  = "key"
    payload = "Tab"
    read_state = true

    # Optional: per-provider grid + style.
    [providers.claude]
    style = "emoji_text"           # "emoji" | "text" | "emoji_text"
    buttons = [
      ["screen", "ctrlc",   "live" ],
      ["mode",   "think",   "esc"  ],
      ["last",   "getfile", "close"],
    ]

Unknown providers fall back to the ``claude`` layout. Any malformed entry
is logged as a warning and skipped — the loader never raises.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger()

ButtonStyle = Literal["emoji", "text", "emoji_text"]
ActionType = Literal["key", "text", "builtin"]

_VALID_STYLES: frozenset[str] = frozenset({"emoji", "text", "emoji_text"})
_VALID_TYPES: frozenset[str] = frozenset({"key", "text", "builtin"})

# Action names appear in callback_data as "tb:<window_id>:<name>". With a
# 64-byte Telegram limit and worst-case ~30-char foreign window IDs (e.g.
# "emdash-claude-main-abc123:@0"), names must stay ≤24 chars to fit.
_MAX_NAME_LEN = 24

# Telegram allows at most 8 inline buttons per row.
_MAX_ROW_WIDTH = 8


@dataclass(frozen=True, slots=True)
class ToolbarAction:
    """A single button action that can appear in any provider's grid.

    Three flavors via ``action_type``:
      - ``key``: ``payload`` is a tmux key string (e.g. ``"C-c"``,
        ``"Tab"``, ``"\\x1b[Z"`` for Shift-Tab). Sent via
        ``tmux_manager.send_keys(payload, enter=False, literal=action.literal)``.
      - ``text``: ``payload`` is literal text (e.g. ``"/clear"``). Sent
        followed by Enter via ``send_keys(payload, enter=True, literal=True)``.
      - ``builtin``: ``payload`` is the builtin handler name. Dispatched
        specially in ``handlers/toolbar_callbacks.py``.
    """

    name: str
    """Stable identifier — used in TOML config and in callback_data."""
    emoji: str
    """Single-glyph emoji (or short symbol like '^D')."""
    text: str
    """Short text label (≤6 chars to fit Telegram's per-cell budget)."""
    action_type: ActionType
    payload: str
    """Type-dependent payload. See class docstring."""
    literal: bool = False
    """For ``key`` actions: whether the key string is literal vs. a named key."""
    read_state: bool = False
    """For ``key``/``text`` actions: capture pane after sending and surface
    the most recent mode-line in the toast (used by Mode/Think/YOLO)."""

    def render(self, style: ButtonStyle) -> str:
        """Return the button label in the requested style."""
        if style == "emoji":
            return self.emoji
        if style == "text":
            return self.text
        return f"{self.emoji} {self.text}"


@dataclass(frozen=True, slots=True)
class ToolbarLayout:
    """Per-provider toolbar configuration."""

    style: ButtonStyle
    buttons: tuple[tuple[str, ...], ...]
    """Grid of action names: outer = rows, inner = cells per row."""


# ──────────────────────────────────────────────────────────────────────
# Built-in actions — always available. User TOML may override by name.
# ──────────────────────────────────────────────────────────────────────


def _b(
    name: str,
    emoji: str,
    text: str,
    action_type: ActionType,
    payload: str,
    *,
    literal: bool = False,
    read_state: bool = False,
) -> ToolbarAction:
    return ToolbarAction(
        name=name,
        emoji=emoji,
        text=text,
        action_type=action_type,
        payload=payload,
        literal=literal,
        read_state=read_state,
    )


BUILTIN_ACTIONS: dict[str, ToolbarAction] = {
    a.name: a
    for a in (
        # Builtin handlers (specialized dispatch in toolbar_callbacks.py)
        _b("screen", "\U0001f4f7", "Screen", "builtin", "screenshot"),
        _b("ctrlc", "\u23f9", "Ctrl-C", "builtin", "ctrlc"),
        _b("live", "\U0001f4fa", "Live", "builtin", "live"),
        _b("getfile", "\U0001f4e5", "Get File", "builtin", "getfile"),
        _b("last", "\U0001f4c4", "Last", "builtin", "lastreply"),
        _b("close", "\u2716", "Close", "builtin", "dismiss"),
        # Mode toggle — cycles Claude's permission modes via Shift-Tab.
        # read_state=True so the button label updates to Def/Edit/Plan/Full.
        _b(
            "mode",
            "\U0001f500",
            "Mode",
            "key",
            "\x1b[Z",
            literal=True,
            read_state=True,
        ),
        # Claude Code uses Meta+T (Alt+T) to toggle extended thinking.
        # Thinking has no persistent chrome indicator, so no read_state.
        _b("think", "\U0001f4ad", "Think", "key", "M-t"),
        _b("yolo", "\U0001f1fe", "YOLO", "key", "C-y", read_state=True),
        # Plain key sends.
        _b("esc", "\u238b", "Esc", "key", "Escape"),
        _b("enter", "\u23ce", "Enter", "key", "Enter"),
        _b("tab", "\u21e5", "Tab", "key", "Tab"),
        _b("eof", "^D", "EOF", "key", "C-d"),
        _b("susp", "^Z", "Susp", "key", "C-z"),
        _b("up", "\U0001f53c", "Up", "key", "Up"),
        _b("down", "\U0001f53d", "Down", "key", "Down"),
        # π — Pi flagship: opens the model picker. Greek letter (U+03C0)
        # for the pi.dev brand; reused by other providers that support /model.
        _b("model", "π", "Model", "text", "/model"),
    )
}


# ──────────────────────────────────────────────────────────────────────
# Default per-provider layouts (emoji_text). Claude/Codex/Gemini/Pi are 4
# rows × 3 buttons; shell is 3 rows (last row has 4 buttons).
# ──────────────────────────────────────────────────────────────────────

DEFAULT_LAYOUTS: dict[str, ToolbarLayout] = {
    "claude": ToolbarLayout(
        style="emoji_text",
        buttons=(
            ("screen", "ctrlc", "live"),
            ("mode", "think", "esc"),
            ("up", "enter", "down"),
            ("last", "getfile", "close"),
        ),
    ),
    "codex": ToolbarLayout(
        style="emoji_text",
        buttons=(
            ("screen", "ctrlc", "live"),
            ("esc", "tab", "mode"),
            ("up", "enter", "down"),
            ("last", "getfile", "close"),
        ),
    ),
    "gemini": ToolbarLayout(
        style="emoji_text",
        buttons=(
            ("screen", "ctrlc", "live"),
            ("mode", "yolo", "esc"),
            ("up", "enter", "down"),
            ("last", "getfile", "close"),
        ),
    ),
    "pi": ToolbarLayout(
        style="emoji_text",
        buttons=(
            ("screen", "ctrlc", "live"),
            ("esc", "tab", "model"),
            ("up", "enter", "down"),
            ("last", "getfile", "close"),
        ),
    ),
    "shell": ToolbarLayout(
        style="emoji_text",
        buttons=(
            ("screen", "ctrlc", "live"),
            ("enter", "eof", "susp"),
            ("last", "getfile", "esc", "close"),
        ),
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Resolved config + loader
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolbarConfig:
    """Resolved per-provider toolbar layouts and the merged action pool."""

    layouts: dict[str, ToolbarLayout] = field(default_factory=dict)
    actions: dict[str, ToolbarAction] = field(default_factory=dict)

    def for_provider(self, provider_name: str) -> ToolbarLayout:
        """Return the layout for ``provider_name``, falling back to claude."""
        return self.layouts.get(provider_name) or self.layouts["claude"]


def _parse_action(name: str, raw: object) -> ToolbarAction | None:
    """Parse one ``[actions.NAME]`` table from TOML."""
    if not isinstance(name, str) or not name:
        logger.warning("Toolbar config: action name must be a non-empty string")
        return None
    if len(name) > _MAX_NAME_LEN:
        logger.warning(
            "Toolbar config: action %r exceeds %d chars (callback_data overflow risk), ignoring",
            name,
            _MAX_NAME_LEN,
        )
        return None
    if not isinstance(raw, dict):
        logger.warning("Toolbar config: action %r is not a table, ignoring", name)
        return None
    emoji = str(raw.get("emoji", "")).strip()
    text = str(raw.get("text", "")).strip()
    action_type = str(raw.get("type", "")).strip()
    payload = str(raw.get("payload", "")).strip()
    if not (emoji or text):
        logger.warning("Toolbar config: action %r needs emoji or text, ignoring", name)
        return None
    if action_type not in _VALID_TYPES:
        logger.warning(
            "Toolbar config: action %r type=%r invalid (use one of %s), ignoring",
            name,
            action_type,
            sorted(_VALID_TYPES),
        )
        return None
    if action_type == "builtin":
        logger.warning(
            "Toolbar config: action %r type=builtin is reserved for built-ins; "
            "use type=key or type=text instead",
            name,
        )
        return None
    if not payload:
        logger.warning("Toolbar config: action %r missing payload, ignoring", name)
        return None
    return ToolbarAction(
        name=name,
        emoji=emoji or text,  # fall back to text if emoji omitted
        text=text or name,  # fall back to name if text omitted
        action_type=action_type,  # type: ignore[arg-type]
        payload=payload,
        literal=bool(raw.get("literal", False)),
        read_state=bool(raw.get("read_state", False)),
    )


def _parse_style(provider: str, raw_style: object) -> ButtonStyle:
    """Coerce a style value into a valid ButtonStyle, defaulting on error."""
    style = str(raw_style)
    if style not in _VALID_STYLES:
        logger.warning(
            "Toolbar config: %s.style=%r invalid (use one of %s), defaulting to emoji_text",
            provider,
            style,
            sorted(_VALID_STYLES),
        )
        return "emoji_text"
    return style  # type: ignore[return-value]


def _parse_row(
    provider: str,
    row_idx: int,
    row: object,
    action_pool: dict[str, ToolbarAction],
) -> tuple[str, ...]:
    """Parse a single row of action names. Returns possibly-empty tuple."""
    if not isinstance(row, list):
        logger.warning(
            "Toolbar config: %s.buttons[%d] is not a list, skipping",
            provider,
            row_idx,
        )
        return ()
    cells: list[str] = []
    for name in row:
        if isinstance(name, str) and name in action_pool:
            cells.append(name)
        else:
            logger.warning(
                "Toolbar config: unknown action %r in %s row %d, skipping cell",
                name,
                provider,
                row_idx,
            )
    if len(cells) > _MAX_ROW_WIDTH:
        logger.warning(
            "Toolbar config: %s row %d has %d cells (max %d), trimming",
            provider,
            row_idx,
            len(cells),
            _MAX_ROW_WIDTH,
        )
        cells = cells[:_MAX_ROW_WIDTH]
    return tuple(cells)


def _parse_layout(
    provider: str, raw: object, action_pool: dict[str, ToolbarAction]
) -> ToolbarLayout | None:
    """Parse one ``[providers.NAME]`` table from TOML."""
    if not isinstance(raw, dict):
        logger.warning("Toolbar config: provider %r is not a table, ignoring", provider)
        return None
    buttons_raw = raw.get("buttons")
    if not isinstance(buttons_raw, list) or not buttons_raw:
        logger.warning(
            "Toolbar config: %s.buttons missing or empty, ignoring", provider
        )
        return None
    style = _parse_style(provider, raw.get("style", "emoji_text"))
    rows = [
        cells
        for row_idx, row in enumerate(buttons_raw)
        if (cells := _parse_row(provider, row_idx, row, action_pool))
    ]
    if not rows:
        logger.warning("Toolbar config: %s has no usable rows, ignoring", provider)
        return None
    return ToolbarLayout(style=style, buttons=tuple(rows))


def _read_toml(path: Path) -> dict | None:
    """Read and parse a TOML file. Returns None on any error."""
    if not path.exists():
        # Optional config — its absence is the normal default case, not a warning.
        logger.debug("Toolbar config file not found: %s — using defaults", path)
        return None
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.warning("Toolbar config %s unreadable (%s) — using defaults", path, e)
        return None
    if not isinstance(raw, dict):
        logger.warning("Toolbar config %s root must be a table — using defaults", path)
        return None
    return raw


def _apply_user_actions(cfg: ToolbarConfig, raw: dict) -> None:
    """Add user-defined actions from a parsed TOML doc to the config pool."""
    user_actions = raw.get("actions") or {}
    if not isinstance(user_actions, dict):
        return
    for name, raw_action in user_actions.items():
        action = _parse_action(name, raw_action)
        if action is not None:
            cfg.actions[name] = action


def _apply_user_layouts(cfg: ToolbarConfig, raw: dict) -> None:
    """Replace per-provider default layouts with user overrides from TOML."""
    user_providers = raw.get("providers") or {}
    if not isinstance(user_providers, dict):
        return
    for provider, raw_layout in user_providers.items():
        layout = _parse_layout(provider, raw_layout, cfg.actions)
        if layout is not None:
            cfg.layouts[provider] = layout


def load_toolbar_config(path: str | Path | None = None) -> ToolbarConfig:
    """Load toolbar layouts from a TOML file or fall back to defaults.

    Per-provider entries in the TOML override the corresponding default;
    providers absent from the TOML keep their defaults. User-defined
    actions extend the built-in pool (and may shadow built-ins by name).
    Missing or malformed config falls back to defaults with a warning.
    """
    cfg = ToolbarConfig(
        layouts=dict(DEFAULT_LAYOUTS),
        actions=dict(BUILTIN_ACTIONS),
    )
    if not path:
        return cfg
    raw = _read_toml(Path(path).expanduser())
    if raw is None:
        return cfg
    # User actions first so providers can reference them.
    _apply_user_actions(cfg, raw)
    _apply_user_layouts(cfg, raw)
    return cfg
