"""/agent — manually override the auto-detected provider for a topic's window.

Manual provider selections pin routing only when the live foreground is
recognized as that provider. Unknown processes and mismatches fail closed;
this command does not start or redirect a process in the terminal.

Bare command shows an inline-keyboard picker; one-arg form skips the
picker (``/agent shell``). Manual selections must match the live foreground
provider; they cannot start, stop, or redirect an agent in the terminal. A
manual override reconciles the map against the live destination, retaining
only a matching entry and filtering later hooks from other providers. The
``provider_manual_override`` flag on WindowState blocks the periodic
``_detect_and_apply_provider`` from overwriting the choice on the next
poll. ``/agent auto`` clears the flag and re-runs detection.

Aliased as ``/provider``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)

from ..config import config
from ..session import session_manager
from ..telegram_client import PTBTelegramClient, TelegramClient
from ..thread_router import thread_router
from ..window_state_ports import identity_state
from .callback_data import CB_AGENT_CANCEL, CB_AGENT_SET
from .callback_helpers import get_thread_id, user_owns_window
from .callback_tokens import compact_callback_data, resolve_callback_data
from .callback_registry import register
from .messaging_pipeline.message_sender import safe_edit, safe_reply
from .provider_display import provider_label
from .status.provider_switch import remember_provider_selection
from .status.topic_emoji import sync_topic_name

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_BUTTONS_PER_ROW = 3

# Stable order also defines the provider-button layout; labels stay in one place.
_PROVIDER_ORDER = ("antigravity", "claude", "codex", "gemini", "pi", "shell")
_VALID_NAMES = frozenset(_PROVIDER_ORDER) | {"auto"}


def _resolve_window(update: Update) -> tuple[int, int, str] | None:
    """Return ``(user_id, thread_id, window_id)`` or None if not in a bound topic."""
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return None
    thread_id = get_thread_id(update)
    if thread_id is None:
        return None
    chat_id = update.effective_chat.id if update.effective_chat else None
    window_id = (
        thread_router.get_window_for_thread(user.id, thread_id, chat_id)
        if isinstance(chat_id, int)
        else thread_router.get_window_for_thread(user.id, thread_id)
    )
    if not window_id:
        return None
    return user.id, thread_id, window_id


def _build_keyboard(window_id: str, current: str) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for name in _PROVIDER_ORDER:
        prefix = "✓ " if name == current else ""
        row.append(
            InlineKeyboardButton(
                f"{prefix}{provider_label(name)}",
                callback_data=compact_callback_data(
                    CB_AGENT_SET, f"{CB_AGENT_SET}{window_id}:{name}", window_id
                ),
            )
        )
        if len(row) == _BUTTONS_PER_ROW:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(
                "🔄 Auto",
                callback_data=compact_callback_data(
                    CB_AGENT_SET, f"{CB_AGENT_SET}{window_id}:auto", window_id
                ),
            ),
            InlineKeyboardButton(
                "Cancel",
                callback_data=compact_callback_data(
                    CB_AGENT_CANCEL, f"{CB_AGENT_CANCEL}{window_id}", window_id
                ),
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _picker_text(window_id: str) -> str:
    current = identity_state.get_provider_name(window_id) or "unknown"
    name = identity_state.get_window_name(window_id) or thread_router.get_display_name(
        window_id
    )
    if not name or name == window_id:
        name = "This topic"
    mode = (
        "Manual"
        if identity_state.is_provider_manually_overridden(window_id)
        else "Auto"
    )
    return (
        f"{name}\nProvider: **{provider_label(current)}** · {mode}\n\n"
        "Choose a provider, or **Auto** to follow the running agent."
    )


async def _apply_switch(
    window_id: str,
    chosen: str,
    *,
    client: TelegramClient | None = None,
    chat_id: int = 0,
    thread_id: int = 0,
) -> tuple[str, str]:
    """Apply the provider switch and return ``(resolved_name, reply_text)``.

    ``client``/``chat_id``/``thread_id`` are threaded to ``ensure_setup`` so
    the shell-switch path can show the "Set up / Skip" offer keyboard instead
    of silently mutating PS1.
    """
    current = identity_state.get_provider_name(window_id) or ""
    was_manual = identity_state.is_provider_manually_overridden(window_id)
    target, manual, reply_intro = await _resolve_provider_selection(window_id, chosen)
    if target is None:
        return current or "unknown", reply_intro

    hold_manual_filter = manual or (chosen == "auto" and was_manual)
    _commit_switch(
        window_id,
        target,
        current,
        manual=hold_manual_filter,
        preserve_session_map=hold_manual_filter,
    )
    if hold_manual_filter:
        reply_intro += _readmit_destination_entry(
            window_id, target, release_manual=chosen == "auto" and was_manual
        )
    if client is not None and chat_id and thread_id:
        remember_provider_selection(chat_id, thread_id, window_id, target)
        if current != target:
            await sync_topic_name(
                client, chat_id, thread_id, thread_router.get_display_name(window_id)
            )

    if target == "shell":
        # Always offer prompt-marker setup on a shell-target switch —
        # whether the user picked shell explicitly or auto-detect resolved
        # to it. ``ensure_setup`` no-ops when the marker is already present
        # or the user previously chose Skip.
        # Lazy: shell prompt orchestrator hits the recovery subpackage via
        # send-keys callbacks; loading at module level would cycle.
        from .shell.shell_prompt_orchestrator import ensure_setup

        await ensure_setup(
            window_id,
            "provider_switch",
            client=client,
            chat_id=chat_id,
            thread_id=thread_id,
        )
        reply = f"{reply_intro} Text routes to shell commands. Use `!command` for a direct command."
    else:
        reply = reply_intro
    return target, reply


def _readmit_destination_entry(
    window_id: str, target: str, *, release_manual: bool
) -> str:
    # Lazy: session_map imports provider/identity wiring used by bootstrap.
    from ..session_map import session_map_sync

    result = session_map_sync.clear_session_map_entry(window_id)
    if release_manual:
        identity_state.set_provider_manual_override(window_id, value=result is None)
    if result is None:
        return " Stale session mapping could not be cleared; provider remains pinned."
    if target == "shell":
        return ""
    if result == "preserved":
        return " Existing session restored."
    return _tracking_wait_message(target)


def _tracking_wait_message(provider_name: str) -> str:
    # Lazy: provider registry loads concrete providers and their capabilities.
    from ..providers.registry import UnknownProviderError, registry

    try:
        supports_hook = registry.get(provider_name).capabilities.supports_hook
    except UnknownProviderError:
        supports_hook = False
    if supports_hook:
        return " Waiting for the next session hook to resume tracking."
    return " Waiting for transcript discovery to resume tracking."


async def _resolve_provider_selection(
    window_id: str, chosen: str
) -> tuple[str | None, bool, str]:
    if chosen == "auto":
        detected = await _redetect_provider(window_id)
        if detected is None:
            return (
                None,
                False,
                (
                    "⚠ Could not reach the multiplexer, so the agent was left "
                    "as it is. Try again in a moment."
                ),
            )
        return detected, False, f"Auto-detected: **{provider_label(detected)}**."

    live_provider = await _redetect_provider(window_id, allow_shell_fallback=False)
    if live_provider is None:
        return (
            None,
            False,
            (
                "⚠ Could not verify the live process, so the provider was left "
                "unchanged. Try again in a moment."
            ),
        )
    if live_provider != chosen:
        return (
            None,
            False,
            (
                f"⚠ {provider_label(live_provider)} is running in this pane. "
                "Exit it or start the desired agent first; then use **Auto**. "
                "Provider unchanged."
            ),
        )
    intro = (
        f"Selected **{provider_label(chosen)}** (manual)."
        if chosen != "shell"
        else "Selected **Terminal** (manual)."
    )
    return chosen, True, intro


async def _redetect_provider(
    window_id: str, *, allow_shell_fallback: bool = True
) -> str | None:
    """Read the pane provider; shell fallback is enabled only for Auto.

    Manual selections fail closed when the process is unknown or the
    multiplexer is unavailable, because a guessed provider can route input to
    the wrong foreground application.
    """
    # Lazy: detect_provider_from_pane pulls the providers package — only
    # needed for /agent selection verification or Auto re-detection.
    from ..providers import detect_provider_from_pane

    # Lazy: tmux_manager imports providers; same cycle-break as above.
    from ..multiplexer import multiplexer as tmux_manager

    # Lazy: importing the reconciliation seam at module load forms a cycle.
    from ..multiplexer.reconciliation import window_presence

    if await window_presence(window_id, tmux_manager) is None:
        return None

    w = await tmux_manager.find_window_by_id(window_id)
    if w is None:
        # Either the window went in the meantime or this second read failed;
        # neither is a confirmed "not an agent", and the caller persists what
        # comes back. Report unknown instead of resolving to shell.
        return None
    if not w.pane_current_command:
        return "shell" if allow_shell_fallback else None
    detected = await detect_provider_from_pane(
        w.pane_current_command, window_id=window_id
    )
    return detected or ("shell" if allow_shell_fallback else None)


def _commit_switch(
    window_id: str,
    chosen: str,
    current: str,
    *,
    manual: bool,
    preserve_session_map: bool = False,
) -> None:
    """Switch routing state and retire the old live transcript identity.

    The caller reconciles any retained map entry against the live destination
    before releasing a temporary manual filter. Same-provider selections keep
    the primary identity and toggle the manual-override flag.
    """
    # Lazy: session_map shares the SessionManager bootstrap wiring.
    from ..session_map import session_map_sync

    session_map_sync.invalidate_selection_snapshots()
    same_provider = current == chosen
    session_manager.set_window_provider(
        window_id, chosen, preserve_session_map=preserve_session_map
    )
    if not same_provider:
        identity_state.clear_session_identity(window_id)
    identity_state.set_provider_manual_override(window_id, value=manual)
    if same_provider:
        # Re-admit the map with primary preference, while retaining session
        # and transcript state for the current same-provider topic.
        return
    if chosen != "shell":
        # The caller's map reconciliation keeps only a matching destination
        # entry before any temporary manual filter is released.
        return
    # Leaving a hookful provider for shell: drop monitor/orchestrator state.
    # Lazy: shell subpackage pulls shell_infra; only needed on the shell-switch branch.
    from .shell.shell_capture import clear_shell_monitor_state

    # Lazy: same shell-switch-only reason as above.
    from .shell.shell_prompt_orchestrator import clear_state as clear_orchestrator

    clear_shell_monitor_state(window_id)
    clear_orchestrator(window_id)


async def agent_command(update: Update, _context: "ContextTypes.DEFAULT_TYPE") -> None:
    """Handle ``/agent [provider|auto]`` — show picker or apply switch."""
    resolved = _resolve_window(update)
    if resolved is None or not update.message:
        if update.message:
            await safe_reply(update.message, "Use /agent inside a bound topic.")
        return
    _user_id, thread_id, window_id = resolved

    args = (update.message.text or "").split(maxsplit=1)
    arg = args[1].strip().lower() if len(args) > 1 else ""
    if not arg:
        await safe_reply(
            update.message,
            _picker_text(window_id),
            reply_markup=_build_keyboard(
                window_id, identity_state.get_provider_name(window_id) or ""
            ),
        )
        return
    if arg not in _VALID_NAMES:
        await safe_reply(
            update.message,
            f"Unknown agent `{arg}`. Use one of: {', '.join(sorted(_VALID_NAMES))}.",
        )
        return
    client = PTBTelegramClient(update.get_bot())
    chat_id = update.message.chat.id
    _, reply = await _apply_switch(
        window_id, arg, client=client, chat_id=chat_id, thread_id=thread_id
    )
    await safe_reply(update.message, reply)


@register(CB_AGENT_SET, CB_AGENT_CANCEL)
async def _dispatch(update: Update, _context: "ContextTypes.DEFAULT_TYPE") -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    user = update.effective_user
    if user is None:
        return
    data = resolve_callback_data(query.data, user.id, user_owns_window)
    if data is None:
        await query.answer("This button has expired", show_alert=True)
        return
    if data.startswith(CB_AGENT_CANCEL):
        window_id = data[len(CB_AGENT_CANCEL) :]
        if not user_owns_window(user.id, window_id):
            await query.answer("Not your window")
            return
        await _ack_and_strip(
            query,
            f"Cancelled. Agent still **{identity_state.get_provider_name(window_id) or '(unknown)'}**.",
        )
        return
    payload = data[len(CB_AGENT_SET) :]
    if ":" not in payload:
        await query.answer("Bad callback")
        return
    window_id, chosen = payload.rsplit(":", 1)
    if not user_owns_window(user.id, window_id):
        await query.answer("Not your window")
        return
    if chosen not in _VALID_NAMES:
        await query.answer("Unknown provider")
        return
    client = PTBTelegramClient(query.get_bot())
    chat_id = query.message.chat.id if query.message else 0
    thread_id = get_thread_id(update) or 0
    _, reply = await _apply_switch(
        window_id, chosen, client=client, chat_id=chat_id, thread_id=thread_id
    )
    await _ack_and_strip(query, reply)


async def _ack_and_strip(query: CallbackQuery, text: str) -> None:
    """Answer the callback and edit the picker message in place, removing the keyboard."""
    await query.answer()
    await safe_edit(query, text, reply_markup=None)
