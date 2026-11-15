"""Window picker callback handlers.

Handles inline keyboard callbacks for the window picker UI:
  - CB_WIN_BIND: Bind an existing unbound tmux window to the current topic
  - CB_WIN_NEW: Transition from window picker to directory browser for new session
  - CB_WIN_CANCEL: Cancel the window picker

Key function: handle_window_callback (uniform callback handler signature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import structlog
from pathlib import Path

from telegram import CallbackQuery, Chat, Update
from telegram.error import TelegramError
from ... import window_query
from ...config import config
from ...topic_selection import select_topic_target, verify_topic_target
from .topic_orchestration import pending_creation_transaction
from ...telegram_client import PTBTelegramClient, TelegramClient
from ...session import session_manager
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...window_state_ports.identity_state import get_initial_provider_name
from ...window_state_ports import lifecycle_state
from ..telegram_origin import send_telegram_to_window
from ...session_protocol.telegram import TelegramSessionAdapter, adapter_for
from ...session_protocol.models import ProtocolError
from ..callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from ..callback_helpers import get_thread_id
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    clear_window_picker_state,
)
from ..callback_registry import register
from ..messaging_pipeline.message_sender import safe_edit, safe_send
from ..status.topic_emoji import format_topic_name_for_mode
from ..user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT

if TYPE_CHECKING:
    from telegram.ext import ContextTypes
    from ...multiplexer.base import WindowRef

logger = structlog.get_logger()


def _get_topic_chat(update: Update, query: CallbackQuery) -> Chat | None:
    """Resolve the chat object for the current callback topic, if available."""
    query_message = (
        update.callback_query.message if update.callback_query else None
    ) or query.message
    return query_message.chat if query_message else None


def _store_group_chat_id(
    user_id: int, thread_id: int, update: Update, query: CallbackQuery
) -> None:
    """Persist group chat routing for a topic thread (best-effort)."""
    chat = _get_topic_chat(update, query)
    if chat and chat.type in ("group", "supergroup"):
        thread_router.set_group_chat_id(user_id, thread_id, chat.id)


async def _rename_forum_topic(
    client: TelegramClient, chat_id: int, thread_id: int, display: str, window_id: str
) -> None:
    """Rename a topic in either a forum or a private topic chat."""
    try:
        await client.edit_forum_topic(
            chat_id=chat_id,
            message_thread_id=thread_id,
            name=format_topic_name_for_mode(
                display, window_query.get_approval_mode(window_id)
            ),
        )
    except TelegramError as exc:
        logger.debug("Failed to rename topic: %s", exc)


async def handle_window_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle window picker callbacks.

    Dispatches to the appropriate sub-handler based on callback data prefix.
    """
    if data.startswith(CB_WIN_BIND):
        await _handle_bind(query, user_id, data, update, context)
    elif data == CB_WIN_NEW:
        await _handle_new(query, user_id, update, context)
    elif data == CB_WIN_CANCEL:
        await _handle_cancel(query, update, context)


async def _detect_and_setup_provider(
    window_id: str,
    pane_current_command: str | None,
    *,
    adapter: TelegramSessionAdapter,
    client: TelegramClient | None = None,
    user_id: int = 0,
    thread_id: int = 0,
) -> str:
    """Detect provider from pane process and set up prompt if shell.

    Uses foreground-process detection (via the multiplexer seam) when the
    pane command is a JS runtime, falling back to basename-only matching.
    Returns the detected provider name (empty string if undetected).
    """
    # Lazy: providers/__init__ loads process_detection; gate behind actual
    # adoption.
    # Lazy: providers package heavy bootstrap
    from ...providers import detect_provider_from_pane

    # Pass through even when the command is empty: herdr leaves
    # pane_current_command empty for a bare shell pane, and
    # detect_provider_from_pane consults the foreground process via window_id
    # to classify it (e.g. as "shell"). tmux always reports a command.
    detected = await detect_provider_from_pane(
        pane_current_command or "", window_id=window_id
    )
    if detected:
        session_manager.set_window_provider(window_id, detected)
        # Lazy: same providers cycle as detect_provider_from_pane.
        from ...providers import get_provider_for_window

        provider = get_provider_for_window(window_id, detected)
        if provider and provider.capabilities.chat_first_command_path:
            # Lazy: shell ↔ topics cycle via prompt-marker callbacks.
            from ..shell.shell_prompt_orchestrator import ensure_setup

            await ensure_setup(
                window_id,
                "external_bind",
                adapter=adapter,
                client=client,
                chat_id=thread_router.resolve_chat_id(user_id, thread_id),
                thread_id=thread_id,
            )
    return detected


async def _forward_pending_text(
    client: TelegramClient,
    user_id: int,
    thread_id: int,
    window_id: str,
    text: str,
    provider_name: str,
    *,
    adapter,
    is_existing_window: bool = False,
    chat_id: int | None = None,
    shell_exit_detected: bool = False,
) -> None:
    """Forward pending text to an existing target or a new shell via LLM.

    Args:
        is_existing_window: True when binding an existing window (not a fresh
            one from directory browser). Existing shells receive explicit
            terminal input to avoid racing _ensure_prompt_marker with the offer UI.
    """
    # Lazy: same providers cycle as _detect_and_setup_provider.
    from ...providers import get_provider_for_window

    provider = get_provider_for_window(window_id, provider_name)
    is_chat_first = bool(provider and provider.capabilities.chat_first_command_path)
    if is_chat_first and is_existing_window:
        if shell_exit_detected:
            send_ok = False
            send_msg = (
                "Agent exited or shell access is not confirmed; recover the session "
                "or run /agent shell before sending input."
            )
        else:
            send_ok, send_msg = await _send_verified_shell_input(
                user_id,
                window_id,
                thread_id,
                text,
                chat_id,
                adapter,
            )
        if not send_ok:
            logger.warning(
                "Failed to forward pending text to window %s (user %s): %s",
                window_id,
                user_id,
                send_msg,
            )
            await safe_send(
                client,
                thread_router.resolve_chat_id(user_id, thread_id),
                f"❌ Failed to forward pending message: {send_msg}",
                message_thread_id=thread_id,
            )
    elif is_chat_first:
        # Lazy: shell ↔ topics cycle.
        from ..shell.shell_commands import handle_shell_message

        await handle_shell_message(
            client,
            user_id,
            thread_id,
            window_id,
            text,
            chat_id=chat_id,
            adapter=adapter,
        )
    else:
        # Agent sessions use the message action. It is rejected by the protocol
        # if this target has already fallen back to a shell.
        send_ok, send_msg = await send_telegram_to_window(
            user_id, window_id, thread_id, text, chat_id, adapter=adapter
        )
        if not send_ok:
            logger.warning(
                "Failed to forward pending text to window %s (user %s): %s",
                window_id,
                user_id,
                send_msg,
            )
            await safe_send(
                client,
                thread_router.resolve_chat_id(user_id, thread_id),
                f"❌ Failed to forward pending message: {send_msg}",
                message_thread_id=thread_id,
            )


async def _send_verified_shell_input(
    user_id: int,
    window_id: str,
    thread_id: int,
    text: str,
    chat_id: int | None,
    adapter,
) -> tuple[bool, str]:
    """Submit to an existing shell only when the protocol confirms its process."""
    # Lazy: Telegram origin tracking depends on session and provider state.
    from ..telegram_origin import (
        forget_telegram_injection,
        remember_telegram_injection,
    )

    try:
        target, snapshot = await adapter.resolve(window_id)
    except ProtocolError as exc:
        return False, exc.message
    if target.provider != "shell" or not target.foreground_verified:
        return False, "Shell access is not confirmed; recover the session first."

    injection = remember_telegram_injection(
        user_id, window_id, thread_id, text, chat_id
    )
    success = False
    try:
        success, message = await adapter.send_text(
            window_id,
            text,
            submit=True,
            terminal_input=True,
            expected_binding_version=snapshot.binding_version,
        )
        return success, message
    except ProtocolError as exc:
        return False, exc.message
    finally:
        if not success:
            forget_telegram_injection(user_id, window_id, thread_id, injection, chat_id)


async def _select_and_bind(
    query: CallbackQuery,
    user_id: int,
    thread_id: int,
    selected_wid: str,
    window: WindowRef,
    update: Update,
) -> bool:
    owner = window.topic_mark_owner
    with pending_creation_transaction():
        if thread_router.has_target_provisioning(selected_wid) is True:
            await query.answer(
                "Session is still being provisioned; please retry", show_alert=True
            )
            return False
        if lifecycle_state.is_topic_mark_quarantined(selected_wid, owner=owner) or (
            config.topic_discovery_mode == "marked"
            and window.topic_markable
            and owner is None
        ):
            await query.answer(
                "Could not verify topic selection; please retry", show_alert=True
            )
            return False
        selected, owner = await select_topic_target(selected_wid, expected_owner=owner)
        if not selected:
            if (
                config.topic_discovery_mode == "marked"
                or lifecycle_state.is_topic_mark_quarantined(selected_wid, owner=owner)
                or lifecycle_state.is_topic_detach_pending(
                    selected_wid,
                    owner=owner,
                )
            ):
                await query.answer(
                    "Could not verify topic selection; please retry", show_alert=True
                )
                return False
            logger.warning("topic_selection_pending_on_bind", window_id=selected_wid)
        current = await verify_topic_target(
            selected_wid,
            owner=owner,
            require_marked=config.topic_discovery_mode == "marked",
        )
        if (
            current is None
            or lifecycle_state.is_topic_mark_quarantined(
                selected_wid, owner=current.topic_mark_owner
            )
            or not lifecycle_state.record_topic_mark_owner(
                selected_wid, current.topic_mark_owner
            )
        ):
            await query.answer(
                "Session changed; refresh the window picker", show_alert=True
            )
            return False
        session_manager.flush_state()
        thread_router.bind_thread(
            user_id,
            thread_id,
            selected_wid,
            window_name=window.window_name,
            chat_id=(
                update.callback_query.message.chat.id
                if update.callback_query and update.callback_query.message
                else None
            ),
        )
        completed = lifecycle_state.complete_topic_admission(
            selected_wid, current.topic_mark_owner
        )
        session_manager.flush_state()
        if not completed:
            logger.warning(
                "topic_admission_lifecycle_unfinished", window_id=selected_wid
            )
    return True


async def _handle_bind(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_WIN_BIND: bind existing unbound window to current topic."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and get_thread_id(update) != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    try:
        idx = int(data[len(CB_WIN_BIND) :])
    except ValueError:
        await query.answer("Invalid data")
        return

    cached_windows: list[str] = (
        context.user_data.get(UNBOUND_WINDOWS_KEY, []) if context.user_data else []
    )
    if idx < 0 or idx >= len(cached_windows):
        await query.answer("Window list changed, please retry", show_alert=True)
        return
    selected_wid = cached_windows[idx]

    w = await tmux_manager.find_window_by_id(selected_wid)
    if not w:
        display = thread_router.get_display_name(selected_wid)
        await query.answer(f"Window '{display}' no longer exists", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Not in a topic", show_alert=True)
        return

    display = w.window_name
    # Native selection may create an otherwise empty WindowState. Capture
    # provider history first so that empty state is not mistaken for an exit.
    previous_provider = window_query.get_window_provider(selected_wid)
    initial_provider = get_initial_provider_name(selected_wid)
    if not await _select_and_bind(query, user_id, thread_id, selected_wid, w, update):
        return
    clear_window_picker_state(context.user_data)
    _store_group_chat_id(user_id, thread_id, update, query)
    client: TelegramClient = PTBTelegramClient(context.bot)
    # Capture origin before detection updates the window's current provider.
    # One foreground observation must drive both routing and exit policy: a
    # second probe can see a different process during an agent exit.
    detected = await _detect_and_setup_provider(
        selected_wid,
        w.pane_current_command,
        adapter=adapter_for(context.bot_data),
        client=client,
        user_id=user_id,
        thread_id=thread_id,
    )
    shell_exit_detected = detected == "shell" and (
        initial_provider != "shell"
        and (previous_provider is not None or bool(initial_provider))
    )

    await _rename_forum_topic(
        client,
        thread_router.resolve_chat_id(user_id, thread_id),
        thread_id,
        display,
        selected_wid,
    )

    await safe_edit(
        query,
        f"✅ Bound to window `{display}`",
    )

    pending_text = (
        context.user_data.get(PENDING_THREAD_TEXT) if context.user_data else None
    )
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_TEXT, None)
        context.user_data.pop(PENDING_THREAD_ID, None)
    if pending_text:
        await _forward_pending_text(
            client,
            user_id,
            thread_id,
            selected_wid,
            pending_text,
            detected,
            adapter=adapter_for(context.bot_data),
            is_existing_window=True,
            shell_exit_detected=shell_exit_detected,
            chat_id=(
                update.callback_query.message.chat.id
                if update.callback_query and update.callback_query.message
                else None
            ),
        )
    await query.answer("Bound")


async def _handle_new(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_WIN_NEW: transition from window picker to directory browser."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    # A live picker always has a pending thread (set when it was shown).
    # None means the flow was reset (e.g. /start) and this is a stale tap —
    # rebuilding the browser here would leave it pending-thread-less and
    # let a later confirm spawn an unbound window in the bot's own cwd.
    if pending_tid is None:
        await query.answer("Stale picker (flow reset)", show_alert=True)
        return
    if get_thread_id(update) != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    clear_window_picker_state(context.user_data)
    start_path = str(Path.cwd())
    msg_text, keyboard, subdirs = build_directory_browser(start_path, user_id=user_id)
    if context.user_data is not None:
        context.user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
        context.user_data[BROWSE_PATH_KEY] = start_path
        context.user_data[BROWSE_PAGE_KEY] = 0
        context.user_data[BROWSE_DIRS_KEY] = subdirs
    await safe_edit(query, msg_text, reply_markup=keyboard)
    await query.answer()


async def _handle_cancel(
    query: CallbackQuery,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_WIN_CANCEL: cancel the window picker."""
    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and get_thread_id(update) != pending_tid:
        await query.answer("Stale picker (topic mismatch)", show_alert=True)
        return
    clear_window_picker_state(context.user_data)
    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_ID, None)
        context.user_data.pop(PENDING_THREAD_TEXT, None)
    await safe_edit(query, "Cancelled")
    await query.answer("Cancelled")


# --- Registry dispatch entry point ---


@register(CB_WIN_BIND, CB_WIN_NEW, CB_WIN_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await handle_window_callback(query, user.id, query.data, update, context)
