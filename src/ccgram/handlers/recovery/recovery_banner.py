"""Dead-window recovery banner UX flow.

Owns the banner the user sees in three situations:
  - a tmux window died proactively (``dead`` mode),
  - the user invoked ``/restore`` (``restore`` mode),
  - the user opened the resume picker (``resume`` mode).

Public surface:
  - :class:`RecoveryBanner` / :data:`RecoveryMode`
  - :func:`render_banner`, :func:`build_recovery_keyboard`
  - :func:`_create_and_bind_window` (used by :mod:`resume_picker` to wire a
    new window after the user picks a session)
  - the per-button handlers ``_handle_back/_fresh/_continue/_resume/
    _send_empty_state/_handle_browse/_handle_cancel``

The dispatcher in :mod:`recovery_callbacks` routes button taps here. The
sibling cycle with :mod:`resume_picker` is one-way at the top level
(this module imports from the picker for ``scan_sessions_for_cwd``); the
reverse direction lives behind a lazy import inside the picker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError

from ... import window_query
from ...providers import get_provider, get_provider_for_window, resolve_launch_command
from ...session import session_manager
from ...session_map import session_map_sync
from ...telegram_client import PTBTelegramClient
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.base import canonical_window_id
from ..telegram_origin import send_telegram_to_window
from ...window_state_store import CCGRAM_CREATED_WINDOW_ORIGIN
from ..callback_data import (
    CB_RECOVERY_BACK,
    CB_RECOVERY_BROWSE,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_RESUME,
)
from ..callback_helpers import get_thread_id
from ..callback_tokens import compact_callback_data
from ..messaging_pipeline.message_sender import safe_edit, safe_send
from ..status.topic_emoji import format_topic_name_for_mode
from ..user_state import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    RECOVERY_SESSIONS,
    RECOVERY_WINDOW_ID,
)
from .recovery_callbacks import _clear_recovery_state
from .resume_picker import (
    _build_empty_resume_keyboard,
    _build_resume_picker_keyboard,
    scan_sessions_for_cwd,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

RecoveryMode = Literal["dead", "restore", "resume"]


@dataclass(frozen=True)
class RecoveryBanner:
    """Inputs for the unified recovery banner.

    The banner is the dead-window notification ccgram shows in three
    situations: a window died proactively (``dead``), the user invoked
    /restore (``restore``), or the user opened the resume picker
    (``resume``). All three flow through ``render_banner`` so the keyboard,
    subtitle, and copy stay consistent across entry points.
    """

    chat_id: int
    thread_id: int
    window_id: str
    mode: RecoveryMode
    provider: str | None = None
    display: str = ""
    cwd: str = ""


def _validate_recovery_state(
    data_suffix: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> tuple[int, str] | None:
    """Validate common recovery preconditions.

    Supports two paths:
      1. Text-handler path: PENDING_THREAD_ID and RECOVERY_WINDOW_ID in user_data.
      2. Proactive notification path: no user_data state, validate via binding.

    Returns ``(thread_id, old_window_id)`` on success, or ``None`` on
    failure (caller should return early and call ``query.answer``).
    """
    thread_id = get_thread_id(update)
    if thread_id is None:
        return None

    user_id = update.effective_user.id if update.effective_user else None
    if user_id is None:
        return None

    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    stored_wid = (
        context.user_data.get(RECOVERY_WINDOW_ID) if context.user_data else None
    )

    if pending_tid is not None:
        if thread_id != pending_tid or stored_wid != data_suffix:
            return None
    else:
        bound_wid = thread_router.get_window_for_thread(user_id, thread_id)
        if bound_wid != data_suffix:
            return None
        if context.user_data is not None:
            context.user_data[PENDING_THREAD_ID] = thread_id
            context.user_data[RECOVERY_WINDOW_ID] = data_suffix

    return thread_id, data_suffix


def render_banner(banner: RecoveryBanner) -> tuple[str, InlineKeyboardMarkup]:
    """Render the recovery banner text and inline keyboard.

    Returns the message body and a :class:`InlineKeyboardMarkup` ready to
    pass to ``safe_reply`` / ``rate_limit_send_message``. The keyboard is
    the provider-aware action keyboard from :func:`build_recovery_keyboard`
    in every mode — modes only differ in the surrounding copy so the user
    knows whether the banner appeared on its own or in response to a
    request.
    """

    keyboard = build_recovery_keyboard(banner.window_id)
    help_text = _recovery_help_text(banner.window_id)
    cwd_line = f"\n\U0001f4c2 `{banner.cwd}`" if banner.cwd else ""
    label = banner.display or banner.window_id

    if banner.mode == "restore":
        title = f"\U0001f504 Restore `{label}`."
        prompt = f"Choose how to continue.\n{help_text}"
    elif banner.mode == "resume":
        title = f"⏪ Resume `{label}`."
        prompt = f"Pick a session below or use the menu.\n{help_text}"
    else:
        title = f"⚠ Session `{label}` ended."
        prompt = f"Tap a button or send a message to recover.\n{help_text}"

    text = f"{title}{cwd_line}\n\n{prompt}"
    return text, keyboard


def _recovery_help_text(window_id: str) -> str:
    """Return a one-line subtitle explaining the available recovery actions.

    Mirrors the keyboard layout in ``build_recovery_keyboard`` so users can
    read what each button does without trial and error. Buttons hidden by
    the active provider's capabilities are omitted from the subtitle too.
    """

    caps = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    ).capabilities
    parts = ["Start fresh"]
    if caps.supports_continue:
        parts.append("Continue last session")
    if caps.supports_resume and caps.supports_resume_picker:
        parts.append("Resume from list")
    return " · ".join(parts)


def build_recovery_keyboard(window_id: str) -> InlineKeyboardMarkup:
    """Build inline keyboard for dead window recovery options.

    Buttons for Continue and Resume are only shown when the active provider
    declares support for those capabilities.
    """

    caps = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    ).capabilities
    options: list[InlineKeyboardButton] = [
        InlineKeyboardButton(
            "\U0001f195 Fresh",
            callback_data=compact_callback_data(
                CB_RECOVERY_FRESH, f"{CB_RECOVERY_FRESH}{window_id}", window_id
            ),
        ),
    ]
    if caps.supports_continue:
        options.append(
            InlineKeyboardButton(
                "▶ Continue",
                callback_data=compact_callback_data(
                    CB_RECOVERY_CONTINUE,
                    f"{CB_RECOVERY_CONTINUE}{window_id}",
                    window_id,
                ),
            )
        )
    if caps.supports_resume and caps.supports_resume_picker:
        options.append(
            InlineKeyboardButton(
                "⏪ Resume",
                callback_data=compact_callback_data(
                    CB_RECOVERY_RESUME, f"{CB_RECOVERY_RESUME}{window_id}", window_id
                ),
            )
        )
    return InlineKeyboardMarkup(
        [
            options,
            [InlineKeyboardButton("✖ Cancel", callback_data=CB_RECOVERY_CANCEL)],
        ]
    )


async def _stale_recovery_offer(old_window_id: str) -> str | None:
    """Why this recovery offer must not be acted on, or None to proceed.

    Every Fresh / Continue / Resume route converges on the replacement below,
    and the banner offering them may have been drawn from an ambiguous lookup
    minutes ago. The replacement unbinds before it creates, so a stale offer
    leaves the old terminal running with its routing handed to a new window.

    One confirmed read answers both questions — does it exist, and is what
    holds it still an agent — because a presence check followed by a second
    lookup is a second chance to fail.
    """
    if not old_window_id:
        return None

    # Lazy: importing the reconciliation seam at module load forms a cycle.
    from ...multiplexer.reconciliation import window_snapshot

    # Lazy: same cycle through provider/session initialization.
    from ..telegram_origin import agent_origin_returned_to_shell

    confirmed, window = await window_snapshot(old_window_id)
    if not confirmed:
        return (
            "\u26a0 Could not reach the multiplexer, so nothing was changed. "
            "Try again in a moment."
        )
    if window is None:
        # Confirmed gone: recovery is exactly right.
        return None
    if await agent_origin_returned_to_shell(old_window_id, window):
        # Alive, but the agent exited and left a shell — the case the text
        # handler deliberately offers recovery for.
        return None
    return "That session is running again, so it was left alone."


def _is_definitive_create_failure(message: str) -> bool:
    """Return whether a failed create response proves no target exists."""
    normalized = message.casefold()
    return any(
        marker in normalized
        for marker in (
            "directory does not exist",
            "not a directory",
            "selected workspace no longer exists",
            "requires a sessionful agent",
            "does not create worktrees",
        )
    )


async def _await_creation_result(creation):  # noqa: ANN001
    """Wait for creation while retaining a result delivered with cancellation."""
    task = asyncio.create_task(creation)
    try:
        return await asyncio.shield(task), False
    except asyncio.CancelledError:
        return await task, True


def _mark_provisioning_uncertain(claim_id: str | None) -> None:
    """Release runtime ownership while retaining durable recovery evidence."""
    if claim_id is None:
        return
    try:
        thread_router.mark_provisioning_uncertain(claim_id)
    except KeyError:
        logger.warning("Provisioning claim disappeared while marking uncertain")
        return
    try:
        session_manager.flush_state()
    except BaseException:  # noqa: BLE001
        logger.warning("Could not persist uncertain provisioning claim")


def _abort_provisioning_claim(claim_id: str | None) -> None:
    """Release a claim whose backend operation definitely never started."""
    if claim_id is None:
        return
    try:
        thread_router.abort_topic_provisioning(
            claim_id,
            target_confirmed_absent=True,
        )
    except KeyError:
        logger.warning("Provisioning claim disappeared while aborting")
        return
    try:
        session_manager.flush_state()
    except BaseException:  # noqa: BLE001
        logger.warning("Could not persist aborted provisioning claim")


def _clear_failed_target_state(window_id: str) -> None:
    """Clear state scoped only to a failed replacement target."""
    # Lazy: cleanup/state registries import recovery and polling modules.
    from ...session_map import session_map_prefix

    # Lazy: cleanup/state registries import recovery and polling modules.
    from ...topic_state_registry import topic_state

    # Lazy: cleanup/state registries import recovery and polling modules.
    from ..callback_tokens import revoke_window_tokens

    topic_state.clear_window(window_id)
    topic_state.clear_qualified(f"{session_map_prefix()}{window_id}")
    revoke_window_tokens(window_id)


def _follow_provisioning_supersession(window_id: str, *, claim_id: str | None) -> str:
    """Persist a backend target identity that changed during hook startup."""
    current = window_query.resolve_window_alias(window_id)
    if current == window_id:
        return window_id
    # Lazy: topic orchestration imports polling and topic handlers.
    from ..topics import topic_orchestration

    if canonical_window_id(current) == canonical_window_id(window_id):
        return current
    if claim_id is not None:
        try:
            thread_router.attach_provisioning_target(claim_id, current)
            session_manager.flush_state()
        except BaseException:  # noqa: BLE001
            _mark_provisioning_uncertain(claim_id)
            raise
    topic_orchestration.register_pending_creation(current)
    topic_orchestration.clear_pending_creation(window_id)
    logger.info("Recovery target superseded: %s -> %s", window_id, current)
    return current


async def _cleanup_failed_topic(
    user_id: int,
    thread_id: int,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Delete an exact topic after a target-for-topic failure."""
    # Lazy: cleanup imports the recovery state registry.
    from ..cleanup import clear_topic_state

    # Lazy: topic deletion imports cleanup and polling state.
    from ..topics.topic_deletion import cleanup_retired_topic

    retired = next(
        (
            topic
            for topic in thread_router.iter_retired_topics()
            if topic.user_id == user_id
            and topic.chat_id == chat_id
            and topic.thread_id == thread_id
        ),
        None,
    )
    if retired is None:
        return

    async def clear_state_before_delete() -> None:
        await clear_topic_state(
            user_id,
            thread_id,
            PTBTelegramClient(context.bot),
            context.user_data,
            window_id=retired.target_id,
            chat_id=chat_id,
            window_dead=True,
        )

    await cleanup_retired_topic(
        PTBTelegramClient(context.bot),
        retired,
        router=thread_router,
        before_delete=clear_state_before_delete,
    )


async def _finish_failed_provisioning(  # noqa: C901
    *,
    claim_id: str | None,
    target_id: str | None,
    user_id: int,
    thread_id: int,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    failure_message: str = "",
    preserve_topic: bool = False,
    target_confirmed_absent: bool = False,
) -> bool:
    """Abort a proven-absent target or quarantine an uncertain target."""
    if claim_id is None:
        if target_id is None:
            return False
        try:
            absent = await tmux_manager.kill_window(target_id)
        except BaseException:  # noqa: BLE001
            return False
        if absent:
            # Lazy: topic orchestration imports polling and topic handlers.
            from ..topics import topic_orchestration

            topic_orchestration.clear_pending_creation(target_id)
            _clear_failed_target_state(target_id)
        return bool(absent)

    if target_id is None:
        if not target_confirmed_absent and not _is_definitive_create_failure(
            failure_message
        ):
            _mark_provisioning_uncertain(claim_id)
            return False
        target_confirmed_absent = True
    elif target_confirmed_absent:
        pass
    else:
        try:
            target_confirmed_absent = await tmux_manager.kill_window(target_id)
        except BaseException:  # noqa: BLE001
            _mark_provisioning_uncertain(claim_id)
            return False

    # Lazy: topic orchestration imports polling and topic handlers.
    from ..topics import topic_orchestration

    if target_confirmed_absent:
        _abort_provisioning_claim(claim_id)
        if target_id is not None:
            topic_orchestration.clear_pending_creation(target_id)
            _clear_failed_target_state(target_id)
        if not preserve_topic:
            await _cleanup_failed_topic(user_id, thread_id, chat_id, context)
        return True

    _mark_provisioning_uncertain(claim_id)
    topic_orchestration.clear_pending_creation(target_id or "")
    if failure_message:
        logger.warning("Provisioned target remains quarantined: %s", failure_message)
    return False


async def _create_and_bind_window(  # noqa: C901, PLR0912, PLR0915
    query: CallbackQuery,
    user_id: int,
    thread_id: int,
    cwd: str,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    agent_args: str = "",
    success_label: str = "Session started.",
    old_window_id: str = "",
    provider_name: str = "",
) -> bool:
    """Create a new tmux window, bind it, rename topic, forward pending text.

    ``provider_name`` overrides the provider inherited from ``old_window_id``.
    A resume pick needs it: the picker widens to every provider precisely when
    the old window has no provider name, so inheriting from it would resolve to
    the config default and launch that agent with another agent's resume args.

    Returns True on success, False on failure.
    """
    chat = query.message.chat if query.message else None
    raw_chat_id = getattr(chat, "id", None)
    chat_id = (
        raw_chat_id
        if isinstance(raw_chat_id, int)
        else thread_router.resolve_chat_id(user_id, thread_id)
    )
    claim_id: str | None = None
    try:
        provisioning_kwargs = {
            "thread_id": thread_id,
            "kind": "replacement" if old_window_id else "target_for_topic",
        }
        if old_window_id:
            provisioning_kwargs["previous_target_id"] = old_window_id
        claim = thread_router.begin_topic_provisioning(
            user_id,
            chat_id,
            **provisioning_kwargs,
        )
    except (TypeError, ValueError) as exc:
        await safe_edit(query, f"❌ {exc}", reply_markup=None)
        await query.answer("Failed")
        return False
    claim_id = claim.claim_id
    # Persist the exact topic claim before the first multiplexer read.
    try:
        session_manager.flush_state()
    except BaseException:  # noqa: BLE001
        _abort_provisioning_claim(claim_id)
        raise

    # Every Fresh / Continue / Resume route converges here, and the banner
    # offering them may have been drawn from an ambiguous lookup minutes ago.
    # Keep the old binding in place until a replacement has committed.
    try:
        refusal = await _stale_recovery_offer(old_window_id)
    except BaseException:  # noqa: BLE001
        # The replacement request has not started, so release only its claim;
        # the previous binding remains untouched for retry/recovery.
        _abort_provisioning_claim(claim_id)
        raise
    if refusal is not None:
        await _finish_failed_provisioning(
            claim_id=claim_id,
            target_id=None,
            user_id=user_id,
            thread_id=thread_id,
            chat_id=chat_id,
            context=context,
            failure_message=refusal,
            preserve_topic=bool(old_window_id),
            target_confirmed_absent=True,
        )
        await safe_edit(query, refusal, reply_markup=None)
        return False

    if old_window_id:
        old_view = window_query.view_window(old_window_id)
        provider = get_provider_for_window(
            old_window_id,
            provider_name=provider_name
            or (old_view.provider_name if old_view else None),
        )
        approval_mode = old_view.approval_mode if old_view else "normal"
    elif provider_name:
        provider = get_provider_for_window("", provider_name=provider_name)
        approval_mode = "normal"
    else:
        provider = get_provider()
        approval_mode = "normal"
    launch_command = resolve_launch_command(
        provider.capabilities.name, approval_mode=approval_mode
    )

    creation_was_cancelled = False
    try:
        creation, creation_was_cancelled = await _await_creation_result(
            tmux_manager.create_window(
                cwd, agent_args=agent_args, launch_command=launch_command
            )
        )
    except BaseException:  # noqa: BLE001
        _mark_provisioning_uncertain(claim_id)
        raise
    success, message, created_wname, created_wid = creation
    if success and not created_wid:
        success = False
        message = "Backend returned no target ID"
    # Lazy: topic orchestration imports polling and topic handlers.
    from ..topics import topic_orchestration

    if not success:
        if created_wid:
            try:
                thread_router.attach_provisioning_target(claim_id, created_wid)
                session_manager.flush_state()
                topic_orchestration.register_pending_creation(created_wid)
            except BaseException:  # noqa: BLE001
                _mark_provisioning_uncertain(claim_id)
                raise
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=created_wid,
                user_id=user_id,
                thread_id=thread_id,
                chat_id=chat_id,
                context=context,
                failure_message=message,
                preserve_topic=bool(old_window_id),
            )
        elif _is_definitive_create_failure(message):
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=None,
                user_id=user_id,
                thread_id=thread_id,
                chat_id=chat_id,
                context=context,
                failure_message=message,
                preserve_topic=bool(old_window_id),
            )
        else:
            _mark_provisioning_uncertain(claim_id)
        await safe_edit(query, f"❌ {message}")
        _clear_recovery_state(context.user_data)
        await query.answer("Failed")
        if creation_was_cancelled:
            raise asyncio.CancelledError
        return False

    try:
        thread_router.attach_provisioning_target(claim_id, created_wid)
        session_manager.flush_state()
    except BaseException:  # noqa: BLE001
        _mark_provisioning_uncertain(claim_id)
        raise

    topic_orchestration.register_pending_creation(created_wid)
    if creation_was_cancelled:
        await _finish_failed_provisioning(
            claim_id=claim_id,
            target_id=created_wid,
            user_id=user_id,
            thread_id=thread_id,
            chat_id=chat_id,
            context=context,
            failure_message="creation cancelled",
            preserve_topic=bool(old_window_id),
        )
        raise asyncio.CancelledError

    map_entry_found = True
    if provider.capabilities.supports_hook:
        try:
            map_entry_found = await session_map_sync.wait_for_session_map_entry(
                created_wid,
                timeout=5.0,
                resolve_window_id=window_query.resolve_window_alias,
            )
        except BaseException as exc:  # noqa: BLE001
            created_wid = _follow_provisioning_supersession(
                created_wid, claim_id=claim_id
            )
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=created_wid,
                user_id=user_id,
                thread_id=thread_id,
                chat_id=chat_id,
                context=context,
                failure_message=str(exc),
                preserve_topic=bool(old_window_id),
            )
            raise
    created_wid = _follow_provisioning_supersession(created_wid, claim_id=claim_id)

    if not map_entry_found:
        # A hook-registration timeout does not prove that the new terminal is
        # dead. Re-read presence before deciding whether the target may be
        # aborted; live or unknown targets remain quarantined for recovery.
        # Lazy: reconciliation imports the active multiplexer backend.
        from ...multiplexer.reconciliation import window_presence

        try:
            presence = await window_presence(created_wid, tmux_manager)
        except BaseException as exc:  # noqa: BLE001
            _mark_provisioning_uncertain(claim_id)
            topic_orchestration.clear_pending_creation(created_wid)
            if isinstance(exc, asyncio.CancelledError):
                raise
            message = (
                "Could not verify the new session; ccgram kept it quarantined "
                "for recovery."
            )
            await safe_edit(query, f"⚠ {message}")
            await query.answer("Still starting")
            return False
        if presence is False:
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=created_wid,
                user_id=user_id,
                thread_id=thread_id,
                chat_id=chat_id,
                context=context,
                failure_message="session target is gone",
                preserve_topic=bool(old_window_id),
                target_confirmed_absent=True,
            )
            message = "Session did not register with ccgram and is gone"
            await safe_edit(query, f"❌ {message}")
            _clear_recovery_state(context.user_data)
            await query.answer("Failed")
            return False

        _mark_provisioning_uncertain(claim_id)
        topic_orchestration.clear_pending_creation(created_wid)
        message = (
            "Session is still starting; ccgram kept it quarantined for recovery. "
            "Try again after it finishes registering."
        )
        await safe_edit(query, f"⚠ {message}")
        await query.answer("Still starting")
        return False

    session_manager.set_window_origin(created_wid, CCGRAM_CREATED_WINDOW_ORIGIN)
    session_manager.set_window_provider(created_wid, provider.capabilities.name)
    session_manager.set_window_approval_mode(created_wid, approval_mode)

    try:
        committed = thread_router.commit_topic_provisioning(
            claim_id,
            window_name=created_wname,
        )
        session_manager.flush_state()
    except BaseException:  # noqa: BLE001
        _mark_provisioning_uncertain(claim_id)
        topic_orchestration.clear_pending_creation(created_wid)
        raise
    if not committed:
        _mark_provisioning_uncertain(claim_id)
        topic_orchestration.clear_pending_creation(created_wid)
        message = "Session could not be bound to this topic"
        await safe_edit(query, f"❌ {message}")
        _clear_recovery_state(context.user_data)
        await query.answer("Failed")
        return False

    # The replacement is committed. The old binding was kept intact until
    # this point, so every failed path above leaves it usable.
    topic_orchestration.clear_pending_creation(created_wid)
    # Lazy: polling_state → recovery_banner via callback_registry side effects.
    from ..polling.polling_state import lifecycle_strategy

    lifecycle_strategy.clear_dead_notification(user_id, thread_id)

    client = PTBTelegramClient(context.bot)
    try:
        await client.edit_forum_topic(
            chat_id=thread_router.resolve_chat_id(user_id, thread_id),
            message_thread_id=thread_id,
            name=format_topic_name_for_mode(created_wname, approval_mode),
        )
    except TelegramError as e:
        logger.debug("Failed to rename topic: %s", e)

    await safe_edit(query, f"✅ {message}\n\n{success_label}")

    pending_text = (
        context.user_data.get(PENDING_THREAD_TEXT) if context.user_data else None
    )
    _clear_recovery_state(context.user_data)
    if pending_text:
        send_ok, send_msg = await send_telegram_to_window(
            user_id, created_wid, thread_id, pending_text, chat.id if chat else None
        )
        if not send_ok:
            logger.warning(
                "Failed to forward pending text to window %s (user %s): %s",
                created_wid,
                user_id,
                send_msg,
            )
            await safe_send(
                client,
                thread_router.resolve_chat_id(user_id, thread_id),
                f"❌ Failed to send pending message: {send_msg}",
                message_thread_id=thread_id,
            )
    await query.answer("Created")
    return True


def _cwd_for_window(window_id: str) -> str:
    """Return the bound cwd for ``window_id`` or empty string."""
    view = window_query.view_window(window_id)
    return view.cwd if view else ""


async def _recovery_cwd_or_report(
    query: CallbackQuery,
    window_id: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> str | None:
    """Return the recovery cwd, or report which of two failures happened.

    Fresh/Continue/Resume all need the directory, and both ways of not having
    it used to share one message that claimed the directory was gone (#176).
    A missing window state means the directory is *unknown*, which is a
    different problem with a different way out: Browse still works without
    state, so offer it rather than ending the flow on a false statement about
    the filesystem.
    """
    cwd = _cwd_for_window(window_id)
    if not cwd:
        await safe_edit(
            query,
            "⚠ This topic's session state is gone, so its folder is unknown.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "\U0001f5c2 Browse other projects",
                            callback_data=compact_callback_data(
                                CB_RECOVERY_BROWSE,
                                f"{CB_RECOVERY_BROWSE}{window_id}",
                                window_id,
                            ),
                        )
                    ]
                ]
            ),
        )
        # Deliberately not cleared: Browse re-validates against this state.
        await query.answer("State gone")
        return None
    if not Path(cwd).is_dir():
        await safe_edit(query, "\u274c Directory no longer exists.")
        _clear_recovery_state(context.user_data)
        await query.answer("Project gone")
        return None
    return cwd


async def _handle_back(
    query: CallbackQuery,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_BACK: return to the recovery options menu."""
    window_id = data[len(CB_RECOVERY_BACK) :]
    validated = _validate_recovery_state(window_id, update, context)
    if validated is None:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return
    thread_id, _ = validated
    if query.message is None or query.message.chat is None:
        await query.answer("Chat unavailable", show_alert=True)
        return
    chat_id = query.message.chat.id
    display = thread_router.get_display_name(window_id) or window_id
    banner = RecoveryBanner(
        chat_id=chat_id,
        thread_id=thread_id,
        window_id=window_id,
        mode="restore",
        provider=window_query.get_window_provider(window_id),
        display=display,
        cwd=_cwd_for_window(window_id),
    )
    text, kb = render_banner(banner)
    await safe_edit(query, text, reply_markup=kb)
    await query.answer()


async def _handle_fresh(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_FRESH: create fresh session in same directory."""
    old_wid = data[len(CB_RECOVERY_FRESH) :]
    validated = _validate_recovery_state(old_wid, update, context)
    if validated is None:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    thread_id, _ = validated
    cwd = await _recovery_cwd_or_report(query, old_wid, context)
    if cwd is None:
        return

    await _create_and_bind_window(
        query,
        user_id,
        thread_id,
        cwd,
        context,
        success_label="Fresh session started.",
        old_window_id=old_wid,
    )


async def _handle_continue(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_CONTINUE: resume most recent session via --continue.

    If there are no sessions on disk for ``cwd``, ``--continue`` would fail
    silently inside the agent. Surface the empty-state UI instead so the
    user can pick another project or start fresh.
    """
    old_wid = data[len(CB_RECOVERY_CONTINUE) :]
    validated = _validate_recovery_state(old_wid, update, context)
    if validated is None:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    thread_id, _ = validated
    cwd = await _recovery_cwd_or_report(query, old_wid, context)
    if cwd is None:
        return

    provider_name = window_query.get_window_provider(old_wid)
    provider = get_provider_for_window(old_wid, provider_name=provider_name)
    # Probe with the resolved name, not the raw three-valued one. Resume may
    # widen an unknown provider to every picker-capable one because each entry
    # carries its own provider to the relaunch; Continue cannot, because it
    # launches exactly this ``provider``. Probing wider would find another
    # agent's sessions, skip the empty state, and run `<default> --continue`
    # into a folder it has nothing to continue — the silent failure the empty
    # state exists to prevent.
    if provider.capabilities.supports_resume_picker and not await asyncio.to_thread(
        scan_sessions_for_cwd,
        cwd,
        provider.capabilities.name,
    ):
        await _send_empty_state(query, old_wid, cwd)
        return

    launch_args = provider.make_launch_args(use_continue=True)
    await _create_and_bind_window(
        query,
        user_id,
        thread_id,
        cwd,
        context,
        agent_args=launch_args,
        success_label="Continuing previous session.",
        old_window_id=old_wid,
    )


async def _handle_resume(
    query: CallbackQuery,
    _user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_RESUME: show session picker for --resume."""
    old_wid = data[len(CB_RECOVERY_RESUME) :]
    validated = _validate_recovery_state(old_wid, update, context)
    if validated is None:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    cwd = await _recovery_cwd_or_report(query, old_wid, context)
    if cwd is None:
        return

    provider_name = window_query.get_window_provider(old_wid)
    sessions = await asyncio.to_thread(
        scan_sessions_for_cwd,
        cwd,
        provider_name,
    )
    if not sessions:
        await _send_empty_state(query, old_wid, cwd)
        return

    if context.user_data is not None:
        context.user_data[RECOVERY_SESSIONS] = [
            {
                "session_id": s.session_id,
                "summary": s.summary,
                "mtime": s.mtime,
                "provider_name": s.provider_name,
            }
            for s in sessions
        ]

    keyboard = _build_resume_picker_keyboard(sessions, old_wid)
    await safe_edit(
        query,
        f"⏪ Select a session to resume:\n(`{cwd}`)",
        reply_markup=keyboard,
    )
    await query.answer()


async def _send_empty_state(
    query: CallbackQuery,
    window_id: str,
    cwd: str,
) -> None:
    """Edit the recovery message to the no-sessions empty-state UI.

    Replaces the legacy ``query.answer("No sessions ...", show_alert=True)``
    toast with an inline keyboard so the user has explicit next steps
    instead of being trapped on a dismissable alert.
    """

    keyboard = _build_empty_resume_keyboard(window_id)
    await safe_edit(
        query,
        f"⚠ No sessions in this folder yet.\n(`{cwd}`)",
        reply_markup=keyboard,
    )
    await query.answer()


async def _handle_browse(
    query: CallbackQuery,
    _user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_BROWSE: switch to the cross-project resume picker.

    The user explicitly chose to look outside the bound cwd, so the pending
    text — which targeted the original project — is dropped before
    delegating to the /resume cross-project flow.
    """

    # Lazy: sibling cycle — resume_command imports from this package.
    from ..user_state import RESUME_SESSIONS

    # Lazy: recovery_banner ↔ resume_command cycle through the picker
    from .resume_command import _build_resume_keyboard, scan_all_sessions

    old_wid = data[len(CB_RECOVERY_BROWSE) :]
    validated = _validate_recovery_state(old_wid, update, context)
    if validated is None:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    provider_name = window_query.get_window_provider(old_wid)
    sessions = await asyncio.to_thread(scan_all_sessions, provider_name)
    if not sessions:
        await safe_edit(query, "⚠ No past sessions found in any project.")
        _clear_recovery_state(context.user_data)
        await query.answer("Nothing to resume")
        return

    if context.user_data is not None:
        context.user_data.pop(PENDING_THREAD_TEXT, None)
        context.user_data.pop(RECOVERY_SESSIONS, None)
        context.user_data[RESUME_SESSIONS] = [
            {
                "session_id": s.session_id,
                "summary": s.summary,
                "cwd": s.cwd,
                "mtime": s.mtime,
                "msg_count": s.msg_count,
                "provider_name": s.provider_name,
            }
            for s in sessions
        ]

    keyboard = _build_resume_keyboard(
        context.user_data[RESUME_SESSIONS] if context.user_data else [], page=0
    )
    await safe_edit(query, "⏪ Select a session to resume:", reply_markup=keyboard)
    await query.answer()


async def _handle_cancel(
    query: CallbackQuery,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_RECOVERY_CANCEL: cancel recovery."""
    # Lazy: callback_helpers ↔ recovery cycle

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    pending_tid = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )
    if pending_tid is not None and thread_id != pending_tid:
        await query.answer("Stale recovery (topic mismatch)", show_alert=True)
        return

    _clear_recovery_state(context.user_data)
    await safe_edit(query, "Cancelled. Send a message to try again.")
    await query.answer("Cancelled")
