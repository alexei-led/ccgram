"""Window creation and topic-binding service for the directory-browser flow.

Extracts the launch sequence from directory_callbacks._create_window_and_bind
into a self-contained service module.  Callers build a ``WindowLaunchRequest``
and call ``launch_window``; the result is a ``WindowLaunchResult``.

Creation is protected by a transaction guard until the durable target is
registered. This keeps the SessionMonitor from adopting it before binding.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from telegram import CallbackQuery, Message
from telegram.error import TelegramError

from ...agent_startup import answer_first_run_prompt, preseed_trust
from ...config import config
from ...providers import registry as provider_registry
from ...session import session_manager
from ...session_map import session_map_sync
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.base import canonical_window_id
from ..telegram_origin import send_telegram_to_window
from ...session_protocol.telegram import TelegramSessionAdapter, adapter_for
from ...session_protocol.models import ProtocolError
from ...user_preferences import user_preferences
from ... import window_query
from ...window_state_store import CCGRAM_CREATED_WINDOW_ORIGIN
from ..messaging_pipeline.message_sender import safe_edit, safe_send
from ..status.topic_emoji import format_topic_name_for_mode
from .directory_browser import clear_worktree_state, clear_workspace_state
from .topic_creation_draft import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKSPACE_ID,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_PATH,
    PENDING_WORKTREE_REPO,
)
from . import topic_orchestration

if TYPE_CHECKING:
    from telegram import CallbackQuery
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_YOLO_CREATION_GUARD_GRACE_S = 10.0

__all__ = [
    "WindowLaunchRequest",
    "WindowLaunchResult",
    "launch_window",
]


@dataclass
class WindowLaunchRequest:
    """Parameters for the window-creation + topic-binding step."""

    user_id: int
    thread_id: int | None
    provider_name: str
    cwd: str
    mode: Literal["normal", "yolo"]
    pending_text: str | None
    chat_id: int | None = None
    # Topic/window label. Pinned as the display name so a backend or agent
    # title change cannot rename the Telegram topic. None keeps the legacy
    # behaviour of using the backend's derived window name.
    label: str | None = None
    # Worktree metadata. The typed-plan path fills these in directly; the
    # directory-browser path leaves them None and keeps using the
    # PENDING_WORKTREE_* user_data keys set by its own steps.
    worktree_repo: str | None = None
    worktree_branch: str | None = None
    worktree_path: str | None = None
    # A typed plan owns its worktree decision: it must never inherit the
    # PENDING_WORKTREE_* keys of a half-finished browser flow, which would
    # launch this plan into another repository's checkout.
    ignore_pending_worktree: bool = False


@dataclass
class WindowLaunchResult:
    """Outcome of ``launch_window``."""

    success: bool
    window_id: str | None = None
    error_message: str | None = None


# ── helpers ──────────────────────────────────────────────────────────────────


def _cwd_within(cwd: str, worktree_path: str) -> bool:
    """True if *cwd* is the worktree root or nested inside it."""
    try:
        c = Path(cwd).resolve()
        w = Path(worktree_path).resolve()
    except OSError:
        return False
    return c == w or c.is_relative_to(w)


def _persist_worktree_state(
    window_id: str,
    cwd: str,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    worktree_path: str | None = None,
    worktree_branch: str | None = None,
) -> None:
    """Persist a pending worktree path/branch onto the new window state.

    Only persists when the window's *cwd* is the pending worktree path
    (or a subdirectory of it — the new topic may be rooted at a subdir
    of the fresh checkout) so a stale path from an earlier aborted
    attempt can't attach to an unrelated window. Always clears the
    worktree flow keys afterwards.
    """
    user_data = context.user_data
    if worktree_path is None:
        worktree_path = user_data.get(PENDING_WORKTREE_PATH) if user_data else None
    if worktree_branch is None:
        worktree_branch = user_data.get(PENDING_WORKTREE_BRANCH) if user_data else None
    if worktree_path and worktree_branch and _cwd_within(cwd, worktree_path):
        session_manager.set_window_worktree(window_id, worktree_path, worktree_branch)
    clear_worktree_state(user_data)


async def _abort_topic_creation(
    query: CallbackQuery | Message, message: str, context: ContextTypes.DEFAULT_TYPE
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
    clear_workspace_state(context.user_data)


async def _create_topic_window(
    selected_path: str,
    provider_name: str,
    approval_mode: Literal["normal", "yolo"],
    chosen_workspace_id: str | None,
    context: ContextTypes.DEFAULT_TYPE,
    request: "WindowLaunchRequest | None" = None,
) -> tuple[bool, str, str, str]:
    """Create a topic session through the shared session service."""
    ud = None if request and request.ignore_pending_worktree else context.user_data
    wt_repo = (
        request.worktree_repo
        if request and request.worktree_repo
        else (ud.get(PENDING_WORKTREE_REPO) if ud else None)
    )
    wt_branch = (
        request.worktree_branch
        if request and request.worktree_branch
        else (ud.get(PENDING_WORKTREE_BRANCH) if ud else None)
    )
    wt_path = (
        request.worktree_path
        if request and request.worktree_path
        else (ud.get(PENDING_WORKTREE_PATH) if ud else None)
    )
    # Seed the agent's own trust record so its first-run dialog never shows.
    # Offloaded: it reads/writes a JSON/TOML file on disk.
    await asyncio.to_thread(preseed_trust, provider_name, Path(selected_path))
    data: dict[str, Any] = {"provider": provider_name, "directory": selected_path}
    if chosen_workspace_id:
        data["workspaceId"] = chosen_workspace_id
    if tmux_manager.capabilities.native_worktrees and wt_repo and wt_branch and wt_path:
        # The native worktree API cannot pin a preselected workspace.  Creating
        # into an implicit workspace would violate the user's selection.
        if chosen_workspace_id:
            return (
                False,
                "Selected workspace cannot create a native worktree",
                "",
                "",
            )
        data["directory"] = wt_repo
        data["worktree"] = True
        data["branch"] = wt_branch
    try:
        created = await adapter_for(context.bot_data).create(
            data, approval_mode=approval_mode
        )
    except ProtocolError as exc:
        if exc.status < HTTPStatus.INTERNAL_SERVER_ERROR:
            return False, exc.message, "", ""
        raise
    return (
        True,
        f"Created topic target '{created.title}'",
        created.title,
        created.target_id,
    )


async def _wait_for_shell_ready(
    window_id: str, adapter: TelegramSessionAdapter, *, attempts: int = 5
) -> None:
    """Wait for a freshly created tmux window to show a shell prompt."""
    # Lazy: only needed inside the shell-detection branch
    import os

    # Lazy: providers package heavy bootstrap
    from ccgram.providers.shell import KNOWN_SHELLS

    for _ in range(attempts):
        try:
            target, _ = await adapter.resolve(window_id)
        except ProtocolError:
            target = None
        if target and target.foreground_argv:
            cmd = os.path.basename(target.foreground_argv[0]).lstrip("-")
            if cmd in KNOWN_SHELLS:
                return
        await asyncio.sleep(0.2)


async def agent_process_started(
    window_id: str,
    adapter: TelegramSessionAdapter,
    *,
    expected_provider: str | None = None,
) -> bool | None:
    """Inspect launch readiness once, without waiting.

    Return None if resolution fails or no foreground argv is reported.
    Return False for a shell or, with ``expected_provider``, an unavailable
    target or another provider. True means a non-shell process, also available
    and matching when ``expected_provider`` is supplied.
    """
    # Lazy: only launch/recovery flows need process inspection.
    import os

    # Lazy: providers package heavy bootstrap
    from ccgram.providers.shell import KNOWN_SHELLS

    try:
        target, _ = await adapter.resolve(window_id)
    except ProtocolError:
        return None
    if not target.foreground_argv:
        return None
    if expected_provider is not None and (
        not target.available or target.provider != expected_provider
    ):
        return False
    cmd = os.path.basename(target.foreground_argv[0]).lstrip("-")
    return cmd not in KNOWN_SHELLS


async def _wait_for_agent_process(
    window_id: str,
    adapter: TelegramSessionAdapter,
    *,
    attempts: int = 50,
    expected_provider: str | None = None,
) -> bool:
    """Wait for a non-shell foreground, matching the provider when specified.

    SessionStart can precede terminal readiness. A timeout leaves the launch
    quarantined rather than sending its first prompt to an unknown process.
    """
    for _ in range(attempts):
        if await agent_process_started(
            window_id, adapter, expected_provider=expected_provider
        ):
            return True
        await asyncio.sleep(0.2)
    return False


async def _accept_yolo_confirmation(
    window_id: str,
    adapter: TelegramSessionAdapter,
    *,
    timeout: float | None = None,
) -> bool:
    """Detect and accept Claude Code's bypass permissions confirmation prompt.

    When launched with --dangerously-skip-permissions, Claude Code shows a
    TUI confirmation where "No, exit" is the default selection. Sends
    Down+Enter to select the "Yes" option so the session can start.
    """
    loop = asyncio.get_running_loop()
    timeout = config.yolo_confirmation_timeout if timeout is None else timeout
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        text = await tmux_manager.capture_pane(window_id)
        lower = text.lower() if text else ""
        if "bypass permissions" in lower and (
            "no, exit" in lower or "i accept" in lower
        ):
            await asyncio.sleep(0.3)
            success, _ = await adapter.send_keys(window_id, ["Down", "Enter"])
            if not success:
                return False
            logger.info("Accepted bypass permissions prompt for window %s", window_id)
            return True
        if "⏵⏵ bypass permissions" in lower:
            return False
        await asyncio.sleep(0.5)
    logger.warning(
        "Bypass permissions prompt not detected within %.0fs for window %s",
        timeout,
        window_id,
    )
    return False


def _follow_supersession(window_id: str, *, claim_id: str | None = None) -> str:
    """Re-point creation at the id its window answers to now.

    Reconciliation runs on the monitor cycle, so a backend that firms up
    window identity late (Herdr, once the agent session is published) can
    rename the target while creation is still waiting on the hook. Every
    step after the wait — releasing the guard, cleaning up a failure,
    reporting the window — has to act on the current id; acting on the one
    creation minted killed nothing and quarantined a healthy topic.

    Carries the creation guard across to the new id so the monitor cannot
    adopt it into a second topic in the window before the flow finishes.
    """
    current = window_query.resolve_window_alias(window_id)
    if current == window_id:
        return window_id
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
    logger.info("Creation target superseded: %s -> %s", window_id, current)
    return current


def _is_definitive_create_failure(message: str) -> bool:
    """Return whether a failed create response proves no target exists."""
    normalized = message.casefold()
    return any(
        marker in normalized
        for marker in (
            "directory does not exist",
            "not a directory",
            "selected workspace cannot create",
            "selected herdr workspace no longer exists",
            "requires a sessionful agent",
            "does not create worktrees natively",
            "does not create worktrees",
        )
    )


async def _await_creation_result(creation):  # noqa: ANN001
    """Wait for creation while retaining a result delivered with cancellation."""
    task = asyncio.create_task(creation)
    try:
        return await asyncio.shield(task), False
    except asyncio.CancelledError:
        # ``shield`` keeps the backend request alive.  Await it so a target
        # created just before cancellation is still available for cleanup.
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


async def _cleanup_failed_topic(
    user_id: int,
    thread_id: int | None,
    chat_id: int | None,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Delete a failed exact topic after its target is confirmed absent."""
    if thread_id is None or chat_id is None:
        return

    # The storage abort records an unbound exact topic as a retired cleanup
    # candidate. Use the shared deletion flow so state is cleared before the
    # Telegram delete and failed deletes remain retryable.
    # Lazy: cleanup → shell/polling → topics cycle.
    from ...telegram_client import PTBTelegramClient

    # Lazy: cleanup → shell/polling → topics cycle.
    from ..cleanup import clear_topic_state

    # Lazy: topic deletion imports cleanup and polling state.
    from .topic_deletion import cleanup_retired_topic

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

    client = PTBTelegramClient(context.bot)

    async def clear_state_before_delete() -> None:
        await clear_topic_state(
            user_id,
            thread_id,
            client,
            context.user_data,
            window_id=retired.target_id,
            chat_id=chat_id,
            window_dead=True,
        )

    await cleanup_retired_topic(
        client,
        retired,
        router=thread_router,
        before_delete=clear_state_before_delete,
    )


async def _finish_failed_provisioning(  # noqa: C901
    *,
    claim_id: str | None,
    target_id: str | None,
    user_id: int,
    thread_id: int | None,
    chat_id: int | None,
    context: ContextTypes.DEFAULT_TYPE,
    failure_message: str = "",
    target_already_absent: bool = False,
) -> bool:
    """Abort a proven-absent target or quarantine an uncertain target."""
    if claim_id is None:
        if target_id is None:
            return False
        if target_already_absent:
            topic_orchestration.clear_pending_creation(target_id)
            return True
        try:
            target_confirmed_absent, _ = await adapter_for(context.bot_data).close(
                target_id
            )
        except BaseException:  # noqa: BLE001
            return False
        if target_confirmed_absent:
            topic_orchestration.clear_pending_creation(target_id)
        return bool(target_confirmed_absent)

    if target_id is None and not _is_definitive_create_failure(failure_message):
        _mark_provisioning_uncertain(claim_id)
        return False

    target_confirmed_absent = target_id is None or target_already_absent
    if target_id is not None and not target_already_absent:
        try:
            target_confirmed_absent, _ = await adapter_for(context.bot_data).close(
                target_id
            )
        except BaseException:  # noqa: BLE001
            _mark_provisioning_uncertain(claim_id)
            return False

    if target_confirmed_absent:
        _abort_provisioning_claim(claim_id)
        if target_id is not None:
            topic_orchestration.clear_pending_creation(target_id)
        await _cleanup_failed_topic(user_id, thread_id, chat_id, context)
        return True

    _mark_provisioning_uncertain(claim_id)
    if target_id is not None:
        topic_orchestration.clear_pending_creation(target_id)
    if failure_message:
        logger.warning("Provisioned target remains quarantined: %s", failure_message)
    return False


async def _await_agent_ready(
    window_id: str,
    provider_name: str,
    provider: Any,
    provider_caps: Any,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """Wait until the created target reports a live, matching agent foreground."""
    adapter = adapter_for(context.bot_data)
    if provider.capabilities.supports_hook:
        if not await session_map_sync.wait_for_session_map_entry(
            window_id, resolve_window_id=window_query.resolve_window_alias
        ):
            return False
        # SessionStart can arrive before the terminal reports the CLI.
        return await _wait_for_agent_process(
            window_id, adapter, expected_provider=provider_name
        )
    if provider_caps.chat_first_command_path:
        return True
    # Only a pane that actually left its launch shell may be bound; a timed-out
    # wait reuses the quarantine path instead of reporting success for a window
    # the next poll would kill.
    return await _wait_for_agent_process(
        window_id, adapter, expected_provider=provider_name
    )


# ── main entry point ──────────────────────────────────────────────────────────


async def launch_window(  # noqa: C901, PLR0911, PLR0912, PLR0915
    query: CallbackQuery | Message,
    context: ContextTypes.DEFAULT_TYPE,
    request: WindowLaunchRequest,
) -> WindowLaunchResult:
    """Create a tmux window, bind to the pending topic, and forward pending text.

    Shared by the typed-plan entry (which passes the reply Message it just
    sent), _handle_provider_select (provider picker) and the legacy
    _handle_mode_select callback from keyboards sent by older versions.

    CRITICAL (MC-2967): creation starts inside a global transaction guard,
    then its durable target is registered before the transaction releases. This
    prevents the SessionMonitor from auto-creating a topic while Herdr publishes
    a session identity or before ``bind_thread`` runs below.
    """
    user_id = request.user_id
    pending_thread_id = request.thread_id
    selected_path = request.cwd
    provider_name = request.provider_name
    approval_mode = request.mode

    # A CallbackQuery carries the message in .message; a Message is its own.
    # Duck-typed (Any) so test doubles and both real types behave alike.
    query_message: Any = getattr(query, "message", None) or query
    chat = query_message.chat if query_message else None
    raw_chat_id = (
        request.chat_id if request.chat_id is not None else getattr(chat, "id", None)
    )
    chat_id = raw_chat_id if isinstance(raw_chat_id, int) else None

    claim_id: str | None = None
    if pending_thread_id is not None and chat_id is not None:
        try:
            claim = thread_router.begin_topic_provisioning(
                user_id,
                chat_id,
                thread_id=pending_thread_id,
                kind="target_for_topic",
            )
        except (TypeError, ValueError) as exc:
            message = str(exc)
            await _abort_topic_creation(query, message, context)
            return WindowLaunchResult(success=False, error_message=message)
        claim_id = claim.claim_id
        # The exact topic claim must reach disk before the first backend await.
        try:
            session_manager.flush_state()
        except BaseException:  # noqa: BLE001
            _abort_provisioning_claim(claim_id)
            raise

    chosen_workspace_id: str | None = (
        context.user_data.get(PENDING_WORKSPACE_ID) if context.user_data else None
    ) or None

    creation_was_cancelled = False
    with topic_orchestration.pending_creation_transaction():
        try:
            creation, creation_was_cancelled = await _await_creation_result(
                _create_topic_window(
                    selected_path,
                    provider_name,
                    approval_mode,
                    chosen_workspace_id,
                    context,
                    request,
                )
            )
        except BaseException:  # noqa: BLE001
            _mark_provisioning_uncertain(claim_id)
            raise

        success, message, created_wname, created_wid = creation
        if success and not created_wid:
            success = False
            message = "Backend returned no target ID"

        if success:
            if claim_id is not None:
                try:
                    thread_router.attach_provisioning_target(claim_id, created_wid)
                    session_manager.flush_state()
                except BaseException:  # noqa: BLE001
                    _mark_provisioning_uncertain(claim_id)
                    raise
            if approval_mode == "yolo":
                topic_orchestration.register_pending_creation(
                    created_wid,
                    ttl_s=(
                        config.yolo_confirmation_timeout + _YOLO_CREATION_GUARD_GRACE_S
                    ),
                )
            else:
                topic_orchestration.register_pending_creation(created_wid)

        if not success:
            if created_wid:
                try:
                    if claim_id is not None:
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
                    thread_id=pending_thread_id,
                    chat_id=chat_id,
                    context=context,
                    failure_message=message,
                )
            elif claim_id is not None:
                if _is_definitive_create_failure(message):
                    await _finish_failed_provisioning(
                        claim_id=claim_id,
                        target_id=None,
                        user_id=user_id,
                        thread_id=pending_thread_id,
                        chat_id=chat_id,
                        context=context,
                        failure_message=message,
                    )
                else:
                    _mark_provisioning_uncertain(claim_id)
            await _abort_topic_creation(query, message, context)
            if creation_was_cancelled:
                raise asyncio.CancelledError
            return WindowLaunchResult(success=False, error_message=message)

        if creation_was_cancelled:
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=created_wid,
                user_id=user_id,
                thread_id=pending_thread_id,
                chat_id=chat_id,
                context=context,
                failure_message="creation cancelled",
            )
            raise asyncio.CancelledError

    if not success:
        await _abort_topic_creation(query, message, context)
        return WindowLaunchResult(success=False, error_message=message)

    user_preferences.update_user_mru(user_id, selected_path)
    session_manager.set_window_origin(created_wid, CCGRAM_CREATED_WINDOW_ORIGIN)
    session_manager.set_window_cwd(created_wid, selected_path)
    session_manager.set_window_provider(created_wid, provider_name)
    session_manager.set_window_approval_mode(created_wid, approval_mode)
    _persist_worktree_state(
        created_wid,
        selected_path,
        context,
        worktree_path=request.worktree_path,
        worktree_branch=request.worktree_branch,
    )
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
    try:
        await tmux_manager.stamp_pane_title(created_wid, provider_name)
    except BaseException as exc:  # noqa: BLE001
        created_wid = _follow_supersession(created_wid, claim_id=claim_id)
        await _finish_failed_provisioning(
            claim_id=claim_id,
            target_id=created_wid,
            user_id=user_id,
            thread_id=pending_thread_id,
            chat_id=chat_id,
            context=context,
            failure_message=str(exc),
        )
        raise

    provider_caps = provider_registry.get(provider_name).capabilities
    if provider_caps.chat_first_command_path:
        # Lazy: shell ↔ topics cycle via window_callbacks adoption flow.
        from ..shell.shell_prompt_orchestrator import ensure_setup

        try:
            await _wait_for_shell_ready(created_wid, adapter_for(context.bot_data))
            await ensure_setup(
                created_wid, "auto", adapter=adapter_for(context.bot_data)
            )
        except BaseException as exc:  # noqa: BLE001
            created_wid = _follow_supersession(created_wid, claim_id=claim_id)
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=created_wid,
                user_id=user_id,
                thread_id=pending_thread_id,
                chat_id=chat_id,
                context=context,
                failure_message=str(exc),
            )
            raise

    provider = provider_registry.get(provider_name)
    yolo_confirm = (
        approval_mode == "yolo" and provider.capabilities.has_yolo_confirmation
    )
    try:
        if yolo_confirm:
            await _accept_yolo_confirmation(created_wid, adapter_for(context.bot_data))

        map_entry_found = await _await_agent_ready(
            created_wid, provider_name, provider, provider_caps, context
        )
        if not map_entry_found:
            # A declared first-run screen (folder trust, hook trust) blocks the
            # agent before it can report readiness. Answer it and wait once
            # more, instead of quarantining a session the user can see running.
            answered = await answer_first_run_prompt(
                created_wid,
                provider_name,
                adapter=adapter_for(context.bot_data),
                poll=False,
            )
            if answered:
                if yolo_confirm:
                    await _accept_yolo_confirmation(
                        created_wid, adapter_for(context.bot_data)
                    )
                map_entry_found = await _await_agent_ready(
                    created_wid, provider_name, provider, provider_caps, context
                )
    except BaseException as exc:  # noqa: BLE001
        created_wid = _follow_supersession(created_wid, claim_id=claim_id)
        await _finish_failed_provisioning(
            claim_id=claim_id,
            target_id=created_wid,
            user_id=user_id,
            thread_id=pending_thread_id,
            chat_id=chat_id,
            context=context,
            failure_message=str(exc),
        )
        raise

    created_wid = _follow_supersession(created_wid, claim_id=claim_id)

    if not map_entry_found:
        # Registration/readiness timeout does not prove the terminal is dead.
        # Re-read presence before deciding whether the target may be
        # aborted; live or unknown targets remain quarantined for recovery.
        # Lazy: reconciliation imports the active multiplexer backend.
        from ...multiplexer.reconciliation import window_presence

        try:
            presence = await window_presence(created_wid, tmux_manager)
        except BaseException as exc:  # noqa: BLE001
            _mark_provisioning_uncertain(claim_id)
            if claim_id is not None:
                topic_orchestration.clear_pending_creation(created_wid)
            if isinstance(exc, asyncio.CancelledError):
                raise
            message = (
                "Could not verify the new session; ccgram kept it quarantined "
                "for recovery."
            )
            await safe_edit(query, f"⚠ {message}")
            return WindowLaunchResult(success=False, error_message=message)
        if presence is False:
            await _finish_failed_provisioning(
                claim_id=claim_id,
                target_id=created_wid,
                user_id=user_id,
                thread_id=pending_thread_id,
                chat_id=chat_id,
                context=context,
                failure_message="session target is gone",
                target_already_absent=True,
            )
            message = "Session target is gone"
            await _abort_topic_creation(query, message, context)
            return WindowLaunchResult(success=False, error_message=message)

        _mark_provisioning_uncertain(claim_id)
        if claim_id is not None:
            topic_orchestration.clear_pending_creation(created_wid)
        message = (
            "Session is still starting or its foreground is unverified; "
            "ccgram kept it quarantined for recovery. Try again once the selected "
            "provider is running. Custom launch scripts must exec the provider CLI."
        )
        # Retain the exact topic and first-message state while registration or
        # matching foreground evidence is still unavailable.
        await safe_edit(query, f"⚠ {message}")
        return WindowLaunchResult(success=False, error_message=message)

    if claim_id is not None and pending_thread_id is not None:
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
            await _abort_topic_creation(query, message, context)
            return WindowLaunchResult(success=False, error_message=message)
    elif pending_thread_id is not None:
        # A malformed callback without an observable chat cannot carry a
        # durable exact-topic claim; retain the legacy synchronous fallback.
        thread_router.bind_thread(
            user_id,
            pending_thread_id,
            created_wid,
            window_name=created_wname,
            chat_id=chat_id,
        )
        session_manager.flush_state()

    # The provider-specific creation checks passed. Release the guard also for
    # the no-thread path, which has no topic bind to release it.
    topic_orchestration.clear_pending_creation(created_wid)
    if pending_thread_id is None:
        await safe_edit(query, f"✅ {message}")
        return WindowLaunchResult(success=True, window_id=created_wid)

    # Pin the label before renaming: pinned names survive backend/agent title
    # changes, so the topic stops drifting to "<name> <target id>".
    topic_label = request.label or created_wname
    thread_router.set_display_name(created_wid, topic_label, pin=True)
    chat_id = thread_router.resolve_chat_id(user_id, pending_thread_id)
    try:
        await context.bot.edit_forum_topic(
            chat_id=chat_id,
            message_thread_id=pending_thread_id,
            name=format_topic_name_for_mode(topic_label, approval_mode),
        )
    except TelegramError as e:
        logger.debug("Failed to rename topic: %s", e)

    await safe_edit(
        query,
        f"✅ {message}\n\nBound to this topic. Send messages here.",
    )

    pending_text = request.pending_text
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
                chat_id=chat.id if chat else None,
                adapter=adapter_for(context.bot_data),
            )
        else:
            send_ok, send_msg = await send_telegram_to_window(
                user_id,
                created_wid,
                pending_thread_id,
                pending_text,
                chat.id if chat else None,
                adapter=adapter_for(context.bot_data),
            )
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
    return WindowLaunchResult(success=True, window_id=created_wid)
