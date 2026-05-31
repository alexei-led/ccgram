"""Transcript discovery for hookless providers.

Discovers and registers transcripts for providers without hook support
(Codex, Gemini). Also handles provider auto-detection from pane process
and shell ↔ agent transitions.

Key components:
  - discover_and_register_transcript: main discovery function called per topic
  - _detect_and_apply_provider: provider auto-detection from running process
  - _find_and_register_transcript: transcript search for hookless providers
"""

import asyncio
from typing import TYPE_CHECKING

import structlog

from ...config import config
from ...providers import (
    detect_provider_from_pane,
    detect_provider_from_runtime,
    detect_provider_from_transcript_path,
    get_provider_for_window,
    should_probe_pane_title_for_provider_detection,
)
from ...session import session_manager
from ...session_map import session_map_sync
from ...telegram_client import TelegramClient
from ...tmux_manager import tmux_manager
from ...window_state_ports import identity_state

if TYPE_CHECKING:
    from ...providers.base import AgentProvider
    from ...tmux_manager import TmuxWindow

logger = structlog.get_logger()


def _session_id_already_bound(session_id: str, window_id: str) -> bool:
    """Return True if another currently bound window already uses ``session_id``."""
    # Lazy: thread_router may not be installed in some test paths; fail open
    # if it isn't available so discovery can still continue with this window.
    from ...thread_router import thread_router

    try:
        iterator = thread_router.iter_thread_bindings()
    except RuntimeError:
        return False

    for _user_id, _thread_id, bound_window_id in iterator:
        if bound_window_id == window_id:
            continue
        if identity_state.get_session_id(bound_window_id) == session_id:
            return True
    return False


async def _detect_and_apply_provider(
    window_id: str,
    identity: identity_state.IdentityProjection,
    w: "TmuxWindow",
    *,
    client: TelegramClient | None = None,
    chat_id: int = 0,
    thread_id: int = 0,
) -> None:
    """Detect provider from pane process and apply transitions."""
    if identity_state.is_provider_manually_overridden(window_id):
        return
    detected = await detect_provider_from_pane(
        w.pane_current_command, pane_tty=w.pane_tty, window_id=window_id
    )
    if not detected and should_probe_pane_title_for_provider_detection(
        w.pane_current_command
    ):
        pane_title = await tmux_manager.get_pane_title(window_id)
        detected = detect_provider_from_runtime(
            w.pane_current_command,
            pane_title=pane_title,
        )

    if detected and detected != identity.provider_name:
        old_provider = identity.provider_name
        session_manager.set_window_provider(window_id, detected, cwd=w.cwd or None)
        # Lazy: providers/__init__.py reaches back into transcript code
        # via provider format modules.
        from ...providers import get_provider_for_window

        new_caps = get_provider_for_window(window_id, detected)
        old_caps = (
            get_provider_for_window(window_id, old_provider) if old_provider else None
        )
        if new_caps and new_caps.capabilities.chat_first_command_path:
            identity_state.clear_transcript_path(window_id)
            # Lazy: shell.shell_prompt_orchestrator hits the recovery
            # subpackage's discovery code via send-keys callbacks.
            from ..shell.shell_prompt_orchestrator import ensure_setup

            await ensure_setup(
                window_id,
                "provider_switch",
                client=client,
                chat_id=chat_id,
                thread_id=thread_id,
            )
        elif old_caps and old_caps.capabilities.chat_first_command_path:
            # Lazy: same shell ↔ recovery cycle as above.
            from ..shell.shell_capture import clear_shell_monitor_state

            # Lazy: same shell ↔ recovery cycle as above.
            from ..shell.shell_prompt_orchestrator import (
                clear_state as clear_orchestrator,
            )

            clear_shell_monitor_state(window_id)
            clear_orchestrator(window_id)
    elif not detected and identity.transcript_path:
        inferred = detect_provider_from_transcript_path(str(identity.transcript_path))
        if inferred and inferred != identity.provider_name:
            session_manager.set_window_provider(window_id, inferred, cwd=w.cwd or None)


def _resolve_providers_to_try(
    window_id: str,
    identity: identity_state.IdentityProjection,
    w: "TmuxWindow | None",
) -> list[tuple[str, "AgentProvider"]] | None:
    """Determine which providers to probe for transcripts.

    Returns a list of (name, provider) pairs, or ``None`` to signal the
    caller should set up a shell provider.
    """
    # Lazy: hoisting forms polling/__init__ → window_tick →
    # recovery.transcript_discovery → polling_state partial-init
    # cycle (worker-order-dependent; verified during F6.2). polling_types
    # is leaf-level — Task 5 of Round 5 may hoist this once cycle test covers it.
    # Lazy: polling_types is leaf-pure; importing here at module load would touch the polling subpackage __init__
    from ..polling.polling_types import is_shell_prompt

    # Lazy: providers registry reaches back through transcripts
    from ...providers import registry

    if identity.provider_name:
        provider = get_provider_for_window(window_id, identity.provider_name)
        if provider.capabilities.chat_first_command_path:
            return []
        return [(provider.capabilities.name, provider)]

    if w and is_shell_prompt(w.pane_current_command):
        return None  # signals caller to set up shell

    return [
        (name, registry.get(name))
        for name in registry.provider_names()
        if not registry.get(name).capabilities.supports_hook and name != "shell"
    ]


async def _find_and_register_transcript(
    window_id: str,
    identity: identity_state.IdentityProjection,
    providers_to_try: list[tuple[str, "AgentProvider"]],
    pane_alive: bool,
) -> None:
    """Search for transcripts among candidate providers and register if found."""
    window_key = f"{config.tmux_session_name}:{window_id}"

    transcript_path_str = (
        str(identity.transcript_path) if identity.transcript_path else ""
    )

    for provider_name, provider in providers_to_try:
        max_age = 0 if pane_alive else None
        event = await asyncio.to_thread(
            provider.discover_transcript,
            identity.cwd,
            window_key,
            max_age=max_age,
        )
        if not event:
            continue

        if _session_id_already_bound(event.session_id, window_id):
            logger.debug(
                "Skipping discover result for window %s: session_id %s already bound",
                window_id,
                event.session_id,
            )
            continue

        if (
            identity.session_id == event.session_id
            and transcript_path_str == event.transcript_path
            and identity.provider_name == provider_name
        ):
            return

        session_map_sync.register_hookless_session(
            window_id=window_id,
            session_id=event.session_id,
            cwd=event.cwd,
            transcript_path=event.transcript_path,
            provider_name=provider_name,
        )
        await asyncio.to_thread(
            session_map_sync.write_hookless_session_map,
            window_id=window_id,
            session_id=event.session_id,
            cwd=event.cwd,
            transcript_path=event.transcript_path,
            provider_name=provider_name,
        )
        return


def _hook_already_resolved(
    window_id: str, identity: identity_state.IdentityProjection
) -> bool:
    """True when a hookful provider has already populated transcript_path."""
    if not identity.provider_name:
        return False
    provider = get_provider_for_window(window_id, identity.provider_name)
    return bool(provider.capabilities.supports_hook and identity.transcript_path)


async def _switch_to_shell(
    window_id: str,
    *,
    client: TelegramClient | None,
    chat_id: int,
    thread_id: int,
) -> None:
    """Provider-switch to shell and clear transcript bookkeeping."""
    session_manager.set_window_provider(window_id, "shell")
    identity_state.clear_transcript_path(window_id)
    # Lazy: same shell ↔ recovery cycle as _detect_and_apply_provider.
    from ..shell.shell_prompt_orchestrator import ensure_setup

    await ensure_setup(
        window_id,
        "provider_switch",
        client=client,
        chat_id=chat_id,
        thread_id=thread_id,
    )


async def discover_and_register_transcript(
    window_id: str,
    *,
    _window: "TmuxWindow | None" = None,
    client: TelegramClient | None = None,
    user_id: int = 0,
    thread_id: int = 0,
) -> None:
    """Discover and register transcript for hookless providers (Codex, Gemini).

    Also handles provider auto-detection from pane process name
    and shell ↔ agent transitions with prompt marker setup.
    """
    # Lazy: same polling/__init__ cycle as _resolve_providers_to_try.
    from ..polling.polling_types import is_shell_prompt

    # Lazy: thread_router proxy resolved when transcript discovery is invoked
    from ...thread_router import thread_router

    identity = identity_state.get_identity(window_id)
    if identity is None:
        return

    chat_id = thread_router.resolve_chat_id(user_id, thread_id) if user_id else 0

    w = _window or await tmux_manager.find_window_by_id(window_id)

    if w and w.pane_current_command:
        await _detect_and_apply_provider(
            window_id, identity, w, client=client, chat_id=chat_id, thread_id=thread_id
        )
        refreshed = identity_state.get_identity(window_id)
        if refreshed is None:
            return
        identity = refreshed

    if _hook_already_resolved(window_id, identity):
        return

    if not identity.cwd:
        if not w or not w.cwd:
            return
        session_manager.set_window_provider(
            window_id, identity.provider_name or "", cwd=w.cwd
        )
        refreshed = identity_state.get_identity(window_id)
        if refreshed is None:
            return
        identity = refreshed

    providers_to_try = _resolve_providers_to_try(window_id, identity, w)
    if providers_to_try is None:
        await _switch_to_shell(
            window_id, client=client, chat_id=chat_id, thread_id=thread_id
        )
        return
    if not providers_to_try:
        return

    pane_alive = w is not None and not is_shell_prompt(w.pane_current_command)
    await _find_and_register_transcript(
        window_id, identity, providers_to_try, pane_alive
    )
