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

from ..config import config
from ..providers import (
    detect_provider_from_pane,
    detect_provider_from_runtime,
    detect_provider_from_transcript_path,
    get_provider_for_window,
    should_probe_pane_title_for_provider_detection,
)
from ..providers.claude import _transcript_indicates_session_end
from ..session import session_manager
from ..tmux_manager import tmux_manager
from ..window_resolver import is_foreign_window
from .polling_strategies import is_shell_prompt

if TYPE_CHECKING:
    from ..providers.base import AgentProvider
    from ..session import WindowState
    from ..tmux_manager import TmuxWindow

logger = structlog.get_logger()


def should_attempt_hooked_transcript_recovery(
    state: object,
    provider_name: str,
    *,
    pane_alive: bool,
) -> bool:
    """Allow conservative transcript discovery for missed Claude SessionStart."""
    if not pane_alive or provider_name != "claude":
        return False
    transcript_path = getattr(state, "transcript_path", "")
    if not transcript_path:
        return False
    return _transcript_indicates_session_end(transcript_path)


async def _detect_and_apply_provider(
    window_id: str, state: "WindowState", w: "TmuxWindow"
) -> None:
    """Detect provider from pane process and apply transitions."""
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

    if detected and detected != state.provider_name:
        old_provider = state.provider_name
        session_manager.set_window_provider(window_id, detected, cwd=w.cwd or None)
        if detected == "shell":
            state.transcript_path = ""
            from ..providers.shell import setup_shell_prompt

            await setup_shell_prompt(window_id, clear=False)
        elif old_provider == "shell":
            from .shell_capture import clear_shell_monitor_state

            clear_shell_monitor_state(window_id)
    elif not detected and state.transcript_path:
        inferred = detect_provider_from_transcript_path(state.transcript_path)
        if inferred and inferred != state.provider_name:
            session_manager.set_window_provider(window_id, inferred, cwd=w.cwd or None)


def _resolve_providers_to_try(
    window_id: str, state: "WindowState", w: "TmuxWindow | None"
) -> list[tuple[str, "AgentProvider"]] | None:
    """Determine which providers to probe for transcripts.

    Returns a list of (name, provider) pairs, or ``None`` to signal the
    caller should set up a shell provider.
    """
    from ..providers import registry

    if state.provider_name:
        provider = get_provider_for_window(window_id)
        if not provider.capabilities.supports_mailbox_delivery:
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
    state: "WindowState",
    providers_to_try: list[tuple[str, "AgentProvider"]],
    pane_alive: bool,
) -> None:
    """Search for transcripts among candidate providers and register if found."""
    window_key = (
        window_id
        if is_foreign_window(window_id)
        else f"{config.tmux_session_name}:{window_id}"
    )

    for provider_name, provider in providers_to_try:
        max_age = 0 if pane_alive and not provider.capabilities.supports_hook else None
        event = await asyncio.to_thread(
            provider.discover_transcript,
            state.cwd,
            window_key,
            max_age=max_age,
        )
        if not event:
            continue

        if (
            state.session_id == event.session_id
            and state.transcript_path == event.transcript_path
            and state.provider_name == provider_name
        ):
            return

        session_manager.register_hookless_session(
            window_id=window_id,
            session_id=event.session_id,
            cwd=event.cwd,
            transcript_path=event.transcript_path,
            provider_name=provider_name,
        )
        await asyncio.to_thread(
            session_manager.write_hookless_session_map,
            window_id=window_id,
            session_id=event.session_id,
            cwd=event.cwd,
            transcript_path=event.transcript_path,
            provider_name=provider_name,
        )
        return


async def discover_and_register_transcript(
    window_id: str,
    *,
    _window: "TmuxWindow | None" = None,
    bot: "object | None" = None,  # noqa: ARG001
    user_id: int = 0,  # noqa: ARG001
    thread_id: int = 0,  # noqa: ARG001
) -> None:
    """Discover and register transcript for hookless providers (Codex, Gemini).

    Also handles provider auto-detection from pane process name
    and shell ↔ agent transitions with prompt marker setup.
    """
    state = session_manager.window_states.get(window_id)
    if not state:
        return

    w = _window or await tmux_manager.find_window_by_id(window_id)

    if w and w.pane_current_command:
        await _detect_and_apply_provider(window_id, state, w)

    pane_alive = w is not None and not is_shell_prompt(w.pane_current_command)

    if state.provider_name:
        provider = get_provider_for_window(window_id)
        if provider.capabilities.supports_hook and not should_attempt_hooked_transcript_recovery(
            state,
            provider.capabilities.name,
            pane_alive=pane_alive,
        ):
            return

    if not state.cwd:
        if not w or not w.cwd:
            return
        session_manager.set_window_provider(
            window_id, state.provider_name or "", cwd=w.cwd
        )

    providers_to_try = _resolve_providers_to_try(window_id, state, w)
    if providers_to_try is None:
        session_manager.set_window_provider(window_id, "shell")
        state.transcript_path = ""
        from ..providers.shell import setup_shell_prompt

        await setup_shell_prompt(window_id, clear=False)
        return
    if not providers_to_try:
        return

    await _find_and_register_transcript(window_id, state, providers_to_try, pane_alive)
