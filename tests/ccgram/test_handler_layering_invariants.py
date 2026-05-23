"""Structural tests pinning the F5 / F2 handler-layer invariants.

Two AST walks over ``src/ccgram/handlers/**/*.py`` codify what is
currently allowed to bypass the documented seams. Any NEW handler that
reaches for a PTB ``Bot`` directly or pokes one of the SessionManager
substores has to update the allow-list — which forces a deliberate
"is this coupling justified?" decision in code review.

* :func:`test_no_new_ptb_bot_escapes` — F5 / F5.7. Handlers should
  depend on :class:`ccgram.telegram_client.TelegramClient`. The handful
  of sites that need PTB-only helpers (``get_file``, ``send_chat_action``,
  ``edit_forum_topic``, etc.) are listed in :data:`_PTB_BOT_ALLOWLIST`.
* :func:`test_no_new_lower_level_singleton_access` — F2 / Round-5 F2.
  The query layer is the documented read seam. Handlers may still touch
  the stateful singletons (``window_store`` / ``thread_router`` /
  ``user_preferences`` / ``session_map_sync``) for legacy reasons;
  this test snapshots the file-level coupling so a new handler can't
  silently grow a dependency.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HANDLERS_ROOT = _REPO_ROOT / "src" / "ccgram" / "handlers"

# F5 PTB-Bot escape allow-list — files that legitimately reach below
# the TelegramClient Protocol because they call PTB-only helpers
# (``getFile`` / ``send_chat_action`` / ``edit_forum_topic`` /
# ``get_chat_member``) or rely on PTB-specific request flows
# (``do_api_request`` for ``DraftStream``).
_PTB_BOT_ALLOWLIST = frozenset(
    {
        # /agent wraps get_bot() in PTBTelegramClient to pass through to
        # ensure_setup (so the shell-setup offer keyboard can render).
        "agent_command.py",
        "cleanup.py",
        "command_history.py",
        "commands/__init__.py",
        "commands/forward.py",
        "commands/menu_sync.py",
        "file_handler.py",
        "last_reply.py",  # wraps get_bot() in PTBTelegramClient for send_last_reply
        "live/pane_callbacks.py",
        "live/screenshot_callbacks.py",
        "messaging/msg_spawn.py",
        "messaging_pipeline/topic_commands.py",
        "status/status_bar_actions.py",
        "sync_command.py",
        "toolbar/toolbar_callbacks.py",
        "topics/directory_callbacks.py",
        "voice/voice_callbacks.py",
        "voice/voice_handler.py",
    }
)

# F2 / Round-5 F2 lower-level singleton allow-list. Captures every
# handler file that currently touches one of the SessionManager
# substores directly. New entries require an explicit decision —
# either go through ``window_query`` / ``session_query`` instead, or
# justify the direct access in the diff.
_SINGLETON_ATTRS = frozenset(
    {"window_store", "thread_router", "user_preferences", "session_map_sync"}
)
_SINGLETON_ALLOWLIST = frozenset(
    {
        # /agent command resolves window_id via thread_router (same routing
        # pattern as sync_command/sessions_dashboard) and clears session_map
        # for hookful provider switches (no equivalent in query layer).
        "agent_command.py",
        "callback_helpers.py",
        "callback_registry.py",
        "cleanup.py",
        "commands/__init__.py",
        "commands/forward.py",
        "commands/menu_sync.py",
        "file_handler.py",
        "hook_events.py",
        "interactive/interactive_ui.py",
        "last_reply.py",  # reads thread_router for window/chat resolution in last_command
        "live/pane_callbacks.py",
        "live/screenshot_callbacks.py",
        "messaging/msg_telegram.py",
        "messaging_pipeline/message_queue.py",
        "messaging_pipeline/message_routing.py",
        "messaging_pipeline/tool_batch.py",
        "messaging_pipeline/topic_commands.py",
        "polling/polling_coordinator.py",
        "polling/polling_state.py",
        "polling/window_tick/apply.py",
        "recovery/history.py",
        "recovery/recovery_banner.py",
        "recovery/recovery_callbacks.py",
        "recovery/restore_command.py",
        "recovery/resume_command.py",
        "recovery/transcript_discovery.py",
        "send/send_callbacks.py",
        "send/send_command.py",
        "sessions_dashboard.py",
        "shell/shell_capture.py",
        "shell/shell_commands.py",
        # rc_probe iterates thread bindings to route the RC outcome reply
        # to the bound topic — same notification-routing pattern as
        # hook_events.py / msg_telegram.py (read-only over thread_router).
        "status/rc_probe.py",
        "status/status_bar_actions.py",
        "status/status_bubble.py",
        "status/topic_emoji.py",
        "sync_command.py",
        "text/text_handler.py",
        "toolbar/toolbar_callbacks.py",
        "topics/directory_browser.py",
        "topics/directory_callbacks.py",
        "topics/topic_lifecycle.py",
        "topics/topic_orchestration.py",
        "topics/window_callbacks.py",
        "voice/voice_callbacks.py",
        "voice/voice_handler.py",
    }
)


def _iter_handler_files() -> Iterator[Path]:
    for path in sorted(_HANDLERS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == "__init__.py" and path.parent == _HANDLERS_ROOT:
            continue  # top-level package init is just a marker.
        yield path


def _rel(path: Path) -> str:
    return path.relative_to(_HANDLERS_ROOT).as_posix()


def _uses_ptb_bot(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # message.get_bot(), update.message.get_bot(), etc.
        if node.attr == "get_bot":
            return True
        # context.bot.<anything>
        if isinstance(node.value, ast.Attribute) and node.value.attr == "bot":
            return True
    return False


def _uses_singleton(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id in _SINGLETON_ATTRS:
            return True
    return False


@pytest.mark.parametrize("path", list(_iter_handler_files()), ids=_rel)
def test_no_new_ptb_bot_escapes(path: Path) -> None:
    rel = _rel(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if rel in _PTB_BOT_ALLOWLIST:
        return  # legitimate — PTB-only helpers documented in F5.7.
    assert not _uses_ptb_bot(tree), (
        f"{rel} reaches the raw PTB Bot via .get_bot() or context.bot.X. "
        "Use the TelegramClient Protocol (PTBTelegramClient(bot)) instead, "
        "or add this file to _PTB_BOT_ALLOWLIST with a one-line note in "
        "the PR explaining why a PTB-only helper is required."
    )


@pytest.mark.parametrize("path", list(_iter_handler_files()), ids=_rel)
def test_no_new_lower_level_singleton_access(path: Path) -> None:
    rel = _rel(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if rel in _SINGLETON_ALLOWLIST:
        return
    assert not _uses_singleton(tree), (
        f"{rel} touches one of {sorted(_SINGLETON_ATTRS)} directly. "
        "Read through ccgram.window_query / ccgram.session_query, or add "
        "this file to _SINGLETON_ALLOWLIST with a one-line note in the "
        "PR explaining why direct singleton access is justified."
    )
