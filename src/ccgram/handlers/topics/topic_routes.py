"""Route auto-created topics to the Telegram chat configured for their workspace.

``CCGRAM_TOPIC_CHAT_ROUTES`` maps a workspace label (a Herdr space) to a group
chat, so each workspace gets its own forum group and its sessions become topics
there: a two-level layout (group = workspace, topic = session). Windows whose
workspace has no route keep the default behaviour, minus the routed groups.

  - route_target_chats(): pure choice of destination chats for one window.
  - window_workspace(): the workspace label the backend reports for a window.
  - relocate_misrouted_topics(): move live topics bound in the wrong chat (the
    route was added after the topic existed, or the tab moved to another space).
  - routed_workspace(): the reverse direction, so a session started from a topic
    in a workspace's group opens in that workspace.
  - log_group_join(): log a group's chat id when the bot joins it or the group
    becomes a supergroup, so routes can be filled without a third-party bot.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

import structlog
from telegram.error import TelegramError

from ... import session_query, window_query
from ...config import config
from ...multiplexer import multiplexer
from ...multiplexer.base import WindowRef, WorkspaceRef
from ...session import session_manager
from ...telegram_client import TelegramClient
from ..cleanup import clear_topic_state
from .topic_deletion import retire_topic_binding

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

RELOCATED_REASON = "relocated"


def configured_routes() -> Mapping[str, int]:
    """Workspace label -> chat id routes from ``CCGRAM_TOPIC_CHAT_ROUTES``."""
    return config.topic_chat_routes


def route_target_chats(
    seen_chats: Collection[int], workspace: str, routes: Mapping[str, int]
) -> set[int]:
    """Chats a window's topic belongs in.

    A routed workspace goes to its own chat only. An unrouted one goes to the
    usual chats except the routed groups, so a group shows one workspace.
    """
    if not routes:
        return set(seen_chats)
    target = routes.get(workspace)
    if target is not None:
        return {target}
    routed = set(routes.values())
    return {chat for chat in seen_chats if chat not in routed}


def _wanted(chat_id: int, workspace: str, routes: Mapping[str, int]) -> bool:
    target = routes.get(workspace)
    if target is not None:
        return chat_id == target
    return chat_id not in set(routes.values())


def routed_label(chat_id: int | None, routes: Mapping[str, int]) -> str | None:
    """Workspace label whose topics live in *chat_id*, if it is a routed group."""
    if chat_id is None:
        return None
    return next((label for label, chat in routes.items() if chat == chat_id), None)


async def routed_workspace(chat_id: int | None) -> WorkspaceRef | None:
    """The workspace a session started from *chat_id* belongs in, if routed.

    Without this a launch from a workspace's group lands wherever the backend
    places it, and relocation would then move the new topic out of the group.
    """
    label = routed_label(chat_id, configured_routes())
    if label is None or not multiplexer.capabilities.supports_workspace_selection:
        return None
    for workspace in await multiplexer.list_workspaces():
        if workspace.label == label:
            return workspace
    logger.warning("Routed workspace not found", workspace=label, chat_id=chat_id)
    return None


async def window_workspace(window_id: str) -> str:
    """Workspace label of a live window, or "" when unknown."""
    window = await multiplexer.find_window_by_id(window_id)
    return window.workspace if window is not None else ""


_Binding = tuple[int, int, int, str]


async def _create_where_it_belongs(
    client: TelegramClient, window: WindowRef, routes: Mapping[str, int]
) -> bool:
    """Create the window's topic in its destination chat(s); True if any worked."""
    # Lazy: topic_orchestration imports this module for routing.
    from .topic_orchestration import collect_target_chats, create_topic_in_chat

    view = window_query.view_window(window.window_id)
    name = (view.window_name if view else "") or window.window_name
    dest = route_target_chats(
        collect_target_chats(window.window_id), window.workspace, routes
    )
    created = False
    for chat_id in sorted(dest):
        try:
            if await create_topic_in_chat(client, chat_id, window.window_id, name):
                created = True
        except TelegramError:
            logger.exception(
                "Topic relocation: creation failed",
                window_id=window.window_id,
                chat_id=chat_id,
            )
    return created


async def _retire_live_binding(client: TelegramClient, binding: _Binding) -> None:
    """Delete one topic of a live window, leaving the session running."""
    uid, chat_id, tid, wid = binding

    async def clear_before_delete() -> None:
        await clear_topic_state(
            uid, tid, client, window_id=wid, chat_id=chat_id, window_dead=False
        )

    await retire_topic_binding(
        client,
        uid,
        tid,
        wid,
        chat_id=chat_id,
        before_delete=clear_before_delete,
        retirement_reason=RELOCATED_REASON,
    )


async def relocate_misrouted_topics(client: TelegramClient) -> int:
    """Move topics of live windows into the chat their workspace routes to.

    The new topic is created first; the old binding is retired (its topic
    deleted, the session left running) only once the window is bound where it
    belongs, so a failed creation never leaves a session without a topic.
    Returns the number of windows moved.
    """
    routes = configured_routes()
    if not routes:
        return 0
    # Lazy: same reconciliation-seam cycle as topic_orchestration.still_adoptable.
    from ...multiplexer.reconciliation import list_windows_for_reconciliation

    windows = await list_windows_for_reconciliation(multiplexer)
    if windows is None:
        return 0
    bindings = [
        (uid, chat, tid, wid)
        for uid, chat, tid, wid in session_query.iter_bound_topics()
        if chat is not None
    ]
    moved = 0
    for window in windows:
        if not window.topic_eligible:
            continue
        mine = [b for b in bindings if window.matches(b[3])]
        drop = [b for b in mine if not _wanted(b[1], window.workspace, routes)]
        if not drop:
            continue
        # Not bound anywhere it belongs yet: create the topic there first.
        if len(drop) == len(mine) and not await _create_where_it_belongs(
            client, window, routes
        ):
            continue
        for binding in drop:
            await _retire_live_binding(client, binding)
        session_manager.flush_state()
        moved += 1
        logger.info(
            "Topic relocated to its workspace chat",
            window_id=window.window_id,
            workspace=window.workspace,
            dropped=[chat for _u, chat, _t, _w in drop],
        )
    return moved


async def log_group_join(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the chat id of a group the bot was added to, created in or migrated.

    Telegram always delivers these service messages to bots, so the id needed
    for ``CCGRAM_TOPIC_CHAT_ROUTES`` shows up in the log. Turning on Topics
    upgrades a basic group to a supergroup with a new ``-100…`` id, logged as
    ``migrated_to``.
    """
    msg = update.effective_message
    if msg is None or msg.chat.type not in ("group", "supergroup"):
        return
    user = update.effective_user
    logger.info(
        "Group service message: chat id for CCGRAM_TOPIC_CHAT_ROUTES",
        chat_id=msg.chat.id,
        title=msg.chat.title,
        chat_type=msg.chat.type,
        is_forum=bool(msg.chat.is_forum),
        migrated_to=msg.migrate_to_chat_id,
        by_allowed_user=bool(user and config.is_user_allowed(user.id)),
    )
