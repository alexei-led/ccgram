"""Pinned, topic-scoped command panels using Telegram inline keyboards."""

from __future__ import annotations

import contextlib
import json
from collections import OrderedDict
from dataclasses import dataclass
import structlog
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.error import TelegramError
from telegram.ext import CommandHandler

from ... import session_query, window_query
from ...cc_commands import CONTROL_COMMANDS, discover_provider_commands
from ...config import config
from ...providers import AgentProvider, get_provider_for_window
from ...utils import is_general_topic
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..callback_tokens import compact_callback_data, resolve_callback_data
from ..messaging_pipeline.message_sender import interactive_edit, safe_edit, safe_reply
from .forward import _command_update, forward_command_handler

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

_PANEL_PREFIX = "cmdpanel:"
_PAGE_SIZE = 8
_TELEGRAM_BUTTON_TEXT_LIMIT = 64
_TWO_COLUMN_LABEL_LIMIT = 22
_PAYLOAD_FIELDS = 7
_CONFIRMATION_KEY = "command_panel_confirmation"
_CONFIRM_COMMANDS = frozenset({"upgrade", "unbind", "clear", "new", "rewind", "reset"})
_MAX_PANEL_MESSAGES = 512
_panel_messages: OrderedDict[tuple[int, int, int | None], Message] = OrderedDict()
_CONTROL_LABELS = {
    "commands": "Commands",
    "sessions": "Sessions",
    "sync": "Audit state",
    "upgrade": "Update ccgram",
}
_SESSION_COMMANDS = (
    ("screenshot", "Screenshot"),
    ("live", "Live terminal"),
    ("panes", "Panes"),
    ("toolbar", "Keyboard controls"),
    ("recall", "Recent inputs"),
    ("send", "Get file"),
    ("last", "Last reply"),
    ("agent", "Agent selection"),
    ("unbind", "Unbind topic"),
)


@dataclass(frozen=True, slots=True)
class PanelTarget:
    owner_id: int
    chat_id: int
    thread_id: int | None
    window_id: str
    provider_name: str


def resolve_panel_target(
    message: Message, owner_id: int
) -> tuple[PanelTarget, AgentProvider | None]:
    update = Update(0, message=message)
    thread_id = get_thread_id(update)
    window_id = ""
    provider = None
    if thread_id is not None and not is_general_topic(message):
        window_id = (
            session_query.resolve_window_for_topic(owner_id, thread_id, message.chat.id)
            or ""
        )
        if window_id:
            provider = get_provider_for_window(
                window_id, provider_name=window_query.get_window_provider(window_id)
            )
    target = PanelTarget(
        0 if thread_id is None else owner_id,
        message.chat.id,
        thread_id,
        window_id,
        provider.capabilities.name if provider else "",
    )
    return target, provider


def _callback(target: PanelTarget, action: str, value: str = "") -> str:
    payload = _PANEL_PREFIX + json.dumps(
        [
            target.owner_id,
            target.chat_id,
            target.thread_id,
            target.window_id,
            target.provider_name,
            action,
            value,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return compact_callback_data(_PANEL_PREFIX, payload, target.window_id)


def _agent_names(provider: AgentProvider | None) -> list[str]:
    if provider is None or provider.capabilities.chat_first_command_path:
        return []
    return list(
        dict.fromkeys(
            command.name if command.name.startswith("/") else f"/{command.name}"
            for command in discover_provider_commands(provider)
            if command.name and not any(char.isspace() for char in command.name)
        )
    )


def _bot_actions(
    target: PanelTarget, provider: AgentProvider | None
) -> list[tuple[str, str]]:
    if not target.window_id:
        return (
            [("resume", "Resume past session")] if target.thread_id is not None else []
        )
    actions = list(_SESSION_COMMANDS)
    if provider and provider.capabilities.chat_first_command_path:
        actions = [(name, label) for name, label in actions if name != "agent"]
    if provider and provider.capabilities.supports_structured_transcript:
        actions += [
            ("history", "History"),
            ("verbose", "Tool verbosity"),
            ("toolcalls", "Tool details"),
        ]
    if (
        provider
        and provider.capabilities.supports_resume
        and provider.capabilities.supports_resume_picker
    ):
        actions += [("resume", "Resume session")]
    return actions


def _rows(
    target: PanelTarget, items: list[tuple[str, str, str]]
) -> list[list[InlineKeyboardButton]]:
    rows: list[list[InlineKeyboardButton]] = []
    for label, action, value in items:
        button = InlineKeyboardButton(
            label, callback_data=_callback(target, action, value)
        )
        # Preserve long native names verbatim instead of squeezing or truncating them.
        if len(label) > _TWO_COLUMN_LABEL_LIMIT:
            rows.append([button])
        elif (
            rows
            and len(rows[-1]) == 1
            and len(rows[-1][0].text) <= _TWO_COLUMN_LABEL_LIMIT
        ):
            rows[-1].append(button)
        else:
            rows.append([button])
    return rows


def _command_page(
    target: PanelTarget,
    group: str,
    page: int,
    description: str,
    controls: list[tuple[str, str, str]],
    native_names: list[str],
    bot_actions: list[tuple[str, str]],
) -> tuple[str, InlineKeyboardMarkup]:
    items = (
        controls
        if group == "global"
        else [(name, "agent", name) for name in native_names]
        if group == "agent"
        else [(label, "bot", name) for name, label in bot_actions]
    )
    pages = max(1, (len(items) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = min(max(0, page), pages - 1)
    long_native_names = (
        [
            name
            for name in native_names[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]
            if len(name) > _TELEGRAM_BUTTON_TEXT_LIMIT
        ]
        if group == "agent"
        else []
    )
    title = (
        "Global controls"
        if group == "global"
        else f"{target.provider_name} commands"
        if group == "agent"
        else "ccgram actions"
    )
    context = (
        "Manage ccgram and sessions in other topics."
        if group == "global"
        else description
    )
    text = f"*{title}*\n{context}\n\n" + (
        "Native names are sent unchanged to this agent."
        if group == "agent"
        else "Choose an action for this topic."
    )
    if not items:
        text += "\nNo discoverable commands."
    if long_native_names:
        text += "\n\nLong native command names (shown in full):\n" + "\n".join(
            f"{page * _PAGE_SIZE + offset + 1}. {name}"
            for offset, name in enumerate(
                native_names[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]
            )
            if len(name) > _TELEGRAM_BUTTON_TEXT_LIMIT
        )
    page_items = items[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]
    if group == "agent":
        first_item = page * _PAGE_SIZE
        page_items = [
            (
                f"Command {first_item + offset + 1}"
                if len(name) > _TELEGRAM_BUTTON_TEXT_LIMIT
                else name,
                "agent",
                name,
            )
            for offset, (name, _action, _value) in enumerate(page_items)
        ]
    rows = _rows(target, page_items)
    navigation = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton(
                "‹ Previous",
                callback_data=_callback(target, "view", f"{group}:{page - 1}"),
            )
        )
    if pages > 1:
        navigation.append(
            InlineKeyboardButton(f"{page + 1} / {pages}", callback_data="noop")
        )
    if page + 1 < pages:
        navigation.append(
            InlineKeyboardButton(
                "Next ›", callback_data=_callback(target, "view", f"{group}:{page + 1}")
            )
        )
    if navigation:
        rows.append(navigation)
    rows.append(
        [
            InlineKeyboardButton(
                "‹ Command groups", callback_data=_callback(target, "view", "groups:0")
            )
        ]
    )
    return text, InlineKeyboardMarkup(rows)


def build_command_panel(
    target: PanelTarget,
    provider: AgentProvider | None,
    *,
    group: str = "home",
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render from the current binding; agent button labels are native names."""
    controls = [(label, "bot", name) for name, label in _CONTROL_LABELS.items()]
    bot_actions = _bot_actions(target, provider)
    native_names = _agent_names(provider)
    view = window_query.view_window(target.window_id) if target.window_id else None
    display = view.window_name if view and view.window_name else target.window_id
    description = f"Provider: `{target.provider_name}`\nSession: `{display}`"
    rows: list[list[InlineKeyboardButton]]
    if not target.window_id and target.thread_id is None:
        text = "*General controls*\nNo agent or terminal is attached here.\n\nManage ccgram and sessions in other topics."
        rows = _rows(target, controls)
    elif not target.window_id and group != "global":
        text = "*This topic is not attached*\nSet up a session or resume a past one. Terminal and agent actions appear after binding."
        rows = _rows(
            target,
            [("Set up session", "setup", "")]
            + [(label, "bot", name) for name, label in bot_actions],
        )
        rows.append(
            [
                InlineKeyboardButton(
                    "Global controls",
                    callback_data=_callback(target, "view", "global:0"),
                )
            ]
        )
    elif group == "home":
        quick = [("Screenshot", "bot", "screenshot")]
        quick.append(
            ("History", "bot", "history")
            if provider and provider.capabilities.supports_structured_transcript
            else ("Live terminal", "bot", "live")
        )
        preferred = (
            ["/context", "/compact"]
            if target.provider_name == "claude"
            else ["/status", "/compact"]
        )
        quick += [(name, "agent", name) for name in preferred if name in native_names]
        if not native_names:
            quick.append(("Keyboard controls", "bot", "toolbar"))
        text = f"*Commands for this topic*\n{description}\n\nActions affect this session only. Tap an agent command to send its original name."
        rows = _rows(target, quick)
        rows.append(
            [
                InlineKeyboardButton(
                    "All commands ›",
                    callback_data=_callback(target, "view", "groups:0"),
                )
            ]
        )
    elif group == "groups":
        text = f"*Commands for this topic*\n{description}\n\nChoose a group. This panel updates in place."
        rows = [
            [
                InlineKeyboardButton(
                    "ccgram actions",
                    callback_data=_callback(target, "view", "ccgram:0"),
                )
            ]
        ]
        if native_names:
            rows.append(
                [
                    InlineKeyboardButton(
                        "Agent commands",
                        callback_data=_callback(target, "view", "agent:0"),
                    )
                ]
            )
        rows += [
            [
                InlineKeyboardButton(
                    "Global controls",
                    callback_data=_callback(target, "view", "global:0"),
                )
            ],
            [
                InlineKeyboardButton(
                    "‹ Quick actions", callback_data=_callback(target, "view", "home:0")
                )
            ],
        ]
    else:
        return _command_page(
            target, group, page, description, controls, native_names, bot_actions
        )
    return text, InlineKeyboardMarkup(rows)


def _remember_panel_message(key: tuple[int, int, int | None], message: Message) -> None:
    _panel_messages.pop(key, None)
    _panel_messages[key] = message
    while len(_panel_messages) > _MAX_PANEL_MESSAGES:
        _panel_messages.popitem(last=False)


async def send_command_panel(
    message: Message, owner_id: int, *, replace_message: Message | None = None
) -> None:
    target, provider = resolve_panel_target(message, owner_id)
    key = (owner_id, target.chat_id, target.thread_id)
    if replace_message is None:
        replace_message = _panel_messages.get(key)
        if replace_message is not None:
            _panel_messages.move_to_end(key)
    if replace_message is None and target.thread_id is None:
        with contextlib.suppress(TelegramError):
            chat = await message.get_bot().get_chat(target.chat_id)
            pinned = chat.pinned_message
            if (
                pinned
                and pinned.from_user
                and pinned.from_user.id == message.get_bot().id
                and pinned.message_thread_id in (None, 1)
                and pinned.reply_markup
                and (
                    "named topic" in (pinned.text or "").lower()
                    or any(
                        isinstance(button.callback_data, str)
                        and button.callback_data.startswith(_PANEL_PREFIX)
                        for row in pinned.reply_markup.inline_keyboard
                        for button in row
                    )
                )
            ):
                replace_message = pinned
    text, keyboard = build_command_panel(target, provider)
    if replace_message is not None:
        await safe_edit(replace_message, text, reply_markup=keyboard)
        _remember_panel_message(key, replace_message)
        return
    sent = await safe_reply(message, text, reply_markup=keyboard)
    if sent:
        _remember_panel_message(key, sent)
        with contextlib.suppress(TelegramError):
            await sent.pin(disable_notification=True)


def _decode(data: str) -> tuple[PanelTarget, str, str] | None:
    if not data.startswith(_PANEL_PREFIX):
        return None
    try:
        parts = json.loads(data[len(_PANEL_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(parts, list) or len(parts) != _PAYLOAD_FIELDS:
        return None
    owner, chat, thread, window, provider, action, value = parts
    if (
        not isinstance(owner, int)
        or not isinstance(chat, int)
        or (thread is not None and not isinstance(thread, int))
    ):
        return None
    if not all(isinstance(item, str) for item in (window, provider, action, value)):
        return None
    return PanelTarget(owner, chat, thread, window, provider), action, value


async def _dispatch_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message: Message,
    kind: str,
    command: str,
) -> None:
    if kind == "bot" and command == "commands" and update.effective_user:
        await send_command_panel(
            message, update.effective_user.id, replace_message=message
        )
        return
    adapted = _command_update(
        update, message, command if command.startswith("/") else f"/{command}"
    )
    if kind == "agent":
        await forward_command_handler(adapted, context, native_command=command)
        return
    context.args = []
    for handlers in context.application.handlers.values():
        for handler in handlers:
            if isinstance(handler, CommandHandler) and command in handler.commands:
                await handler.callback(adapted, context)
                return
    await safe_reply(message, "Command is unavailable. Open /commands again.")


async def _show_confirmation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: PanelTarget,
    kind: str,
    command: str,
) -> None:
    query = update.callback_query
    assert query is not None and isinstance(query.message, Message)
    if context.user_data is None:
        await query.answer(
            "Cannot confirm this action. Open /commands again.", show_alert=True
        )
        return
    context.user_data[_CONFIRMATION_KEY] = (
        target,
        query.message.message_id,
        kind,
        command,
    )
    warning = (
        "This updates and restarts ccgram; all topics may briefly lose their connection."
        if command == "upgrade"
        else "This disconnects the topic. The terminal stays open."
        if command == "unbind"
        else "This can reset or rewind the attached agent conversation. Project files may be affected by the agent."
    )
    buttons = [
        [
            InlineKeyboardButton(
                "Cancel", callback_data=_callback(target, "view", "home:0")
            ),
            InlineKeyboardButton(
                "Confirm",
                callback_data=_callback(target, "confirm", f"{kind}:{command}"),
            ),
        ],
    ]
    await query.answer()
    await interactive_edit(
        query,
        f"*Confirm `/{command.lstrip('/')}`?*\n{warning}",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def _execute_panel_action(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: PanelTarget,
    provider: AgentProvider | None,
    action: str,
    value: str,
) -> None:
    query = update.callback_query
    assert query is not None and isinstance(query.message, Message)
    kind, _, command = (
        value.partition(":") if action == "confirm" else (action, "", value)
    )
    allowed_bot = {name for name, _ in CONTROL_COMMANDS} | {
        name for name, _ in _bot_actions(target, provider)
    }
    if (
        kind == "agent"
        and command not in _agent_names(provider)
        or kind == "bot"
        and command not in allowed_bot
        or kind not in {"agent", "bot"}
    ):
        await query.answer(
            "This command is not available in this topic.", show_alert=True
        )
        return
    if action == "confirm":
        expected = (target, query.message.message_id, kind, command)
        pending = (
            context.user_data.pop(_CONFIRMATION_KEY, None)
            if context.user_data is not None
            else None
        )
        if pending != expected:
            await query.answer(
                "Confirmation expired. Open /commands again.", show_alert=True
            )
            return
    elif command.lstrip("/") in _CONFIRM_COMMANDS:
        await _show_confirmation(update, context, target, kind, command)
        return
    if context.user_data is not None:
        context.user_data.pop(_CONFIRMATION_KEY, None)
    if kind == "agent":
        await query.answer("Sending to the attached agent…")
        if action == "confirm":
            text, keyboard = build_command_panel(target, provider, group="agent")
            await interactive_edit(query, text, reply_markup=keyboard)
        try:
            await _dispatch_command(update, context, query.message, kind, command)
        except Exception:
            logger.exception(
                "Topic command panel dispatch failed",
                command=command,
                chat_id=target.chat_id,
                thread_id=target.thread_id,
                provider=target.provider_name,
            )
            await safe_reply(
                query.message, "❌ Could not send the command. Check the ccgram log."
            )
        return
    await query.answer()
    text, keyboard = build_command_panel(target, provider)
    await interactive_edit(query, text, reply_markup=keyboard)
    try:
        await _dispatch_command(update, context, query.message, kind, command)
    except Exception:
        logger.exception(
            "Topic control panel dispatch failed",
            command=command,
            chat_id=target.chat_id,
            thread_id=target.thread_id,
        )
        await safe_reply(
            query.message, "❌ Could not run this control. Check the ccgram log."
        )


@register(_PANEL_PREFIX)
async def handle_command_panel(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query, user = update.callback_query, update.effective_user
    if (
        not query
        or not query.data
        or not user
        or not isinstance(query.message, Message)
    ):
        return
    if (
        not config.is_user_allowed(user.id)
        or config.group_id
        and query.message.chat.id != config.group_id
    ):
        await query.answer("Not authorized", show_alert=True)
        return
    current, provider = resolve_panel_target(query.message, user.id)
    data = resolve_callback_data(
        query.data, user.id, lambda _uid, window: window in {"", current.window_id}
    )
    decoded = _decode(data) if data else None
    if decoded is None or decoded[0] != current:
        await query.answer(
            "This panel belongs to another user or the session changed. Open /commands again.",
            show_alert=True,
        )
        return
    target, action, value = decoded
    logger.info(
        "Topic command panel callback accepted",
        action=action,
        command=value if action in {"agent", "bot", "confirm"} else None,
        user_id=user.id,
        chat_id=target.chat_id,
        thread_id=target.thread_id,
        provider=target.provider_name,
    )
    if action == "view":
        group, _, page_text = value.partition(":")
        if (
            group not in {"home", "groups", "ccgram", "agent", "global"}
            or not page_text.isdecimal()
        ):
            await query.answer("Invalid panel page", show_alert=True)
            return
        if context.user_data is not None:
            context.user_data.pop(_CONFIRMATION_KEY, None)
        text, keyboard = build_command_panel(
            target, provider, group=group, page=int(page_text)
        )
        await query.answer()
        await interactive_edit(query, text, reply_markup=keyboard)
    elif action == "setup" and target.thread_id is not None and not target.window_id:
        # Lazy: text_handler imports the commands package; importing here avoids that cycle.
        from ..text.text_handler import _handle_unbound_topic

        await query.answer()
        await _handle_unbound_topic(
            user.id, target.thread_id, "", context.user_data, query.message
        )
    else:
        await _execute_panel_action(update, context, target, provider, action, value)
