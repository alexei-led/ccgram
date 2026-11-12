"""In-process Telegram adapter over the shared session protocol core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from .backend import CreationOptions, Target
from .models import (
    UNCONFIRMED_DELIVERY_MESSAGE,
    Action,
    ActionResult,
    ProtocolError,
    SessionSnapshot,
)
from .service import LocalSessionService

_SESSION_SERVICE_KEY = "session_service"


@dataclass(frozen=True, slots=True)
class CreatedSession:
    target_id: str
    session: SessionSnapshot

    @property
    def title(self) -> str:
        return self.session.title


def _action_id(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(value))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, value))


def _result_message(result: ActionResult) -> str:
    if result.error and isinstance(result.error.get("message"), str):
        return result.error["message"]
    return "Sent" if result.outcome == "sent" else UNCONFIRMED_DELIVERY_MESSAGE


class TelegramSessionAdapter:
    """Translate Telegram interactions into shared protocol operations."""

    def __init__(self, service: LocalSessionService) -> None:
        self.service = service

    async def get_session(self, window_id: str) -> SessionSnapshot:
        _, snapshot = await self.service.for_target(window_id)
        return snapshot

    async def resolve(self, window_id: str) -> tuple[Target, SessionSnapshot]:
        return await self.service.for_target(window_id)

    async def list_sessions(self) -> list[SessionSnapshot]:
        return await self.service.list_sessions()

    async def _execute(
        self,
        window_id: str,
        kind: str,
        data: dict[str, Any],
        *,
        action_id: str | None = None,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
        expected_binding_version: int | None = None,
        internal: bool = False,
    ) -> tuple[bool, str]:
        try:
            _, snapshot = await self.service.for_target(window_id)
            action = Action(
                _action_id(action_id),
                snapshot.binding_version
                if expected_binding_version is None
                else expected_binding_version,
                kind,
                data,
            )
            if internal:
                result, _ = await self.service.execute_internal(
                    snapshot.id,
                    action,
                    pane_id=pane_id,
                    expected_prompt=expected_prompt,
                )
            else:
                result, _ = await self.service.execute(
                    snapshot.id,
                    action,
                    pane_id=pane_id,
                    expected_prompt=expected_prompt,
                )
            return result.outcome == "sent", _result_message(result)
        except ProtocolError as exc:
            return False, exc.message
        except OSError:
            return False, "Session backend is unavailable"

    async def send_text(
        self,
        window_id: str,
        text: str,
        *,
        pane_id: str | None = None,
        submit: bool = True,
        terminal_input: bool = False,
        action_id: str | None = None,
        expected_prompt: str | None = None,
        expected_binding_version: int | None = None,
    ) -> tuple[bool, str]:
        if terminal_input and submit:
            return await self._execute(
                window_id,
                "terminal.submit",
                {"text": text},
                action_id=action_id,
                pane_id=pane_id,
                expected_prompt=expected_prompt,
                expected_binding_version=expected_binding_version,
                internal=True,
            )
        return await self._execute(
            window_id,
            "message.send" if submit else "terminal.paste",
            {"text": text, **({"dismissRequest": True} if submit else {})},
            action_id=action_id,
            pane_id=pane_id,
            expected_prompt=expected_prompt,
            expected_binding_version=expected_binding_version,
        )

    async def setup_prompt(
        self,
        window_id: str,
        text: str,
        *,
        clear: bool,
        pane_id: str | None,
        expected_binding_version: int,
    ) -> tuple[bool, str]:
        return await self._execute(
            window_id,
            "terminal.setup_prompt",
            {"text": text, "clear": clear},
            pane_id=pane_id,
            expected_binding_version=expected_binding_version,
            internal=True,
        )

    async def send_followup(
        self,
        window_id: str,
        text: str,
        *,
        pane_id: str | None = None,
        action_id: str | None = None,
    ) -> tuple[bool, str]:
        return await self._execute(
            window_id,
            "message.followup",
            {"text": text},
            action_id=action_id,
            pane_id=pane_id,
            internal=True,
        )

    async def send_keys(
        self,
        window_id: str,
        keys: list[str],
        *,
        pane_id: str | None = None,
        action_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> tuple[bool, str]:
        return await self._execute(
            window_id,
            "terminal.key",
            {"keys": keys},
            action_id=action_id,
            pane_id=pane_id,
            expected_prompt=expected_prompt,
        )

    async def close(
        self, window_id: str, *, action_id: str | None = None
    ) -> tuple[bool, str]:
        return await self._execute(window_id, "session.close", {}, action_id=action_id)

    async def rename(
        self, window_id: str, title: str, *, action_id: str | None = None
    ) -> tuple[bool, str]:
        return await self._execute(
            window_id,
            "session.rename",
            {"title": title},
            action_id=action_id,
        )

    async def split(
        self, window_id: str, *, action_id: str | None = None
    ) -> tuple[bool, str]:
        try:
            _, snapshot = await self.service.for_target(window_id)
            pane_id = await self.service.split_session(
                snapshot.id,
                expected_binding_version=snapshot.binding_version,
                action_id=_action_id(action_id),
            )
            return (True, pane_id) if pane_id else (False, "Split is not supported")
        except ProtocolError as exc:
            return False, exc.message

    async def create(
        self,
        data: dict[str, Any],
        *,
        approval_mode: Literal["normal", "yolo"] = "normal",
        resume_id: str | None = None,
        continue_session: bool = False,
        launch_command: str | None = None,
    ) -> CreatedSession:
        options = CreationOptions(
            approval_mode, resume_id, continue_session, launch_command
        )
        target, snapshot = await self.service.create_target(data, options=options)
        return CreatedSession(target.id, snapshot)


def adapter_for(bot_data: Mapping[str, Any]) -> TelegramSessionAdapter:
    service = bot_data.get(_SESSION_SERVICE_KEY)
    if not isinstance(service, LocalSessionService):
        raise RuntimeError("Telegram session service is not initialized")
    return TelegramSessionAdapter(service)


__all__ = ["CreatedSession", "TelegramSessionAdapter", "adapter_for"]
