"""Validate and serialize actions, reserving each ID durably before side effects."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from weakref import WeakValueDictionary
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..providers.pi import PI_LIKE_PROVIDERS
from .backend import Target
from .input_lock import input_lock
from .models import (
    ACTION_KINDS,
    MAX_TEXT_BYTES,
    Action,
    ActionResult,
    ProtocolError,
    wire,
)

if TYPE_CHECKING:
    from .service import LocalSessionService

_MAX_KEYS = 32
_MAX_ATTACHMENTS = 10

_KEY_RE = re.compile(
    r"^(?:[CMS]-){0,3}(?:[a-zA-Z0-9]|Enter|Escape|Tab|Space|BSpace|DC|Up|Down|Left|Right|Home|End|PPage|NPage|F(?:[1-9]|1[0-2]))$"
)


def parse_action(body: Any) -> Action:
    if not isinstance(body, dict):
        raise ProtocolError(400, "invalid_request", "Action must be an object")
    action_id, version, kind, data = (
        body.get(k) for k in ("id", "bindingVersion", "kind", "data")
    )
    try:
        if not isinstance(action_id, str) or str(UUID(action_id)) != action_id.lower():
            raise ValueError
    except ValueError as exc:
        raise ProtocolError(400, "invalid_request", "id must be a UUID") from exc
    if (
        type(version) is not int
        or version < 1
        or not isinstance(kind, str)
        or kind not in ACTION_KINDS
        or not isinstance(data, dict)
    ):
        raise ProtocolError(
            400, "invalid_request", "Invalid bindingVersion, kind, or data"
        )
    _validate_data(kind, data)
    return Action(action_id.lower(), version, kind, data)


def _string(data: dict[str, Any], name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProtocolError(
            400, "invalid_request", f"{name} must be nonempty text without NUL"
        )
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ProtocolError(
            400, "invalid_request", f"{name} must be valid UTF-8"
        ) from exc
    if size > maximum:
        raise ProtocolError(
            400, "invalid_request", f"{name} exceeds {maximum} UTF-8 bytes"
        )
    return value


def _validate_data(kind: str, data: dict[str, Any]) -> None:
    if kind in {"message.send", "terminal.paste"}:
        _string(data, "text")
    if kind == "message.send":
        _validate_message(data)
    if kind == "terminal.key":
        keys = data.get("keys")
        if (
            not isinstance(keys, list)
            or not 1 <= len(keys) <= _MAX_KEYS
            or any(not isinstance(k, str) or not _KEY_RE.fullmatch(k) for k in keys)
        ):
            raise ProtocolError(
                400, "invalid_request", "keys must contain 1 to 32 terminal key names"
            )
    if kind == "request.respond":
        _validate_response_data(data)
    if kind == "session.rename":
        _string(data, "title", maximum=200)
    if kind == "session.resume":
        _string(data, "conversationId", maximum=200)


def _validate_message(data: dict[str, Any]) -> None:
    if "dismissRequest" in data and type(data["dismissRequest"]) is not bool:
        raise ProtocolError(400, "invalid_request", "dismissRequest must be boolean")
    files = data.get("files", [])
    if (
        not isinstance(files, list)
        or len(files) > _MAX_ATTACHMENTS
        or any(
            not isinstance(f, dict) or not isinstance(f.get("id"), str) for f in files
        )
    ):
        raise ProtocolError(
            400,
            "invalid_request",
            "files must be a list of at most ten upload references",
        )


def _validate_response_data(data: dict[str, Any]) -> None:
    _string(data, "requestId", maximum=128)
    if "optionId" in data:
        _string(data, "optionId", maximum=128)
    elif "answer" not in data:
        raise ProtocolError(
            400, "invalid_request", "A response needs optionId or answer"
        )
    if "optionId" in data and "answer" in data:
        raise ProtocolError(400, "invalid_request", "Choose optionId or answer")


def _validate_internal_action(action: Action) -> Action:
    if action.kind not in {
        "terminal.submit",
        "terminal.setup_prompt",
        "message.followup",
        "session.split",
    }:
        raise ProtocolError(422, "unsupported_action", "Unsupported adapter operation")
    try:
        if str(UUID(action.id)) != action.id.lower():
            raise ValueError
    except ValueError as exc:
        raise ProtocolError(400, "invalid_request", "id must be a UUID") from exc
    if type(action.binding_version) is not int or action.binding_version < 1:
        raise ProtocolError(400, "invalid_request", "Invalid bindingVersion")
    if not isinstance(action.data, dict):
        raise ProtocolError(400, "invalid_request", "Action data must be an object")
    if action.kind == "session.split":
        if action.data:
            raise ProtocolError(400, "invalid_request", "session.split takes no data")
    else:
        _string(action.data, "text")
    if (
        action.kind == "terminal.setup_prompt"
        and type(action.data.get("clear")) is not bool
    ):
        raise ProtocolError(400, "invalid_request", "clear must be boolean")
    return action


class ActionExecutor:
    def __init__(self, service: LocalSessionService) -> None:
        self.service = service
        self._action_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )

    async def execute(
        self,
        session_id: str,
        action: Action,
        *,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> tuple[ActionResult, int]:
        # Validate in-process clients too; the HTTP decoder is not a trust boundary.
        action = parse_action(wire(action))
        lock = self._action_locks.setdefault(action.id, asyncio.Lock())
        async with lock:
            return await self._once(
                session_id, action, pane_id=pane_id, expected_prompt=expected_prompt
            )

    async def execute_internal(
        self,
        session_id: str,
        action: Action,
        *,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> tuple[ActionResult, int]:
        """Run adapter-only semantic actions through the shared action lifecycle."""
        action = _validate_internal_action(action)
        lock = self._action_locks.setdefault(action.id, asyncio.Lock())
        async with lock:
            return await self._once(
                session_id,
                action,
                internal=True,
                pane_id=pane_id,
                expected_prompt=expected_prompt,
            )

    async def _once(
        self,
        session_id: str,
        action: Action,
        *,
        internal: bool = False,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> tuple[ActionResult, int]:
        store = self.service.store
        fingerprint = store.action_fingerprint(session_id, action)
        if pane_id is not None or expected_prompt is not None:
            fingerprint = hashlib.sha256(
                json.dumps(
                    [
                        "ccgram/action-execution-context/v1",
                        fingerprint,
                        pane_id,
                        expected_prompt,
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        previous = store.action(action.id, fingerprint)
        if previous is not None:
            return previous
        store.reserve(action.id, fingerprint)
        attempted = False
        try:
            target, _ = await self.service.current(session_id)
            async with input_lock(target.id):
                target = await self._preflight(
                    session_id,
                    action,
                    internal=internal,
                    pane_id=pane_id,
                    expected_prompt=expected_prompt,
                )
                sent_at = time.time()
                attempted = True
                async with asyncio.timeout(10):
                    effect = (
                        await self._perform_split(session_id, target)
                        if action.kind == "session.split"
                        else await self._perform(session_id, target, action)
                    )
                success = bool(effect)
                result = ActionResult(
                    action.id,
                    "sent" if success else "unknown",
                    data={"paneId": effect}
                    if action.kind == "session.split" and isinstance(effect, str)
                    else None,
                    error={
                        "code": "backend_unavailable",
                        "message": "Could not confirm split",
                        "details": {},
                    }
                    if action.kind == "session.split" and not success
                    else None,
                )
                if success:
                    await self._record_delivery(session_id, target, action, sent_at)
        except ProtocolError as exc:
            if attempted:
                result, status = (
                    ActionResult(action.id, "unknown", exc.body()["error"]),
                    200,
                )
            else:
                result, status = (
                    ActionResult(action.id, "rejected", exc.body()["error"]),
                    exc.status,
                )
            store.finish(result, status)
            return result, status
        except OSError, TimeoutError:
            result = ActionResult(
                action.id,
                "unknown" if attempted else "rejected",
                {
                    "code": "backend_unavailable",
                    "message": "Could not confirm delivery",
                    "details": {},
                },
            )
        store.finish(result)
        self.service.hub.publish("action.updated", session_id, result)
        return result, 200

    async def _preflight(
        self,
        session_id: str,
        action: Action,
        *,
        internal: bool = False,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> Target:
        target, snapshot = await self.service.current(session_id)
        self._validate_target(target, snapshot, action)
        if action.kind == "session.close":
            # Observation failures hide input capabilities, but cannot prevent
            # closing a target whose backend still confirms its identity.
            fresh = await self.service.backend.inspect(target.id)
            if not fresh.available:
                raise ProtocolError(
                    503, "backend_unavailable", "Session backend is unavailable"
                )
            if (
                fresh.provider,
                fresh.conversation_id,
                fresh.pane_id,
                fresh.process_id,
                fresh.foreground_argv,
            ) != (
                target.provider,
                target.conversation_id,
                target.pane_id,
                target.process_id,
                target.foreground_argv,
            ):
                raise ProtocolError(409, "binding_conflict", "Session binding changed")
        if action.kind in {
            "message.send",
            "request.respond",
        } and self.service.requests.has_deferred(session_id):
            raise ProtocolError(409, "request_open", "Hook evidence awaits processing")
        if action.kind == "request.respond":
            await self._validate_response(session_id, target, action)
        self._validate_action_capability(target, snapshot, action, internal)
        self._validate_message_request(session_id, action)
        if action.kind == "message.send":
            self.service.uploads.message_text(
                action.data["text"], action.data.get("files", [])
            )
        if pane_id is not None:
            target = await self.service.backend.select_pane(target, pane_id)
        if expected_prompt is not None:
            prompt = await self.service.backend.prompt(target)
            if prompt is None or prompt.fingerprint != expected_prompt:
                raise ProtocolError(
                    409, "binding_conflict", "Interactive prompt changed before input"
                )
        return target

    @staticmethod
    def _validate_target(target: Target, snapshot: Any, action: Action) -> None:
        if snapshot.binding_version != action.binding_version:
            raise ProtocolError(
                409, "binding_conflict", "Session binding changed", snapshot
            )
        if action.kind != "session.close" and not target.foreground_verified:
            raise ProtocolError(
                503,
                "backend_unavailable",
                "Foreground process could not be verified",
            )

    def _validate_action_capability(
        self, target: Target, snapshot: Any, action: Action, internal: bool
    ) -> None:
        if action.kind == "session.split":
            return
        if action.kind == "terminal.setup_prompt" and (
            target.provider != "shell" or len(target.foreground_argv) != 1
        ):
            raise ProtocolError(
                422, "unsupported_action", "Prompt setup requires an interactive shell"
            )
        if action.kind == "session.resume" and (
            target.provider != "shell"
            or action.data["conversationId"] != target.conversation_id
        ):
            raise ProtocolError(
                422,
                "unsupported_action",
                "Resume requires the previous conversation in a shell",
            )
        public_kind = {
            "terminal.submit": "terminal.paste",
            "terminal.setup_prompt": "terminal.paste",
            "message.followup": "message.send",
        }.get(action.kind, action.kind)
        if public_kind not in snapshot.capabilities:
            raise ProtocolError(
                422,
                "unsupported_action",
                "Session does not support this action",
            )
        if (
            internal
            and action.kind == "message.followup"
            and target.provider not in PI_LIKE_PROVIDERS
        ):
            raise ProtocolError(
                422,
                "unsupported_action",
                "Follow-up input is supported for Pi sessions",
            )

    def _validate_message_request(self, session_id: str, action: Action) -> None:
        if (
            action.kind == "message.send"
            and self.service.requests.get(session_id)
            and not action.data.get("dismissRequest", False)
        ):
            raise ProtocolError(
                409, "request_open", "Sending text requires dismissRequest"
            )

    async def _record_delivery(
        self, session_id: str, target: Target, action: Action, sent_at: float
    ) -> None:
        if action.kind == "message.send":
            text = self.service.uploads.message_text(
                action.data["text"], action.data.get("files", [])
            )
            await self.service.remember_delivery(
                action.id, session_id, target.conversation_id, text, sent_at
            )

    async def _validate_response(
        self, session_id: str, target: Target, action: Action
    ) -> None:
        current = self.service.requests.get(session_id)
        if current is None or current.request.id != action.data["requestId"]:
            raise ProtocolError(409, "request_closed", "Request has closed")
        answer = action.data.get("answer")
        valid = (
            action.data.get("optionId") in current.option_keys
            if "optionId" in action.data
            else isinstance(answer, str) and answer in current.answer_keys
        )
        if current.request.source != "hook" or not valid:
            raise ProtocolError(
                422, "unsupported_action", "Use the terminal for this response"
            )
        prompt = await self.service.backend.prompt(target)
        if not self.service.requests.is_current(session_id, current):
            raise ProtocolError(
                409, "request_closed", "Request changed during validation"
            )
        if prompt is None or prompt.fingerprint != current.fingerprint:
            self.service.requests.close(session_id, "vanished")
            raise ProtocolError(
                409, "request_closed", "Prompt changed before the response"
            )

    async def _perform_split(self, session_id: str, target: Target) -> str | None:
        pane_id = await self.service.backend.split(target)
        if pane_id:
            await self.service.current(session_id)
        return pane_id

    async def _perform(self, session_id: str, target: Target, action: Action) -> bool:
        backend = self.service.backend
        kind, data = action.kind, action.data
        if kind in {"terminal.submit", "terminal.setup_prompt", "message.followup"}:
            return await self._perform_internal(target, action)
        if kind == "message.send":
            return await self._send_message(session_id, target, action)
        if kind == "request.respond":
            return await self._respond(session_id, target, action)
        if kind == "terminal.key":
            return await self._send_keys(target, data["keys"])
        if kind == "run.interrupt":
            success = await backend.send_key(target, "C-c")
            if success:
                self.service.requests.close(session_id, "interrupted")
            return success
        if kind == "terminal.paste":
            return await backend.send_text(target, data["text"], enter=False, raw=True)
        if kind in {"session.close", "session.rename"}:
            return await self._perform_session_action(target, action)
        return await backend.resume(target, data["conversationId"])

    async def _perform_internal(self, target: Target, action: Action) -> bool:
        if action.kind == "terminal.setup_prompt":
            backend = self.service.backend
            if not await backend.send_key(target, "C-c"):
                return False
            await asyncio.sleep(0.1)
            if not await backend.send_text(
                target, action.data["text"], enter=True, raw=True
            ):
                return False
            await asyncio.sleep(0.3)
            return not action.data["clear"] or await backend.send_text(
                target, "clear", enter=True, raw=True
            )
        if action.kind == "terminal.submit":
            return await self.service.backend.send_text(
                target, action.data["text"], enter=True, raw=True
            )
        return await self.service.backend.send_followup(target, action.data["text"])

    async def _perform_session_action(self, target: Target, action: Action) -> bool:
        if action.kind == "session.close":
            return await self.service.backend.close(target)
        return await self.service.backend.rename(target, action.data["title"])

    async def _send_keys(self, target: Target, keys: list[str]) -> bool:
        for key in keys:
            if not await self.service.backend.send_key(target, key):
                return False
        return True

    async def _send_message(
        self, session_id: str, target: Target, action: Action
    ) -> bool:
        backend = self.service.backend
        request = self.service.requests.get(session_id)
        if request:
            if not await backend.send_key(target, "Escape"):
                return False
            if not await self._prompt_gone(target):
                return False
            if not self.service.requests.is_current(session_id, request):
                raise ProtocolError(
                    409, "request_open", "Request changed during dismissal"
                )
            self.service.requests.close(session_id, "dismissed")
        text = self.service.uploads.message_text(
            action.data["text"], action.data.get("files", [])
        )
        return await backend.send_text(target, text, enter=True)

    async def _respond(self, session_id: str, target: Target, action: Action) -> bool:
        current = self.service.requests.get(session_id)
        if (
            current is None
            or current.request.id != action.data["requestId"]
            or not self.service.requests.is_current(session_id, current)
        ):
            raise ProtocolError(
                409, "request_closed", "Request changed before dispatch"
            )
        key = (
            current.option_keys[action.data["optionId"]]
            if "optionId" in action.data
            else current.answer_keys[action.data["answer"]]
        )
        if not await self.service.backend.send_text(target, key, enter=False):
            return False
        if not await self._prompt_gone(target):
            return False
        if self.service.requests.is_current(session_id, current):
            self.service.requests.close(session_id, "answered")
        return True

    async def _prompt_gone(self, target: Target) -> bool:
        for _ in range(3):
            await asyncio.sleep(0.3)
            if await self.service.backend.prompt(target) is None:
                return True
        return False
