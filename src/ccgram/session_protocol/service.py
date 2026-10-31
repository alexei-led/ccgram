"""Session operations independent of HTTP and Telegram identities."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Collection
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol

from ..providers.base import HookEvent
from .backend import CreationOptions, Prompt, SessionBackend, Target
from .events import Event, EventHub
from .message_builder import MessageBuilder, TranscriptChangedError
from .messages import Message
from .models import (
    Action,
    ActionResult,
    Conversation,
    HistoryPage,
    Pane,
    ProtocolError,
    SessionSnapshot,
    Status,
    utc_now,
)
from .requests import HookIdentity, PendingHook, RequestStore, hook_input_hash
from .store import ProtocolStore
from .uploads import UploadStore

MAX_HISTORY_PAGE = 200

SUPPORTED_ACTIONS = (
    "message.send",
    "run.interrupt",
    "request.respond",
    "terminal.key",
    "terminal.paste",
    "session.close",
    "session.rename",
    "session.resume",
)


class SessionService(Protocol):
    async def list_sessions(self) -> list[SessionSnapshot]: ...
    async def get_session(self, session_id: str) -> SessionSnapshot: ...
    async def history(
        self,
        session_id: str,
        cursor: str | None,
        limit: int,
        *,
        conversation: str | None = None,
    ) -> HistoryPage: ...
    def watch(self, session_ids: Collection[str] | None) -> AsyncIterator[Event]: ...
    async def execute(
        self, session_id: str, action: Action
    ) -> tuple[ActionResult, int]: ...


class LocalSessionService:
    def __init__(
        self, backend: SessionBackend, store: ProtocolStore, hub: EventHub | None = None
    ) -> None:
        self.backend = backend
        self.store = store
        self.uploads = UploadStore(store.path.parent / "uploads", store)
        self.hub = hub or EventHub()
        self.requests = RequestStore(self.hub)
        self._snapshots: dict[str, SessionSnapshot] = {}
        self._targets: dict[str, Target] = {}
        self._refresh_lock = asyncio.Lock()
        self._transcript_stamps: dict[str, tuple[int, int, int, int]] = {}
        self._builders: dict[str, MessageBuilder] = {}
        self._builder_lock = asyncio.Lock()
        self._republish: set[str] = set()
        self._revisions: dict[str, dict[str, int]] = {}
        # Lazy: ActionExecutor imports the service only for type checking.
        from .actions import ActionExecutor

        self.executor = ActionExecutor(self)

    async def refresh(self) -> None:
        async with self._refresh_lock:
            targets = await self.backend.targets()
            live = set()
            for target in targets:
                _, snapshot = await self._update(target)
                live.add(snapshot.id)
            sources = {self._source_key(target) for target in targets}
            self.store.retain_transcripts(sources)
            async with self._builder_lock:
                for key in self._builders.keys() - sources:
                    self._drop_builder(key)
            for session_id in self._snapshots.keys() - live:
                self.requests.close(session_id, "vanished")
                self.hub.publish(
                    "session.removed", session_id, {"reason": "target_gone"}
                )
                self._snapshots.pop(session_id)
                self._targets.pop(session_id)
                self._transcript_stamps.pop(session_id, None)
                self._republish.discard(session_id)
                self._revisions.pop(session_id, None)

    async def _update(self, target: Target) -> tuple[Target, SessionSnapshot]:
        try:
            for _ in range(3):
                prompt = await self.backend.prompt(target)
                target = await self._source_version(target)
                sid, version = self.store.bind(
                    target.id, target.binding, target.aliases
                )
                snapshot = await self._observe(sid, version, target, prompt)
                if target.transcript is None:
                    return target, snapshot
                stat = await asyncio.to_thread(target.transcript.stat)
                if target.transcript_stamp == (
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                ):
                    return target, snapshot
            raise OSError("Transcript changed repeatedly during observation")
        except (ProtocolError, OSError) as exc:
            target = replace(
                target,
                available=False,
                transcript_stamp=None,
                transcript_digest=None,
                status=Status("unknown", str(exc), "screen", utc_now()),
            )
            sid, version = self.store.bind(target.id, target.binding, target.aliases)
            return target, await self._observe(sid, version, target, None)

    async def _observe(
        self, session_id: str, version: int, target: Target, prompt: Prompt | None
    ) -> SessionSnapshot:
        old = self._snapshots.get(session_id)
        if old and old.binding_version != version:
            self.requests.close(session_id, "vanished")
            self._revisions.pop(session_id, None)
            self._transcript_stamps.pop(session_id, None)
            self._republish.add(session_id)
        self._targets[session_id] = replace(
            target, transcript_stamp=None, transcript_digest=None
        )
        opened = self.requests.reconcile(
            session_id, target.conversation_id, version, prompt
        )
        status = target.status
        if opened:
            status = Status(
                "blocked", opened.request.title, opened.request.source, status.since
            )
        if old and (old.status.state, old.status.detail, old.status.source) == (
            status.state,
            status.detail,
            status.source,
        ):
            status = replace(status, since=old.status.since)
        last_activity = old.last_activity if old else utc_now()
        if old and old.status != status:
            last_activity = utc_now()
        capabilities = tuple(
            k
            for k in SUPPORTED_ACTIONS
            if (k == "session.close" or target.foreground_verified)
            and (k != "message.send" or target.provider not in {"shell", "unknown"})
            and (
                k != "request.respond"
                or (opened is not None and bool(opened.option_keys))
            )
            and (k != "session.resume" or bool(target.resume_provider))
        )
        snapshot = SessionSnapshot(
            session_id,
            version,
            target.title,
            target.host,
            target.cwd,
            None,
            target.provider,
            Conversation(target.conversation_id, "full", target.history_generation)
            if target.transcript
            else None,
            status,
            "default" if target.approval_mode == "normal" else target.approval_mode,
            (Pane("p0", target.pane_title, True),),
            (opened.request,) if opened else (),
            (*capabilities, "terminal.view"),
            last_activity,
        )
        self._snapshots[session_id] = snapshot
        if old != snapshot:
            self.hub.publish(
                "session.snapshot" if old is None else "session.updated",
                session_id,
                snapshot,
            )
        if target.available:
            await self._publish_messages(session_id, target)
        return self._snapshots[session_id]

    async def _source_version(self, target: Target) -> Target:
        if target.transcript is None:
            return target
        source_key = self._source_key(target)
        try:
            observation = await self.store.transcript_snapshot(
                source_key, target.transcript
            )
        except OSError as exc:
            if (
                isinstance(exc, FileNotFoundError)
                and target.provider == "claude"
                and target.foreground_verified
                and target.conversation_id
                and not self.store.has_transcript_observation(source_key)
            ):
                # Claude can create the JSONL only after its first input. Do not
                # confuse pending initial history with an unverified terminal.
                return replace(target, transcript=None)
            raise ProtocolError(
                503, "backend_unavailable", "Transcript source is unavailable"
            ) from exc
        return replace(
            target,
            history_generation=observation.generation,
            transcript_stamp=observation.stamp,
            transcript_digest=observation.record[3],
        )

    @staticmethod
    def _source_key(target: Target) -> str:
        return f"{target.id.casefold()}:{target.conversation_id}"

    async def _built(
        self,
        target: Target,
        *,
        source_range: tuple[int, int | None] | None = None,
        dirty_only: bool = False,
    ) -> tuple[MessageBuilder, tuple[Message, ...]]:
        """Advance the builder by appended bytes only, then select its messages."""
        if target.transcript is None or target.transcript_stamp is None:
            raise ProtocolError(
                503, "backend_unavailable", "Transcript observation is unavailable"
            )
        provider = target.transcript_provider or target.provider
        size = target.transcript_stamp[1]
        key = self._source_key(target)
        async with self._builder_lock:
            builder = self._builders.get(key)
            if (
                builder is None
                or (builder.conversation_id, builder.provider, builder.generation)
                != (target.conversation_id, provider, target.history_generation)
                or builder.offset > size
            ):
                self._drop_builder(key)
                builder = MessageBuilder(
                    target.conversation_id,
                    provider,
                    target.history_generation,
                    self.store.path.parent,
                )
                self._builders[key] = builder
            work = asyncio.ensure_future(
                asyncio.to_thread(
                    builder.update,
                    target.transcript,
                    target.transcript_stamp,
                    source_range,
                    dirty_only=dirty_only,
                    digest=target.transcript_digest,
                )
            )
            try:
                messages = await asyncio.shield(work)
            except asyncio.CancelledError:
                await self._settle(work, key, builder, dirty_only)
                raise
            except Exception as exc:
                # A failed update may leave the index half-written: rebuild next poll.
                if not isinstance(exc, TranscriptChangedError):
                    self._drop_builder(key)
                if not isinstance(exc, OSError):
                    raise
                raise ProtocolError(
                    503, "backend_unavailable", "Transcript is unavailable"
                ) from exc
        return builder, messages

    async def _settle(
        self,
        work: asyncio.Future[tuple[Message, ...]],
        key: str,
        builder: MessageBuilder,
        dirty_only: bool,
    ) -> None:
        """Hold the builder lock until the orphaned worker ends, then clean up."""
        while not work.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait({work})
        failure = None if work.cancelled() else work.exception()
        if failure is None and not work.cancelled():
            if dirty_only:  # the discarded result held the only record of them
                builder.dirty |= {message.id for message in work.result()}
        elif not isinstance(failure, TranscriptChangedError):
            self._drop_builder(key)

    def _drop_builder(self, key: str) -> None:
        builder = self._builders.pop(key, None)
        if builder is not None:
            builder.close()

    async def _messages(
        self,
        target: Target,
        *,
        source_range: tuple[int, int | None] | None = None,
    ) -> tuple[Message, ...]:
        if target.transcript is None:
            return ()
        _, messages = await self._built(target, source_range=source_range)
        return self._correlated(target, messages)

    def _correlated(
        self, target: Target, messages: tuple[Message, ...]
    ) -> tuple[Message, ...]:
        sid = next((sid for sid, t in self._targets.items() if t.id == target.id), "")
        return tuple(self._correlate(sid, message) for message in messages)

    async def remember_delivery(
        self,
        action_id: str,
        session_id: str,
        conversation_id: str,
        text: str,
        sent_at: float,
    ) -> None:
        async with self._refresh_lock:
            self.store.remember_delivery(
                action_id, session_id, conversation_id, text, sent_at
            )
            # A refresh may have observed the user entry before send acknowledgement.
            self._transcript_stamps.pop(session_id, None)
            self._republish.add(session_id)

    async def _publish_messages(self, session_id: str, target: Target) -> None:
        if target.transcript is None:
            return
        stamp = target.transcript_stamp
        if self._transcript_stamps.get(session_id) == stamp:
            return
        builder, selected = await self._built(
            target, dirty_only=session_id not in self._republish
        )
        revisions = self._revisions.setdefault(session_id, {})
        changed = False
        try:
            for message in self._correlated(target, selected):
                if revisions.get(message.id) != message.revision:
                    changed = True
                    self.hub.publish("message.upsert", session_id, message)
                    revisions[message.id] = message.revision
        except Exception:
            builder.dirty |= {message.id for message in selected}
            raise
        self._republish.discard(session_id)
        if stamp is not None:
            self._transcript_stamps[session_id] = stamp
        if changed:
            snapshot = replace(self._snapshots[session_id], last_activity=utc_now())
            self._snapshots[session_id] = snapshot
            self.hub.publish(
                "session.updated",
                session_id,
                {
                    "bindingVersion": snapshot.binding_version,
                    "lastActivity": snapshot.last_activity,
                },
            )

    def _correlate(self, session_id: str, message: Message) -> Message:
        if message.role != "user" or not message.time:
            return message
        try:
            timestamp = datetime.fromisoformat(message.time).timestamp()
        except ValueError:
            return message
        text = " ".join(p.text or "" for p in message.parts if p.type == "text")
        action_id, fresh = self.store.match_delivery(
            session_id, message.conversation_id, message.id, text, timestamp
        )
        if action_id and fresh:
            self.hub.publish(
                "action.updated",
                session_id,
                {"actionId": action_id, "outcome": "sent", "messageId": message.id},
            )
        return (
            replace(message, action_id=action_id, revision=message.revision + 1)
            if action_id
            else message
        )

    async def list_sessions(self) -> list[SessionSnapshot]:
        await self.refresh()
        return list(self._snapshots.values())

    async def get_session(self, session_id: str) -> SessionSnapshot:
        await self.refresh()
        try:
            return self._snapshots[session_id]
        except KeyError as exc:
            raise ProtocolError(404, "not_found", "Session not found") from exc

    async def current(self, session_id: str) -> tuple[Target, SessionSnapshot]:
        if session_id not in self._targets:
            await self.refresh()
        async with self._refresh_lock:
            target = self._targets.get(session_id)
            if target is None:
                raise ProtocolError(404, "not_found", "Session not found")
            return await self._update(await self.backend.inspect(target.id))

    async def for_target(self, target_id: str) -> tuple[Target, SessionSnapshot]:
        async with self._refresh_lock:
            return await self._update(await self.backend.inspect(target_id))

    async def history(
        self,
        session_id: str,
        cursor: str | None,
        limit: int,
        *,
        conversation: str | None = None,
    ) -> HistoryPage:
        if not 1 <= limit <= MAX_HISTORY_PAGE:
            raise ProtocolError(
                400, "invalid_request", "limit must be between 1 and 200"
            )
        target, _ = await self.current(session_id)
        if conversation is not None and conversation != target.conversation_id:
            raise ProtocolError(
                404, "not_found", "Conversation is no longer bound to this session"
            )
        messages = await self._messages(target)
        before = len(messages)
        if cursor:
            try:
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if (
                    decoded["conversation"] != target.conversation_id
                    or decoded.get("generation", 0) != target.history_generation
                ):
                    raise ValueError
                before = next(
                    i for i, m in enumerate(messages) if m.id == decoded["before"]
                )
            except (ValueError, KeyError, TypeError, StopIteration) as exc:
                raise ProtocolError(
                    400, "invalid_request", "Invalid cursor for this conversation"
                ) from exc
        start = max(0, before - limit)
        page = messages[start:before]
        prev = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "conversation": target.conversation_id,
                        "before": messages[start].id,
                        "generation": target.history_generation,
                    }
                ).encode()
            ).decode()
            if start
            else None
        )
        return HistoryPage(
            target.conversation_id or None,
            page,
            prev,
            "full" if target.transcript else "scrollback",
            target.history_generation,
        )

    async def history_by_source_range(
        self,
        session_id: str,
        start_byte: int,
        end_byte: int | None = None,
    ) -> HistoryPage:
        """Read canonical messages contributed by source records in a byte range.

        Telegram's unread view stores transcript byte offsets. Tool calls and
        results still normalize together; a result in the range includes its
        owning tool message even when the call itself preceded the range.
        """
        if start_byte < 0 or (end_byte is not None and end_byte < start_byte):
            raise ProtocolError(400, "invalid_request", "Invalid transcript byte range")
        target, _ = await self.current(session_id)
        messages = await self._messages(target, source_range=(start_byte, end_byte))
        return HistoryPage(
            target.conversation_id or None,
            messages,
            None,
            "full" if target.transcript else "scrollback",
            target.history_generation,
        )

    async def watch(
        self, session_ids: Collection[str] | None = None
    ) -> AsyncIterator[Event]:
        subscriber = self.hub.subscribe(session_ids)
        try:
            while True:
                yield await subscriber.get()
        finally:
            self.hub.unsubscribe(subscriber)

    async def execute(
        self,
        session_id: str,
        action: Action,
        *,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> tuple[ActionResult, int]:
        return await self.executor.execute(
            session_id,
            action,
            pane_id=pane_id,
            expected_prompt=expected_prompt,
        )

    async def execute_internal(
        self,
        session_id: str,
        action: Action,
        *,
        pane_id: str | None = None,
        expected_prompt: str | None = None,
    ) -> tuple[ActionResult, int]:
        return await self.executor.execute_internal(
            session_id,
            action,
            pane_id=pane_id,
            expected_prompt=expected_prompt,
        )

    async def create_session(
        self,
        data: dict[str, Any],
        *,
        options: CreationOptions | None = None,
    ) -> SessionSnapshot:
        _, snapshot = await self.create_target(data, options=options)
        return snapshot

    async def create_target(
        self,
        data: dict[str, Any],
        *,
        options: CreationOptions | None = None,
    ) -> tuple[Target, SessionSnapshot]:
        async with self._refresh_lock:
            created = (
                await self.backend.create(data)
                if options is None
                else await self.backend.create(data, trusted_options=options)
            )
            return await self._update(created)

    async def split_session(
        self,
        session_id: str,
        *,
        expected_binding_version: int,
        action_id: str,
    ) -> str | None:
        action = Action(action_id, expected_binding_version, "session.split", {})
        result, status = await self.execute_internal(session_id, action)
        if result.outcome == "sent" and result.data:
            pane_id = result.data.get("paneId")
            if isinstance(pane_id, str):
                return pane_id
        if result.outcome == "rejected":
            error = result.error or {}
            raise ProtocolError(
                status,
                error.get("code", "backend_unavailable"),
                error.get("message", "Could not confirm split"),
                error.get("details", {}),
            )
        return None

    async def terminal(self, session_id: str, pane: str = "p0") -> dict[str, Any]:
        target, _ = await self.current(session_id)
        return await self.backend.terminal(target, pane)

    async def commands(self, session_id: str) -> list[dict[str, str]]:
        target, _ = await self.current(session_id)
        return await self.backend.commands(target)

    async def observe_hook(self, event: HookEvent, prefix: str) -> None:
        async with self._refresh_lock:
            await self._observe_hook(event, prefix)

    def defer_hook(self, event: HookEvent, prefix: str) -> tuple[str, ...]:
        identity = self._hook_identity(event)
        if identity is None:
            return ()
        sessions = tuple(
            sid
            for sid, _ in self._hook_targets(event, prefix)
            if self.requests.observe_identity(sid, identity)
        )
        for sid in sessions:
            self.requests.defer(sid)
        return sessions

    def resolve_hook(self, sessions: tuple[str, ...]) -> None:
        for sid in sessions:
            self.requests.resolve(sid)

    @staticmethod
    def _hook_identity(event: HookEvent) -> HookIdentity | None:
        tool_name = event.data.get("tool_name", "")
        if event.event_type in {"Stop", "SessionEnd"}:
            kind = "turn_ended"
            tool_name = ""
        elif event.event_type == "PermissionRequest" or (
            event.event_type == "PreToolUse" and tool_name == "AskUserQuestion"
        ):
            kind = "question" if tool_name == "AskUserQuestion" else "permission"
        else:
            return None
        digest = event.data.get("tool_input_hash", "")
        return HookIdentity(
            event.session_id,
            kind,
            tool_name if isinstance(tool_name, str) else "",
            digest if isinstance(digest, str) else "",
            event.timestamp,
        )

    def _hook_targets(self, event: HookEvent, prefix: str) -> list[tuple[str, Target]]:
        target_id = event.window_key.removeprefix(prefix)
        if not event.window_key.startswith(prefix):
            return []
        return [
            (sid, target)
            for sid, target in self._targets.items()
            if target.id.casefold() == target_id.casefold()
            and target.conversation_id == event.session_id
        ]

    async def _observe_hook(self, event: HookEvent, prefix: str) -> None:
        identity = self._hook_identity(event)
        if identity is None:
            return
        for sid, target in self._hook_targets(event, prefix):
            if not self.requests.observe_identity(sid, identity):
                continue
            if event.event_type in {"Stop", "SessionEnd"}:
                self.requests.close(sid, "turn_ended")
                continue
            if event.event_type not in {"PermissionRequest", "PreToolUse"}:
                continue
            tool_name = event.data.get("tool_name")
            if event.event_type == "PreToolUse" and tool_name != "AskUserQuestion":
                continue
            if not await self._match_hook(sid, target, event):
                self.requests.discard_identity(sid, identity)

    async def _match_hook(self, sid: str, target: Target, event: HookEvent) -> bool:
        tool_name = event.data.get("tool_name")
        if not isinstance(tool_name, str):
            return False
        # Hook records can arrive after the last poll; do not reconcile requests
        # while refreshing this evidence, as a newer queued hook may own them.
        target = await self._source_version(await self.backend.inspect(target.id))
        if target.conversation_id != event.session_id:
            return False
        for message in reversed(await self._messages(target)):
            for part in message.parts:
                if (
                    part.type != "tool"
                    or part.name != tool_name
                    or not isinstance(part.input, dict)
                ):
                    continue
                digest = hook_input_hash(part.input)
                if digest == event.data.get("tool_input_hash"):
                    self.requests.observe_hook(
                        sid,
                        PendingHook(
                            event.session_id,
                            "question"
                            if tool_name == "AskUserQuestion"
                            else "permission",
                            tool_name,
                            part.input,
                            event.timestamp,
                        ),
                    )
                    return True
        return False
