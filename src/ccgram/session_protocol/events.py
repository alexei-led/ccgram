"""Bounded durable replay and non-blocking fan-out to remote observers."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from typing import Any

from .models import wire


class ResyncRequiredError(Exception):
    """The subscriber fell behind; reconnect and load snapshots/history."""


@dataclass(frozen=True, slots=True)
class Event:
    epoch: str
    type: str
    session_id: str
    data: Any
    seq: int | None

    def body(self) -> dict[str, Any]:
        result = {
            "v": 1,
            "epoch": self.epoch,
            "type": self.type,
            "class": "durable" if self.seq is not None else "ephemeral",
            "sessionId": self.session_id,
            "data": wire(self.data),
        }
        if self.seq is not None:
            result["seq"] = self.seq
        return result


class Subscription:
    def __init__(self, session_ids: Collection[str] | None, capacity: int) -> None:
        self.session_ids = None if session_ids is None else set(session_ids)
        self.queue: asyncio.Queue[Event | None] = asyncio.Queue(capacity)
        self.lagged = False

    def accepts(self, event: Event) -> bool:
        return self.session_ids is None or event.session_id in self.session_ids

    async def get(self) -> Event:
        event = await self.queue.get()
        if self.lagged or event is None:
            raise ResyncRequiredError
        return event

    def offer(self, event: Event) -> None:
        if self.lagged or not self.accepts(event):
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.lagged = True
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queue.put_nowait(None)


class EventHub:
    def __init__(
        self,
        *,
        capacity: int = 1000,
        max_age: float = 600,
        queue_capacity: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.epoch = "e_" + secrets.token_urlsafe(16)
        self.seq = 0
        self.capacity = capacity
        self.max_age = max_age
        self.queue_capacity = queue_capacity
        self.clock = clock
        self._ring: deque[tuple[float, Event]] = deque()
        self._gap_seq: int | None = None
        self._subscribers: set[Subscription] = set()
        self._durable_consumers: dict[str, Callable[[Any], Awaitable[None]]] = {}

    def register_durable_consumer(
        self, name: str, handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """Register an awaited in-process sink, separate from remote replay."""
        if not name:
            raise ValueError("durable consumer name must not be empty")
        if name in self._durable_consumers:
            raise ValueError(f"durable consumer already registered: {name}")
        self._durable_consumers[name] = handler

    def unregister_durable_consumer(self, name: str) -> None:
        self._durable_consumers.pop(name, None)

    def durable_consumer_names(self) -> tuple[str, ...]:
        return tuple(self._durable_consumers)

    async def dispatch_durable(self, name: str, payload: Any) -> None:
        """Await one named in-process sink; callers own retry/checkpoint policy."""
        handler = self._durable_consumers.get(name)
        if handler is None:
            raise KeyError(f"durable consumer is not registered: {name}")
        await handler(payload)

    def _prune(self) -> None:
        cutoff = self.clock() - self.max_age
        while self._ring and (
            len(self._ring) > self.capacity or self._ring[0][0] <= cutoff
        ):
            self._ring.popleft()

    def publish(
        self, event_type: str, session_id: str, data: Any, *, durable: bool = True
    ) -> Event:
        if durable:
            self.seq += 1
        # Freeze the payload at publication, including mutable provider tool data.
        event = Event(
            self.epoch,
            event_type,
            session_id,
            wire(data),
            self.seq if durable else None,
        )
        if durable:
            self._ring.append((self.clock(), event))
            self._prune()
        for subscriber in self._subscribers:
            subscriber.offer(event)
        return event

    def mark_gap(self) -> None:
        """Record that message events stopped being published at the current seq.

        A resume at or before this point cannot replay exactly, so it fails and
        the client reloads its snapshots and history.
        """
        self._gap_seq = self.seq

    def replay(self, epoch: str, seq: int) -> tuple[bool, list[Event]]:
        self._prune()
        if epoch != self.epoch or seq < 0 or seq > self.seq:
            return False, []
        if self._gap_seq is not None and seq <= self._gap_seq:
            return False, []
        if seq == self.seq:
            return True, []
        if not self._ring or seq < (self._ring[0][1].seq or 0) - 1:
            return False, []
        return True, [event for _, event in self._ring if (event.seq or 0) > seq]

    def has_subscriber(self, session_id: str) -> bool:
        """Whether a live subscription receives events for this session."""
        return any(
            subscriber.session_ids is None or session_id in subscriber.session_ids
            for subscriber in self._subscribers
        )

    def subscribe(self, session_ids: Collection[str] | None = None) -> Subscription:
        subscriber = Subscription(session_ids, self.queue_capacity)
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscription) -> None:
        self._subscribers.discard(subscriber)
