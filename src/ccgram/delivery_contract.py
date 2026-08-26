"""Dependency-light acknowledgement contract for transcript delivery.

Producers create a receipt for one transcript item. Queue workers settle each
tracked outbound task. Persistence may advance only after every receipt closes
without a failed task. This module deliberately knows nothing about Telegram,
handlers, queues, or session monitoring.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum


class DeliveryOutcome(Enum):
    """Terminal result of one outbound task at the delivery boundary."""

    DELIVERED = "delivered"
    INTENTIONALLY_DROPPED = "intentionally_dropped"
    FAILED = "failed"


@dataclass
class DeliveryReceipt:
    """Acknowledgement for all outbound tasks created from one transcript item."""

    checkpoint: int | None = None
    _pending: int = 0
    _closed: bool = False
    failed: bool = False

    def track(self) -> None:
        if self._closed:
            raise RuntimeError("cannot add work to a closed delivery receipt")
        self._pending += 1

    def settle(self, outcome: DeliveryOutcome) -> None:
        if self._pending <= 0:
            raise RuntimeError("delivery receipt settled without queued work")
        self._pending -= 1
        if outcome is DeliveryOutcome.FAILED:
            self.failed = True

    def fail(self) -> None:
        """Record a producer-side failure that prevented safe enqueueing."""
        self.failed = True

    def close(self) -> None:
        self._closed = True

    @property
    def commit_ready(self) -> bool:
        return self._closed and self._pending == 0 and not self.failed


_active_receipt: ContextVar[DeliveryReceipt | None] = ContextVar(
    "delivery_receipt", default=None
)


def new_delivery_receipt(*, checkpoint: int | None = None) -> DeliveryReceipt:
    """Create an acknowledgement token for one producer delivery cycle."""
    return DeliveryReceipt(checkpoint=checkpoint)


def activate_delivery_receipt(
    receipt: DeliveryReceipt,
) -> Token[DeliveryReceipt | None]:
    """Associate outbound work created in this context with ``receipt``."""
    return _active_receipt.set(receipt)


def deactivate_delivery_receipt(token: Token[DeliveryReceipt | None]) -> None:
    """End a producer cycle without leaking its receipt into later work."""
    _active_receipt.reset(token)


def get_active_delivery_receipt() -> DeliveryReceipt | None:
    """Return the receipt associated with the current producer context."""
    return _active_receipt.get()


def delivery_receipts_ready(receipts: list[DeliveryReceipt]) -> bool:
    """Whether every receipt is explicitly acknowledged or intentionally dropped."""
    return all(receipt.commit_ready for receipt in receipts)
