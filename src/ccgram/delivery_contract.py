"""Dependency-light acknowledgement contract for transcript delivery.

Producers create a receipt for one transcript item. Queue workers settle each
tracked outbound task. Persistence advances over the leading run of receipts
that closed without a failed task (see ``settled_prefix`` and
``settled_run_offset``); the first pending, failed, or unclosed receipt, or
one without a checkpoint, fences the rest. This module deliberately knows
nothing about Telegram, handlers, queues, or session monitoring.
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


def settled_prefix(receipts: list[DeliveryReceipt]) -> list[DeliveryReceipt]:
    """The leading run of receipts whose delivery may commit now.

    The queue settles tasks in dispatch order, so the leading run of
    commit-ready receipts is durable progress even while later receipts of
    the same session are still in flight (#205): waiting for every receipt
    to close never commits under sustained output, and a restart then
    replays the whole settled run. The first pending, failed, or unclosed
    receipt, or one without a checkpoint, fences the rest; replay resumes
    from the fence, preserving at-least-once delivery.

    The run alone does not fix a commit VALUE: receipts registered in one
    parse cycle share that cycle's batch-end checkpoint, so the run's last
    checkpoint can equal the fence's. Use ``settled_run_offset`` for the
    durable offset.
    """
    prefix: list[DeliveryReceipt] = []
    for receipt in receipts:
        if receipt.checkpoint is None or not receipt.commit_ready:
            break
        prefix.append(receipt)
    return prefix


def settled_run_offset(
    run: list[DeliveryReceipt], fence: DeliveryReceipt | None
) -> int | None:
    """Durable commit offset for a settled run, honoring checkpoint ties.

    Receipts registered in one parse cycle share that cycle's batch-end
    checkpoint: an unsettled sibling sitting below the shared checkpoint
    would be lost by a restart replaying from it. The offset is therefore
    the run's last checkpoint STRICTLY below the fence's, so persistence
    marks only fully settled batches and lags delivery by at most one
    in-flight batch. Without a fence the whole run commits. A fence
    without a checkpoint is unorderable and blocks the commit
    conservatively.
    """
    if not run:
        return None
    if fence is None:
        boundary = run[-1].checkpoint
        assert boundary is not None  # settled_prefix fences None out
        return boundary
    fence_checkpoint = fence.checkpoint
    if fence_checkpoint is None:
        return None
    for receipt in reversed(run):
        checkpoint = receipt.checkpoint
        assert checkpoint is not None  # settled_prefix fences None out
        if checkpoint < fence_checkpoint:
            return checkpoint
    return None
