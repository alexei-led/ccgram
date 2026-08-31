"""Tests for the delivery receipt contract (settled-prefix policy, #205)."""

import pytest

from ccgram.delivery_contract import (
    DeliveryOutcome,
    DeliveryReceipt,
    settled_prefix,
    settled_run_offset,
)


def _ready(checkpoint: int | None) -> DeliveryReceipt:
    receipt = DeliveryReceipt(checkpoint=checkpoint)
    receipt.track()
    receipt.settle(DeliveryOutcome.DELIVERED)
    receipt.close()
    return receipt


def _failed(checkpoint: int) -> DeliveryReceipt:
    receipt = DeliveryReceipt(checkpoint=checkpoint)
    receipt.track()
    receipt.settle(DeliveryOutcome.FAILED)
    receipt.close()
    return receipt


def _tracked(checkpoint: int) -> DeliveryReceipt:
    receipt = DeliveryReceipt(checkpoint=checkpoint)
    receipt.track()
    return receipt


class TestSettledPrefix:
    """The leading commit-ready run, fenced by the first receipt that
    cannot commit (pending, failed, unclosed, or uncheckpointed)."""

    @pytest.mark.parametrize(
        ("receipts", "expected_len"),
        [
            ([], 0),
            ([_ready(10)], 1),
            ([_ready(10), _ready(20), _ready(30)], 3),
            ([_ready(10), _tracked(20)], 1),
            ([_ready(10), _failed(20), _ready(30)], 1),
            ([_ready(10), DeliveryReceipt(checkpoint=30), _ready(40)], 1),
            ([_ready(10), _ready(None), _ready(40)], 1),
            ([_tracked(10), _ready(20)], 0),
            ([DeliveryReceipt(checkpoint=10), _ready(20)], 0),
        ],
    )
    def test_prefix_run(self, receipts, expected_len: int) -> None:
        assert len(settled_prefix(receipts)) == expected_len

    def test_prefix_members_are_the_leading_ready_receipts(self) -> None:
        first, second = _ready(10), _ready(20)
        fence = _tracked(30)
        tail = _ready(40)
        prefix = settled_prefix([first, second, fence, tail])
        assert prefix == [first, second]


class TestSettledRunOffset:
    """The durable commit value for a settled run under the shared-batch
    checkpoint tie rule (persistence lags delivery by at most one batch)."""

    @pytest.mark.parametrize(
        ("run", "fence", "expected"),
        [
            ([_ready(10)], None, 10),
            ([_ready(10), _ready(20)], None, 20),
            ([_ready(10)], _tracked(20), 10),
            ([_ready(10)], _failed(20), 10),
            ([_ready(10)], _ready(30), 10),
            # The tie: a sibling sharing the batch-end checkpoint sits
            # below it, so the run's checkpoint must NOT commit; the
            # largest strictly-below checkpoint does.
            ([_ready(10)], _tracked(10), None),
            ([_ready(10)], _ready(10), None),
            ([_ready(10), _ready(20)], _tracked(20), 10),
            ([_ready(300), _ready(400)], _tracked(400), 300),
            # Unorderable fence: conservative block.
            ([_ready(10)], DeliveryReceipt(checkpoint=None), None),
            ([], None, None),
            ([], _tracked(10), None),
        ],
    )
    def test_run_offset(self, run, fence, expected) -> None:
        assert settled_run_offset(run, fence) == expected
