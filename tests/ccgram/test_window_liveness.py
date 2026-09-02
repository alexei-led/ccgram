from ccgram.multiplexer.base import WindowRef
from ccgram.multiplexer.window_liveness import (
    is_window_live,
    note_live_windows,
    reset_window_liveness,
)


def setup_function() -> None:
    reset_window_liveness()


def teardown_function() -> None:
    reset_window_liveness()


def test_tracked_window_missing_from_confirmed_listing_is_dead() -> None:
    note_live_windows(
        [WindowRef(window_id="live", window_name="live", cwd="/tmp")],
        tracked_window_ids=["closed"],
    )

    assert is_window_live("live")
    assert not is_window_live("closed")


def test_unseen_window_fails_open_until_it_is_tracked() -> None:
    note_live_windows([], tracked_window_ids=[])

    assert is_window_live("newly-created")


def test_reset_returns_to_unknown_fail_open_state() -> None:
    note_live_windows([], tracked_window_ids=["closed"])
    reset_window_liveness()

    assert is_window_live("closed")
