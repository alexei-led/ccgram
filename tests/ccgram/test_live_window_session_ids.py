"""Tests for ``live_window_session_ids`` — the backend-neutral join used by
herdr restart re-resolution to map a live window id to its agent session id.

Pure function, no SessionManager wiring required.
"""

from __future__ import annotations

from ccgram.session_map import live_window_session_ids


def test_tmux_key_matches_by_suffix() -> None:
    raw = {"ccgram:@12": {"session_id": "S1", "cwd": "/repo"}}
    assert live_window_session_ids(raw, {"@12"}) == {"@12": "S1"}


def test_herdr_key_with_colon_in_window_id() -> None:
    # herdr key is ``herdr:w2:p1`` — the id itself contains a colon, so a naive
    # rsplit(":", 1) would break. Suffix match handles it.
    raw = {"herdr:w2:p1": {"session_id": "S1"}}
    assert live_window_session_ids(raw, {"w2:p1"}) == {"w2:p1": "S1"}


def test_stale_entry_not_live_is_ignored() -> None:
    # The session_map still lists the pre-restart pane id; only live ids count.
    raw = {
        "herdr:w2:p1": {"session_id": "S1"},  # stale (not live)
        "herdr:w3:p1": {"session_id": "S2"},  # live
    }
    assert live_window_session_ids(raw, {"w3:p1"}) == {"w3:p1": "S2"}


def test_missing_session_id_skipped() -> None:
    raw = {"herdr:w2:p1": {"cwd": "/repo"}}
    assert live_window_session_ids(raw, {"w2:p1"}) == {}


def test_non_dict_entry_skipped() -> None:
    raw = {"herdr:w2:p1": "garbage", "herdr:w3:p1": {"session_id": "S2"}}
    assert live_window_session_ids(raw, {"w2:p1", "w3:p1"}) == {"w3:p1": "S2"}


def test_similar_ids_do_not_cross_match() -> None:
    # ``w2:p1`` must not match the ``herdr:w12:p1`` key (the leading ':' anchor
    # prevents the partial suffix match).
    raw = {"herdr:w12:p1": {"session_id": "S12"}}
    assert live_window_session_ids(raw, {"w2:p1"}) == {}


def test_empty_inputs() -> None:
    assert live_window_session_ids({}, set()) == {}
    assert live_window_session_ids({}, {"w2:p1"}) == {}


# --- Tab-identity (Task 1 flip): ids are w2:t1, not w2:p1 ---


def test_herdr_tab_id_key_matches_by_suffix() -> None:
    # After Task 1, herdr uses tab ids (``wN:tM``); the suffix match must work
    # identically because ``live_window_session_ids`` is format-agnostic.
    raw = {"herdr:w2:t1": {"session_id": "S1"}}
    assert live_window_session_ids(raw, {"w2:t1"}) == {"w2:t1": "S1"}


def test_herdr_tab_id_no_cross_match_on_similar_prefix() -> None:
    # ``w2:t1`` must NOT match ``herdr:w12:t1`` — the leading ``:`` in the
    # suffix anchor prevents partial-prefix collisions.
    raw = {"herdr:w12:t1": {"session_id": "S12"}}
    assert live_window_session_ids(raw, {"w2:t1"}) == {}


def test_herdr_tab_id_stale_pre_restart_entry_ignored() -> None:
    # After restart, session_map still has the pre-restart entry (``herdr:w2:t1``);
    # the live set contains only the new id (``herdr:w3:t1``).  Only live ids count.
    raw = {
        "herdr:w2:t1": {"session_id": "S1"},  # stale — not in live set
        "herdr:w3:t1": {"session_id": "S1"},  # new id, same session
    }
    assert live_window_session_ids(raw, {"w3:t1"}) == {"w3:t1": "S1"}
