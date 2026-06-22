"""Round-trip tests for pane callback-data encoding (Task 5).

Verifies that CB_PANE_DELIMITER (``|``) is used as the window↔pane
separator in all four parsers and their matching builders, so both tmux
ids (``@12`` / ``%5``) and herdr ids (``w2:t1`` / ``w2:p1``) round-trip
without colliding on the colons inside herdr ids.

Also asserts the 64-byte Telegram callback-data limit for the worst-case
herdr id pair.
"""

import pytest

from ccgram.handlers.callback_data import (
    CB_ASK_ENTER,
    CB_KEYS_PREFIX,
    CB_LIVE_START,
    CB_LIVE_STOP,
    CB_PANE_DELIMITER,
    CB_PANE_RENAME,
    CB_PANE_SCREENSHOT,
    CB_PANE_SUBSCRIBE,
)
from ccgram.handlers.callback_helpers import parse_target
from ccgram.handlers.interactive.interactive_callbacks import match_interactive_prefix
from ccgram.handlers.live.pane_callbacks import _parse_target as pane_parse_target


# ---------------------------------------------------------------------------
# CB_PANE_DELIMITER sanity
# ---------------------------------------------------------------------------


def test_pane_delimiter_is_pipe() -> None:
    assert CB_PANE_DELIMITER == "|"
    assert CB_PANE_DELIMITER != ":"


# ---------------------------------------------------------------------------
# parse_target (shared helper — used by screenshot_callbacks + status_bar_actions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,expected_window,expected_pane",
    [
        # tmux: window only
        ("@12", "@12", None),
        # tmux: window + pane
        (f"@12{CB_PANE_DELIMITER}%5", "@12", "%5"),
        # herdr: tab only (contains colons — must NOT split on them)
        ("w2:t1", "w2:t1", None),
        # herdr: tab + pane
        (f"w2:t1{CB_PANE_DELIMITER}w2:p1", "w2:t1", "w2:p1"),
        # herdr: large ids
        (f"w99:t99{CB_PANE_DELIMITER}w99:p99", "w99:t99", "w99:p99"),
    ],
)
def test_parse_target(
    target: str, expected_window: str, expected_pane: str | None
) -> None:
    window_id, pane_id = parse_target(target)
    assert window_id == expected_window
    assert pane_id == expected_pane


def test_parse_target_herdr_bare_no_split() -> None:
    """A bare herdr window id must parse to (window_id, None), not split on ':'."""
    window_id, pane_id = parse_target("w2:t1")
    assert window_id == "w2:t1"
    assert pane_id is None


# ---------------------------------------------------------------------------
# pane_callbacks._parse_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix,data_suffix,expected_window,expected_pane",
    [
        # tmux
        (CB_PANE_SUBSCRIBE, f"@12{CB_PANE_DELIMITER}%5", "@12", "%5"),
        (CB_PANE_RENAME, f"@0{CB_PANE_DELIMITER}%3", "@0", "%3"),
        # herdr
        (CB_PANE_SUBSCRIBE, f"w2:t1{CB_PANE_DELIMITER}w2:p1", "w2:t1", "w2:p1"),
        (CB_PANE_RENAME, f"w99:t99{CB_PANE_DELIMITER}w99:p99", "w99:t99", "w99:p99"),
    ],
)
def test_pane_parse_target(
    prefix: str, data_suffix: str, expected_window: str, expected_pane: str
) -> None:
    data = prefix + data_suffix
    result = pane_parse_target(data, prefix)
    assert result is not None
    window_id, pane_id = result
    assert window_id == expected_window
    assert pane_id == expected_pane


def test_pane_parse_target_missing_delimiter_returns_none() -> None:
    # Data without the delimiter must fail gracefully
    result = pane_parse_target(CB_PANE_SUBSCRIBE + "@12", CB_PANE_SUBSCRIBE)
    assert result is None


def test_pane_parse_target_herdr_no_internal_split() -> None:
    # herdr window id without pane — no delimiter — must return None
    result = pane_parse_target(CB_PANE_SUBSCRIBE + "w2:t1", CB_PANE_SUBSCRIBE)
    assert result is None


# ---------------------------------------------------------------------------
# match_interactive_prefix (interactive_callbacks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data,expected_prefix,expected_window,expected_pane",
    [
        # window only
        (f"{CB_ASK_ENTER}@12", CB_ASK_ENTER, "@12", None),
        # tmux: window + pane
        (f"{CB_ASK_ENTER}@12{CB_PANE_DELIMITER}%5", CB_ASK_ENTER, "@12", "%5"),
        # herdr: tab only (colons in id must not split)
        (f"{CB_ASK_ENTER}w2:t1", CB_ASK_ENTER, "w2:t1", None),
        # herdr: tab + pane
        (
            f"{CB_ASK_ENTER}w2:t1{CB_PANE_DELIMITER}w2:p1",
            CB_ASK_ENTER,
            "w2:t1",
            "w2:p1",
        ),
    ],
)
def test_match_interactive_prefix(
    data: str,
    expected_prefix: str,
    expected_window: str,
    expected_pane: str | None,
) -> None:
    result = match_interactive_prefix(data)
    assert result is not None
    prefix, window_id, pane_id = result
    assert prefix == expected_prefix
    assert window_id == expected_window
    assert pane_id == expected_pane


def test_match_interactive_prefix_herdr_bare_no_split() -> None:
    """herdr tab id ``w2:t1`` must parse as window_id without splitting on ':'."""
    result = match_interactive_prefix(f"{CB_ASK_ENTER}w2:t1")
    assert result is not None
    _, window_id, pane_id = result
    assert window_id == "w2:t1"
    assert pane_id is None


# ---------------------------------------------------------------------------
# Builder round-trips — pane_buttons
# ---------------------------------------------------------------------------


def test_pane_buttons_builder_round_trips_tmux() -> None:
    from ccgram.handlers.live.pane_callbacks import build_pane_buttons

    buttons = build_pane_buttons("@12", "%5", subscribed=False)
    raw = buttons[0].callback_data
    assert isinstance(raw, str)
    assert f"@12{CB_PANE_DELIMITER}%5" in raw
    result = pane_parse_target(raw, CB_PANE_SCREENSHOT)
    assert result == ("@12", "%5")


def test_pane_buttons_builder_round_trips_herdr() -> None:
    from ccgram.handlers.live.pane_callbacks import build_pane_buttons

    buttons = build_pane_buttons("w2:t1", "w2:p1", subscribed=True)
    raw = buttons[0].callback_data
    assert isinstance(raw, str)
    assert f"w2:t1{CB_PANE_DELIMITER}w2:p1" in raw
    result = pane_parse_target(raw, CB_PANE_SCREENSHOT)
    assert result == ("w2:t1", "w2:p1")


# ---------------------------------------------------------------------------
# Builder round-trips — screenshot keyboard
# ---------------------------------------------------------------------------


def _all_callback_data(kb_markup: object) -> list[str]:
    """Extract all non-None str callback_data values from an InlineKeyboardMarkup."""
    from telegram import InlineKeyboardMarkup

    assert isinstance(kb_markup, InlineKeyboardMarkup)
    result = []
    for row in kb_markup.inline_keyboard:
        for btn in row:
            cd = btn.callback_data
            if isinstance(cd, str):
                result.append(cd)
    return result


def test_screenshot_keyboard_target_tmux() -> None:
    from ccgram.handlers.live.screenshot_callbacks import build_screenshot_keyboard

    kb = build_screenshot_keyboard("@12", pane_id="%5")
    datas = _all_callback_data(kb)
    assert any(f"@12{CB_PANE_DELIMITER}%5" in d for d in datas)


def test_screenshot_keyboard_target_herdr() -> None:
    from ccgram.handlers.live.screenshot_callbacks import build_screenshot_keyboard

    kb = build_screenshot_keyboard("w2:t1", pane_id="w2:p1")
    datas = _all_callback_data(kb)
    assert any(f"w2:t1{CB_PANE_DELIMITER}w2:p1" in d for d in datas)


def test_screenshot_keyboard_window_only() -> None:
    from ccgram.handlers.live.screenshot_callbacks import build_screenshot_keyboard

    kb = build_screenshot_keyboard("@5")
    datas = _all_callback_data(kb)
    # No delimiter should appear — window-only target
    assert not any(CB_PANE_DELIMITER in d for d in datas)


# ---------------------------------------------------------------------------
# Builder round-trips — live keyboard
# ---------------------------------------------------------------------------


def test_live_keyboard_round_trips_tmux() -> None:
    from ccgram.handlers.live.live_view import build_live_keyboard

    kb = build_live_keyboard("@12", pane_id="%5")
    datas = _all_callback_data(kb)
    stop_datas = [d for d in datas if d.startswith(CB_LIVE_STOP)]
    assert stop_datas, "No LIVE_STOP button found"
    target = stop_datas[0][len(CB_LIVE_STOP) :]
    window_id, pane_id = parse_target(target)
    assert window_id == "@12"
    assert pane_id == "%5"


def test_live_keyboard_round_trips_herdr() -> None:
    from ccgram.handlers.live.live_view import build_live_keyboard

    kb = build_live_keyboard("w2:t1", pane_id="w2:p1")
    datas = _all_callback_data(kb)
    stop_datas = [d for d in datas if d.startswith(CB_LIVE_STOP)]
    assert stop_datas, "No LIVE_STOP button found"
    target = stop_datas[0][len(CB_LIVE_STOP) :]
    window_id, pane_id = parse_target(target)
    assert window_id == "w2:t1"
    assert pane_id == "w2:p1"


def test_live_keyboard_window_only() -> None:
    from ccgram.handlers.live.live_view import build_live_keyboard

    kb = build_live_keyboard("@5")
    datas = _all_callback_data(kb)
    stop_datas = [d for d in datas if d.startswith(CB_LIVE_STOP)]
    assert stop_datas
    target = stop_datas[0][len(CB_LIVE_STOP) :]
    window_id, pane_id = parse_target(target)
    assert window_id == "@5"
    assert pane_id is None


# ---------------------------------------------------------------------------
# CB_KEYS_PREFIX path (status_bar_actions._handle_keys uses parse_target)
# ---------------------------------------------------------------------------


def test_kb_prefix_target_tmux() -> None:
    """kb:<key_id>:<window_id>|<pane_id> — parse_target recovers the pair."""
    target = f"@12{CB_PANE_DELIMITER}%5"
    data = f"{CB_KEYS_PREFIX}ent:{target}"

    rest = data[len(CB_KEYS_PREFIX) :]
    colon_idx = rest.find(":")
    recovered_target = rest[colon_idx + 1 :]
    w, p = parse_target(recovered_target)
    assert w == "@12"
    assert p == "%5"


def test_kb_prefix_target_herdr() -> None:
    """kb:<key_id>:w2:t1|w2:p1 — parse_target recovers the herdr pair."""
    target = f"w2:t1{CB_PANE_DELIMITER}w2:p1"
    data = f"{CB_KEYS_PREFIX}ent:{target}"

    rest = data[len(CB_KEYS_PREFIX) :]
    colon_idx = rest.find(":")
    recovered_target = rest[colon_idx + 1 :]
    w, p = parse_target(recovered_target)
    assert w == "w2:t1"
    assert p == "w2:p1"


# ---------------------------------------------------------------------------
# 64-byte Telegram callback-data limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        # interactive — longest prefix + worst herdr pair
        f"{CB_ASK_ENTER}w99:t99{CB_PANE_DELIMITER}w99:p99",
        # kb: — key_id "spc" (longest) + worst herdr pair
        f"{CB_KEYS_PREFIX}spc:w99:t99{CB_PANE_DELIMITER}w99:p99",
        # pn:ss: + worst herdr pair
        f"{CB_PANE_SCREENSHOT}w99:t99{CB_PANE_DELIMITER}w99:p99",
        # lv:go: + worst herdr pair
        f"{CB_LIVE_START}w99:t99{CB_PANE_DELIMITER}w99:p99",
        # lv:stop: + worst herdr pair
        f"{CB_LIVE_STOP}w99:t99{CB_PANE_DELIMITER}w99:p99",
    ],
)
def test_64_byte_limit(data: str) -> None:
    assert len(data) <= 64, f"Exceeds 64 bytes ({len(data)}): {data!r}"
