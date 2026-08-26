"""Tests for command history storage and recall."""

import pytest

from collections.abc import Iterator

from ccgram.handlers.command_history import (
    HISTORY_MAX,
    _history,
    clear_history,
    get_history,
    record_command,
    truncate_for_display,
)


@pytest.fixture(autouse=True)
def _clean_history() -> Iterator[None]:
    _history.clear()
    yield
    _history.clear()


class TestRecordCommand:
    def test_stores_command(self) -> None:
        record_command(1, 100, "hello")
        assert get_history(1, 100) == ["hello"]

    def test_newest_first(self) -> None:
        record_command(1, 100, "first")
        record_command(1, 100, "second")
        assert get_history(1, 100) == ["second", "first"]

    def test_deduplicates_consecutive(self) -> None:
        record_command(1, 100, "same")
        record_command(1, 100, "same")
        record_command(1, 100, "same")
        assert get_history(1, 100) == ["same"]

    def test_non_consecutive_duplicates_kept(self) -> None:
        record_command(1, 100, "a")
        record_command(1, 100, "b")
        record_command(1, 100, "a")
        assert get_history(1, 100) == ["a", "b", "a"]

    def test_caps_at_history_max(self) -> None:
        for i in range(HISTORY_MAX + 5):
            record_command(1, 100, f"cmd-{i}")
        history = get_history(1, 100)
        assert len(history) == HISTORY_MAX
        assert history[0] == f"cmd-{HISTORY_MAX + 4}"
        assert history[-1] == "cmd-5"


class TestGetHistory:
    def test_limit(self) -> None:
        for i in range(10):
            record_command(1, 100, f"cmd-{i}")
        assert len(get_history(1, 100, limit=2)) == 2
        assert get_history(1, 100, limit=2) == ["cmd-9", "cmd-8"]

    def test_unknown_key_returns_empty(self) -> None:
        assert get_history(999, 999) == []

    def test_users_isolated(self) -> None:
        record_command(1, 100, "user1-cmd")
        record_command(2, 100, "user2-cmd")
        assert get_history(1, 100) == ["user1-cmd"]
        assert get_history(2, 100) == ["user2-cmd"]

    def test_topics_isolated(self) -> None:
        record_command(1, 100, "topic100")
        record_command(1, 200, "topic200")
        assert get_history(1, 100) == ["topic100"]
        assert get_history(1, 200) == ["topic200"]


class TestClearHistory:
    def test_removes_entry(self) -> None:
        record_command(1, 100, "hello")
        clear_history(1, 100)
        assert get_history(1, 100) == []

    def test_no_error_on_missing(self) -> None:
        clear_history(999, 999)


class TestTruncateForDisplay:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("hello", "hello", id="shorter-than-limit"),
            pytest.param("a" * 20, "a" * 20, id="exactly-at-limit"),
            pytest.param("a" * 30, "a" * 19 + "\u2026", id="over-limit-gets-ellipsis"),
        ],
    )
    def test_truncation(self, text: str, expected: str) -> None:
        result = truncate_for_display(text, 20)
        assert result == expected
        assert len(result) <= 20
