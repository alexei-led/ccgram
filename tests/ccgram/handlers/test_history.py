"""Tests for history helpers."""

import pytest

from ccgram.handlers.history import _format_timestamp


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("ts", "expected"),
        [
            ("2024-01-15T14:32:00.000Z", "14:32"),
            ("2024-01-15T14:32:00Z", "14:32"),
            ("2024-01-15T14:32:00+05:30", "14:32"),
            ("2024-01-15T14:32:59", "14:32"),
            ("2024-01-15 14:32:00", "14:32"),
            ("not-a-timestamp", ""),
            ("", ""),
            (None, ""),
        ],
        ids=[
            "standard-iso-with-Z",
            "no-millis-with-Z",
            "timezone-offset",
            "no-timezone",
            "space-separator",
            "invalid-string",
            "empty-string",
            "none",
        ],
    )
    def test_format_timestamp(self, ts: str | None, expected: str) -> None:
        assert _format_timestamp(ts) == expected
