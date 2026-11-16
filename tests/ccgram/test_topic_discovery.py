import pytest

from ccgram.topic_discovery import (
    DEFAULT_TOPIC_DISCOVERY_MODE,
    TOPIC_DISCOVERY_ENV,
    TOPIC_DISCOVERY_MODES,
    parse_topic_discovery_mode,
)


def test_discovery_vocabulary():
    assert TOPIC_DISCOVERY_MODES == ("auto", "marked")
    assert DEFAULT_TOPIC_DISCOVERY_MODE == "auto"
    assert TOPIC_DISCOVERY_ENV == "CCGRAM_TOPIC_DISCOVERY"


@pytest.mark.parametrize(
    ("raw", "expected"), [("auto", "auto"), (" MARKED ", "marked")]
)
def test_discovery_parser_normalizes_supported_modes(raw, expected):
    assert parse_topic_discovery_mode(raw) == expected


@pytest.mark.parametrize("raw", ["", " ", "all", "mark", "true"])
def test_discovery_parser_rejects_invalid_modes(raw):
    with pytest.raises(ValueError, match="CCGRAM_TOPIC_DISCOVERY"):
        parse_topic_discovery_mode(raw)
