"""Discovery mode vocabulary shared by configuration, CLI, and admission policy."""

TOPIC_DISCOVERY_MODES = ("auto", "marked")
DEFAULT_TOPIC_DISCOVERY_MODE = "auto"
TOPIC_DISCOVERY_ENV = "CCGRAM_TOPIC_DISCOVERY"


def parse_topic_discovery_mode(value: str) -> str:
    """Validate discovery configuration without loading bot credentials."""
    mode = value.strip().lower()
    if mode not in TOPIC_DISCOVERY_MODES:
        raise ValueError(
            f"{TOPIC_DISCOVERY_ENV} must be {' or '.join(TOPIC_DISCOVERY_MODES)}"
        )
    return mode
