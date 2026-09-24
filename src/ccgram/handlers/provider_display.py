"""Small, searchable provider labels for Telegram's constrained topic headers."""

PROVIDER_LABELS = {
    "claude": "Claude",
    "codex": "Codex",
    "gemini": "Gemini",
    "pi": "Pi",
    "antigravity": "Antigravity",
    "shell": "Terminal",
}
PROVIDER_SEPARATOR = " · "


def provider_label(name: str, *, compact: bool = False) -> str:
    if compact and name == "shell":
        return "Term"
    return PROVIDER_LABELS.get(name, name.capitalize())


def strip_provider_prefix(name: str, *, legacy: bool = False) -> str:
    """Strip only our delimited labels, including the existing Herdr format."""
    separators = (PROVIDER_SEPARATOR, " ▸ ") if legacy else (PROVIDER_SEPARATOR,)
    for label in (*PROVIDER_LABELS.values(), "Term", "Shell"):
        for separator in separators:
            if name.startswith(label + separator):
                return name[len(label + separator) :]
    return name


def provider_topic_name(name: str, provider: str) -> str:
    if not provider:
        return name
    return (
        provider_label(provider, compact=True)
        + PROVIDER_SEPARATOR
        + strip_provider_prefix(name, legacy=True)
    )
