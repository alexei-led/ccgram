"""Provider registry — maps provider names to classes, caches instances.

The module-level ``registry`` singleton starts empty; providers are
registered lazily via ``_ensure_registered()`` in ``ccgram.providers.__init__``
before first use. ``get()`` caches one instance per provider name.
"""

import structlog

from ccgram.providers.base import AgentProvider

logger = structlog.get_logger()


class UnknownProviderError(LookupError):
    """Raised when requesting a provider name that is not registered."""


class ProviderRegistry:
    """Maps provider name strings to AgentProvider classes.

    Instances are cached per name — ``get()`` returns the same instance
    for repeated calls with the same name.
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[AgentProvider]] = {}
        self._instances: dict[str, AgentProvider] = {}

    def register(self, name: str, provider_cls: type[AgentProvider]) -> None:
        """Register a provider class under *name* (overwrites silently)."""
        self._providers[name] = provider_cls
        self._instances.pop(name, None)  # invalidate cached instance
        logger.debug("Registered provider %r", name)

    def provider_names(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._providers)

    def is_valid(self, name: str) -> bool:
        """Return True if *name* is a registered provider."""
        return name in self._providers

    def get(self, name: str) -> AgentProvider:
        """Return a cached provider instance for *name*.

        Raises ``UnknownProviderError`` if *name* is not registered.
        """
        if name in self._instances:
            return self._instances[name]
        cls = self._providers.get(name)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise UnknownProviderError(
                f"Unknown provider {name!r}. Available: {available}"
            )
        instance = cls()
        self._instances[name] = instance
        return instance


registry = ProviderRegistry()
