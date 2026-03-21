"""LLM command generation provider abstraction.

Provides a pluggable interface for generating shell commands from
natural language descriptions using OpenAI-compatible or Anthropic APIs.
"""

import os

from .base import CommandGenerator, CommandResult
from .httpx_completer import AnthropicCompleter, OpenAICompatCompleter

_PROVIDERS: dict[str, dict[str, str | None]] = {
    "openai": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "anthropic": {
        "base_url": None,
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "api_key_env": "",
    },
}


def get_completer() -> CommandGenerator | None:
    """Create and return an LLM command generator based on config.

    Returns None if llm_provider is not configured (empty string).
    """
    from ccgram.config import config

    provider = config.llm_provider
    if not provider:
        return None

    provider_info = _PROVIDERS.get(provider)
    if not provider_info:
        msg = f"Unknown LLM provider: {provider}"
        raise ValueError(msg)

    api_key = config.llm_api_key
    if not api_key:
        api_key_env = provider_info.get("api_key_env", "")
        if api_key_env:
            api_key = os.getenv(api_key_env, "")
        if not api_key and provider != "ollama":
            env_name = api_key_env or "CCGRAM_LLM_API_KEY"
            msg = f"No API key found: set {env_name} or CCGRAM_LLM_API_KEY"
            raise ValueError(msg)

    base_url = config.llm_base_url or provider_info.get("base_url")
    model = config.llm_model or provider_info.get("model") or ""

    if provider == "anthropic":
        return AnthropicCompleter(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    return OpenAICompatCompleter(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


__all__ = [
    "CommandGenerator",
    "CommandResult",
    "get_completer",
]
