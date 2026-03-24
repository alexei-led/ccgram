"""OpenAI-compatible chat completions via httpx for command generation.

Supports any API that follows OpenAI's chat completions endpoint
(OpenAI, Groq, Ollama, etc.) plus a thin Anthropic adapter.
Uses raw httpx — zero new dependencies.
"""

import abc
import json
import platform
import re
from collections.abc import Callable
from typing import Any

import httpx

from .base import CommandResult

_OPENAI_BASE_URL = "https://api.openai.com/v1"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

_MAX_RECENT_OUTPUT_CHARS = 500

_SYSTEM_PROMPT = """\
You are a shell command generator. Given a natural language description, \
generate the appropriate shell command or pipeline.

When available tools are listed in the context, ALWAYS use them instead of \
their traditional counterparts (e.g. use fd instead of find, rg instead of grep). \
Use the correct syntax for the available tool, not the syntax of the tool it replaces.

Return ONLY valid JSON with these fields:
- "command": the shell command (string)
- "explanation": brief explanation of what it does (string)
- "dangerous": true if the command could destroy data or is irreversible (boolean)

Examples of dangerous commands: rm -rf, dd, mkfs, DROP TABLE, \
format, shutdown, reboot, kill -9.

Do NOT wrap the JSON in markdown code fences. Return raw JSON only."""

_DANGEROUS_RE = re.compile(
    r"rm\s+(-\w*[rR]\w*\s+|--recursive)"
    r"|\bdd\s+"
    r"|\bmkfs\b"
    r"|\b(shutdown|reboot|halt|poweroff)\b"
    r"|\bkill\s+-9\b|\bkillall\b"
    r"|\bchmod\s+(-\w*R\w*\s+)?777\b"
    r"|>\s*/dev/sd|>\s*/dev/nvme"
    r"|\bDROP\s+(TABLE|DATABASE)\b"
    r"|\bsudo\s+rm\b",
    re.IGNORECASE,
)


def _is_dangerous_heuristic(command: str) -> bool:
    """Check if a command matches known dangerous patterns."""
    return bool(_DANGEROUS_RE.search(command))


_SHELL_SYNTAX_NOTES: dict[str, str] = {
    "fish": (
        "\n\nIMPORTANT: The target shell is fish. Fish is NOT POSIX-compatible.\n"
        "- No && or || — use `; and` / `; or` or separate lines with `and`/`or`\n"
        "- No heredocs (<<EOF) — use `printf '...' | command` or write to a temp file\n"
        "- Variables: `set VAR value`, NOT `export VAR=value`\n"
        "- Command substitution: `(command)` not `$(command)`\n"
        "- No brace expansion {a,b} — use explicit arguments\n"
        "- Conditionals: `if command; ...; end` not `if command; then ...; fi`\n"
        "- Loops: `for x in a b c; ...; end` not `for x in a b c; do ...; done`\n"
        "- Multi-line scripts: use `begin; ...; end` blocks\n"
        "- For inline Python: `python3 -c 'code'` (single quotes, no heredoc)"
    ),
    "zsh": (
        "\n\nTarget shell is zsh. Use zsh-compatible syntax.\n"
        "- Arrays are 1-indexed: $arr[1] not $arr[0]\n"
        "- Glob qualifiers available: *(.) for files, *(/) for dirs"
    ),
    "bash": (
        "\n\nTarget shell is bash. Use bash-compatible syntax.\n"
        "- Use [[ ]] for conditionals (safer than [ ])"
    ),
}


def _build_system_prompt(shell: str = "") -> str:
    """Build the system prompt with shell-specific syntax notes."""
    notes = _SHELL_SYNTAX_NOTES.get(shell.lower()) if shell else None
    if notes:
        return _SYSTEM_PROMPT + notes
    if shell:
        return _SYSTEM_PROMPT + f"\n\nTarget shell is {shell}."
    return _SYSTEM_PROMPT


def _build_user_message(
    description: str,
    *,
    cwd: str = "",
    shell: str = "",
    os_info: str = "",
    recent_output: str = "",
    shell_tools: str = "",
) -> str:
    """Build the user message with context."""
    parts = [description]
    context_parts: list[str] = []
    if cwd:
        context_parts.append(f"CWD: {cwd}")
    if shell:
        context_parts.append(f"Shell: {shell}")
    if os_info:
        context_parts.append(f"OS: {os_info}")
    if shell_tools:
        context_parts.append(f"Available tools: {shell_tools}")
    if recent_output:
        trimmed = (
            recent_output[-_MAX_RECENT_OUTPUT_CHARS:]
            if len(recent_output) > _MAX_RECENT_OUTPUT_CHARS
            else recent_output
        )
        context_parts.append(f"Recent output:\n{trimmed}")
    if context_parts:
        parts.append("\nContext:\n" + "\n".join(context_parts))
    return "\n".join(parts)


def _parse_command_result(text: str) -> CommandResult:
    """Parse LLM response text into a CommandResult."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return CommandResult(command=cleaned, explanation="", is_dangerous=True)

    if not isinstance(data, dict):
        return CommandResult(command=cleaned, explanation="", is_dangerous=True)

    command = data.get("command", "")
    if not isinstance(command, str) or not command:
        return CommandResult(command=cleaned, explanation="", is_dangerous=True)

    explanation = data.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = ""
    dangerous = bool(data.get("dangerous", False)) or _is_dangerous_heuristic(command)

    return CommandResult(
        command=command, explanation=explanation, is_dangerous=dangerous
    )


class _BaseCompleter(abc.ABC):
    """Shared base for LLM command generators using httpx.

    Subclasses provide ``_url()``, ``_headers()``, ``_payload()``, and
    ``_extract()`` for API-specific details.  The shared ``_request()``
    handles HTTP execution and error handling.  Creates a fresh httpx
    client per request to avoid lifecycle issues in long-running bots.
    """

    _default_base_url: str = _OPENAI_BASE_URL

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._base_url = (base_url or self._default_base_url).rstrip("/")

    async def generate_command(
        self,
        description: str,
        *,
        cwd: str = "",
        shell: str = "",
        os_info: str = "",
        recent_output: str = "",
        shell_tools: str = "",
    ) -> CommandResult:
        """Generate a shell command from a natural language description."""
        if not os_info:
            os_info = f"{platform.system()} {platform.release()}"
        user_msg = _build_user_message(
            description,
            cwd=cwd,
            shell=shell,
            os_info=os_info,
            recent_output=recent_output,
            shell_tools=shell_tools,
        )
        text = await self._request(user_msg, shell=shell)
        return _parse_command_result(text)

    async def _post_and_extract(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        extract: Callable[[dict[str, Any]], str],
    ) -> str:
        """Post to LLM API and extract response text with shared error handling."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                msg = f"LLM request failed: {exc.response.status_code}"
                raise RuntimeError(msg) from exc
            except httpx.HTTPError as exc:
                msg = f"LLM request failed: {exc}"
                raise RuntimeError(msg) from exc

            try:
                return extract(response.json())
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                msg = f"Unexpected LLM response: {response.text[:200]}"
                raise RuntimeError(msg) from exc

    @abc.abstractmethod
    async def _request(self, user_msg: str, *, shell: str = "") -> str:
        """Send the request and return the response text."""
        ...


class OpenAICompatCompleter(_BaseCompleter):
    """LLM command generator using OpenAI-compatible chat completions API."""

    _default_base_url: str = _OPENAI_BASE_URL

    async def _request(self, user_msg: str, *, shell: str = "") -> str:
        prompt = _build_system_prompt(shell)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
        }
        return await self._post_and_extract(
            f"{self._base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload,
            lambda data: data["choices"][0]["message"]["content"],
        )


class AnthropicCompleter(_BaseCompleter):
    """LLM command generator using the Anthropic Messages API."""

    _default_base_url: str = _ANTHROPIC_BASE_URL

    async def _request(self, user_msg: str, *, shell: str = "") -> str:
        prompt = _build_system_prompt(shell)
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": prompt,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": self.temperature,
        }
        return await self._post_and_extract(
            f"{self._base_url}/messages",
            {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
            lambda data: data["content"][0]["text"],
        )
