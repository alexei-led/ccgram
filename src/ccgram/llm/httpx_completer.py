"""OpenAI-compatible chat completions via httpx for command generation.

Supports any API that follows OpenAI's chat completions endpoint
(OpenAI, Groq, Ollama, etc.) plus a thin Anthropic adapter.
Uses raw httpx — zero new dependencies.
"""

import json
import platform

import httpx

from .base import CommandResult

_OPENAI_BASE_URL = "https://api.openai.com/v1"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

_MAX_RECENT_OUTPUT_CHARS = 500

_SYSTEM_PROMPT = """\
You are a shell command generator. Given a natural language description, \
generate the appropriate shell command or pipeline.

Return ONLY valid JSON with these fields:
- "command": the shell command (string)
- "explanation": brief explanation of what it does (string)
- "dangerous": true if the command could destroy data or is irreversible (boolean)

Examples of dangerous commands: rm -rf, dd, mkfs, DROP TABLE, \
format, shutdown, reboot, kill -9.

Do NOT wrap the JSON in markdown code fences. Return raw JSON only."""


def _build_user_message(
    description: str,
    *,
    cwd: str = "",
    shell: str = "",
    os_info: str = "",
    recent_output: str = "",
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
        return CommandResult(command=cleaned, explanation="", is_dangerous=False)

    if not isinstance(data, dict):
        return CommandResult(command=cleaned, explanation="", is_dangerous=False)

    command = data.get("command", "")
    if not isinstance(command, str) or not command:
        return CommandResult(command=cleaned, explanation="", is_dangerous=False)

    explanation = data.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = ""
    dangerous = bool(data.get("dangerous", False))

    return CommandResult(
        command=command, explanation=explanation, is_dangerous=dangerous
    )


class _BaseCompleter:
    """Shared base for LLM command generators using httpx.

    Subclasses implement ``_request()`` for API-specific payload and
    response parsing.  The httpx client is created once and reused.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        temperature: float = 0.1,
        default_base_url: str = _OPENAI_BASE_URL,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._base_url = (base_url or default_base_url).rstrip("/")
        self._client = httpx.AsyncClient()

    async def generate_command(
        self,
        description: str,
        *,
        cwd: str = "",
        shell: str = "",
        os_info: str = "",
        recent_output: str = "",
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
        )
        text = await self._request(user_msg)
        return _parse_command_result(text)

    async def _request(self, user_msg: str) -> str:
        """Send the request and return the response text."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()


class OpenAICompatCompleter(_BaseCompleter):
    """LLM command generator using OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            api_key,
            model,
            base_url,
            temperature=temperature,
            default_base_url=_OPENAI_BASE_URL,
        )

    async def _request(self, user_msg: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
        }
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"LLM request failed: {exc.response.status_code} {exc.response.text[:200]}"
            raise RuntimeError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"LLM request failed: {exc}"
            raise RuntimeError(msg) from exc

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            msg = f"Unexpected LLM response: {response.text[:200]}"
            raise RuntimeError(msg) from exc


class AnthropicCompleter(_BaseCompleter):
    """LLM command generator using the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            api_key,
            model,
            base_url,
            temperature=temperature,
            default_base_url=_ANTHROPIC_BASE_URL,
        )

    async def _request(self, user_msg: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": self.temperature,
        }
        try:
            response = await self._client.post(
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"LLM request failed: {exc.response.status_code} {exc.response.text[:200]}"
            raise RuntimeError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"LLM request failed: {exc}"
            raise RuntimeError(msg) from exc

        try:
            return response.json()["content"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            msg = f"Unexpected LLM response: {response.text[:200]}"
            raise RuntimeError(msg) from exc
