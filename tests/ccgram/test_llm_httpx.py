"""Tests for LLM HTTP completers — request/response and error handling.

Covers the HTTP layer that test_llm_completer.py does not: actual request
payload construction, response parsing from mock HTTP, error propagation,
header verification, and temperature passthrough.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ccgram.llm.httpx_completer import (
    AnthropicCompleter,
    OpenAICompatCompleter,
    _SYSTEM_PROMPT,
)

# ── Response builders ────────────────────────────────────────────────────


def _openai_response(
    command: str, explanation: str = "", dangerous: bool = False
) -> dict:
    content = json.dumps(
        {"command": command, "explanation": explanation, "dangerous": dangerous}
    )
    return {"choices": [{"message": {"content": content}}]}


def _anthropic_response(
    command: str, explanation: str = "", dangerous: bool = False
) -> dict:
    content = json.dumps(
        {"command": command, "explanation": explanation, "dangerous": dangerous}
    )
    return {"content": [{"text": content}]}


def _mock_http_response(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.text = json.dumps(json_data)
    return resp


# ── OpenAI-compatible completer ──────────────────────────────────────────


class TestOpenAICompleterRequest:
    @pytest.fixture
    def completer(self) -> OpenAICompatCompleter:
        return OpenAICompatCompleter(api_key="sk-test", model="test-model")

    async def test_payload_structure(self, completer: OpenAICompatCompleter) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(_openai_response("ls"))
            await completer.generate_command("list files", os_info="Linux")

            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "test-model"
            assert payload["messages"][0] == {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            }
            assert "list files" in payload["messages"][1]["content"]

    async def test_authorization_header(self, completer: OpenAICompatCompleter) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(_openai_response("ls"))
            await completer.generate_command("test", os_info="Linux")

            headers = mock_post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer sk-test"
            assert headers["Content-Type"] == "application/json"

    async def test_posts_to_chat_completions_endpoint(
        self, completer: OpenAICompatCompleter
    ) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(_openai_response("ls"))
            await completer.generate_command("test", os_info="Linux")

            url = mock_post.call_args[0][0]
            assert url.endswith("/chat/completions")

    async def test_returns_parsed_command(
        self, completer: OpenAICompatCompleter
    ) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(
                _openai_response("echo hi", "Print hi")
            )
            result = await completer.generate_command("print hi", os_info="Linux")

        assert result.command == "echo hi"
        assert result.explanation == "Print hi"
        assert result.is_dangerous is False

    async def test_dangerous_flag_passthrough(
        self, completer: OpenAICompatCompleter
    ) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(
                _openai_response("rm -rf /", "Delete all", dangerous=True)
            )
            result = await completer.generate_command("delete all", os_info="Linux")

        assert result.is_dangerous is True


# ── Anthropic completer ──────────────────────────────────────────────────


class TestAnthropicCompleterRequest:
    @pytest.fixture
    def completer(self) -> AnthropicCompleter:
        return AnthropicCompleter(api_key="sk-ant-test", model="claude-test")

    async def test_payload_structure(self, completer: AnthropicCompleter) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(_anthropic_response("ls"))
            await completer.generate_command("list files", os_info="Linux")

            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "claude-test"
            assert payload["system"] == _SYSTEM_PROMPT
            assert payload["max_tokens"] == 1024
            assert payload["messages"][0]["role"] == "user"

    async def test_anthropic_headers(self, completer: AnthropicCompleter) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(_anthropic_response("ls"))
            await completer.generate_command("test", os_info="Linux")

            headers = mock_post.call_args.kwargs["headers"]
            assert headers["x-api-key"] == "sk-ant-test"
            assert headers["anthropic-version"] == "2023-06-01"

    async def test_posts_to_messages_endpoint(
        self, completer: AnthropicCompleter
    ) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(_anthropic_response("ls"))
            await completer.generate_command("test", os_info="Linux")

            url = mock_post.call_args[0][0]
            assert url.endswith("/messages")

    async def test_returns_parsed_command(self, completer: AnthropicCompleter) -> None:
        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(
                _anthropic_response("pwd", "Print directory")
            )
            result = await completer.generate_command("current dir", os_info="Linux")

        assert result.command == "pwd"
        assert result.explanation == "Print directory"


# ── Shared error handling ────────────────────────────────────────────────


class TestCompleterErrors:
    @pytest.mark.parametrize(
        ("cls", "api_key"),
        [
            (OpenAICompatCompleter, "sk-test"),
            (AnthropicCompleter, "sk-ant-test"),
        ],
        ids=["openai", "anthropic"],
    )
    async def test_http_status_error_raises_runtime(
        self, cls: type, api_key: str
    ) -> None:
        completer = cls(api_key=api_key, model="m")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limited"

        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_resp
            )
            with pytest.raises(RuntimeError, match="LLM request failed.*429"):
                await completer.generate_command("test", os_info="Linux")

    @pytest.mark.parametrize(
        ("cls", "api_key"),
        [
            (OpenAICompatCompleter, "sk-test"),
            (AnthropicCompleter, "sk-ant-test"),
        ],
        ids=["openai", "anthropic"],
    )
    async def test_connection_error_raises_runtime(
        self, cls: type, api_key: str
    ) -> None:
        completer = cls(api_key=api_key, model="m")

        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(RuntimeError, match="LLM request failed"):
                await completer.generate_command("test", os_info="Linux")

    @pytest.mark.parametrize(
        ("cls", "api_key", "bad_response"),
        [
            (OpenAICompatCompleter, "sk-test", {"unexpected": "format"}),
            (AnthropicCompleter, "sk-ant-test", {"content": []}),
        ],
        ids=["openai-bad-shape", "anthropic-empty-content"],
    )
    async def test_unexpected_response_raises_runtime(
        self, cls: type, api_key: str, bad_response: dict
    ) -> None:
        completer = cls(api_key=api_key, model="m")

        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(bad_response)
            with pytest.raises(RuntimeError, match="Unexpected LLM response"):
                await completer.generate_command("test", os_info="Linux")


# ── Temperature passthrough ──────────────────────────────────────────────


class TestCompleterTemperature:
    @pytest.mark.parametrize(
        ("cls", "response_factory", "temp"),
        [
            (OpenAICompatCompleter, _openai_response, 0.0),
            (OpenAICompatCompleter, _openai_response, 0.5),
            (OpenAICompatCompleter, _openai_response, 1.0),
            (AnthropicCompleter, _anthropic_response, 0.0),
            (AnthropicCompleter, _anthropic_response, 0.7),
        ],
        ids=[
            "openai-0.0",
            "openai-0.5",
            "openai-1.0",
            "anthropic-0.0",
            "anthropic-0.7",
        ],
    )
    async def test_temperature_in_payload(
        self, cls: type, response_factory, temp: float
    ) -> None:
        completer = cls(api_key="sk-test", model="m", temperature=temp)

        with patch.object(
            completer._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _mock_http_response(response_factory("ls"))
            await completer.generate_command("test", os_info="Linux")

            payload = mock_post.call_args.kwargs["json"]
            assert payload["temperature"] == temp


# ── Base URL configuration ───────────────────────────────────────────────


class TestCompleterBaseUrl:
    @pytest.mark.parametrize(
        ("cls", "default_url"),
        [
            (OpenAICompatCompleter, "https://api.openai.com/v1"),
            (AnthropicCompleter, "https://api.anthropic.com/v1"),
        ],
        ids=["openai", "anthropic"],
    )
    def test_default_base_url(self, cls: type, default_url: str) -> None:
        c = cls(api_key="sk-test", model="m")
        assert c._base_url == default_url

    @pytest.mark.parametrize(
        "cls",
        [OpenAICompatCompleter, AnthropicCompleter],
        ids=["openai", "anthropic"],
    )
    def test_custom_base_url(self, cls: type) -> None:
        c = cls(api_key="sk-test", model="m", base_url="https://custom.api/v1")
        assert c._base_url == "https://custom.api/v1"

    @pytest.mark.parametrize(
        "cls",
        [OpenAICompatCompleter, AnthropicCompleter],
        ids=["openai", "anthropic"],
    )
    def test_trailing_slash_stripped(self, cls: type) -> None:
        c = cls(api_key="sk-test", model="m", base_url="https://custom.api/v1/")
        assert not c._base_url.endswith("/")
