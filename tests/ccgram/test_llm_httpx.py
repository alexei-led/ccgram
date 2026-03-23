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

_MOD = "ccgram.llm.httpx_completer"


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


def _patch_httpx_client(mock_post: AsyncMock):
    """Create a patch for httpx.AsyncClient that returns a mock with given post."""
    mock_client = AsyncMock()
    mock_client.post = mock_post
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return patch(f"{_MOD}.httpx.AsyncClient", return_value=mock_ctx)


class TestOpenAICompleterRequest:
    @pytest.fixture
    def completer(self) -> OpenAICompatCompleter:
        return OpenAICompatCompleter(api_key="sk-test", model="test-model")

    async def test_payload_structure(self, completer: OpenAICompatCompleter) -> None:
        mock_post = AsyncMock(return_value=_mock_http_response(_openai_response("ls")))
        with _patch_httpx_client(mock_post):
            await completer.generate_command("list files", os_info="Linux")

            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "test-model"
            assert payload["messages"][0] == {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            }
            assert "list files" in payload["messages"][1]["content"]

    async def test_authorization_header(self, completer: OpenAICompatCompleter) -> None:
        mock_post = AsyncMock(return_value=_mock_http_response(_openai_response("ls")))
        with _patch_httpx_client(mock_post):
            await completer.generate_command("test", os_info="Linux")

            headers = mock_post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer sk-test"
            assert headers["Content-Type"] == "application/json"

    async def test_posts_to_chat_completions_endpoint(
        self, completer: OpenAICompatCompleter
    ) -> None:
        mock_post = AsyncMock(return_value=_mock_http_response(_openai_response("ls")))
        with _patch_httpx_client(mock_post):
            await completer.generate_command("test", os_info="Linux")

            url = mock_post.call_args[0][0]
            assert url.endswith("/chat/completions")

    async def test_returns_parsed_command(
        self, completer: OpenAICompatCompleter
    ) -> None:
        mock_post = AsyncMock(
            return_value=_mock_http_response(_openai_response("echo hi", "Print hi"))
        )
        with _patch_httpx_client(mock_post):
            result = await completer.generate_command("print hi", os_info="Linux")

        assert result.command == "echo hi"
        assert result.explanation == "Print hi"
        assert result.is_dangerous is False

    async def test_dangerous_flag_passthrough(
        self, completer: OpenAICompatCompleter
    ) -> None:
        mock_post = AsyncMock(
            return_value=_mock_http_response(
                _openai_response("rm -rf /", "Delete all", dangerous=True)
            )
        )
        with _patch_httpx_client(mock_post):
            result = await completer.generate_command("delete all", os_info="Linux")

        assert result.is_dangerous is True


class TestAnthropicCompleterRequest:
    @pytest.fixture
    def completer(self) -> AnthropicCompleter:
        return AnthropicCompleter(api_key="sk-ant-test", model="claude-test")

    async def test_payload_structure(self, completer: AnthropicCompleter) -> None:
        mock_post = AsyncMock(
            return_value=_mock_http_response(_anthropic_response("ls"))
        )
        with _patch_httpx_client(mock_post):
            await completer.generate_command("list files", os_info="Linux")

            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "claude-test"
            assert payload["system"] == _SYSTEM_PROMPT
            assert payload["max_tokens"] == 1024
            assert payload["messages"][0]["role"] == "user"

    async def test_anthropic_headers(self, completer: AnthropicCompleter) -> None:
        mock_post = AsyncMock(
            return_value=_mock_http_response(_anthropic_response("ls"))
        )
        with _patch_httpx_client(mock_post):
            await completer.generate_command("test", os_info="Linux")

            headers = mock_post.call_args.kwargs["headers"]
            assert headers["x-api-key"] == "sk-ant-test"
            assert headers["anthropic-version"] == "2023-06-01"

    async def test_posts_to_messages_endpoint(
        self, completer: AnthropicCompleter
    ) -> None:
        mock_post = AsyncMock(
            return_value=_mock_http_response(_anthropic_response("ls"))
        )
        with _patch_httpx_client(mock_post):
            await completer.generate_command("test", os_info="Linux")

            url = mock_post.call_args[0][0]
            assert url.endswith("/messages")

    async def test_returns_parsed_command(self, completer: AnthropicCompleter) -> None:
        mock_post = AsyncMock(
            return_value=_mock_http_response(
                _anthropic_response("pwd", "Print directory")
            )
        )
        with _patch_httpx_client(mock_post):
            result = await completer.generate_command("current dir", os_info="Linux")

        assert result.command == "pwd"
        assert result.explanation == "Print directory"


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

        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_resp
            )
        )
        with (
            _patch_httpx_client(mock_post),
            pytest.raises(RuntimeError, match="LLM request failed.*429"),
        ):
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

        mock_post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        with (
            _patch_httpx_client(mock_post),
            pytest.raises(RuntimeError, match="LLM request failed"),
        ):
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

        mock_post = AsyncMock(return_value=_mock_http_response(bad_response))
        with (
            _patch_httpx_client(mock_post),
            pytest.raises(RuntimeError, match="Unexpected LLM response"),
        ):
            await completer.generate_command("test", os_info="Linux")

    @pytest.mark.parametrize(
        ("cls", "api_key"),
        [
            (OpenAICompatCompleter, "sk-test"),
            (AnthropicCompleter, "sk-ant-test"),
        ],
        ids=["openai", "anthropic"],
    )
    async def test_timeout_error_raises_runtime(self, cls: type, api_key: str) -> None:
        completer = cls(api_key=api_key, model="m")

        mock_post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
        with (
            _patch_httpx_client(mock_post),
            pytest.raises(RuntimeError, match="LLM request failed"),
        ):
            await completer.generate_command("test", os_info="Linux")


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

        mock_post = AsyncMock(return_value=_mock_http_response(response_factory("ls")))
        with _patch_httpx_client(mock_post):
            await completer.generate_command("test", os_info="Linux")

            payload = mock_post.call_args.kwargs["json"]
            assert payload["temperature"] == temp


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


class TestBuildSystemPrompt:
    def test_empty_shell_returns_base(self) -> None:
        from ccgram.llm.httpx_completer import _SYSTEM_PROMPT, _build_system_prompt

        assert _build_system_prompt() == _SYSTEM_PROMPT
        assert _build_system_prompt("") == _SYSTEM_PROMPT

    @pytest.mark.parametrize(
        ("shell", "expected_substring"),
        [
            ("fish", "NOT POSIX-compatible"),
            ("fish", "No && or ||"),
            ("fish", "No heredocs"),
            ("zsh", "1-indexed"),
            ("bash", "bash-compatible"),
        ],
        ids=["fish-posix", "fish-no-and", "fish-no-heredoc", "zsh-arrays", "bash"],
    )
    def test_known_shell_notes_included(
        self, shell: str, expected_substring: str
    ) -> None:
        from ccgram.llm.httpx_completer import _build_system_prompt

        prompt = _build_system_prompt(shell)
        assert expected_substring in prompt

    def test_unknown_shell_gets_generic_note(self) -> None:
        from ccgram.llm.httpx_completer import _build_system_prompt

        prompt = _build_system_prompt("tcsh")
        assert "Target shell is tcsh" in prompt

    def test_case_insensitive(self) -> None:
        from ccgram.llm.httpx_completer import _build_system_prompt

        prompt_lower = _build_system_prompt("fish")
        prompt_upper = _build_system_prompt("FISH")
        assert prompt_lower == prompt_upper

    async def test_openai_passes_shell_to_prompt(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = _openai_response("ls")
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            c = OpenAICompatCompleter(api_key="sk-test", model="m")
            await c.generate_command("list files", shell="fish")

        payload = mock_client.post.call_args[1]["json"]
        system_msg = payload["messages"][0]["content"]
        assert "NOT POSIX-compatible" in system_msg
