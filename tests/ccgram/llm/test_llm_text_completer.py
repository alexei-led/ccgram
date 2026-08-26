"""Tests for generic LLM text completion (complete() method and get_text_completer)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ccgram.llm.httpx_completer import AnthropicCompleter, OpenAICompatCompleter


class TestOpenAIComplete:
    @pytest.fixture
    def completer(self):
        return OpenAICompatCompleter(
            api_key="sk-test", model="test-model", temperature=0.0
        )

    async def test_complete_sends_system_and_user(self, completer):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello world"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await completer.complete("Be helpful.", "Summarize this.")

        assert result == "hello world"
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert payload["messages"][1] == {"role": "user", "content": "Summarize this."}
        assert payload["model"] == "test-model"

    async def test_complete_http_error_raises_runtime(self, completer):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPError("connection failed")
            with pytest.raises(RuntimeError, match="LLM request failed"):
                await completer.complete("sys", "usr")

    async def test_complete_bad_response_raises_runtime(self, completer):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = MagicMock()
        mock_response.text = '{"unexpected": "format"}'

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(RuntimeError, match="Unexpected LLM response"):
                await completer.complete("sys", "usr")


class TestAnthropicComplete:
    @pytest.fixture
    def completer(self):
        return AnthropicCompleter(
            api_key="sk-ant-test", model="claude-test", temperature=0.0
        )

    async def test_complete_sends_anthropic_format(self, completer):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": [{"text": "summary result"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await completer.complete("Be concise.", "What happened?")

        assert result == "summary result"
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["system"] == "Be concise."
        assert payload["messages"] == [{"role": "user", "content": "What happened?"}]
        assert payload["model"] == "claude-test"
        headers = call_kwargs.kwargs["headers"]
        assert "x-api-key" in headers

    async def test_complete_http_error_raises_runtime(self, completer):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPError("timeout")
            with pytest.raises(RuntimeError, match="LLM request failed"):
                await completer.complete("sys", "usr")


class TestGetTextCompleter:
    def test_delegates_to_the_shared_completer_factory(self, monkeypatch):
        """get_text_completer is a typed alias of the get_completer factory —
        provider/API-key resolution is covered by test_llm_completer.py."""
        mock_cfg = MagicMock(
            llm_provider="openai",
            llm_api_key="sk-test",
            llm_base_url="",
            llm_model="",
            llm_temperature=0.1,
        )
        monkeypatch.setattr("ccgram.config.config", mock_cfg)

        from ccgram.llm import get_text_completer

        completer = get_text_completer()
        assert isinstance(completer, OpenAICompatCompleter)
        assert callable(completer.complete)
