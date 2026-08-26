"""Unit tests for the whisper transcriber and its config-driven factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ccgram.whisper import get_transcriber
from ccgram.whisper.httpx_transcriber import OpenAICompatTranscriber


@pytest.fixture
def _mock_httpx():
    """Provide a mock httpx async client and response for transcription tests."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"text": "hello"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "ccgram.whisper.httpx_transcriber.httpx.AsyncClient",
        return_value=mock_client,
    ):
        yield mock_client, mock_response


class TestTranscribe:
    """Tests for OpenAICompatTranscriber.transcribe()."""

    async def test_success(self, _mock_httpx: tuple[AsyncMock, MagicMock]) -> None:
        mock_client, mock_response = _mock_httpx
        mock_response.json.return_value = {"text": "hello world"}

        t = OpenAICompatTranscriber(api_key="k", model="whisper-1")
        result = await t.transcribe(b"audio", "voice.ogg")

        assert result.text == "hello world"
        call_kw = mock_client.post.call_args.kwargs
        assert call_kw["data"] == {"model": "whisper-1"}
        assert call_kw["files"] == {"file": ("voice.ogg", b"audio")}
        assert "Bearer k" in call_kw["headers"]["Authorization"]

    async def test_language_forwarded(
        self, _mock_httpx: tuple[AsyncMock, MagicMock]
    ) -> None:
        mock_client, mock_response = _mock_httpx
        mock_response.json.return_value = {"text": "你好"}

        t = OpenAICompatTranscriber(api_key="k", model="whisper-1", language="zh")
        result = await t.transcribe(b"audio", "voice.ogg")

        assert result.text == "你好"
        assert mock_client.post.call_args.kwargs["data"] == {
            "model": "whisper-1",
            "language": "zh",
        }

    async def test_empty_result(self, _mock_httpx: tuple[AsyncMock, MagicMock]) -> None:
        _, mock_response = _mock_httpx
        mock_response.json.return_value = {"text": ""}

        t = OpenAICompatTranscriber(api_key="k", model="m")
        assert (await t.transcribe(b"audio", "v.ogg")).text == ""

    async def test_too_large(self) -> None:
        t = OpenAICompatTranscriber(api_key="k", model="m")
        with pytest.raises(ValueError, match="too large"):
            await t.transcribe(b"x" * (25 * 1024 * 1024 + 1), "v.ogg")

    async def test_exactly_at_limit(
        self, _mock_httpx: tuple[AsyncMock, MagicMock]
    ) -> None:
        _, mock_response = _mock_httpx
        mock_response.json.return_value = {"text": "ok"}

        t = OpenAICompatTranscriber(api_key="k", model="m")
        result = await t.transcribe(b"x" * (25 * 1024 * 1024), "v.ogg")
        assert result.text == "ok"

    async def test_http_status_error(
        self, _mock_httpx: tuple[AsyncMock, MagicMock]
    ) -> None:
        mock_client, _ = _mock_httpx
        resp = MagicMock(status_code=401, text="Unauthorized")
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=resp)
        )

        t = OpenAICompatTranscriber(api_key="k", model="m")
        with pytest.raises(RuntimeError, match="401"):
            await t.transcribe(b"audio", "v.ogg")

    async def test_network_error(
        self, _mock_httpx: tuple[AsyncMock, MagicMock]
    ) -> None:
        mock_client, _ = _mock_httpx
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        t = OpenAICompatTranscriber(api_key="k", model="m")
        with pytest.raises(RuntimeError, match="Transcription failed"):
            await t.transcribe(b"audio", "v.ogg")

    async def test_unexpected_json_response(
        self, _mock_httpx: tuple[AsyncMock, MagicMock]
    ) -> None:
        _, mock_response = _mock_httpx
        mock_response.json.return_value = {"result": "no text key"}

        t = OpenAICompatTranscriber(api_key="k", model="m")
        with pytest.raises(RuntimeError, match="Unexpected API response"):
            await t.transcribe(b"audio", "v.ogg")

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            pytest.param(None, "https://api.openai.com/v1", id="default"),
            pytest.param(
                "https://api.groq.com/openai/v1/",
                "https://api.groq.com/openai/v1",
                id="strips_trailing_slash",
            ),
        ],
    )
    def test_base_url_resolution(self, base_url: str | None, expected: str) -> None:
        t = OpenAICompatTranscriber(api_key="k", model="m", base_url=base_url)
        assert t._base_url == expected


class TestGetTranscriber:
    """Only the whisper-specific key rule lives here — provider defaults,
    config overrides and the unknown-provider failure are covered against a real
    Config in tests/integration/test_whisper_integration.py."""

    def test_openai_key_does_not_satisfy_another_provider(self, monkeypatch) -> None:
        """Unlike the LLM factory, whisper has no universal OPENAI_API_KEY
        fallback: a groq transcriber must not silently use an OpenAI key."""
        monkeypatch.setattr(
            "ccgram.config.config",
            MagicMock(
                whisper_provider="groq",
                whisper_api_key="",
                whisper_base_url="",
                whisper_model="",
                whisper_language="",
            ),
        )
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong-service")

        with pytest.raises(ValueError, match="set GROQ_API_KEY"):
            get_transcriber()
