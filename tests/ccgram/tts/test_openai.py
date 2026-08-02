from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ccgram.tts.base import TtsAudio, TtsSynthesisError
from ccgram.tts.openai import OpenAITtsSynthesizer


@pytest.fixture
def _mock_httpx():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b"audio-bytes"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "ccgram.tts.openai.httpx.AsyncClient",
        return_value=mock_client,
    ):
        yield mock_client, mock_response


class TestOpenAITtsSynthesizer:
    async def test_success(self, _mock_httpx):
        mock_client, mock_response = _mock_httpx
        mock_response.content = b"mp3-data"

        synth = OpenAITtsSynthesizer(
            api_key="sk-test", model="gpt-4o-mini-tts", voice="alloy"
        )
        result = await synth.synthesize("Hello world")

        assert result == TtsAudio(data=b"mp3-data")

    async def test_request_payload(self, _mock_httpx):
        mock_client, _ = _mock_httpx

        synth = OpenAITtsSynthesizer(
            api_key="sk-test", model="gpt-4o-mini-tts", voice="nova"
        )
        await synth.synthesize("Say something")

        call_kw = mock_client.post.call_args.kwargs
        assert call_kw["json"] == {
            "model": "gpt-4o-mini-tts",
            "input": "Say something",
            "voice": "nova",
            "response_format": "mp3",
        }
        assert "Bearer sk-test" in call_kw["headers"]["Authorization"]

    async def test_raises_on_empty_text(self):
        synth = OpenAITtsSynthesizer(api_key="k", model="m", voice="alloy")
        with pytest.raises(ValueError, match="empty"):
            await synth.synthesize("   ")

    async def test_http_status_error_raises_tts_error(self, _mock_httpx):
        mock_client, _ = _mock_httpx
        resp = MagicMock(status_code=401, text="Unauthorized")
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=resp)
        )

        synth = OpenAITtsSynthesizer(api_key="k", model="m", voice="alloy")
        with pytest.raises(TtsSynthesisError, match="401"):
            await synth.synthesize("Hello")

    async def test_network_error_raises_tts_error(self, _mock_httpx):
        mock_client, _ = _mock_httpx
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        synth = OpenAITtsSynthesizer(api_key="k", model="m", voice="alloy")
        with pytest.raises(TtsSynthesisError, match="TTS failed"):
            await synth.synthesize("Hello")

    async def test_empty_response_raises_tts_error(self, _mock_httpx):
        _, mock_response = _mock_httpx
        mock_response.content = b""

        synth = OpenAITtsSynthesizer(api_key="k", model="m", voice="alloy")
        with pytest.raises(TtsSynthesisError, match="No audio bytes"):
            await synth.synthesize("Hello")

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            pytest.param(None, "https://api.openai.com/v1", id="default"),
            pytest.param(
                "https://api.openai.com/v1/",
                "https://api.openai.com/v1",
                id="strips_trailing_slash",
            ),
        ],
    )
    def test_base_url_resolution(self, base_url, expected):
        synth = OpenAITtsSynthesizer(
            api_key="k", model="m", voice="v", base_url=base_url
        )
        assert synth._base_url == expected


class TestGetSynthesizerOpenAI:
    def test_returns_openai_synthesizer(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_TTS_PROVIDER", "openai")
        monkeypatch.setenv("CCGRAM_TTS_MODEL", "gpt-4o-mini-tts")
        monkeypatch.setenv("CCGRAM_TTS_VOICE", "alloy")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from ccgram.config import Config
        from ccgram.tts import get_synthesizer

        cfg = Config()
        with patch("ccgram.tts.config", cfg):
            synth = get_synthesizer()

        assert isinstance(synth, OpenAITtsSynthesizer)
        assert synth._model == "gpt-4o-mini-tts"
        assert synth._voice == "alloy"

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_TTS_PROVIDER", "openai")
        monkeypatch.delenv("CCGRAM_TTS_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "ccgram.config.load_dotenv", lambda *_args, **_kwargs: False
        )

        from ccgram.config import Config
        from ccgram.tts import get_synthesizer

        cfg = Config()
        with (
            patch("ccgram.tts.config", cfg),
            pytest.raises(ValueError, match="API key"),
        ):
            get_synthesizer()

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("CCGRAM_TTS_PROVIDER", "elevenlabs")

        from ccgram.config import Config
        from ccgram.tts import get_synthesizer

        cfg = Config()
        with (
            patch("ccgram.tts.config", cfg),
            pytest.raises(ValueError, match="Unknown TTS provider"),
        ):
            get_synthesizer()
