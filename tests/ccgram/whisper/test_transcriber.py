"""Unit tests for OpenAICompatTranscriber — mocks the openai AsyncOpenAI client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError

from ccgram.whisper import get_transcriber
from ccgram.whisper.openai_compat import OpenAICompatTranscriber


class TestOpenAICompatTranscriber:
    """Tests for OpenAICompatTranscriber."""

    @patch("ccgram.whisper.openai_compat.openai.AsyncOpenAI")
    async def test_transcribe_success(self, mock_openai_class: MagicMock) -> None:
        """Test successful transcription returns the text."""
        mock_response = MagicMock()
        mock_response.text = "hello"

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        transcriber = OpenAICompatTranscriber(
            api_key="test-key",
            model="whisper-1",
        )
        result = await transcriber.transcribe(b"fake audio data", "voice.ogg")

        assert result.text == "hello"
        mock_client.audio.transcriptions.create.assert_called_once_with(
            model="whisper-1",
            file=("voice.ogg", b"fake audio data"),
        )

    @patch("ccgram.whisper.openai_compat.openai.AsyncOpenAI")
    async def test_transcribe_passes_language_when_configured(
        self, mock_openai_class: MagicMock
    ) -> None:
        """Test configured language is forwarded to the API call."""
        mock_response = MagicMock()
        mock_response.text = "你好"

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        transcriber = OpenAICompatTranscriber(
            api_key="test-key",
            model="whisper-1",
            language="zh",
        )
        result = await transcriber.transcribe(b"fake audio data", "voice.ogg")

        assert result.text == "你好"
        mock_client.audio.transcriptions.create.assert_called_once_with(
            model="whisper-1",
            file=("voice.ogg", b"fake audio data"),
            language="zh",
        )

    @patch("ccgram.whisper.openai_compat.openai.AsyncOpenAI")
    async def test_transcribe_empty_result(self, mock_openai_class: MagicMock) -> None:
        """Test empty transcription result returns empty string."""
        mock_response = MagicMock()
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        transcriber = OpenAICompatTranscriber(
            api_key="test-key",
            model="whisper-1",
        )
        result = await transcriber.transcribe(b"fake audio data", "voice.ogg")

        assert result.text == ""

    @patch("ccgram.whisper.openai_compat.openai.AsyncOpenAI")
    async def test_transcribe_api_error(self, mock_openai_class: MagicMock) -> None:
        """Test API error is re-raised as RuntimeError."""
        # Create a minimal request object for APIError
        mock_request = MagicMock()
        mock_request.url = "https://api.openai.com/v1/audio/transcriptions"

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=APIError("401 Unauthorized", request=mock_request, body=None)
        )
        mock_openai_class.return_value = mock_client

        transcriber = OpenAICompatTranscriber(
            api_key="test-key",
            model="whisper-1",
        )

        with pytest.raises(RuntimeError) as exc_info:
            await transcriber.transcribe(b"fake audio data", "voice.ogg")

        assert "Transcription failed" in str(exc_info.value)

    async def test_transcribe_too_large(self) -> None:
        """Test audio over 25 MB raises ValueError."""
        transcriber = OpenAICompatTranscriber(
            api_key="test-key",
            model="whisper-1",
        )

        # Create audio bytes larger than 25 MB
        large_audio = b"x" * (25 * 1024 * 1024 + 1)

        with pytest.raises(ValueError) as exc_info:
            await transcriber.transcribe(large_audio, "voice.ogg")

        assert "too large" in str(exc_info.value)


class TestGetTranscriber:
    """Tests for get_transcriber factory function."""

    def test_get_transcriber_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_transcriber returns None when whisper_provider is empty."""
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "")
        result = get_transcriber()
        assert result is None

    @patch("ccgram.whisper.openai_compat.openai.AsyncOpenAI")
    def test_get_transcriber_openai(
        self, mock_openai_class: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_transcriber returns OpenAICompatTranscriber with correct defaults."""
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "openai")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", None)
        monkeypatch.setattr("ccgram.config.config.whisper_model", None)
        monkeypatch.setattr("ccgram.config.config.whisper_language", None)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        result = get_transcriber()

        assert isinstance(result, OpenAICompatTranscriber)
        assert result.language is None
        # Verify AsyncOpenAI was called without base_url (default OpenAI)
        mock_openai_class.assert_called_once_with(api_key="test-key")

    @patch("ccgram.whisper.openai_compat.openai.AsyncOpenAI")
    def test_get_transcriber_groq(
        self, mock_openai_class: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_transcriber returns Groq-compatible transcriber."""
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "groq")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", None)
        monkeypatch.setattr("ccgram.config.config.whisper_model", None)
        monkeypatch.setattr("ccgram.config.config.whisper_language", None)
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")

        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        result = get_transcriber()

        assert isinstance(result, OpenAICompatTranscriber)
        assert result.model == "whisper-large-v3"
        # Verify AsyncOpenAI was called with Groq base_url
        mock_openai_class.assert_called_once_with(
            api_key="groq-key", base_url="https://api.groq.com/openai/v1"
        )
