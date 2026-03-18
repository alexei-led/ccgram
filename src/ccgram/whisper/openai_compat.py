"""OpenAI-compatible Whisper transcription implementation.

Supports any API that follows OpenAI's audio transcription endpoint
(e.g., OpenAI, Groq, local servers).
"""

import openai

from .base import TranscriptionResult


class OpenAICompatTranscriber:
    """Whisper transcriber using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        language: str | None = None,
    ) -> None:
        """Initialize the transcriber.

        Args:
            api_key: API key for the service.
            model: Model name (e.g., "whisper-1", "whisper-large-v3").
            base_url: Base URL for the API (None for OpenAI default).
            language: Optional language code (e.g., "en", "zh").
        """
        self.model = model
        self.language = language

        if base_url:
            self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = openai.AsyncOpenAI(api_key=api_key)

    async def transcribe(
        self, audio_bytes: bytes, filename: str
    ) -> TranscriptionResult:
        """Transcribe audio bytes using the OpenAI-compatible API.

        Args:
            audio_bytes: Raw audio file content.
            filename: Original filename (used as the multipart upload filename so the
                API can detect the audio format from the extension).

        Returns:
            TranscriptionResult with text and detected language.

        Raises:
            ValueError: If audio file exceeds 25 MB.
            RuntimeError: If the API call fails.
        """
        if len(audio_bytes) > 25 * 1024 * 1024:
            msg = "Audio file too large (max 25 MB)"
            raise ValueError(msg)

        try:
            if self.language:
                response = await self._client.audio.transcriptions.create(
                    model=self.model,
                    file=(filename, audio_bytes),
                    language=self.language,
                )
            else:
                response = await self._client.audio.transcriptions.create(
                    model=self.model,
                    file=(filename, audio_bytes),
                )
            return TranscriptionResult(text=response.text, language=None)
        except openai.APIError as e:
            msg = f"Transcription failed: {e}"
            raise RuntimeError(msg) from e
