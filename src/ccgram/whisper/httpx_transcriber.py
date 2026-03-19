"""OpenAI-compatible Whisper transcription via httpx.

Supports any API that follows OpenAI's audio transcription endpoint
(e.g., OpenAI, Groq, local servers). Uses raw httpx instead of the
openai SDK to avoid a heavy dependency for a single API call.
"""

import httpx

from .base import TranscriptionResult

_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatTranscriber:
    """Whisper transcriber using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        language: str | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self._api_key = api_key
        self._base_url = (base_url or _OPENAI_BASE_URL).rstrip("/")

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

        data = {"model": self.model}
        if self.language:
            data["language"] = self.language

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self._base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": (filename, audio_bytes)},
                    data=data,
                    timeout=60.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                msg = f"Transcription failed: {exc.response.status_code} {exc.response.text}"
                raise RuntimeError(msg) from exc
            except httpx.HTTPError as exc:
                msg = f"Transcription failed: {exc}"
                raise RuntimeError(msg) from exc

        try:
            text = response.json()["text"]
        except (KeyError, ValueError) as exc:
            msg = f"Unexpected API response: {response.text[:200]}"
            raise RuntimeError(msg) from exc

        return TranscriptionResult(text=text, language=None)
