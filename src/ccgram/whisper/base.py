"""Base types for Whisper transcription providers.

Defines the protocol and result types that all Whisper transcriber
implementations must follow.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TranscriptionResult:
    """Result of an audio transcription."""

    text: str
    language: str | None = None


class WhisperTranscriber(Protocol):
    """Protocol for Whisper-compatible transcription services.

    Implementations must provide an async transcribe method that takes
    audio bytes and returns a TranscriptionResult.
    """

    async def transcribe(
        self, audio_bytes: bytes, filename: str
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file content.
            filename: Original filename (for format detection).

        Returns:
            TranscriptionResult with text and detected language.
        """
        ...
