from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
from ccgram.tts import TtsAudio, TtsSynthesisError, get_synthesizer, prepare_tts_text


class TestPrepareTtsText:
    def test_strips_pagination_and_markdown(self):
        parts = ("Hello **world**\n\n[1/2]", "More _text_\n\n[2/2]")
        assert prepare_tts_text(parts) == "Hello world\nMore text"

    def test_strips_user_prefix(self):
        assert prepare_tts_text(("\U0001f464 Hello there",)) == "Hello there"

    def test_skips_blank_parts(self):
        assert prepare_tts_text(("", "   ", "hello")) == "hello"

    def test_returns_empty_for_all_blank(self):
        assert prepare_tts_text(("", "  ")) == ""


class TestTtsTypes:
    def test_audio_is_frozen(self):
        audio = TtsAudio(data=b"abc")
        with pytest.raises(FrozenInstanceError):
            audio.data = b"xyz"  # type: ignore[misc]

    def test_audio_default_filename(self):
        assert TtsAudio(data=b"abc").filename == "reply.mp3"

    def test_synthesis_error_is_exception(self):
        err = TtsSynthesisError("boom")
        assert isinstance(err, Exception)
        assert str(err) == "boom"


class TestGetSynthesizer:
    @staticmethod
    def _config(monkeypatch, **attrs: str) -> None:
        defaults = {
            "tts_provider": "",
            "tts_voice": "en-US-AriaNeural",
            "tts_model": "gpt-4o-mini-tts",
            "tts_api_key": "",
        }
        fake = MagicMock()
        for name, value in (defaults | attrs).items():
            setattr(fake, name, value)
        monkeypatch.setattr("ccgram.tts.config", fake)

    def test_returns_none_when_unconfigured(self, monkeypatch):
        self._config(monkeypatch)
        assert get_synthesizer() is None

    def test_edge_provider_builds_edge_synthesizer(self, monkeypatch):
        from ccgram.tts.edge import EdgeTtsSynthesizer

        self._config(monkeypatch, tts_provider="edge", tts_voice="de-DE-KatjaNeural")

        synth = get_synthesizer()

        assert isinstance(synth, EdgeTtsSynthesizer)
        assert synth._voice == "de-DE-KatjaNeural"

    def test_llm_api_key_is_not_reused_for_openai_tts(self, monkeypatch):
        """CCGRAM_LLM_API_KEY may be a non-OpenAI key — it must not leak here."""
        self._config(monkeypatch, tts_provider="openai")
        monkeypatch.setenv("CCGRAM_LLM_API_KEY", "xai-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="No API key for OpenAI TTS"):
            get_synthesizer()
