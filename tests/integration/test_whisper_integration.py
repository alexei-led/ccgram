"""Integration tests for Whisper transcription — real Config + filesystem.

Tests get_transcriber() factory with real env vars and .env file loading,
covering all provider resolution paths and configuration edge cases.
"""

import pytest

pytestmark = pytest.mark.integration


class TestWhisperConfigIntegration:
    """Config correctly loads all whisper_* fields from env / .env file."""

    def test_whisper_fields_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-test")
        monkeypatch.setenv("ALLOWED_USERS", "1")
        monkeypatch.setenv("CCGRAM_WHISPER_PROVIDER", "groq")
        monkeypatch.setenv("CCGRAM_WHISPER_MODEL", "whisper-large-v3-turbo")
        monkeypatch.setenv("CCGRAM_WHISPER_LANGUAGE", "zh")

        from ccgram.config import Config

        cfg = Config()

        assert cfg.whisper_provider == "groq"
        assert cfg.whisper_model == "whisper-large-v3-turbo"
        assert cfg.whisper_language == "zh"

    def test_whisper_fields_from_dotenv(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=tok-dotenv\n"
            "ALLOWED_USERS=1\n"
            "CCGRAM_WHISPER_PROVIDER=openai\n"
        )
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("CCGRAM_WHISPER_PROVIDER", raising=False)

        from ccgram.config import Config

        cfg = Config()

        assert cfg.whisper_provider == "openai"

    def test_whisper_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-test")
        monkeypatch.setenv("ALLOWED_USERS", "1")
        monkeypatch.delenv("CCGRAM_WHISPER_PROVIDER", raising=False)

        from ccgram.config import Config

        cfg = Config()

        assert cfg.whisper_provider == ""

    def test_whisper_language_empty_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-test")
        monkeypatch.setenv("ALLOWED_USERS", "1")
        monkeypatch.delenv("CCGRAM_WHISPER_LANGUAGE", raising=False)

        from ccgram.config import Config

        cfg = Config()

        assert cfg.whisper_language == ""

    def test_custom_api_key_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-test")
        monkeypatch.setenv("ALLOWED_USERS", "1")
        monkeypatch.setenv("CCGRAM_WHISPER_API_KEY", "custom-override-key")

        from ccgram.config import Config

        cfg = Config()

        assert cfg.whisper_api_key == "custom-override-key"


class TestGetTranscriberIntegration:
    """get_transcriber() factory returns correct transcriber for each config."""

    def test_returns_none_when_provider_empty(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "")

        from ccgram.whisper import get_transcriber

        assert get_transcriber() is None

    def test_returns_openai_transcriber(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "openai")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", "")
        monkeypatch.setattr("ccgram.config.config.whisper_model", "")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from ccgram.whisper import get_transcriber
        from ccgram.whisper.openai_compat import OpenAICompatTranscriber

        transcriber = get_transcriber()

        assert transcriber is not None
        assert isinstance(transcriber, OpenAICompatTranscriber)
        assert transcriber.model == "whisper-1"  # built-in default

    def test_returns_groq_transcriber_with_correct_model(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "groq")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", "")
        monkeypatch.setattr("ccgram.config.config.whisper_model", "")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        from ccgram.whisper import get_transcriber
        from ccgram.whisper.openai_compat import OpenAICompatTranscriber

        transcriber = get_transcriber()

        assert transcriber is not None
        assert isinstance(transcriber, OpenAICompatTranscriber)
        assert transcriber.model == "whisper-large-v3"  # Groq built-in default

    def test_model_override_applies(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "groq")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", "")
        monkeypatch.setattr(
            "ccgram.config.config.whisper_model", "whisper-large-v3-turbo"
        )
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        from ccgram.whisper import get_transcriber
        from ccgram.whisper.openai_compat import OpenAICompatTranscriber

        transcriber = get_transcriber()

        assert isinstance(transcriber, OpenAICompatTranscriber)
        assert transcriber.model == "whisper-large-v3-turbo"

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "openai")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", "")
        monkeypatch.setattr("ccgram.config.config.whisper_model", "")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from ccgram.whisper import get_transcriber

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_transcriber()

    def test_custom_api_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "groq")
        monkeypatch.setattr("ccgram.config.config.whisper_api_key", "custom-key")
        monkeypatch.setattr("ccgram.config.config.whisper_base_url", "")
        monkeypatch.setattr("ccgram.config.config.whisper_model", "")
        # No GROQ_API_KEY set — custom key must be used instead
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from ccgram.whisper import get_transcriber
        from ccgram.whisper.openai_compat import OpenAICompatTranscriber

        # Should succeed without raising despite missing GROQ_API_KEY
        transcriber = get_transcriber()

        assert isinstance(transcriber, OpenAICompatTranscriber)

    def test_raises_on_unknown_provider(self, monkeypatch):
        monkeypatch.setattr("ccgram.config.config.whisper_provider", "unknown-llm")

        from ccgram.whisper import get_transcriber

        with pytest.raises(ValueError, match="Unknown whisper provider"):
            get_transcriber()
