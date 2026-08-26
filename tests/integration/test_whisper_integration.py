"""Integration tests for Whisper transcription — real Config + filesystem.

Covers how ``Config`` picks up the whisper settings (env vars and a real
``.env`` file) and how ``get_transcriber()`` turns them into a transcriber:
provider defaults, per-field overrides, and the failure paths.
"""

import pytest

from ccgram.whisper.httpx_transcriber import OpenAICompatTranscriber

pytestmark = pytest.mark.integration

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OPENAI_BASE_URL = "https://api.openai.com/v1"


@pytest.fixture
def make_config(tmp_path, monkeypatch):
    """Factory: build a real Config rooted at an isolated CCGRAM_DIR."""

    def _make(env: dict[str, str], *, dotenv: str | None = None):
        if dotenv is not None:
            (tmp_path / ".env").write_text(dotenv)
        monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-test")
        monkeypatch.setenv("ALLOWED_USERS", "1")
        for key in (
            "CCGRAM_WHISPER_PROVIDER",
            "CCGRAM_WHISPER_MODEL",
            "CCGRAM_WHISPER_LANGUAGE",
            "CCGRAM_WHISPER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        from ccgram.config import Config

        return Config()

    return _make


@pytest.fixture
def whisper_config(monkeypatch):
    """Factory: patch the config singleton's whisper fields for the factory tests."""

    def _set(
        *,
        provider: str = "",
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        language: str = "",
    ) -> None:
        for field, value in (
            ("whisper_provider", provider),
            ("whisper_api_key", api_key),
            ("whisper_base_url", base_url),
            ("whisper_model", model),
            ("whisper_language", language),
        ):
            monkeypatch.setattr(f"ccgram.config.config.{field}", value)

    return _set


class TestWhisperConfigIntegration:
    @pytest.mark.parametrize(
        ("env", "field", "expected"),
        [
            pytest.param(
                {"CCGRAM_WHISPER_PROVIDER": "groq"},
                "whisper_provider",
                "groq",
                id="provider-from-env",
            ),
            pytest.param(
                {"CCGRAM_WHISPER_MODEL": "whisper-large-v3-turbo"},
                "whisper_model",
                "whisper-large-v3-turbo",
                id="model-from-env",
            ),
            pytest.param(
                {"CCGRAM_WHISPER_LANGUAGE": "zh"},
                "whisper_language",
                "zh",
                id="language-from-env",
            ),
            pytest.param(
                {"CCGRAM_WHISPER_API_KEY": "custom-override-key"},
                "whisper_api_key",
                "custom-override-key",
                id="api-key-from-env",
            ),
            pytest.param({}, "whisper_provider", "", id="provider-disabled-by-default"),
            pytest.param({}, "whisper_language", "", id="language-empty-by-default"),
        ],
    )
    def test_whisper_field_resolution(
        self, make_config, env: dict[str, str], field: str, expected: str
    ) -> None:
        assert getattr(make_config(env), field) == expected

    def test_whisper_fields_read_from_dotenv_file(self, make_config, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        cfg = make_config(
            {},
            dotenv=(
                "TELEGRAM_BOT_TOKEN=tok-dotenv\n"
                "ALLOWED_USERS=1\n"
                "CCGRAM_WHISPER_PROVIDER=openai\n"
            ),
        )
        assert cfg.whisper_provider == "openai"


class TestGetTranscriberIntegration:
    def test_returns_none_when_provider_empty(self, whisper_config):
        whisper_config(provider="")

        from ccgram.whisper import get_transcriber

        assert get_transcriber() is None

    @pytest.mark.parametrize(
        ("provider", "key_env", "expected_model", "expected_base_url"),
        [
            pytest.param(
                "openai", "OPENAI_API_KEY", "whisper-1", _OPENAI_BASE_URL, id="openai"
            ),
            pytest.param(
                "groq", "GROQ_API_KEY", "whisper-large-v3", _GROQ_BASE_URL, id="groq"
            ),
        ],
    )
    def test_provider_defaults_applied(
        self,
        whisper_config,
        monkeypatch,
        provider: str,
        key_env: str,
        expected_model: str,
        expected_base_url: str,
    ) -> None:
        whisper_config(provider=provider)
        monkeypatch.setenv(key_env, "key-from-env")

        from ccgram.whisper import get_transcriber

        transcriber = get_transcriber()

        assert isinstance(transcriber, OpenAICompatTranscriber)
        assert transcriber.model == expected_model
        assert transcriber._base_url == expected_base_url
        assert transcriber.language is None

    @pytest.mark.parametrize(
        ("overrides", "attr", "expected"),
        [
            pytest.param(
                {"model": "whisper-large-v3-turbo"},
                "model",
                "whisper-large-v3-turbo",
                id="model",
            ),
            pytest.param(
                {"base_url": "http://localhost:8080/v1"},
                "_base_url",
                "http://localhost:8080/v1",
                id="base-url",
            ),
            pytest.param({"language": "de"}, "language", "de", id="language"),
            pytest.param(
                {"api_key": "custom-key"}, "_api_key", "custom-key", id="api-key"
            ),
        ],
    )
    def test_config_overrides_beat_provider_defaults(
        self, whisper_config, monkeypatch, overrides: dict, attr: str, expected: str
    ) -> None:
        whisper_config(provider="groq", **overrides)
        monkeypatch.setenv("GROQ_API_KEY", "key-from-env")

        from ccgram.whisper import get_transcriber

        assert getattr(get_transcriber(), attr) == expected

    def test_raises_on_missing_api_key(self, whisper_config, monkeypatch):
        whisper_config(provider="openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from ccgram.whisper import get_transcriber

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_transcriber()

    def test_raises_on_unknown_provider(self, whisper_config):
        whisper_config(provider="unknown-llm")

        from ccgram.whisper import get_transcriber

        with pytest.raises(ValueError, match="Unknown whisper provider"):
            get_transcriber()
