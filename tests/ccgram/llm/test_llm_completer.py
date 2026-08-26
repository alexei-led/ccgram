"""Tests for LLM command generation modules."""

import json
from unittest.mock import MagicMock

import pytest

from ccgram.llm.httpx_completer import _build_user_message, _parse_command_result


class TestParseCommandResult:
    @pytest.mark.parametrize(
        ("text", "command", "explanation", "dangerous"),
        [
            pytest.param(
                '{"command": "ls -la", "explanation": "List all files", "dangerous": false}',
                "ls -la",
                "List all files",
                False,
                id="all_fields",
            ),
            pytest.param(
                '{"command": "touch a", "explanation": "Delete everything", "dangerous": true}',
                "touch a",
                "Delete everything",
                True,
                id="dangerous_flag_honoured",
            ),
            pytest.param(
                '```json\n{"command": "echo hi", "explanation": "Print hi", "dangerous": false}\n```',
                "echo hi",
                "Print hi",
                False,
                id="markdown_code_fences",
            ),
            pytest.param(
                '```\n{"command": "pwd", "explanation": "Print dir", "dangerous": false}\n```',
                "pwd",
                "Print dir",
                False,
                id="plain_code_fences",
            ),
            pytest.param(
                '  \n{"command": "ls", "explanation": "list", "dangerous": false}\n  ',
                "ls",
                "list",
                False,
                id="surrounding_whitespace",
            ),
            pytest.param(
                '{"command": "ls", "explanation": "list"}',
                "ls",
                "list",
                False,
                id="dangerous_defaults_to_false",
            ),
            pytest.param(
                '{"command": "ls", "explanation": 42, "dangerous": false}',
                "ls",
                "",
                False,
                id="non_string_explanation_dropped",
            ),
        ],
    )
    def test_parses_well_formed_response(
        self, text: str, command: str, explanation: str, dangerous: bool
    ) -> None:
        result = _parse_command_result(text)
        assert (result.command, result.explanation, result.is_dangerous) == (
            command,
            explanation,
            dangerous,
        )

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("this is not json at all", id="not_json"),
            pytest.param("", id="empty"),
            pytest.param(
                '{"explanation": "No command here", "dangerous": false}',
                id="no_command_key",
            ),
            pytest.param(
                '{"command": "", "explanation": "nothing", "dangerous": false}',
                id="empty_command",
            ),
            pytest.param(
                '{"command": 42, "explanation": "nothing"}', id="non_string_command"
            ),
            pytest.param('[{"command": "ls"}]', id="json_array"),
            pytest.param('"just a string"', id="json_scalar"),
        ],
    )
    def test_unusable_response_falls_back_to_raw_text_marked_dangerous(
        self, text: str
    ) -> None:
        result = _parse_command_result(text)
        assert result.command == text.strip()
        assert result.explanation == ""
        assert result.is_dangerous is True

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /tmp/x",
            "rm --recursive /tmp/x",
            "sudo rm /etc/passwd",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "shutdown -h now",
            "kill -9 1",
            "killall -q python",
            "chmod -R 777 /",
            "echo x > /dev/sda",
            "psql -c 'DROP TABLE users'",
        ],
    )
    def test_heuristic_overrides_a_benign_dangerous_flag(self, command: str) -> None:
        """The model claiming dangerous:false must not disarm the local guard —
        it gates the extra confirmation step in the shell approval flow."""
        text = json.dumps(
            {"command": command, "explanation": "harmless", "dangerous": False}
        )
        result = _parse_command_result(text)
        assert result.is_dangerous is True
        assert result.command == command

    def test_heuristic_leaves_safe_commands_alone(self) -> None:
        text = json.dumps(
            {"command": "rm /tmp/one-file", "explanation": "", "dangerous": False}
        )
        assert _parse_command_result(text).is_dangerous is False


class TestBuildUserMessage:
    def test_description_only(self) -> None:
        result = _build_user_message("list files")
        assert result == "list files"

    def test_with_cwd(self) -> None:
        result = _build_user_message("list files", cwd="/home/user")
        assert "list files" in result
        assert "CWD: /home/user" in result

    def test_with_shell(self) -> None:
        result = _build_user_message("list files", shell="bash")
        assert "Shell: bash" in result

    def test_with_os_info(self) -> None:
        result = _build_user_message("list files", os_info="Linux 6.1")
        assert "OS: Linux 6.1" in result

    def test_recent_output_trimmed_when_long(self) -> None:
        long_output = "x" * 1000
        result = _build_user_message("list files", recent_output=long_output)
        assert "Recent output:" in result
        lines = result.split("\n")
        output_line = [ln for ln in lines if ln.startswith("x")]
        assert len(output_line[0]) == 500

    def test_recent_output_kept_when_short(self) -> None:
        short_output = "$ ls\nfile1.txt"
        result = _build_user_message("list files", recent_output=short_output)
        assert short_output in result

    def test_all_context_fields(self) -> None:
        result = _build_user_message(
            "find large files",
            cwd="/tmp",
            shell="zsh",
            os_info="Darwin 24.0",
            recent_output="$ df -h",
        )
        assert "find large files" in result
        assert "CWD: /tmp" in result
        assert "Shell: zsh" in result
        assert "OS: Darwin 24.0" in result
        assert "Recent output:" in result
        assert "$ df -h" in result

    def test_with_shell_tools(self) -> None:
        result = _build_user_message(
            "find python files", shell_tools="fd (find replacement)"
        )
        assert "Available tools: fd (find replacement)" in result

    def test_context_section_starts_with_newline(self) -> None:
        result = _build_user_message("do something", cwd="/home")
        assert "\nContext:\n" in result


class TestGetCompleter:
    def _mock_config(self, monkeypatch, **attrs) -> None:
        mock_cfg = MagicMock()
        for key, value in attrs.items():
            setattr(mock_cfg, key, value)
        monkeypatch.setattr("ccgram.config.config", mock_cfg)

    def test_no_provider_returns_none(self, monkeypatch) -> None:
        self._mock_config(monkeypatch, llm_provider="")

        from ccgram.llm import get_completer

        assert get_completer() is None

    def test_unknown_provider_raises(self, monkeypatch) -> None:
        self._mock_config(monkeypatch, llm_provider="unknown_provider_xyz")

        from ccgram.llm import get_completer

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_completer()

    @pytest.mark.parametrize(
        ("provider", "api_key", "expected_cls", "expected_url"),
        [
            ("openai", "sk-test", "OpenAICompatCompleter", None),
            ("anthropic", "sk-ant-test", "AnthropicCompleter", None),
            ("xai", "xai-key", "OpenAICompatCompleter", "https://api.x.ai/v1"),
            (
                "deepseek",
                "ds-key",
                "OpenAICompatCompleter",
                "https://api.deepseek.com/v1",
            ),
            (
                "groq",
                "gsk-key",
                "OpenAICompatCompleter",
                "https://api.groq.com/openai/v1",
            ),
        ],
        ids=["openai", "anthropic", "xai", "deepseek", "groq"],
    )
    def test_provider_with_api_key(
        self, monkeypatch, provider, api_key, expected_cls, expected_url
    ) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider=provider,
            llm_api_key=api_key,
            llm_base_url="",
            llm_model="",
        )

        from ccgram.llm import get_completer

        result = get_completer()
        assert result is not None
        assert type(result).__name__ == expected_cls
        if expected_url:
            assert result._base_url == expected_url  # type: ignore[union-attr]

    def test_ollama_no_api_key_required(self, monkeypatch) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider="ollama",
            llm_api_key="",
            llm_base_url="",
            llm_model="",
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from ccgram.llm import get_completer
        from ccgram.llm.httpx_completer import OpenAICompatCompleter

        result = get_completer()
        assert isinstance(result, OpenAICompatCompleter)

    def test_missing_api_key_for_openai_raises(self, monkeypatch) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider="openai",
            llm_api_key="",
            llm_base_url="",
            llm_model="",
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from ccgram.llm import get_completer

        with pytest.raises(ValueError, match="No API key found"):
            get_completer()

    def test_api_key_from_provider_env_var(self, monkeypatch) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider="xai",
            llm_api_key="",
            llm_base_url="",
            llm_model="",
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-from-env")

        from ccgram.llm import get_completer
        from ccgram.llm.httpx_completer import OpenAICompatCompleter

        result = get_completer()
        assert isinstance(result, OpenAICompatCompleter)

    def test_fallback_to_openai_api_key(self, monkeypatch) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider="xai",
            llm_api_key="",
            llm_base_url="",
            llm_model="",
        )
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")

        from ccgram.llm import get_completer
        from ccgram.llm.httpx_completer import OpenAICompatCompleter

        result = get_completer()
        assert isinstance(result, OpenAICompatCompleter)

    def test_no_api_key_anywhere_raises(self, monkeypatch) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider="deepseek",
            llm_api_key="",
            llm_base_url="",
            llm_model="",
        )
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from ccgram.llm import get_completer

        with pytest.raises(ValueError, match="No API key found"):
            get_completer()

    def test_custom_model_and_base_url(self, monkeypatch) -> None:
        self._mock_config(
            monkeypatch,
            llm_provider="openai",
            llm_api_key="sk-test",
            llm_base_url="https://custom.api.com/v1",
            llm_model="custom-model",
        )

        from ccgram.llm import get_completer
        from ccgram.llm.httpx_completer import OpenAICompatCompleter

        result = get_completer()
        assert isinstance(result, OpenAICompatCompleter)
        assert result.model == "custom-model"
