"""Tests for ps-based foreground process detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ccgram.providers.process_detection import (
    _pgid_cache,
    classify_provider_from_args,
    clear_detection_cache,
    detect_provider_cached,
    detect_provider_from_tty,
    get_foreground_args,
)


class TestClassifyProviderFromArgs:
    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ("bun /Users/x/.bun/bin/claude", "claude"),
            ("bun /Users/x/.bun/install/global/node_modules/cc-team/cli.js", "claude"),
            ("node /path/to/claude-code/cli.js", "claude"),
            ("claude --resume abc", "claude"),
            ("ce --current", "claude"),
            ("cc-mirror", "claude"),
            ("zai", "claude"),
            ("bun /Users/x/.bun/bin/codex --full-auto", "codex"),
            ("node /path/to/@openai/codex/bin/codex.js", "codex"),
            ("codex", "codex"),
            ("bun /Users/x/.bun/bin/gemini", "gemini"),
            ("node /path/to/gemini-cli/dist/index.js", "gemini"),
            ("gemini", "gemini"),
            ("-fish", "shell"),
            ("-bash", "shell"),
            ("bash ./scripts/restart.sh run", "shell"),
            ("zsh", "shell"),
            ("fish", "shell"),
            ("sudo codex", "codex"),
            ("env node /path/to/claude", "claude"),
            ("sudo env bun /Users/x/.bun/bin/codex", "codex"),
            ("python /path/to/gemini-cli/index.js", "gemini"),
            ("", ""),
            ("vim /some/file.py", ""),
            ("htop", ""),
            ("tmux", ""),
        ],
    )
    def test_classification(self, args: str, expected: str) -> None:
        assert classify_provider_from_args(args) == expected

    def test_claude_prefix_match(self) -> None:
        assert classify_provider_from_args("claude-code-wrapper") == "claude"

    def test_codex_prefix_match(self) -> None:
        assert classify_provider_from_args("codex-sandbox") == "codex"

    def test_gemini_prefix_match(self) -> None:
        assert classify_provider_from_args("gemini-pro") == "gemini"

    def test_stops_at_first_non_wrapper(self) -> None:
        assert classify_provider_from_args("vim /path/to/claude") == ""


PS_OUTPUT_CLAUDE = (
    " 8617  8617 Ss   -fish\n"
    " 8668  8668 S+   bun /Users/x/.bun/bin/claude\n"
    " 8690  8668 S+   bun /var/folders/context7-mcp\n"
)

PS_OUTPUT_CODEX = (
    "10001 10001 Ss   -zsh\n10050 10050 S+   bun /Users/x/.bun/bin/codex --full-auto\n"
)

PS_OUTPUT_SHELL_ONLY = " 5000  5000 Ss+  -bash\n"

PS_OUTPUT_NO_LEADER = " 9000  9000 Ss   -fish\n 9100  9050 S+   node /some/script.js\n"


class TestGetForegroundArgs:
    async def test_returns_group_leader_args(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_CLAUDE.encode(), b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            args, pgid = await get_foreground_args("/dev/ttys003")

        assert args == "bun /Users/x/.bun/bin/claude"
        assert pgid == 8668

    async def test_returns_fallback_when_no_leader(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_NO_LEADER.encode(), b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            args, pgid = await get_foreground_args("/dev/ttys005")

        assert args == "node /some/script.js"
        assert pgid == 9050

    async def test_returns_empty_on_error(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"error")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            args, pgid = await get_foreground_args("/dev/ttys003")

        assert args == ""
        assert pgid == 0

    async def test_returns_empty_on_timeout(self) -> None:
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=TimeoutError,
        ):
            args, pgid = await get_foreground_args("/dev/ttys003")

        assert args == ""
        assert pgid == 0

    async def test_returns_empty_for_empty_tty(self) -> None:
        args, pgid = await get_foreground_args("")
        assert args == ""
        assert pgid == 0

    async def test_returns_empty_on_oserror(self) -> None:
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("no such file"),
        ):
            args, pgid = await get_foreground_args("/dev/ttys003")

        assert args == ""
        assert pgid == 0


class TestDetectProviderFromTty:
    async def test_detects_claude(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_CLAUDE.encode(), b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await detect_provider_from_tty("/dev/ttys003")

        assert result == "claude"

    async def test_detects_codex(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_CODEX.encode(), b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await detect_provider_from_tty("/dev/ttys004")

        assert result == "codex"

    async def test_detects_shell(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_SHELL_ONLY.encode(), b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await detect_provider_from_tty("/dev/ttys005")

        assert result == "shell"


class TestDetectProviderCached:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        _pgid_cache.clear()

    async def test_cache_miss_calls_ps(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_CLAUDE.encode(), b"")

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            result = await detect_provider_cached("@0", "/dev/ttys003")

        assert result == "claude"
        assert mock_exec.called

    async def test_cache_hit_returns_cached(self) -> None:
        _pgid_cache["@0"] = (8668, "claude")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_CLAUDE.encode(), b"")

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch(
                "ccgram.providers.process_detection.classify_provider_from_args"
            ) as mock_classify,
        ):
            result = await detect_provider_cached("@0", "/dev/ttys003")

        assert result == "claude"
        # PGID unchanged → classify skipped (cache hit)
        mock_classify.assert_not_called()

    async def test_cache_invalidates_on_pgid_change(self) -> None:
        _pgid_cache["@0"] = (9999, "shell")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (PS_OUTPUT_CODEX.encode(), b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await detect_provider_cached("@0", "/dev/ttys003")

        assert result == "codex"
        assert _pgid_cache["@0"] == (10050, "codex")

    async def test_empty_args_returns_empty(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"error")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await detect_provider_cached("@0", "/dev/ttys003")

        assert result == ""
        assert "@0" not in _pgid_cache


class TestClearDetectionCache:
    def test_clear_specific(self) -> None:
        _pgid_cache["@0"] = (100, "claude")
        _pgid_cache["@1"] = (200, "codex")
        clear_detection_cache("@0")
        assert "@0" not in _pgid_cache
        assert "@1" in _pgid_cache

    def test_clear_all(self) -> None:
        _pgid_cache["@0"] = (100, "claude")
        _pgid_cache["@1"] = (200, "codex")
        clear_detection_cache()
        assert len(_pgid_cache) == 0

    def test_clear_nonexistent(self) -> None:
        clear_detection_cache("@99")
