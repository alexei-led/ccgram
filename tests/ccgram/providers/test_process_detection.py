from __future__ import annotations

from unittest.mock import patch

import pytest

from ccgram.multiplexer.base import ForegroundInfo
from ccgram.providers.process_detection import (
    _pgid_cache,
    classify_provider_from_args,
    classify_provider_from_argv,
    clear_detection_cache,
    detect_provider_cached,
)


def _fg(argv: list[str], pgid: int) -> ForegroundInfo:
    """Build a ForegroundInfo as the multiplexer seam would return it."""
    return ForegroundInfo(pid=pgid, pgid=pgid, argv=argv, cwd="/tmp")


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


class TestClassifyProviderFromArgv:
    """``classify_provider_from_argv`` is the list-native core; the string
    variant just splits and delegates."""

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["bun", "/Users/x/.bun/bin/claude"], "claude"),
            (["sudo", "env", "bun", "/Users/x/.bun/bin/codex"], "codex"),
            (["-bash"], "shell"),
            (["bash", "./scripts/restart.sh", "run"], "shell"),
            (["vim", "/path/to/claude"], ""),
            ([], ""),
        ],
    )
    def test_classification(self, argv: list[str], expected: str) -> None:
        assert classify_provider_from_argv(argv) == expected

    def test_matches_string_variant(self) -> None:
        args: str = "sudo env bun /Users/x/.bun/bin/codex --full-auto"
        assert classify_provider_from_argv(args.split()) == classify_provider_from_args(
            args
        )


class TestDetectProviderCached:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        _pgid_cache.clear()

    async def test_cache_miss_classifies(self) -> None:
        fg = _fg(["bun", "/Users/x/.bun/bin/claude"], 8668)
        result = await detect_provider_cached("@0", fg)
        assert result == "claude"
        assert _pgid_cache["@0"] == (8668, "claude")

    async def test_cache_hit_skips_classification(self) -> None:
        _pgid_cache["@0"] = (8668, "claude")
        fg = _fg(["bun", "/Users/x/.bun/bin/claude"], 8668)

        with patch(
            "ccgram.providers.process_detection.classify_provider_from_argv"
        ) as mock_classify:
            result = await detect_provider_cached("@0", fg)

        assert result == "claude"
        mock_classify.assert_not_called()

    async def test_cache_invalidates_on_pgid_change(self) -> None:
        _pgid_cache["@0"] = (9999, "shell")
        fg = _fg(["bun", "/Users/x/.bun/bin/codex", "--full-auto"], 10050)

        result = await detect_provider_cached("@0", fg)

        assert result == "codex"
        assert _pgid_cache["@0"] == (10050, "codex")

    async def test_none_foreground_returns_empty(self) -> None:
        result = await detect_provider_cached("@0", None)
        assert result == ""
        assert "@0" not in _pgid_cache

    async def test_empty_argv_returns_empty(self) -> None:
        result = await detect_provider_cached("@0", _fg([], 1234))
        assert result == ""
        assert "@0" not in _pgid_cache

    async def test_zero_pgid_returns_empty(self) -> None:
        fg = ForegroundInfo(pid=0, pgid=0, argv=["bun", "claude"], cwd="/tmp")
        result = await detect_provider_cached("@0", fg)
        assert result == ""
        assert "@0" not in _pgid_cache

    async def test_unrecognized_argv_not_cached(self) -> None:
        result = await detect_provider_cached("@0", _fg(["vim", "x.py"], 555))
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
