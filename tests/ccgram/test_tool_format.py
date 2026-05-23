from ccgram.tool_format import (
    TOOL_EMOJI,
    compact_arg,
    format_tool_line,
    tool_emoji,
)


class TestToolEmoji:
    def test_exact_match(self) -> None:
        assert tool_emoji("Bash") == "\U0001f4bb"

    def test_exact_match_read(self) -> None:
        assert tool_emoji("Read") == "\U0001f4d6"

    def test_case_insensitive(self) -> None:
        assert tool_emoji("bash") == "\U0001f4bb"
        assert tool_emoji("READ") == "\U0001f4d6"
        assert tool_emoji("TASKCREATE") == "\U0001f4cb"

    def test_mcp_prefix_stripped(self) -> None:
        assert tool_emoji("mcp__deepwiki__ask_question") == tool_emoji("ask_question")

    def test_mcp_prefix_unknown_bare_name_falls_back(self) -> None:
        assert tool_emoji("mcp__server__totally_unknown_xyz") == "\U0001f527"

    def test_fallback_for_unknown(self) -> None:
        assert tool_emoji("ZZZUnknownTool") == "\U0001f527"

    def test_never_returns_empty(self) -> None:
        assert tool_emoji("") != ""
        assert tool_emoji("mcp__x__y") != ""

    def test_skill_emoji(self) -> None:
        assert tool_emoji("Skill") == "\U0001f4da"

    def test_grep_emoji(self) -> None:
        assert tool_emoji("Grep") == "\U0001f50e"

    def test_edit_emoji(self) -> None:
        assert tool_emoji("Edit") == "✏️"

    def test_multiedit_emoji(self) -> None:
        assert tool_emoji("MultiEdit") == "✏️"

    def test_pi_alias_bash(self) -> None:
        assert tool_emoji("bash") == "\U0001f4bb"

    def test_pi_alias_read_file(self) -> None:
        assert tool_emoji("read_file") == "\U0001f4d6"

    def test_codex_alias_exec_command(self) -> None:
        assert tool_emoji("exec_command") == "\U0001f4bb"

    def test_codex_alias_apply_patch(self) -> None:
        assert tool_emoji("apply_patch") == "✏️"

    def test_gemini_search_files(self) -> None:
        assert tool_emoji("search_files") == "\U0001f50e"

    def test_task_create_emoji(self) -> None:
        assert tool_emoji("TaskCreate") == "\U0001f4cb"

    def test_tool_emoji_dict_has_bash(self) -> None:
        assert "Bash" in TOOL_EMOJI
        assert TOOL_EMOJI["Bash"] == "\U0001f4bb"


class TestCompactArg:
    def test_collapses_whitespace(self) -> None:
        result = compact_arg("  hello   world  ")
        assert result == "hello world"

    def test_collapses_newlines(self) -> None:
        result = compact_arg("line1\nline2\nline3")
        assert "\n" not in result
        assert result == "line1 line2 line3"

    def test_collapses_mixed_whitespace(self) -> None:
        result = compact_arg("set -e\n  printf 'x'\n  git --version")
        assert "\n" not in result
        assert "\t" not in result

    def test_trims_at_cap(self) -> None:
        long_text = "a" * 90
        result = compact_arg(long_text)
        assert result == "a" * 50 + "…"
        assert len(result) == 51

    def test_no_trim_at_cap(self) -> None:
        text = "a" * 50
        result = compact_arg(text)
        assert result == text
        assert "…" not in result

    def test_custom_cap(self) -> None:
        result = compact_arg("hello world", cap=5)
        assert result == "hello…"

    def test_backtick_replaced_with_single_quote(self) -> None:
        result = compact_arg("run `make test`")
        assert "`" not in result
        assert "'" in result
        assert result == "run 'make test'"

    def test_empty_string(self) -> None:
        assert compact_arg("") == ""

    def test_whitespace_only(self) -> None:
        assert compact_arg("   \n\t  ") == ""


class TestFormatToolLine:
    def test_with_summary(self) -> None:
        result = format_tool_line("Bash", "ls -la")
        assert result == "\U0001f4bb **bash**: `ls -la`"

    def test_without_summary(self) -> None:
        result = format_tool_line("TodoRead", "")
        assert result == "\U0001f4cb **todoread**"

    def test_action_word_is_bold(self) -> None:
        result = format_tool_line("Read", "src/config.py")
        assert "**read**" in result

    def test_backticks_in_input_replaced_with_quotes(self) -> None:
        """Input backticks must be neutralized to avoid breaking inline-mono wrap."""
        result = format_tool_line("Bash", "run `make`")
        # Output has exactly two backticks — the wrapping pair around the arg.
        assert result.count("`") == 2
        assert result == "\U0001f4bb **bash**: `run 'make'`"

    def test_no_double_quote_around_summary(self) -> None:
        result = format_tool_line("Read", "src/config.py")
        assert '"' not in result

    def test_multiline_command_collapsed(self) -> None:
        cmd = "set -e\nprintf 'git: '\ngit --version"
        result = format_tool_line("Bash", cmd)
        assert "\n" not in result
        assert result.startswith("\U0001f4bb **bash**: `")

    def test_mcp_tool_fallback_emoji(self) -> None:
        result = format_tool_line(
            "mcp__deepwiki__totally_unknown_xyz", "how does X work"
        )
        assert result.startswith("\U0001f527 **mcp__deepwiki__totally_unknown_xyz**")

    def test_grep_with_summary(self) -> None:
        result = format_tool_line(
            "Grep", "config.yaml|auth.json|state.db|longer_pattern"
        )
        assert result.startswith("\U0001f50e **grep**: `")

    def test_skill_with_summary(self) -> None:
        result = format_tool_line("Skill", "github-repo-management")
        assert result == "\U0001f4da **skill**: `github-repo-management`"

    def test_read_with_path(self) -> None:
        result = format_tool_line("Read", "src/ccgram/config.py")
        assert result == "\U0001f4d6 **read**: `src/ccgram/config.py`"

    def test_summary_trimmed_to_cap(self) -> None:
        long_arg = "x" * 90
        result = format_tool_line("Bash", long_arg)
        assert "…" in result
        assert "x" * 50 in result
        assert "x" * 51 not in result

    def test_format_preserves_real_tool_name(self) -> None:
        result = format_tool_line("exec_command", "ls")
        assert "exec_command" in result

    def test_unknown_tool_uses_wrench(self) -> None:
        result = format_tool_line("UnknownXYZ", "some arg")
        assert result.startswith("\U0001f527 **unknownxyz**")
