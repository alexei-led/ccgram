import pytest

from ccgram.tool_format import (
    TOOL_EMOJI,
    compact_arg,
    format_tool_line,
    tool_emoji,
)

BASH = "\U0001f4bb"
READ = "\U0001f4d6"
WRITE = "\U0001f4dd"
EDIT = "✏️"
GREP = "\U0001f50e"
FOLDER = "\U0001f4c2"
CLIPBOARD = "\U0001f4cb"
SKILL = "\U0001f4da"
QUESTION = "❓"
WEB = "\U0001f310"
WRENCH = "\U0001f527"


class TestToolEmoji:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Bash", BASH),
            ("Read", READ),
            ("Write", WRITE),
            ("Edit", EDIT),
            ("MultiEdit", EDIT),
            ("Grep", GREP),
            ("Glob", FOLDER),
            ("Skill", SKILL),
            ("TaskCreate", CLIPBOARD),
            ("AskUserQuestion", QUESTION),
            ("WebFetch", WEB),
        ],
    )
    def test_claude_canonical_names(self, name: str, expected: str) -> None:
        assert tool_emoji(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("bash", BASH),
            ("read_file", READ),
            ("write_file", WRITE),
            ("apply_patch", EDIT),
            ("exec_command", BASH),
            ("search_files", GREP),
            ("find", FOLDER),
            ("web_search", GREP),
            ("fetch", WEB),
        ],
    )
    def test_provider_aliases(self, name: str, expected: str) -> None:
        assert tool_emoji(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("bash", BASH),
            ("BASH", BASH),
            ("READ", READ),
            ("TASKCREATE", CLIPBOARD),
            ("mUlTiEdIt", EDIT),
        ],
    )
    def test_lookup_is_case_insensitive(self, name: str, expected: str) -> None:
        assert tool_emoji(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("mcp__deepwiki__ask_question", QUESTION),
            ("mcp__server__ASK_QUESTION", QUESTION),
            ("mcp__server__Bash", BASH),
            ("mcp__server__totally_unknown_xyz", WRENCH),
        ],
    )
    def test_mcp_prefix_stripped_before_lookup(self, name: str, expected: str) -> None:
        assert tool_emoji(name) == expected

    @pytest.mark.parametrize(
        "name", ["ZZZUnknownTool", "", "mcp__x__y", "mcp__malformed", "  "]
    )
    def test_unknown_names_fall_back_to_wrench(self, name: str) -> None:
        assert tool_emoji(name) == WRENCH

    @pytest.mark.parametrize("name", list(TOOL_EMOJI))
    def test_every_mapped_name_renders_non_empty(self, name: str) -> None:
        assert tool_emoji(name) != ""


class TestCompactArg:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  hello   world  ", "hello world"),
            ("line1\nline2\nline3", "line1 line2 line3"),
            ("set -e\n\tprintf 'x'", "set -e printf 'x'"),
            ("", ""),
            ("   \n\t  ", ""),
        ],
    )
    def test_collapses_whitespace(self, raw: str, expected: str) -> None:
        assert compact_arg(raw) == expected

    def test_backticks_become_single_quotes(self) -> None:
        assert compact_arg("run `make test`") == "run 'make test'"

    @pytest.mark.parametrize(
        ("raw", "cap", "expected"),
        [
            ("a" * 50, 50, "a" * 50),
            ("a" * 51, 50, "a" * 50 + "…"),
            ("a" * 90, 50, "a" * 50 + "…"),
            ("hello world", 5, "hello…"),
            ("hello", 5, "hello"),
        ],
    )
    def test_trims_at_cap(self, raw: str, cap: int, expected: str) -> None:
        assert compact_arg(raw, cap=cap) == expected

    def test_cap_applies_after_whitespace_collapse(self) -> None:
        """Collapsing runs first, so padded input under the cap is not trimmed."""
        assert compact_arg("a" * 40 + "\n\n\n" + "b" * 9, cap=50) == (
            "a" * 40 + " " + "b" * 9
        )


class TestFormatToolLine:
    @pytest.mark.parametrize(
        ("name", "summary", "expected"),
        [
            ("Bash", "ls -la", f"{BASH} **bash**: `ls -la`"),
            (
                "Read",
                "src/ccgram/config.py",
                f"{READ} **read**: `src/ccgram/config.py`",
            ),
            (
                "Skill",
                "github-repo-management",
                f"{SKILL} **skill**: `github-repo-management`",
            ),
            ("TodoRead", "", f"{CLIPBOARD} **todoread**"),
            ("UnknownXYZ", "some arg", f"{WRENCH} **unknownxyz**: `some arg`"),
        ],
    )
    def test_renders_emoji_bold_name_and_mono_arg(
        self, name: str, summary: str, expected: str
    ) -> None:
        assert format_tool_line(name, summary) == expected

    @pytest.mark.parametrize("summary", ["", "   ", "\n\t "])
    def test_blank_summary_omits_the_arg_section(self, summary: str) -> None:
        assert format_tool_line("TodoRead", summary) == f"{CLIPBOARD} **todoread**"

    def test_backticks_in_input_replaced_with_quotes(self) -> None:
        """Input backticks must be neutralized to avoid breaking inline-mono wrap."""
        result = format_tool_line("Bash", "run `make`")
        assert result == f"{BASH} **bash**: `run 'make'`"
        assert result.count("`") == 2

    def test_multiline_command_collapsed_to_one_line(self) -> None:
        cmd = "set -e\nprintf 'git: '\ngit --version"
        result = format_tool_line("Bash", cmd)
        assert "\n" not in result
        assert result == f"{BASH} **bash**: `set -e printf 'git: ' git --version`"

    def test_summary_trimmed_to_cap(self) -> None:
        result = format_tool_line("Bash", "x" * 90)
        assert result == f"{BASH} **bash**: `{'x' * 50}…`"

    def test_preserves_real_tool_name_lowercased(self) -> None:
        assert (
            format_tool_line("exec_command", "ls") == f"{BASH} **exec_command**: `ls`"
        )

    def test_mcp_name_kept_intact_with_fallback_emoji(self) -> None:
        result = format_tool_line(
            "mcp__deepwiki__totally_unknown_xyz", "how does X work"
        )
        assert result.startswith(f"{WRENCH} **mcp__deepwiki__totally_unknown_xyz**")
