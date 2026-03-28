"""Tests for terminal_parser — regex-based detection of Claude Code UI elements."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ccgram.screen_buffer import ScreenBuffer

from ccgram.terminal_parser import (
    extract_bash_output,
    extract_interactive_content,
    find_chrome_boundary,
    format_status_display,
    is_likely_spinner,
    parse_status_line,
    strip_pane_chrome,
)

# ── is_likely_spinner ────────────────────────────────────────────────────


class TestIsLikelySpinner:
    @pytest.mark.parametrize(
        "char",
        ["·", "✻", "✽", "✶", "✳", "✢"],
        ids=[
            "middle_dot",
            "heavy_asterisk",
            "heavy_teardrop",
            "six_star",
            "eight_star",
            "cross",
        ],
    )
    def test_known_spinners(self, char: str):
        assert is_likely_spinner(char) is True

    @pytest.mark.parametrize(
        "char",
        ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        ids=[f"braille_{i}" for i in range(10)],
    )
    def test_braille_spinners(self, char: str):
        assert is_likely_spinner(char) is True

    @pytest.mark.parametrize(
        "char",
        ["─", "│", "┌", "┐", ">", "|"],
        ids=["h_line", "v_line", "corner_tl", "corner_tr", "gt", "pipe"],
    )
    def test_non_spinners_box_drawing(self, char: str):
        assert is_likely_spinner(char) is False

    @pytest.mark.parametrize(
        "char",
        ["A", "z", "0", " ", ""],
        ids=["upper_a", "lower_z", "digit", "space", "empty"],
    )
    def test_non_spinners_common(self, char: str):
        assert is_likely_spinner(char) is False

    def test_math_symbol_detected(self):
        assert is_likely_spinner("∑") is True

    def test_other_symbol_detected(self):
        assert is_likely_spinner("⚡") is True

    @pytest.mark.parametrize(
        "char",
        ["!", "#", "%", "@", "*", "/", "\\", "~", "?", ",", "."],
        ids=[
            "bang",
            "hash",
            "pct",
            "at",
            "star",
            "slash",
            "bslash",
            "tilde",
            "qmark",
            "comma",
            "dot",
        ],
    )
    def test_ascii_punctuation_rejected(self, char: str):
        assert is_likely_spinner(char) is False


# ── parse_status_line ────────────────────────────────────────────────────


_SEPARATOR = "─" * 30


class TestParseStatusLine:
    @pytest.mark.parametrize(
        ("spinner", "rest", "expected"),
        [
            ("·", "Working on task", "Working on task"),
            ("✻", "  Reading file  ", "Reading file"),
            ("✽", "Thinking deeply", "Thinking deeply"),
            ("✶", "Analyzing code", "Analyzing code"),
            ("✳", "Processing input", "Processing input"),
            ("✢", "Building project", "Building project"),
        ],
    )
    def test_spinner_chars(self, spinner: str, rest: str, expected: str):
        pane = f"some output\n{spinner}{rest}\n{_SEPARATOR}\n"
        assert parse_status_line(pane) == expected

    @pytest.mark.parametrize(
        ("spinner", "text"),
        [
            ("⠋", "Loading modules"),
            ("⠹", "Compiling assets"),
            ("⠏", "Fetching data"),
        ],
    )
    def test_braille_spinners_detected(self, spinner: str, text: str):
        pane = f"some output\n{spinner} {text}\n{_SEPARATOR}\n"
        assert parse_status_line(pane) == text

    @pytest.mark.parametrize(
        "pane",
        [
            pytest.param("just normal text\nno spinners here\n", id="no_spinner"),
            pytest.param("", id="empty"),
            pytest.param(
                f"some output\n· bullet point\nmore text\n{_SEPARATOR}\n",
                id="spinner_not_above_separator",
            ),
        ],
    )
    def test_returns_none(self, pane: str):
        assert parse_status_line(pane) is None

    def test_adaptive_scan_finds_distant_separator(self):
        pane = f"✻ Doing work\n{_SEPARATOR}\n" + "trailing\n" * 16
        assert parse_status_line(pane) == "Doing work"

    def test_ignores_bullet_points(self):
        pane = (
            "Here are some items:\n"
            "· first item\n"
            "· second item\n"
            "normal line\n"
            f"{_SEPARATOR}\n"
        )
        assert parse_status_line(pane) is None

    def test_bottom_up_scan_with_chrome(self):
        pane = f"output\n✻ Doing work\n{_SEPARATOR}\n❯\n"
        assert parse_status_line(pane) == "Doing work"

    def test_two_separator_layout(self):
        pane = (
            "output\n"
            "✶ Perusing… (3m 35s)\n"
            "\n"
            f"{_SEPARATOR}\n"
            "❯ \n"
            f"{_SEPARATOR}\n"
            "   ⎇ main  ~/Workspace/proj  ✱ Opus 4.6\n"
        )
        assert parse_status_line(pane) == "Perusing… (3m 35s)"

    def test_two_separator_no_blank_line(self):
        pane = (
            "output\n"
            "✶ Working hard\n"
            f"{_SEPARATOR}\n"
            "❯ \n"
            f"{_SEPARATOR}\n"
            "   ⎇ main  ✱ Opus 4.6\n"
        )
        assert parse_status_line(pane) == "Working hard"

    def test_uses_fixture(self, sample_pane_status_line: str):
        assert parse_status_line(sample_pane_status_line) == "Reading file src/main.py"


# ── extract_interactive_content ──────────────────────────────────────────


class TestExtractInteractiveContent:
    def test_exit_plan_mode(self, sample_pane_exit_plan: str):
        result = extract_interactive_content(sample_pane_exit_plan)
        assert result is not None
        assert result.name == "ExitPlanMode"
        assert "Would you like to proceed?" in result.content
        assert "ctrl-g to edit in" in result.content

    def test_exit_plan_mode_variant(self):
        pane = (
            "  Claude has written up a plan\n  ─────\n  Details here\n  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "ExitPlanMode"
        assert "Claude has written up a plan" in result.content

    def test_ask_user_multi_tab(self, sample_pane_ask_user_multi_tab: str):
        result = extract_interactive_content(sample_pane_ask_user_multi_tab)
        assert result is not None
        assert result.name == "AskUserQuestion"
        assert "←" in result.content

    def test_ask_user_single_tab(self, sample_pane_ask_user_single_tab: str):
        result = extract_interactive_content(sample_pane_ask_user_single_tab)
        assert result is not None
        assert result.name == "AskUserQuestion"
        assert "Enter to select" in result.content

    def test_permission_prompt(self, sample_pane_permission: str):
        result = extract_interactive_content(sample_pane_permission)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert "Do you want to proceed?" in result.content

    def test_edit_permission_prompt_structural(self):
        """PermissionPrompt now matches 'Do you want to make this edit' directly."""
        pane = (
            "  Do you want to make this edit to status_polling.py?\n"
            "\n"
            "  ❯ Yes    Yes All    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert "make this edit" in result.content

    def test_unknown_selection_ui_structural(self):
        """Catch-all detects any future selection prompt with ❯ + action hint."""
        pane = (
            "  Some brand new question nobody predicted?\n"
            "\n"
            "  ❯ Option A    Option B    Option C\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectionUI"
        assert "❯" in result.content
        assert "brand new question" in result.content

    def test_structural_catchall_enter_to_confirm(self):
        """Catch-all works with 'Enter to confirm' bottom hint too."""
        pane = "  Pick a thing:\n  ❯ Alpha\n    Beta\n  Enter to confirm\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectionUI"
        assert "Pick a thing" in result.content

    def test_numbered_list_without_footer(self):
        """Selection UI with numbered items but no action hint footer."""
        pane = (
            "Remote Control\n"
            "\n"
            "   Remote Control lets you access this CLI session.\n"
            "\n"
            "   ❯ 1. Enable Remote Control for this session\n"
            "     2. Never mind\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectionUI"
        assert "Remote Control" in result.content
        assert "❯" in result.content

    def test_codex_selection_cursor(self):
        """Codex uses › (U+203A) instead of ❯ (U+276F) for selection cursor."""
        pane = "  Which option?\n\n  › Option A    Option B\n\n  Esc to cancel\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectionUI"
        assert "Which option?" in result.content

    @pytest.mark.parametrize(
        "bottom_text",
        [
            "  Press enter to confirm",
            "  Press enter to select",
            "  Press enter to submit",
            "  Press enter to continue",
            "  enter to submit",
            "  enter to confirm",
        ],
        ids=[
            "press-confirm",
            "press-select",
            "press-submit",
            "press-continue",
            "enter-submit",
            "enter-confirm",
        ],
    )
    def test_codex_bottom_text_variants(self, bottom_text: str):
        """Codex uses different bottom action hint phrasing."""
        pane = f"  Question?\n  › Option A\n    Option B\n{bottom_text}\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectionUI"

    def test_restore_checkpoint(self):
        pane = (
            "  Restore the code to a previous state?\n"
            "  ─────\n"
            "  Some details\n"
            "  Enter to continue\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "RestoreCheckpoint"
        assert "Restore the code" in result.content

    def test_settings(self):
        pane = "  Settings: press tab to cycle\n  ─────\n  Option 1\n  Esc to cancel\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "Settings"
        assert "Settings:" in result.content

    def test_select_model(self):
        pane = (
            " Select model\n"
            " Switch between Claude models.\n"
            "\n"
            " ❯ 1. Default (recommended) ✔  Opus 4.6\n"
            "   2. Sonnet                   Sonnet 4.6\n"
            "\n"
            " ▌▌▌ Medium effort ← → to adjust\n"
            "\n"
            " Enter to confirm · Esc to exit\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectModel"
        assert "Select model" in result.content
        assert "Enter to confirm" in result.content

    def test_permission_prompt_edit_tool(self):
        pane = (
            "  Do you want to make this edit to config.py?\n"
            "\n"
            "  ❯ Yes    Yes All    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert "make this edit" in result.content

    def test_network_request_outside_sandbox(self):
        pane = (
            "  Network request outside of sandbox\n"
            "\n"
            "  WebFetch wants to access: https://example.com\n"
            "\n"
            "  ❯ Yes    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert "Network request outside of sandbox" in result.content

    def test_network_request_no_cursor(self):
        """Network sandbox prompt detected even without ❯ cursor."""
        pane = (
            "  Network request outside of sandbox\n"
            "\n"
            "  WebFetch wants to access: https://example.com\n"
            "\n"
            "  Yes    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"

    def test_allow_permission_prompt(self):
        pane = (
            "  Allow mcp__server__tool to access /tmp?\n"
            "\n"
            "  ❯ Yes    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"

    def test_bash_command_requires_approval(self):
        pane = (
            "  Bash command\n"
            "  make test\n"
            "  This command requires approval\n"
            "\n"
            "  ❯ Yes    Yes All    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"
        assert "requires approval" in result.content

    def test_bash_command_requires_approval_no_cursor(self):
        pane = (
            "  Bash command\n"
            "  make test\n"
            "  This command requires approval\n"
            "\n"
            "  Yes    Yes All    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"

    def test_settings_esc_to_exit(self):
        pane = "  Settings: press tab to cycle\n  ─────\n  Option 1\n  Esc to exit\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "Settings"
        assert "Settings:" in result.content

    def test_select_model_esc_to_exit(self):
        pane = (
            " Select model\n"
            " Switch between Claude models.\n"
            "\n"
            " ❯ 1. Default (recommended)\n"
            "   2. Sonnet\n"
            "\n"
            " Esc to exit\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectModel"
        assert "Select model" in result.content

    @pytest.mark.parametrize(
        "pane",
        [
            pytest.param("$ echo hello\nhello\n$\n", id="no_ui"),
            pytest.param("", id="empty"),
        ],
    )
    def test_returns_none(self, pane: str):
        assert extract_interactive_content(pane) is None

    def test_min_gap_too_small_returns_none(self):
        pane = "  Do you want to proceed?\n  Esc to cancel\n"
        assert extract_interactive_content(pane) is None


# ── extract_interactive_content boolean behavior ─────────────────────────


class TestExtractInteractiveContentBoolean:
    def test_true_when_ui_present(self, sample_pane_exit_plan: str):
        assert extract_interactive_content(sample_pane_exit_plan) is not None

    def test_false_when_no_ui(self, sample_pane_no_ui: str):
        assert extract_interactive_content(sample_pane_no_ui) is None

    def test_false_for_empty_string(self):
        assert extract_interactive_content("") is None


# ── strip_pane_chrome ───────────────────────────────────────────────────


class TestStripPaneChrome:
    def test_strips_from_separator(self):
        lines = [
            "some output",
            "more output",
            "─" * 30,
            "❯",
            "─" * 30,
            "  [Opus 4.6] Context: 34%",
        ]
        assert strip_pane_chrome(lines) == ["some output", "more output"]

    def test_no_separator_returns_all(self):
        lines = ["line 1", "line 2", "line 3"]
        assert strip_pane_chrome(lines) == lines

    def test_short_separator_not_triggered(self):
        lines = ["output", "─" * 10, "more output"]
        assert strip_pane_chrome(lines) == lines

    def test_adaptive_scan_finds_distant_separator(self):
        # Separator at line 0 with 15 content lines — adaptive scan finds it
        lines = ["─" * 30] + [f"line {i}" for i in range(14)]
        assert strip_pane_chrome(lines) == []

    def test_content_above_separator_preserved(self):
        content = [f"line {i}" for i in range(20)]
        chrome = ["─" * 30, "❯", "─" * 30, "  [Opus 4.6] Context: 34%"]
        lines = content + chrome
        assert strip_pane_chrome(lines) == content


# ── extract_bash_output ─────────────────────────────────────────────────


class TestExtractBashOutput:
    def test_extracts_command_output(self):
        pane = "some context\n! echo hello\n⎿ hello\n"
        result = extract_bash_output(pane, "echo hello")
        assert result is not None
        assert "! echo hello" in result
        assert "hello" in result

    def test_command_not_found_returns_none(self):
        pane = "some context\njust normal output\n"
        assert extract_bash_output(pane, "echo hello") is None

    def test_chrome_stripped(self):
        pane = (
            "some context\n"
            "! ls\n"
            "⎿ file.txt\n"
            + "─" * 30
            + "\n"
            + "❯\n"
            + "─" * 30
            + "\n"
            + "  [Opus 4.6] Context: 34%\n"
        )
        result = extract_bash_output(pane, "ls")
        assert result is not None
        assert "file.txt" in result
        assert "Opus" not in result

    def test_prefix_match_long_command(self):
        pane = "! long_comma…\n⎿ output\n"
        result = extract_bash_output(pane, "long_command_that_gets_truncated")
        assert result is not None
        assert "output" in result

    def test_trailing_blank_lines_stripped(self):
        pane = "! echo hi\n⎿ hi\n\n\n"
        result = extract_bash_output(pane, "echo hi")
        assert result is not None
        assert not result.endswith("\n")


# ── format_status_display ───────────────────────────────────────────────


class TestFormatStatusDisplay:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Reading src/foo.py", "\U0001f4d6 reading\u2026"),
            ("Thinking about the problem", "\U0001f9e0 thinking\u2026"),
            ("Reasoning through options", "\U0001f9e0 thinking\u2026"),
            ("Editing main.py line 42", "\u270f\ufe0f editing\u2026"),
            ("Writing to file", "\U0001f4dd writing\u2026"),
            ("Running bash command", "\u26a1 running\u2026"),
            ("Searching for pattern", "\U0001f50d searching\u2026"),
            ("grep -r foo .", "\U0001f50d searching\u2026"),
            ("glob **/*.py", "\U0001f4c2 searching\u2026"),
            ("Building the project", "\U0001f3d7\ufe0f building\u2026"),
            ("compiling module", "\U0001f3d7\ufe0f building\u2026"),
            ("Installing dependencies", "\U0001f4e6 installing\u2026"),
            ("Fetching remote refs", "\U0001f310 fetching\u2026"),
            ("git push origin main", "\u2b06\ufe0f pushing\u2026"),
            ("git pull --rebase", "\u2b07\ufe0f pulling\u2026"),
            ("git clone https://repo", "\U0001f4cb cloning\u2026"),
            ("git commit -m msg", "\U0001f4be committing\u2026"),
            ("Deploying to prod", "\U0001f680 deploying\u2026"),
            ("Debugging crash", "\U0001f41b debugging\u2026"),
            ("Formatting code", "\U0001f9f9 formatting\u2026"),
            ("Linting files", "\U0001f9f9 linting\u2026"),
            ("Downloading artifact", "\u2b07\ufe0f downloading\u2026"),
            ("Uploading results", "\u2b06\ufe0f uploading\u2026"),
            ("Testing connection", "\U0001f9ea testing\u2026"),
            ("Deleting old files", "\U0001f5d1\ufe0f deleting\u2026"),
            ("Creating new module", "\u2728 creating\u2026"),
            ("Checking types", "\u2705 checking\u2026"),
            ("Updating dependencies", "\U0001f504 updating\u2026"),
            ("Analyzing output", "\U0001f52c analyzing\u2026"),
            ("Parsing JSON", "\U0001f50d parsing\u2026"),
            ("Verifying results", "\u2705 verifying\u2026"),
            ("esc to interrupt \u00b7 working", "\u2699\ufe0f working\u2026"),
            ("Something completely novel", "\u2699\ufe0f working\u2026"),
            ("", "\u2699\ufe0f working\u2026"),
        ],
    )
    def test_known_patterns(self, raw: str, expected: str) -> None:
        assert format_status_display(raw) == expected

    def test_case_insensitive(self) -> None:
        assert format_status_display("READING file") == "\U0001f4d6 reading\u2026"

    def test_first_word_priority(self) -> None:
        assert (
            format_status_display("Writing tests for module")
            == "\U0001f4dd writing\u2026"
        )

    def test_fallback_to_full_string(self) -> None:
        assert (
            format_status_display("foo bar testing baz") == "\U0001f9ea testing\u2026"
        )


# ── find_chrome_boundary ──────────────────────────────────────────────


class TestFindChromeBoundary:
    def test_empty_lines(self):
        assert find_chrome_boundary([]) is None

    def test_no_separator(self):
        assert find_chrome_boundary(["line 1", "line 2"]) is None

    def test_single_separator(self):
        lines = ["output", "more output", "─" * 30, "❯"]
        assert find_chrome_boundary(lines) == 2

    def test_two_separators(self):
        lines = [
            "output",
            "─" * 30,
            "❯ ",
            "─" * 30,
            "  [Opus 4.6] Context: 34%",
        ]
        assert find_chrome_boundary(lines) == 1

    def test_separator_far_from_bottom(self):
        lines = ["output"] * 50 + ["─" * 30, "❯", "─" * 30, "  status"]
        assert find_chrome_boundary(lines) == 50

    def test_content_separator_not_chrome(self):
        lines = [
            "─" * 30,
            "x" * 100,
            "─" * 30,
            "❯",
        ]
        # First separator has long content below it, so only second is chrome
        assert find_chrome_boundary(lines) == 2


# ── Adaptive terminal size tests ─────────────────────────────────────


class TestVariableTerminalSizes:
    def _build_pane(self, content_lines: int) -> str:
        content = [f"line {i}" for i in range(content_lines)]
        status = "✻ Working on task"
        sep = "─" * 30
        chrome = [sep, "❯ ", sep, "  ⎇ main  ✱ Opus 4.6"]
        return "\n".join(content + [status] + chrome)

    @pytest.mark.parametrize("rows", [24, 50, 100], ids=["24row", "50row", "100row"])
    def test_status_detected_any_size(self, rows: int):
        pane = self._build_pane(content_lines=rows - 5)
        assert parse_status_line(pane) == "Working on task"

    @pytest.mark.parametrize("rows", [24, 50, 100], ids=["24row", "50row", "100row"])
    def test_chrome_stripped_any_size(self, rows: int):
        content = [f"line {i}" for i in range(rows - 5)]
        status = "✻ Working on task"
        sep = "─" * 30
        chrome = [sep, "❯ ", sep, "  ⎇ main  ✱ Opus 4.6"]
        lines = content + [status] + chrome
        result = strip_pane_chrome(lines)
        assert result == content + [status]

    def test_pane_rows_optimization(self):
        pane = self._build_pane(content_lines=80)
        assert parse_status_line(pane, pane_rows=100) == "Working on task"

    def test_extra_padding_below_separator(self):
        lines = [
            "output",
            "─" * 30,
            "❯ ",
            "─" * 30,
            "  status bar",
            "",
            "",
        ]
        assert strip_pane_chrome(lines) == ["output"]


# ── Bottom-up fallback detection ─────────────────────────────────────


class TestBottomUpFallback:
    def test_edit_file_title_detected(self):
        """New-style 'Edit File' prompt without matching top pattern."""
        pane = (
            "  Edit File\n"
            "\n"
            "  src/handlers/sync_command.py\n"
            "\n"
            "  + new_line = True\n"
            "  - old_line = False\n"
            "\n"
            "  ❯ Yes    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert "Edit File" in result.content
        assert "Esc to cancel" in result.content

    def test_unknown_title_with_action_hint(self):
        """Completely unknown title still detected by bottom-up."""
        pane = (
            "  Something Never Seen Before\n"
            "\n"
            "  Description text here\n"
            "\n"
            "  Yes    No\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert "Something Never Seen Before" in result.content

    def test_enter_to_confirm_bottom(self):
        """Bottom-up works with 'Enter to confirm' action hint."""
        pane = "  Pick something:\n\n  Option A\n  Option B\n\n  Enter to confirm\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert "Pick something" in result.content

    def test_section_break_stops_upward_scan(self):
        """Two blank lines above the UI block stop the upward scan.

        Uses plain options (no ❯) so bottom-up fallback is exercised
        instead of the SelectionUI catch-all which has context_above=10.
        """
        pane = (
            "Some earlier output\n"
            "More earlier output\n"
            "\n"
            "\n"
            "  New Prompt Title\n"
            "\n"
            "  Accept    Reject\n"
            "\n"
            "  Esc to cancel\n"
        )
        result = extract_interactive_content(pane)
        assert result is not None
        assert "New Prompt Title" in result.content
        assert "earlier output" not in result.content

    def test_no_action_hint_returns_none(self):
        """Regular output without action hints is not detected."""
        pane = "Some regular output\nMore output\nNo hints here\n"
        result = extract_interactive_content(pane)
        assert result is None

    def test_action_hint_too_far_from_bottom(self):
        """Action hint buried in output (not near bottom) is ignored."""
        lines = [
            "  Esc to cancel",  # action hint
            *[f"  output line {i}" for i in range(10)],  # lots of output after
            "",
        ]
        pane = "\n".join(lines)
        result = extract_interactive_content(pane)
        assert result is None

    def test_infers_selection_ui_name(self):
        """Bottom-up infers SelectionUI when ❯ is present."""
        pane = "  Unknown Title\n  ❯ Option A\n    Option B\n  Esc to cancel\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "SelectionUI"

    def test_infers_ask_user_name(self):
        """Bottom-up infers AskUserQuestion when checkbox chars present."""
        pane = "  Unknown Title\n  ☐ Option A\n  ✔ Option B\n  Enter to select\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "AskUserQuestion"

    def test_infers_generic_name(self):
        """Bottom-up returns InteractiveUI when no cursor/checkbox present."""
        pane = "  Unknown Title\n\n  Some text\n\n  Esc to cancel\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "InteractiveUI"

    def test_pattern_match_takes_precedence(self):
        """Known patterns still match first — bottom-up is only a fallback."""
        pane = "  Do you want to proceed?\n\n  ❯ Yes    No\n\n  Esc to cancel\n"
        result = extract_interactive_content(pane)
        assert result is not None
        assert result.name == "PermissionPrompt"  # not bottom-up


# ── extract_interactive_content with list[str] ───────────────────────


class TestExtractInteractiveContentWithLines:
    def test_accepts_list_of_lines(self):
        lines = [
            "  ☐ Option A",
            "  ☐ Option B",
            "  Enter to select",
        ]
        result = extract_interactive_content(lines)
        assert result is not None
        assert result.name == "AskUserQuestion"
        assert "Enter to select" in result.content

    def test_empty_list_returns_none(self):
        assert extract_interactive_content([]) is None

    def test_list_matches_string_result(self):
        pane = "  ☐ Option A\n  ☐ Option B\n  Enter to select\n"
        lines = ["  ☐ Option A", "  ☐ Option B", "  Enter to select"]
        result_str = extract_interactive_content(pane)
        result_list = extract_interactive_content(lines)
        assert result_str is not None
        assert result_list is not None
        assert result_str.name == result_list.name
        # String variant calls .strip() before splitting, so leading whitespace
        # on the first line may differ — compare the detected pattern name only


# ── pyte screen-based parsing ────────────────────────────────────────


class TestParseFromScreen:
    def _make_screen(
        self, raw: str, columns: int = 80, rows: int = 24
    ) -> "ScreenBuffer":
        from ccgram.screen_buffer import ScreenBuffer

        buf = ScreenBuffer(columns=columns, rows=rows)
        buf.feed(raw)
        return buf

    def test_detects_ask_user_question(self):
        raw = "  \x1b[1m☐ Option A\x1b[0m\r\n  ☐ Option B\r\n  Enter to select"
        from ccgram.terminal_parser import parse_from_screen

        screen = self._make_screen(raw)
        result = parse_from_screen(screen)
        assert result is not None
        assert result.name == "AskUserQuestion"
        assert "Enter to select" in result.content

    def test_detects_exit_plan_mode(self):
        raw = (
            "  Would you like to proceed?\r\n"
            "  \x1b[36m─────────────────────────────────\x1b[0m\r\n"
            "  Yes     No\r\n"
            "  ─────────────────────────────────\r\n"
            "  ctrl-g to edit in vim"
        )
        from ccgram.terminal_parser import parse_from_screen

        screen = self._make_screen(raw)
        result = parse_from_screen(screen)
        assert result is not None
        assert result.name == "ExitPlanMode"

    def test_detects_permission_prompt(self):
        raw = (
            "  Do you want to proceed?\r\n"
            "  \x1b[33mSome permission details\x1b[0m\r\n"
            "  Esc to cancel"
        )
        from ccgram.terminal_parser import parse_from_screen

        screen = self._make_screen(raw)
        result = parse_from_screen(screen)
        assert result is not None
        assert result.name == "PermissionPrompt"

    def test_no_ui_returns_none(self):
        raw = "$ echo hello\r\nhello\r\n$ "
        from ccgram.terminal_parser import parse_from_screen

        screen = self._make_screen(raw)
        assert parse_from_screen(screen) is None

    def test_cursor_at_row_zero(self):
        from ccgram.screen_buffer import ScreenBuffer
        from ccgram.terminal_parser import parse_from_screen

        buf = ScreenBuffer(columns=80, rows=5)
        # Cursor stays at row 0, col 0 — empty screen
        assert parse_from_screen(buf) is None

    def test_ansi_stripped_matches_plain_text(self):
        plain = "  ☐ Option A\n  ☐ Option B\n  Enter to select\n"
        ansi = (
            "  \x1b[1;32m☐ Option A\x1b[0m\r\n"
            "  \x1b[34m☐ Option B\x1b[0m\r\n"
            "  Enter to select"
        )
        from ccgram.terminal_parser import parse_from_screen

        plain_result = extract_interactive_content(plain)
        screen = self._make_screen(ansi)
        screen_result = parse_from_screen(screen)
        assert plain_result is not None
        assert screen_result is not None
        assert plain_result.name == screen_result.name


class TestParseStatusFromScreen:
    def _make_screen(
        self, raw: str, columns: int = 80, rows: int = 24
    ) -> "ScreenBuffer":
        from ccgram.screen_buffer import ScreenBuffer

        buf = ScreenBuffer(columns=columns, rows=rows)
        buf.feed(raw)
        return buf

    def test_detects_spinner_status(self):
        sep = "─" * 30
        raw = (
            "some output\r\n"
            "\x1b[36m✻ Reading file\x1b[0m\r\n"
            f"{sep}\r\n"
            "❯ \r\n"
            f"{sep}\r\n"
            "  \x1b[90m[Opus 4.6]\x1b[0m Context: 34%"
        )
        from ccgram.terminal_parser import parse_status_from_screen

        screen = self._make_screen(raw)
        result = parse_status_from_screen(screen)
        assert result == "Reading file"

    def test_braille_spinner_via_screen(self):
        sep = "─" * 30
        raw = f"output\r\n⠋ Loading modules\r\n{sep}\r\n❯ "
        from ccgram.terminal_parser import parse_status_from_screen

        screen = self._make_screen(raw)
        result = parse_status_from_screen(screen)
        assert result == "Loading modules"

    def test_no_status_returns_none(self):
        raw = "just normal text\r\nno spinners here"
        from ccgram.terminal_parser import parse_status_from_screen

        screen = self._make_screen(raw)
        assert parse_status_from_screen(screen) is None

    def test_empty_screen_returns_none(self):
        from ccgram.screen_buffer import ScreenBuffer
        from ccgram.terminal_parser import parse_status_from_screen

        screen = ScreenBuffer(columns=80, rows=24)
        assert parse_status_from_screen(screen) is None

    def test_matches_regex_result(self):
        sep = "─" * 30
        plain = f"output\n✻ Working on task\n{sep}\n❯ \n{sep}\n  status"
        ansi = (
            "output\r\n"
            "\x1b[36m✻ Working on task\x1b[0m\r\n"
            f"{sep}\r\n"
            "❯ \r\n"
            f"{sep}\r\n"
            "  \x1b[90mstatus\x1b[0m"
        )
        from ccgram.terminal_parser import parse_status_from_screen

        regex_result = parse_status_line(plain)
        screen = self._make_screen(ansi)
        screen_result = parse_status_from_screen(screen)
        assert regex_result == screen_result == "Working on task"


# ── Remote Control detection ───────────────────────────────────────────


class TestDetectRemoteControl:
    def _build_lines(self, *, rc: bool = True) -> list[str]:
        sep = "─" * 30
        status_bar = "  ● Remote Control active" if rc else "  [Opus 4.6] Context: 34%"
        return [
            "output line 1",
            "output line 2",
            sep,
            "❯ ",
            sep,
            status_bar,
        ]

    def test_rc_present(self):
        from ccgram.terminal_parser import detect_remote_control

        assert detect_remote_control(self._build_lines(rc=True)) is True

    def test_rc_absent(self):
        from ccgram.terminal_parser import detect_remote_control

        assert detect_remote_control(self._build_lines(rc=False)) is False

    def test_no_chrome(self):
        from ccgram.terminal_parser import detect_remote_control

        assert detect_remote_control(["output", "no chrome here"]) is False

    def test_empty(self):
        from ccgram.terminal_parser import detect_remote_control

        assert detect_remote_control([]) is False
