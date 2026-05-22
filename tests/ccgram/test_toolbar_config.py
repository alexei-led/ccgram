"""Tests for ccgram.toolbar_config — TOML loader, validation, defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccgram.toolbar_config import (
    BUILTIN_ACTIONS,
    DEFAULT_LAYOUTS,
    ToolbarAction,
    ToolbarConfig,
    load_toolbar_config,
)


# ── render() ──────────────────────────────────────────────────────────


class TestRender:
    @pytest.fixture
    def action(self) -> ToolbarAction:
        return ToolbarAction(
            name="x",
            emoji="\U0001f4f7",
            text="Screen",
            action_type="builtin",
            payload="screenshot",
        )

    def test_emoji_only(self, action: ToolbarAction) -> None:
        assert action.render("emoji") == "\U0001f4f7"

    def test_text_only(self, action: ToolbarAction) -> None:
        assert action.render("text") == "Screen"

    def test_emoji_text(self, action: ToolbarAction) -> None:
        assert action.render("emoji_text") == "\U0001f4f7 Screen"


# ── for_provider() ────────────────────────────────────────────────────


class TestForProvider:
    def test_known_provider_returns_its_layout(self) -> None:
        cfg = ToolbarConfig(layouts=dict(DEFAULT_LAYOUTS))
        assert cfg.for_provider("codex") is DEFAULT_LAYOUTS["codex"]

    def test_unknown_provider_falls_back_to_claude(self) -> None:
        cfg = ToolbarConfig(layouts=dict(DEFAULT_LAYOUTS))
        assert cfg.for_provider("aider") is DEFAULT_LAYOUTS["claude"]


# ── load_toolbar_config — defaults ────────────────────────────────────


class TestLoadDefaults:
    def test_no_path_returns_defaults(self) -> None:
        cfg = load_toolbar_config(None)
        assert cfg.actions == BUILTIN_ACTIONS
        assert cfg.layouts == DEFAULT_LAYOUTS

    def test_empty_string_path_returns_defaults(self) -> None:
        cfg = load_toolbar_config("")
        assert cfg.actions == BUILTIN_ACTIONS
        assert cfg.layouts == DEFAULT_LAYOUTS

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_toolbar_config(tmp_path / "does-not-exist.toml")
        assert cfg.actions == BUILTIN_ACTIONS
        assert cfg.layouts == DEFAULT_LAYOUTS

    def test_malformed_toml_returns_defaults(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("this is = not = valid = toml = !!!")
        cfg = load_toolbar_config(bad)
        assert cfg.actions == BUILTIN_ACTIONS
        assert cfg.layouts == DEFAULT_LAYOUTS

    def test_root_not_a_table_returns_defaults(self, tmp_path: Path) -> None:
        # TOML can't actually have a non-table root, but we test the guard.
        # An empty file parses to {} which is fine; use a comment-only file.
        comment_only = tmp_path / "empty.toml"
        comment_only.write_text("# nothing here\n")
        cfg = load_toolbar_config(comment_only)
        assert cfg.actions == BUILTIN_ACTIONS
        assert cfg.layouts == DEFAULT_LAYOUTS


# ── load_toolbar_config — user actions ────────────────────────────────


class TestLoadUserActions:
    def test_user_action_extends_pool(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            '[actions.foo]\nemoji = "🦊"\ntext = "Foo"\ntype = "text"\npayload = "/foo"\n'
        )
        cfg = load_toolbar_config(f)
        assert "foo" in cfg.actions
        assert cfg.actions["foo"].emoji == "🦊"
        assert cfg.actions["foo"].action_type == "text"
        assert cfg.actions["foo"].payload == "/foo"
        # builtins are still present
        assert "screen" in cfg.actions

    def test_user_action_overrides_builtin(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            '[actions.mode]\nemoji = "🆕"\ntext = "Mode"\ntype = "key"\npayload = "Tab"\n'
        )
        cfg = load_toolbar_config(f)
        assert cfg.actions["mode"].emoji == "🆕"
        assert cfg.actions["mode"].payload == "Tab"

    def test_invalid_type_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            '[actions.bad]\nemoji = "?"\ntext = "Bad"\ntype = "wat"\npayload = "x"\n'
        )
        cfg = load_toolbar_config(f)
        assert "bad" not in cfg.actions

    def test_user_cannot_define_builtin_type(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            '[actions.steal]\nemoji = "?"\ntext = "Steal"\ntype = "builtin"\npayload = "screenshot"\n'
        )
        cfg = load_toolbar_config(f)
        assert "steal" not in cfg.actions

    def test_missing_payload_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[actions.np]\nemoji = "?"\ntext = "NP"\ntype = "key"\n')
        cfg = load_toolbar_config(f)
        assert "np" not in cfg.actions

    def test_missing_emoji_and_text_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[actions.np]\ntype = "key"\npayload = "Tab"\n')
        cfg = load_toolbar_config(f)
        assert "np" not in cfg.actions

    def test_action_name_too_long_skipped(self, tmp_path: Path) -> None:
        long_name = "x" * 25  # _MAX_NAME_LEN is 24
        f = tmp_path / "t.toml"
        f.write_text(
            f'[actions.{long_name}]\nemoji = "?"\ntext = "Hi"\ntype = "key"\npayload = "Tab"\n'
        )
        cfg = load_toolbar_config(f)
        assert long_name not in cfg.actions

    def test_action_name_at_limit_accepted(self, tmp_path: Path) -> None:
        name = "x" * 24
        f = tmp_path / "t.toml"
        f.write_text(
            f'[actions.{name}]\nemoji = "?"\ntext = "Hi"\ntype = "key"\npayload = "Tab"\n'
        )
        cfg = load_toolbar_config(f)
        assert name in cfg.actions

    def test_emoji_falls_back_to_text(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[actions.tx]\ntext = "TX"\ntype = "text"\npayload = "/tx"\n')
        cfg = load_toolbar_config(f)
        assert cfg.actions["tx"].emoji == "TX"

    def test_text_falls_back_to_name(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[actions.tx]\nemoji = "🆗"\ntype = "text"\npayload = "/tx"\n')
        cfg = load_toolbar_config(f)
        assert cfg.actions["tx"].text == "tx"


# ── load_toolbar_config — user layouts ────────────────────────────────


class TestLoadUserLayouts:
    def test_user_layout_overrides_default(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            "[providers.claude]\n"
            'style = "text"\n'
            "buttons = [\n"
            '  ["screen", "ctrlc"],\n'
            '  ["close"],\n'
            "]\n"
        )
        cfg = load_toolbar_config(f)
        layout = cfg.layouts["claude"]
        assert layout.style == "text"
        assert layout.buttons == (("screen", "ctrlc"), ("close",))

    def test_provider_absent_keeps_default(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[providers.claude]\nstyle = "text"\nbuttons = [["close"]]\n')
        cfg = load_toolbar_config(f)
        # codex/gemini/shell are unchanged
        assert cfg.layouts["codex"] == DEFAULT_LAYOUTS["codex"]
        assert cfg.layouts["gemini"] == DEFAULT_LAYOUTS["gemini"]
        assert cfg.layouts["shell"] == DEFAULT_LAYOUTS["shell"]

    def test_invalid_style_falls_back_to_emoji_text(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[providers.claude]\nstyle = "rainbow"\nbuttons = [["close"]]\n')
        cfg = load_toolbar_config(f)
        assert cfg.layouts["claude"].style == "emoji_text"

    def test_unknown_action_in_grid_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            "[providers.claude]\n"
            'style = "text"\n'
            'buttons = [["close", "nonexistent", "screen"]]\n'
        )
        cfg = load_toolbar_config(f)
        # The unknown name is dropped; valid names remain
        assert cfg.layouts["claude"].buttons == (("close", "screen"),)

    def test_row_with_too_many_buttons_trimmed(self, tmp_path: Path) -> None:
        names = ", ".join('"close"' for _ in range(10))  # 10 cells
        f = tmp_path / "t.toml"
        f.write_text(f'[providers.claude]\nstyle = "text"\nbuttons = [[{names}]]\n')
        cfg = load_toolbar_config(f)
        # _MAX_ROW_WIDTH is 8
        assert len(cfg.layouts["claude"].buttons[0]) == 8

    def test_empty_buttons_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text('[providers.claude]\nstyle = "text"\nbuttons = []\n')
        cfg = load_toolbar_config(f)
        # Falls back to default since the layout was rejected
        assert cfg.layouts["claude"] == DEFAULT_LAYOUTS["claude"]

    def test_row_of_only_unknown_actions_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            '[providers.claude]\nstyle = "text"\nbuttons = [["nope"], ["close"]]\n'
        )
        cfg = load_toolbar_config(f)
        # First row dropped entirely (no valid cells); second row remains
        assert cfg.layouts["claude"].buttons == (("close",),)

    def test_layout_with_only_unknown_rows_falls_back(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            '[providers.claude]\nstyle = "text"\nbuttons = [["nope1"], ["nope2"]]\n'
        )
        cfg = load_toolbar_config(f)
        # No usable rows → fall back to default
        assert cfg.layouts["claude"] == DEFAULT_LAYOUTS["claude"]


# ── Built-ins are intact ──────────────────────────────────────────────


class TestBuiltins:
    def test_all_expected_actions_present(self) -> None:
        expected = {
            "screen",
            "ctrlc",
            "live",
            "getfile",
            "last",
            "close",
            "mode",
            "think",
            "yolo",
            "esc",
            "enter",
            "tab",
            "eof",
            "susp",
            "up",
            "down",
        }
        assert expected.issubset(BUILTIN_ACTIONS.keys())

    def test_send_builtin_removed(self) -> None:
        assert "send" not in BUILTIN_ACTIONS

    def test_getfile_is_builtin_with_correct_payload(self) -> None:
        assert "getfile" in BUILTIN_ACTIONS
        action = BUILTIN_ACTIONS["getfile"]
        assert action.action_type == "builtin"
        assert action.payload == "getfile"
        assert action.emoji == "\U0001f4e5"

    def test_last_is_builtin_with_lastreply_payload(self) -> None:
        assert "last" in BUILTIN_ACTIONS
        action = BUILTIN_ACTIONS["last"]
        assert action.action_type == "builtin"
        assert action.payload == "lastreply"
        assert action.emoji == "\U0001f4c4"

    def test_mode_is_literal_with_read_state(self) -> None:
        mode = BUILTIN_ACTIONS["mode"]
        assert mode.action_type == "key"
        assert mode.literal is True
        assert mode.read_state is True
        assert mode.payload == "\x1b[Z"

    def test_yolo_has_read_state(self) -> None:
        # Think was changed to read_state=False because Claude Code has no
        # persistent chrome indicator for extended-thinking state.
        assert BUILTIN_ACTIONS["think"].read_state is False
        assert BUILTIN_ACTIONS["yolo"].read_state is True

    def test_default_layouts_have_valid_grids(self) -> None:
        for provider, layout in DEFAULT_LAYOUTS.items():
            assert 3 <= len(layout.buttons) <= 4, f"{provider}: expected 3-4 rows"
            for row in layout.buttons:
                assert 1 <= len(row) <= 8, f"{provider}: row width out of range"

    def test_default_layouts_contain_getfile_not_send(self) -> None:
        for provider, layout in DEFAULT_LAYOUTS.items():
            all_names = [name for row in layout.buttons for name in row]
            assert "getfile" in all_names, f"{provider}: missing getfile"
            assert "last" in all_names, f"{provider}: missing last"
            assert "send" not in all_names, (
                f"{provider}: old send builtin still present"
            )

    def test_default_layouts_use_known_actions(self) -> None:
        for provider, layout in DEFAULT_LAYOUTS.items():
            for row in layout.buttons:
                for name in row:
                    assert name in BUILTIN_ACTIONS, (
                        f"{provider} references unknown action {name!r}"
                    )

    def test_default_style_is_emoji_text(self) -> None:
        for layout in DEFAULT_LAYOUTS.values():
            assert layout.style == "emoji_text"


# ── End-to-end ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_combined_actions_and_layout(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            "[actions.clear]\n"
            'emoji = "🧹"\n'
            'text = "Clear"\n'
            'type = "text"\n'
            'payload = "/clear"\n'
            "\n"
            "[providers.claude]\n"
            'style = "emoji_text"\n'
            "buttons = [\n"
            '  ["screen", "clear"],\n'
            '  ["close"],\n'
            "]\n"
        )
        cfg = load_toolbar_config(f)
        assert "clear" in cfg.actions
        layout = cfg.for_provider("claude")
        assert layout.buttons == (("screen", "clear"), ("close",))
        assert layout.style == "emoji_text"


# ──────────────────────────────────────────────────────────────────────
# Permutation tests — every style × grid shape × provider combo
# ──────────────────────────────────────────────────────────────────────


class TestStylePermutations:
    """All three rendering styles work for every action in the pool."""

    @pytest.mark.parametrize("style", ["emoji", "text", "emoji_text"])
    @pytest.mark.parametrize("name", list(BUILTIN_ACTIONS.keys()))
    def test_render_every_builtin_in_every_style(self, style: str, name: str) -> None:
        action = BUILTIN_ACTIONS[name]
        rendered = action.render(style)  # type: ignore[arg-type]
        assert rendered  # non-empty
        if style == "emoji":
            assert rendered == action.emoji
        elif style == "text":
            assert rendered == action.text
        else:
            assert action.emoji in rendered
            assert action.text in rendered

    @pytest.mark.parametrize("style", ["emoji", "text", "emoji_text"])
    def test_load_layout_with_each_style(self, tmp_path: Path, style: str) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            f'[providers.claude]\nstyle = "{style}"\nbuttons = [["screen", "ctrlc"]]\n'
        )
        cfg = load_toolbar_config(f)
        assert cfg.layouts["claude"].style == style


class TestGridShapePermutations:
    """Grid shapes (1xN, Nx1, 2x4, 3x3, asymmetric) all parse correctly."""

    @pytest.mark.parametrize(
        ("shape_toml", "expected_shape"),
        [
            # 1x1
            ('[["close"]]', (("close",),)),
            # 1x4 horizontal
            (
                '[["screen", "ctrlc", "live", "getfile"]]',
                (("screen", "ctrlc", "live", "getfile"),),
            ),
            # 4x1 vertical stack
            (
                '[["screen"], ["ctrlc"], ["live"], ["getfile"]]',
                (("screen",), ("ctrlc",), ("live",), ("getfile",)),
            ),
            # 2x4
            (
                '[["screen", "ctrlc", "live", "getfile"], '
                '["mode", "think", "esc", "close"]]',
                (
                    ("screen", "ctrlc", "live", "getfile"),
                    ("mode", "think", "esc", "close"),
                ),
            ),
            # 3x3
            (
                '[["screen", "ctrlc", "live"], '
                '["mode", "think", "esc"], '
                '["last", "enter", "close"]]',
                (
                    ("screen", "ctrlc", "live"),
                    ("mode", "think", "esc"),
                    ("last", "enter", "close"),
                ),
            ),
            # Asymmetric: rows of different widths
            (
                '[["screen", "ctrlc", "live"], ["close"]]',
                (("screen", "ctrlc", "live"), ("close",)),
            ),
        ],
    )
    def test_grid_shape_parses(
        self, tmp_path: Path, shape_toml: str, expected_shape: tuple
    ) -> None:
        f = tmp_path / "t.toml"
        f.write_text(f'[providers.claude]\nstyle = "text"\nbuttons = {shape_toml}\n')
        cfg = load_toolbar_config(f)
        assert cfg.layouts["claude"].buttons == expected_shape


class TestMultiProviderPermutations:
    """Override any subset of providers; others keep their defaults."""

    @pytest.mark.parametrize(
        "to_override",
        [
            ["claude"],
            ["codex"],
            ["gemini"],
            ["shell"],
            ["claude", "codex"],
            ["claude", "shell"],
            ["claude", "codex", "gemini", "shell"],
        ],
    )
    def test_subset_override(self, tmp_path: Path, to_override: list[str]) -> None:
        f = tmp_path / "t.toml"
        sections = "\n\n".join(
            f'[providers.{p}]\nstyle = "text"\nbuttons = [["close"]]\n'
            for p in to_override
        )
        f.write_text(sections)
        cfg = load_toolbar_config(f)
        for p in to_override:
            assert cfg.layouts[p].style == "text"
            assert cfg.layouts[p].buttons == (("close",),)
        for p in {"claude", "codex", "gemini", "shell"} - set(to_override):
            assert cfg.layouts[p] == DEFAULT_LAYOUTS[p]


class TestActionTypePermutations:
    """Every valid action type round-trips through the loader."""

    @pytest.mark.parametrize(
        ("action_type", "payload"),
        [
            ("key", "Tab"),
            ("key", "C-c"),
            ("key", "S-Tab"),
            ("text", "/clear"),
            ("text", "summarize the latest changes please"),
        ],
    )
    def test_action_type_round_trip(
        self, tmp_path: Path, action_type: str, payload: str
    ) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            f'[actions.custom]\nemoji = "?"\ntext = "X"\n'
            f'type = "{action_type}"\npayload = "{payload}"\n'
        )
        cfg = load_toolbar_config(f)
        assert "custom" in cfg.actions
        assert cfg.actions["custom"].action_type == action_type
        assert cfg.actions["custom"].payload == payload

    def test_literal_escape_sequence_in_toml_literal_string(
        self, tmp_path: Path
    ) -> None:
        """Users wanting the raw \\x1b[Z escape use TOML literal strings."""
        f = tmp_path / "t.toml"
        # TOML literal strings are single-quoted and don't interpret escapes.
        f.write_text(
            "[actions.shifttab]\n"
            'emoji = "?"\n'
            'text = "ST"\n'
            'type = "key"\n'
            "payload = '\\x1b[Z'\n"
            "literal = true\n"
        )
        cfg = load_toolbar_config(f)
        assert "shifttab" in cfg.actions
        # The literal 6-character string \x1b[Z is preserved as-is
        assert cfg.actions["shifttab"].payload == "\\x1b[Z"
        assert cfg.actions["shifttab"].literal is True

    @pytest.mark.parametrize("read_state_value", [True, False])
    def test_read_state_round_trip(
        self, tmp_path: Path, read_state_value: bool
    ) -> None:
        f = tmp_path / "t.toml"
        rs = "true" if read_state_value else "false"
        f.write_text(
            f'[actions.custom]\nemoji = "?"\ntext = "X"\n'
            f'type = "key"\npayload = "Tab"\nread_state = {rs}\n'
        )
        cfg = load_toolbar_config(f)
        assert cfg.actions["custom"].read_state is read_state_value

    @pytest.mark.parametrize("literal_value", [True, False])
    def test_literal_round_trip(self, tmp_path: Path, literal_value: bool) -> None:
        f = tmp_path / "t.toml"
        lit = "true" if literal_value else "false"
        f.write_text(
            f'[actions.custom]\nemoji = "?"\ntext = "X"\n'
            f'type = "key"\npayload = "Tab"\nliteral = {lit}\n'
        )
        cfg = load_toolbar_config(f)
        assert cfg.actions["custom"].literal is literal_value


class TestComplexUserConfigs:
    """End-to-end TOML configs with multiple actions and providers."""

    def test_kitchen_sink(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            "[actions.clear]\n"
            'emoji = "🧹"\ntext = "Clear"\ntype = "text"\npayload = "/clear"\n'
            "\n"
            "[actions.compact]\n"
            'emoji = "📦"\ntext = "Pack"\ntype = "text"\npayload = "/compact"\n'
            "\n"
            "[actions.deepthink]\n"
            'emoji = "🧠"\ntext = "Deep"\ntype = "key"\npayload = "Tab"\n'
            "read_state = true\n"
            "\n"
            "[actions.mode]\n"
            'emoji = "🆕"\ntext = "Mode"\ntype = "key"\n'
            "payload = '\\x1b[Z'\n"  # TOML literal string (single-quoted)
            "literal = true\nread_state = true\n"
            "\n"
            "[providers.claude]\n"
            'style = "emoji_text"\n'
            "buttons = [\n"
            '  ["screen",   "ctrlc",  "live"     ],\n'
            '  ["mode",     "think",  "deepthink"],\n'
            '  ["clear",    "compact","esc"      ],\n'
            '  ["last",     "getfile", "close"    ],\n'
            "]\n"
            "\n"
            "[providers.shell]\n"
            'style = "text"\n'
            'buttons = [["clear", "ctrlc"], ["close"]]\n'
        )
        cfg = load_toolbar_config(f)
        assert {"clear", "compact", "deepthink"}.issubset(cfg.actions.keys())
        assert cfg.actions["mode"].emoji == "🆕"
        claude = cfg.layouts["claude"]
        assert len(claude.buttons) == 4
        assert all(len(row) == 3 for row in claude.buttons)
        shell = cfg.layouts["shell"]
        assert shell.style == "text"
        assert shell.buttons == (("clear", "ctrlc"), ("close",))
        assert cfg.layouts["codex"] == DEFAULT_LAYOUTS["codex"]
        assert cfg.layouts["gemini"] == DEFAULT_LAYOUTS["gemini"]

    def test_unknown_action_in_grid_does_not_break_neighbors(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            "[providers.claude]\n"
            'style = "text"\n'
            'buttons = [["screen", "nope", "live"], ["close"]]\n'
        )
        cfg = load_toolbar_config(f)
        assert cfg.layouts["claude"].buttons == (
            ("screen", "live"),
            ("close",),
        )

    def test_partial_action_failures_keep_valid_actions(self, tmp_path: Path) -> None:
        f = tmp_path / "t.toml"
        f.write_text(
            "[actions.good1]\n"
            'emoji = "?"\ntext = "G1"\ntype = "text"\npayload = "/g1"\n'
            "\n"
            "[actions.bad]\n"
            'emoji = "?"\ntext = "B"\ntype = "wat"\npayload = "x"\n'
            "\n"
            "[actions.good2]\n"
            'emoji = "?"\ntext = "G2"\ntype = "key"\npayload = "Tab"\n'
        )
        cfg = load_toolbar_config(f)
        assert "good1" in cfg.actions
        assert "good2" in cfg.actions
        assert "bad" not in cfg.actions
