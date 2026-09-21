"""Tests for ccgram.topic_emoji_config — TOML loader, defaults, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccgram.topic_emoji_config import (
    DEFAULT_SYSTEM_EMOJI,
    DEFAULT_USER_EMOJI,
    EMOJI_DEAD,
    EMOJI_DONE,
    EMOJI_GREEN_CIRCLE,
    EMOJI_YELLOW_CIRCLE,
    LEGACY_DEAD_EMOJI,
    TopicEmojiConfig,
    load_topic_emoji_config,
)


# ── Defaults ──────────────────────────────────────────────────────────


class TestDefaults:
    def test_system_uses_green_for_active_yellow_for_idle(self) -> None:
        """The pre-customization system mode must be preserved as a default."""
        assert DEFAULT_SYSTEM_EMOJI["active"] == EMOJI_GREEN_CIRCLE
        assert DEFAULT_SYSTEM_EMOJI["idle"] == EMOJI_YELLOW_CIRCLE
        assert DEFAULT_SYSTEM_EMOJI["done"] == EMOJI_DONE
        assert DEFAULT_SYSTEM_EMOJI["dead"] == EMOJI_DEAD

    def test_user_mode_swaps_active_and_idle(self) -> None:
        assert DEFAULT_USER_EMOJI["active"] == EMOJI_YELLOW_CIRCLE
        assert DEFAULT_USER_EMOJI["idle"] == EMOJI_GREEN_CIRCLE
        assert DEFAULT_USER_EMOJI["done"] == EMOJI_DONE
        assert DEFAULT_USER_EMOJI["dead"] == EMOJI_DEAD

    def test_legacy_dead_includes_pre_2026_indicators(self) -> None:
        """Pre-2026-02 black circle and pre-2026-03 cross must remain
        strippable so names left by older installs clean up correctly."""
        assert "\u26ab" in LEGACY_DEAD_EMOJI  # ⚫
        assert "\u274c" in LEGACY_DEAD_EMOJI  # ❌

    def test_default_config_has_all_four_states_per_mode(self) -> None:
        """Every mode table must contain the full state set so callers
        never see a missing-key path."""
        for state in ("active", "idle", "done", "dead"):
            assert state in DEFAULT_SYSTEM_EMOJI
            assert state in DEFAULT_USER_EMOJI


# ── for_mode() ────────────────────────────────────────────────────────


class TestForMode:
    def test_system_returns_system_table(self) -> None:
        cfg = TopicEmojiConfig()
        assert cfg.for_mode("system") is cfg.system_emoji

    def test_user_returns_user_table(self) -> None:
        cfg = TopicEmojiConfig()
        assert cfg.for_mode("user") is cfg.user_emoji

    def test_unknown_mode_falls_back_to_system(self) -> None:
        cfg = TopicEmojiConfig()
        # Unknown mode is the same defensive fallback the production
        # code path uses; must not raise and must return the system table.
        assert cfg.for_mode("bogus") is cfg.system_emoji

    def test_empty_string_falls_back_to_system(self) -> None:
        cfg = TopicEmojiConfig()
        assert cfg.for_mode("") is cfg.system_emoji


# ── load_topic_emoji_config — defaults ─────────────────────────────────


class TestLoadDefaults:
    def test_no_path_returns_defaults(self) -> None:
        cfg = load_topic_emoji_config(None)
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI
        assert cfg.user_emoji == DEFAULT_USER_EMOJI
        assert cfg.legacy_dead == LEGACY_DEAD_EMOJI

    def test_empty_string_path_returns_defaults(self) -> None:
        cfg = load_topic_emoji_config("")
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI
        assert cfg.user_emoji == DEFAULT_USER_EMOJI

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_topic_emoji_config(tmp_path / "missing.toml")
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI
        assert cfg.user_emoji == DEFAULT_USER_EMOJI

    def test_malformed_toml_returns_defaults(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not valid TOML = = =")
        cfg = load_topic_emoji_config(bad)
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI
        assert cfg.user_emoji == DEFAULT_USER_EMOJI

    def test_root_not_a_table_returns_defaults(self, tmp_path: Path) -> None:
        # An array at the root is valid TOML but must not be accepted.
        arr = tmp_path / "arr.toml"
        arr.write_text('["active", "idle"]')
        cfg = load_topic_emoji_config(arr)
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI


# ── load_topic_emoji_config — overrides ───────────────────────────────


class TestLoadOverrides:
    def test_system_table_replaces_single_state(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.system]
            active = "🚀"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.system_emoji["active"] == "🚀"
        # Other states fall through to defaults.
        assert cfg.system_emoji["idle"] == DEFAULT_SYSTEM_EMOJI["idle"]
        assert cfg.system_emoji["done"] == DEFAULT_SYSTEM_EMOJI["done"]
        assert cfg.system_emoji["dead"] == DEFAULT_SYSTEM_EMOJI["dead"]

    def test_user_table_replaces_single_state(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.user]
            idle = "🔥"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.user_emoji["idle"] == "🔥"
        assert cfg.user_emoji["active"] == DEFAULT_USER_EMOJI["active"]

    def test_both_modes_overridden_together(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.system]
            active = "🟢"
            idle = "🟡"
            done = "✅"
            dead = "💥"

            [topic_emoji.user]
            active = "🟡"
            idle = "🟢"
            done = "✅"
            dead = "💥"
            """
        )
        cfg = load_topic_emoji_config(f)
        # Loader must not merge the system overrides into the user table.
        assert cfg.system_emoji["active"] == "🟢"
        assert cfg.user_emoji["active"] == "🟡"

    def test_unknown_state_in_table_is_ignored(self, tmp_path: Path) -> None:
        """A typo (``acitve``) must not appear in the resolved table —
        only the four known states are valid keys."""
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.system]
            acitve = "🚀"
            paused = "💤"
            active = "🟢"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert "acitve" not in cfg.system_emoji
        assert "paused" not in cfg.system_emoji
        assert cfg.system_emoji["active"] == "🟢"

    def test_non_string_emoji_value_is_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.system]
            active = 42
            idle = ["list"]
            done = ""
            dead = "💥"
            """
        )
        cfg = load_topic_emoji_config(f)
        # active was a number, idle was a list, done was empty — all skipped.
        assert cfg.system_emoji["active"] == DEFAULT_SYSTEM_EMOJI["active"]
        assert cfg.system_emoji["idle"] == DEFAULT_SYSTEM_EMOJI["idle"]
        assert cfg.system_emoji["done"] == DEFAULT_SYSTEM_EMOJI["done"]
        # dead is valid and applied.
        assert cfg.system_emoji["dead"] == "💥"

    def test_non_table_section_is_ignored(self, tmp_path: Path) -> None:
        """A section that isn't a table (e.g. a string) must not crash."""
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            topic_emoji = "not a table"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI
        assert cfg.user_emoji == DEFAULT_USER_EMOJI

    def test_non_table_mode_section_is_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji]
            system = "not a table"
            user = 123
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.system_emoji == DEFAULT_SYSTEM_EMOJI
        assert cfg.user_emoji == DEFAULT_USER_EMOJI

    def test_string_path_with_tilde_is_expanded(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # ``Path.expanduser`` only honors a leading ``~``; mock the home
        # directory so the test does not depend on the host's $HOME.
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "expandme.toml"
        f.write_text(
            """
            [topic_emoji.system]
            active = "🚀"
            """
        )
        cfg = load_topic_emoji_config("~/expandme.toml")
        assert cfg.system_emoji["active"] == "🚀"


# ── Legacy dead emoji ──────────────────────────────────────────────────


class TestLoadLegacyDead:
    def test_extends_built_in_legacy_list(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.legacy_dead]
            emojis = ["🪦"]
            """
        )
        cfg = load_topic_emoji_config(f)
        # Built-in entries must still be present.
        assert "\u26ab" in cfg.legacy_dead
        assert "\u274c" in cfg.legacy_dead
        # New entry appended.
        assert "🪦" in cfg.legacy_dead

    def test_dedupes_user_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.legacy_dead]
            emojis = ["🪦", "🪦"]
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.legacy_dead.count("🪦") == 1

    def test_skips_empty_and_non_string_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.legacy_dead]
            emojis = ["", "  ", 42, "🪦", ["nested"]]
            """
        )
        cfg = load_topic_emoji_config(f)
        assert "🪦" in cfg.legacy_dead
        # Empty strings must not appear.
        assert "" not in cfg.legacy_dead
        assert "  " not in cfg.legacy_dead

    def test_invalid_emojis_value_uses_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.legacy_dead]
            emojis = "not a list"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.legacy_dead == LEGACY_DEAD_EMOJI

    def test_non_table_section_uses_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.legacy_dead]
            foo = "bar"
            """
        )
        cfg = load_topic_emoji_config(f)
        # No ``emojis`` key → no append, defaults preserved.
        assert cfg.legacy_dead == LEGACY_DEAD_EMOJI

    def test_top_level_topic_emoji_not_a_table(self, tmp_path: Path) -> None:
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            topic_emoji = 42
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.legacy_dead == LEGACY_DEAD_EMOJI


# ── Integration: loader preserves defaults for missing keys ───────────


class TestDefaultsMerging:
    def test_partial_system_override_keeps_done_dead_defaults(
        self, tmp_path: Path
    ) -> None:
        """If the user only overrides active+idle, done/dead must still
        come from the built-in defaults — never silently dropped."""
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.system]
            active = "🔵"
            idle = "⚪"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.system_emoji["active"] == "🔵"
        assert cfg.system_emoji["idle"] == "⚪"
        assert cfg.system_emoji["done"] == DEFAULT_SYSTEM_EMOJI["done"]
        assert cfg.system_emoji["dead"] == DEFAULT_SYSTEM_EMOJI["dead"]

    def test_partial_user_override_does_not_pollute_system(
        self, tmp_path: Path
    ) -> None:
        """Overriding user.active must NOT change system.active — the
        two tables are independent."""
        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.user]
            active = "🟣"
            """
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.user_emoji["active"] == "🟣"
        assert cfg.system_emoji["active"] == DEFAULT_SYSTEM_EMOJI["active"]


# ── Singleton cache ────────────────────────────────────────────────────


class TestSingleton:
    def test_reload_resets_cached_config(self, monkeypatch, tmp_path: Path) -> None:
        """A test that mutates the path between calls must be able to
        reload the config from the new path."""
        from ccgram.config import config
        from ccgram.handlers.status.topic_emoji import (
            get_topic_emoji_config,
            reload_topic_emoji_config,
        )

        f1 = tmp_path / "first.toml"
        f1.write_text(
            """
            [topic_emoji.system]
            active = "🚀"
            """
        )
        f2 = tmp_path / "second.toml"
        f2.write_text(
            """
            [topic_emoji.system]
            active = "🐢"
            """
        )

        monkeypatch.setattr(config, "topic_emoji_config_path", str(f1))
        reload_topic_emoji_config()
        first = get_topic_emoji_config()
        assert first.system_emoji["active"] == "🚀"

        monkeypatch.setattr(config, "topic_emoji_config_path", str(f2))
        reload_topic_emoji_config()
        second = get_topic_emoji_config()
        assert second.system_emoji["active"] == "🐢"


# ── Greptile PR review regressions ──────────────────────────────────────


class TestRejectWhitespaceInEmoji:
    """Greptile PR #269 finding 1: a whitespace-containing emoji value
    would let ``strip_emoji_prefix`` match a non-prefix and corrupt the
    cached clean name (e.g. ``active="A B"`` vs ``idle="A B C"``).
    Loader must reject any ASCII-whitespace character and keep the
    default for that state."""

    @pytest.mark.parametrize(
        "value",
        [
            "A B",  # single space
            "A  B",  # multiple spaces
            "A\tB",  # tab
            "A\nB",  # newline
        ],
    )
    def test_internal_whitespace_value_is_rejected(
        self, tmp_path: Path, value: str
    ) -> None:
        f = tmp_path / "topic_emoji.toml"
        toml_value = value.replace("\n", "\\n").replace("\t", "\\t")
        f.write_text(
            f'''
            [topic_emoji.system]
            active = "{toml_value}"
            '''
        )
        cfg = load_topic_emoji_config(f)
        assert cfg.system_emoji["active"] == DEFAULT_SYSTEM_EMOJI["active"]

    def test_strip_does_not_corrupt_name_after_rejection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """End-to-end: with ``active="A B"`` rejected at load time,
        ``strip_emoji_prefix`` cannot pick up a partial-match prefix
        because ``A B`` is never in the candidate list."""
        from ccgram.config import config
        from ccgram.handlers.status.topic_emoji import (
            reload_topic_emoji_config,
            strip_emoji_prefix,
        )

        f = tmp_path / "topic_emoji.toml"
        f.write_text(
            """
            [topic_emoji.system]
            active = "A B"
            idle = "A B C"
            """
        )
        monkeypatch.setattr(config, "topic_emoji_config_path", str(f))
        reload_topic_emoji_config()
        try:
            # Idle title — must not match anything that strips to a
            # corrupted clean name. With the rejection in place, only
            # the legitimate defaults are candidates.
            assert strip_emoji_prefix("A B C project") == "A B C project"
        finally:
            reload_topic_emoji_config()
