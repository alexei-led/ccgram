"""Tests for the typed-plan grammar used to start a session in one message."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ccgram.handlers.topics.plan_parser import (
    AGENT_ALIASES,
    build_label,
    parse_agent,
    parse_topic_plan,
    resolve_project,
)


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    """A projects root with one git checkout and one plain directory."""
    project = tmp_path / "ccgram"
    (project / ".git").mkdir(parents=True)
    (tmp_path / "private-notes").mkdir()
    return [tmp_path]


class TestParseAgent:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("codex", ("codex", None)),
            ("CX", ("codex", None)),
            ("claude:safe", ("claude", "normal")),
            ("claude:normal", ("claude", "normal")),
            ("pi:yolo", ("pi", "yolo")),
            ("sh", ("shell", None)),
        ],
    )
    def test_resolves_aliases_and_modes(self, token: str, expected) -> None:
        assert parse_agent(token) == expected

    @pytest.mark.parametrize("token", ["gemini", "fix", "codex2", "", "not:known"])
    def test_rejects_non_agents(self, token: str) -> None:
        assert parse_agent(token) is None

    def test_every_alias_maps_to_a_registered_name(self) -> None:
        from ccgram.providers import registry, _ensure_registered  # noqa: PLC0415

        _ensure_registered()
        for target in AGENT_ALIASES.values():
            assert registry.is_valid(target)


class TestResolveProject:
    def test_exact_name_wins(self, roots: list[Path]) -> None:
        resolved, candidates = resolve_project("ccgram", roots)
        assert resolved == (roots[0] / "ccgram").resolve()
        assert candidates == ()

    def test_unique_prefix_resolves(self, roots: list[Path]) -> None:
        resolved, _ = resolve_project("ccg", roots)
        assert resolved is not None and resolved.name == "ccgram"

    def test_non_git_directory_is_not_a_candidate_by_default(
        self, roots: list[Path]
    ) -> None:
        resolved, _ = resolve_project("private-notes", roots)
        assert resolved is None

    def test_all_directories_mode_lists_plain_dirs(self, roots: list[Path]) -> None:
        resolved, _ = resolve_project("private-notes", roots, git_only=False)
        assert resolved is not None and resolved.name == "private-notes"

    def test_explicit_path_wins(self, roots: list[Path]) -> None:
        target = roots[0] / "private-notes"
        resolved, _ = resolve_project(str(target), roots)
        assert resolved == target.resolve()

    def test_ambiguous_prefix_returns_candidates(self, tmp_path: Path) -> None:
        for name in ("app-one", "app-two"):
            (tmp_path / name / ".git").mkdir(parents=True)
        resolved, candidates = resolve_project("app", [tmp_path])
        assert resolved is None
        assert [path.name for path in candidates] == ["app-one", "app-two"]


class TestParseTopicPlan:
    def test_agent_project_branch_and_instruction(self, roots: list[Path]) -> None:
        parsed = parse_topic_plan(
            "codex ccgram @feat-x fix the hook timeout",
            roots=roots,
            default_agent="claude",
        )
        assert parsed.plan is not None
        plan = parsed.plan
        assert plan.agent == "codex"
        assert plan.project.name == "ccgram"
        assert plan.branch == "feat-x"
        assert plan.instruction == "fix the hook timeout"
        assert plan.mode == "yolo"
        assert parsed.agent_typed and parsed.project_typed

    def test_mode_suffix_is_applied(self, roots: list[Path]) -> None:
        parsed = parse_topic_plan(
            "claude:safe ccgram just look", roots=roots, default_agent="codex"
        )
        assert parsed.plan is not None
        assert (parsed.plan.agent, parsed.plan.mode) == ("claude", "normal")

    def test_no_agent_word_uses_the_default_agent(self, roots: list[Path]) -> None:
        parsed = parse_topic_plan(
            "ccgram explain the parser", roots=roots, default_agent="pi"
        )
        assert parsed.plan is not None
        assert parsed.plan.agent == "pi"
        assert parsed.plan.instruction == "explain the parser"
        assert not parsed.agent_typed

    def test_instruction_only_uses_the_last_project(self, roots: list[Path]) -> None:
        last = (roots[0] / "ccgram").resolve()
        parsed = parse_topic_plan(
            "fix the flaky test", roots=roots, default_agent="claude", last_project=last
        )
        assert parsed.plan is not None
        assert parsed.plan.project == last
        assert parsed.plan.instruction == "fix the flaky test"

    def test_unresolvable_project_reports_candidates(self, roots: list[Path]) -> None:
        parsed = parse_topic_plan(
            "codex nosuchproject do it", roots=roots, default_agent="claude"
        )
        assert parsed.plan is None
        assert parsed.reason == "project"
        assert parsed.candidates

    def test_bare_at_reports_a_branch_error(self, roots: list[Path]) -> None:
        parsed = parse_topic_plan(
            "codex ccgram @ do it", roots=roots, default_agent="claude"
        )
        assert parsed.plan is None
        assert parsed.reason == "branch"

    def test_empty_text_is_not_a_plan(self, roots: list[Path]) -> None:
        assert parse_topic_plan("   ", roots=roots, default_agent="claude").reason == (
            "empty"
        )

    def test_multiline_instruction_is_kept_verbatim(self, roots: list[Path]) -> None:
        text = "codex ccgram fix this\n\nsteps:\n1. read\n2. patch"
        parsed = parse_topic_plan(text, roots=roots, default_agent="claude")
        assert parsed.plan is not None
        assert parsed.plan.instruction == "fix this\n\nsteps:\n1. read\n2. patch"


class TestBuildLabel:
    def test_label_carries_project_and_hint(self) -> None:
        label = build_label(Path("/w/ccgram"), "fix the hook timeout now please")
        assert label.startswith("ccgram \u00b7 fix the hook timeout")

    def test_long_labels_are_truncated(self) -> None:
        label = build_label(Path("/w/ccgram"), "x" * 200, max_len=20)
        assert len(label) <= 20
        assert label.endswith("\u2026")

    def test_label_never_contains_a_target_id(self) -> None:
        label = build_label(Path("/w/ccgram"), "bump the deps")
        assert not re.search(r"[0-9a-f]{8,}", label)
