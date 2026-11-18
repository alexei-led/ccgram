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
        spec = parse_agent(token)
        assert spec is not None
        assert (spec.name, spec.mode) == expected

    def test_an_unknown_mode_suffix_is_reported(self) -> None:
        spec = parse_agent("claude:sfae")
        assert spec is not None
        assert spec.name == "claude"
        assert spec.mode is None
        assert spec.unknown_mode == "sfae"

    @pytest.mark.parametrize("token", ["fix", "codex2", "", "not:known"])
    def test_rejects_non_agents(self, token: str) -> None:
        assert parse_agent(token) is None

    def test_gemini_is_recognised_so_it_is_never_run_as_the_default(self) -> None:
        """A registered-but-hidden provider must parse, not fall through."""
        spec = parse_agent("gemini:safe")
        assert spec is not None
        assert (spec.name, spec.mode) == ("gemini", "normal")

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

    def test_ambiguous_project_surfaces_even_with_a_last_project(
        self, tmp_path: Path
    ) -> None:
        for name in ("app-one", "app-two"):
            (tmp_path / name / ".git").mkdir(parents=True)
        last = (tmp_path / "app-one").resolve()

        parsed = parse_topic_plan(
            "codex app @feat-x fix it",
            roots=[tmp_path],
            default_agent="claude",
            last_project=last,
        )

        assert parsed.plan is None
        assert parsed.reason == "project"
        assert sorted(path.name for path in parsed.candidates) == [
            "app-one",
            "app-two",
        ]

    def test_an_unmatched_word_stays_instruction(self, roots: list[Path]) -> None:
        last = (roots[0] / "ccgram").resolve()

        parsed = parse_topic_plan(
            "codex fix the bug",
            roots=roots,
            default_agent="claude",
            last_project=last,
        )

        assert parsed.plan is not None
        assert parsed.plan.project == last
        assert parsed.plan.instruction == "fix the bug"

    async def test_an_unresolved_explicit_path_never_falls_back(
        self, roots: list[Path]
    ) -> None:
        """A typed path that does not exist is an error, not a cue to use MRU."""
        last = (roots[0] / "ccgram").resolve()

        parsed = parse_topic_plan(
            "codex /missing/project @feat-x fix it",
            roots=roots,
            default_agent="claude",
            last_project=last,
        )

        assert parsed.plan is None
        assert parsed.reason == "project"

    async def test_ambiguous_project_keeps_the_requested_branch(
        self, tmp_path: Path
    ) -> None:
        """The candidate callback rebuilds from slots: @branch must survive."""
        for name in ("app-one", "app-two"):
            (tmp_path / name / ".git").mkdir(parents=True)

        parsed = parse_topic_plan(
            "codex app @feat-x fix it", roots=[tmp_path], default_agent="claude"
        )

        assert parsed.plan is None
        assert parsed.reason == "project"
        assert parsed.branch == "feat-x"
        assert parsed.instruction == "fix it"

    def test_an_unmatched_project_before_a_branch_is_reported(
        self, roots: list[Path]
    ) -> None:
        """<word> @branch proves a project slot: report it, do not use MRU."""
        last = (roots[0] / "ccgram").resolve()

        parsed = parse_topic_plan(
            "codex nosuchproject @fix repair the bug",
            roots=roots,
            default_agent="claude",
            last_project=last,
        )

        assert parsed.plan is None
        assert parsed.reason == "project"
        assert parsed.branch == "fix"
        assert parsed.instruction == "repair the bug"

    def test_an_instruction_word_is_never_taken_as_a_branch(
        self, roots: list[Path]
    ) -> None:
        """After a resolved project, an @word inside the instruction stays text."""
        parsed = parse_topic_plan(
            "codex ccgram explain @decorator syntax",
            roots=roots,
            default_agent="claude",
        )

        assert parsed.plan is not None
        assert parsed.plan.branch is None
        assert parsed.plan.instruction == "explain @decorator syntax"

    def test_a_symlinked_project_keeps_its_listed_name(self, tmp_path: Path) -> None:
        """The word the operator types is the link name, not the target name."""
        real = tmp_path / "repos" / "backend"
        (real / ".git").mkdir(parents=True)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "client").symlink_to(real)

        parsed = parse_topic_plan(
            "codex client fix the bug", roots=[root], default_agent="claude"
        )

        assert parsed.plan is not None
        # The plan keeps the listed name; resolution happens at launch time.
        assert parsed.plan.project.name == "client"
        assert parsed.plan.project.resolve() == real.resolve()

    def test_two_names_for_one_checkout_both_resolve(self, tmp_path: Path) -> None:
        """A link and its target are two words the operator may type."""
        real = tmp_path / "alpha"
        (real / ".git").mkdir(parents=True)
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "zclient").symlink_to(real)

        for word, expected in (("zclient", root / "zclient"), ("alpha", real)):
            parsed = parse_topic_plan(
                f"codex {word} fix the bug",
                roots=[root, tmp_path],
                default_agent="claude",
            )
            assert parsed.plan is not None, word
            assert parsed.plan.project == expected

    def test_an_unknown_tilde_user_is_reported_not_crashed(
        self, roots: list[Path]
    ) -> None:
        """expanduser raises RuntimeError for an unknown account."""
        parsed = parse_topic_plan(
            "codex ~nonexistent-user/project fix it",
            roots=roots,
            default_agent="claude",
        )

        assert parsed.plan is None
        assert parsed.reason == "project"


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
