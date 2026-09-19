from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from ccgram.providers.omp import OmpProvider, encode_cwd_dirname
from ccgram.providers.omp_discovery import discover_omp_commands


def _fake_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Fake home + temp roots, both under *tmp_path* (never the real ones)."""
    return tmp_path / "base" / "home", tmp_path / "base" / "tmp"


@pytest.fixture
def fake_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Fake home/temp roots with ``Path.resolve`` pinned to identity.

    ``encode_cwd_dirname`` canonicalizes before bucketing, so tests that are
    not about canonicalization must not let the real filesystem re-map paths.
    """
    home, temp = _fake_roots(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    return home, temp


class TestEncodeCwdDirname:
    @pytest.mark.parametrize(
        ("cwd", "expected"),
        [
            pytest.param(
                lambda home, tmp: str(home / "Code" / "app"),
                "-Code-app",
                id="home-relative",
            ),
            pytest.param(
                lambda home, tmp: str(home / "Code"),
                "-Code",
                id="home-relative-one-level",
            ),
            pytest.param(
                lambda home, tmp: str(home / "Code") + "/",
                "-Code",
                id="home-relative-trailing-separator",
            ),
            pytest.param(lambda home, tmp: str(home), "-", id="home-itself"),
            pytest.param(
                lambda home, tmp: str(tmp / "omp" / "s1"),
                "-tmp-omp-s1",
                id="temp-relative",
            ),
            pytest.param(lambda home, tmp: str(tmp), "-tmp-", id="temp-root-itself"),
            pytest.param(
                lambda home, tmp: "/srv/app", "--srv-app--", id="under-neither-root"
            ),
            pytest.param(lambda home, tmp: "/a/b", "--a-b--", id="absolute-two-parts"),
            pytest.param(lambda home, tmp: "/", "----", id="filesystem-root"),
        ],
    )
    def test_encodes(
        self,
        fake_roots: tuple[Path, Path],
        cwd: Callable[[Path, Path], str],
        expected: str,
    ) -> None:
        home, temp = fake_roots
        assert encode_cwd_dirname(cwd(home, temp)) == expected

    def test_symlink_alias_lands_in_the_canonical_bucket(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, temp = _fake_roots(tmp_path)
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))
        # omp canonicalizes first, so a cwd reaching the CLI through a symlink
        # must bucket exactly like its target.
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda self, strict=False: Path(str(self).replace("/link/", "/Code/")),
        )

        assert encode_cwd_dirname(str(home / "link" / "app")) == "-Code-app"
        assert encode_cwd_dirname(str(home / "Code" / "app")) == "-Code-app"

    def test_oserror_from_resolve_falls_back_to_the_literal_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, temp = _fake_roots(tmp_path)
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))

        def _boom(self: Path, strict: bool = False) -> Path:
            raise OSError("unresolvable")

        monkeypatch.setattr(Path, "resolve", _boom)

        assert encode_cwd_dirname(str(home / "link" / "app")) == "-link-app"


class TestDiscoverTranscript:
    @pytest.fixture
    def sessions_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr("ccgram.providers.omp._omp_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
        return tmp_path

    @staticmethod
    def _write_session(path: Path, session_id: str, cwd: str) -> Path:
        """Write a transcript shaped like a real omp file: title, then header."""
        path.write_text(
            json.dumps({"type": "title", "v": 1, "title": "a title", "source": "auto"})
            + "\n"
            + json.dumps(
                {"type": "session", "version": 3, "id": session_id, "cwd": cwd}
            )
            + "\n"
        )
        return path

    def test_returns_newest_matching_cwd(self, sessions_dir: Path) -> None:
        cwd = "/real/project"
        bucket = sessions_dir / encode_cwd_dirname(cwd)
        bucket.mkdir()
        older = self._write_session(bucket / "old.jsonl", "s1", cwd)
        newest = self._write_session(bucket / "new.jsonl", "s2", cwd)
        # Two newest-mtime entries that are *not* session files: a lock sibling
        # carrying a valid header, and a subagent directory holding its own
        # transcript. Both would win a looser scan.
        lock = self._write_session(bucket / "new.jsonl.lock.os", "s-lock", cwd)
        subagent_dir = bucket / "s3.jsonl"
        subagent_dir.mkdir()
        nested = self._write_session(subagent_dir / "Task.jsonl", "s-nested", cwd)

        now = time.time()
        os.utime(older, (now - 100, now - 100))
        os.utime(newest, (now, now))
        os.utime(lock, (now + 10, now + 10))
        os.utime(nested, (now + 20, now + 20))
        os.utime(subagent_dir, (now + 20, now + 20))

        ev = OmpProvider().discover_transcript(cwd, "ccgram:@0", max_age=0)
        assert ev is not None
        assert ev.session_id == "s2"
        assert ev.transcript_path == str(newest)
        assert ev.cwd == cwd
        assert ev.window_key == "ccgram:@0"

    def test_skips_transcript_whose_header_cwd_differs(
        self, sessions_dir: Path
    ) -> None:
        cwd = "/real/project"
        bucket = sessions_dir / encode_cwd_dirname(cwd)
        bucket.mkdir()
        foreign = self._write_session(bucket / "other.jsonl", "s-other", "/elsewhere")
        mine = self._write_session(bucket / "mine.jsonl", "s-mine", cwd)

        now = time.time()
        os.utime(foreign, (now, now))
        os.utime(mine, (now - 100, now - 100))

        ev = OmpProvider().discover_transcript(cwd, "ccgram:@0", max_age=0)
        assert ev is not None
        assert ev.session_id == "s-mine"

    def test_empty_cwd_returns_none(self, sessions_dir: Path) -> None:
        assert OmpProvider().discover_transcript("", "ccgram:@0") is None

    def test_missing_bucket_dir_returns_none(self, sessions_dir: Path) -> None:
        assert OmpProvider().discover_transcript("/no/such/place", "ccgram:@0") is None

    def test_rejects_stale_files_when_max_age_set(self, sessions_dir: Path) -> None:
        cwd = "/some/project"
        bucket = sessions_dir / encode_cwd_dirname(cwd)
        bucket.mkdir()
        stale = self._write_session(bucket / "old.jsonl", "s1", cwd)

        now = time.time()
        os.utime(stale, (now - 600, now - 600))

        provider = OmpProvider()
        # Default cap (the provider's own staleness bound) and an explicit
        # short one both reject it; a permissive limit accepts it.
        assert provider.discover_transcript(cwd, "ccgram:@0") is None
        assert provider.discover_transcript(cwd, "ccgram:@0", max_age=60) is None
        ev = provider.discover_transcript(cwd, "ccgram:@0", max_age=1200)
        assert ev is not None and ev.session_id == "s1"


class TestCapabilities:
    def test_shape(self) -> None:
        caps = OmpProvider().capabilities
        assert caps.name == "omp"
        assert caps.launch_command == "omp"
        assert caps.supports_hook is False
        assert caps.supports_resume is True
        assert caps.supports_continue is True
        assert caps.supports_incremental_read is True
        assert caps.followup_key == "C-q"
        assert {"/new", "/clear", "/delete", "/followup"} <= set(caps.builtin_commands)
        assert "/resume" not in caps.builtin_commands
        assert caps.tui_picker_commands
        assert caps.tui_picker_commands <= {
            name.lstrip("/") for name in caps.builtin_commands
        }


class TestMakeLaunchArgs:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param({}, "", id="fresh"),
            pytest.param({"use_continue": True}, "--continue", id="continue"),
            pytest.param(
                {"resume_id": "01a0b5ca-d416-71c0-adec-3936173735c6"},
                "--session 01a0b5ca-d416-71c0-adec-3936173735c6",
                id="session_by_uuid",
            ),
            pytest.param(
                {"resume_id": "x", "use_continue": True},
                "--session x",
                id="resume_wins_over_continue",
            ),
        ],
    )
    def test_make_launch_args(self, kwargs: dict, expected: str) -> None:
        assert OmpProvider().make_launch_args(**kwargs) == expected


class TestParseTranscript:
    """omp transcripts are turned into messages through the same pipeline the
    monitor uses: file line → ``parse_transcript_line`` → ``parse_transcript_entries``."""

    ENTRIES: list[dict] = [
        {"type": "title", "v": 1, "title": "Fix the parser", "source": "auto"},
        {
            "type": "session",
            "version": 3,
            "id": "01a0b5ca-d416-71c0-adec-3936173735c6",
            "timestamp": "2026-09-18T18:32:43.798Z",
            "cwd": "/work/app",
        },
        {"type": "model_change", "model": "claude-opus-4-5"},
        {
            "type": "message",
            "id": "m1",
            "parentId": None,
            "timestamp": "2026-09-18T18:32:44.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hi omp"}],
            },
        },
        {
            "type": "message",
            "id": "m2",
            "parentId": "m1",
            "timestamp": "2026-09-18T18:32:45.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "running"},
                    {
                        "type": "toolCall",
                        "id": "t1",
                        "name": "bash",
                        "arguments": {"command": "ls -la"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "id": "m3",
            "parentId": "m2",
            "timestamp": "2026-09-18T18:32:46.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "t1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "a\nb"}],
                "isError": False,
            },
        },
        {
            "type": "message",
            "id": "m4",
            "parentId": "m3",
            "message": {
                "role": "fileMention",
                "content": [{"type": "text", "text": "@src/foo.py"}],
            },
        },
        {
            "type": "message",
            "id": "m5",
            "parentId": "m4",
            "message": {
                "role": "developer",
                "content": [{"type": "text", "text": "internal note"}],
            },
        },
        {
            "type": "message",
            "id": "m6",
            "parentId": "m5",
            "timestamp": "2026-09-18T18:32:47.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        },
    ]

    def test_messages_and_pending(self, tmp_path: Path) -> None:
        path = tmp_path / "20260918_01a0b5ca.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in self.ENTRIES))

        provider = OmpProvider()
        messages = []
        pending: dict = {}
        for line in path.read_text().splitlines():
            entry = provider.parse_transcript_line(line)
            if entry is None:
                continue
            batch, pending = provider.parse_transcript_entries([entry], pending)
            messages.extend(batch)

        assert [m.content_type for m in messages] == [
            "text",
            "text",
            "tool_use",
            "tool_result",
            "text",
        ]
        assert [m.role for m in messages] == [
            "user",
            "assistant",
            "assistant",
            "assistant",
            "assistant",
        ]
        assert messages[0].text == "hi omp"
        assert messages[1].text == "running"
        assert messages[2].text == "\U0001f4bb **bash**: `ls -la`"
        assert messages[2].tool_use_id == "t1"
        assert messages[2].tool_name == "Bash"
        # The bash result is always rendered as a line count + quotable body.
        assert messages[3].tool_use_id == "t1"
        assert messages[3].tool_name == "Bash"
        assert "2 lines" in messages[3].text
        assert "a\nb" in messages[3].text
        assert messages[4].text == "done"
        assert pending == {}


class TestDiscoverCommands:
    def test_returns_builtins_and_user_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        agent = home / ".omp" / "agent"
        skill_dir = agent / "skills" / "brave-search"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: brave-search\ndescription: Search the web\nuser-invocable: true\n---\n"
        )
        (agent / "prompts").mkdir(parents=True)
        (agent / "prompts" / "review.md").write_text(
            "---\ndescription: Review staged changes\nargument-hint: <PR>\n---\n"
        )
        (agent / "commands").mkdir(parents=True)
        (agent / "commands" / "deploy.md").write_text(
            "---\ndescription: Deploy the app\n---\n"
        )
        (agent / "extensions").mkdir(parents=True)
        (agent / "extensions" / "metrics.ts").write_text(
            'export default function (pi) { pi.registerCommand("metrics", '
            '{ description: "Metrics", handler: async () => {} }); }\n'
        )
        (agent / "hooks" / "pre").mkdir(parents=True)
        (agent / "hooks" / "pre" / "guard.ts").write_text(
            'export default function (pi) { pi.registerCommand("guard", '
            "{ handler: async () => {} }); }\n"
        )
        monkeypatch.setattr(Path, "home", lambda: home)

        by_name = {c.name: c for c in discover_omp_commands(str(tmp_path / "proj"))}

        assert {"/new", "/clear", "/delete", "/followup"} <= set(by_name)
        assert "/resume" not in by_name
        # omp registers skill commands as ``/skill:<name>``, so the advertised
        # name carries the prefix the CLI actually accepts.
        assert by_name["skill:brave-search"].source == "skill"
        assert by_name["skill:brave-search"].description == "Search the web"
        assert "brave-search" not in by_name
        assert by_name["review"].source == "command"
        assert by_name["review"].description == "<PR> — Review staged changes"
        assert by_name["deploy"].source == "command"
        assert by_name["deploy"].description == "Deploy the app"
        assert by_name["metrics"].source == "command"
        assert by_name["guard"].source == "command"

    def test_walks_project_ancestors_stopping_at_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)

        project = tmp_path / "outer" / "project"
        (project / ".git").mkdir(parents=True)
        (project / ".omp" / "commands").mkdir(parents=True)
        (project / ".omp" / "commands" / "projcmd.md").write_text(
            "---\ndescription: Project command\n---\n"
        )
        # Above the .git boundary: must not be discovered.
        (tmp_path / "outer" / ".omp" / "commands").mkdir(parents=True)
        (tmp_path / "outer" / ".omp" / "commands" / "outside.md").write_text(
            "---\ndescription: Outside the repo\n---\n"
        )

        names = {c.name for c in discover_omp_commands(str(project / "src" / "deep"))}
        assert "projcmd" in names
        assert "outside" not in names

    def test_dedup_is_first_source_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        agent = home / ".omp" / "agent"
        # ``alpha`` exists as a skill, a prompt, and an extension command. The
        # skill is namespaced (``skill:alpha``), so it no longer competes for
        # the bare name the other two share: the prompt must win that one.
        (agent / "skills" / "alpha").mkdir(parents=True)
        (agent / "skills" / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Skill alpha\n---\n"
        )
        (agent / "prompts").mkdir(parents=True)
        (agent / "prompts" / "alpha.md").write_text(
            "---\ndescription: Prompt alpha\n---\n"
        )
        # beta: prompt + extension — the prompt must win.
        (agent / "prompts" / "beta.md").write_text(
            "---\ndescription: Prompt beta\n---\n"
        )
        (agent / "extensions").mkdir(parents=True)
        (agent / "extensions" / "collide.ts").write_text(
            'export default function (pi) { pi.registerCommand("alpha", '
            '{ description: "Extension alpha" }); '
            'pi.registerCommand("beta", { description: "Extension beta" }); }\n'
        )
        monkeypatch.setattr(Path, "home", lambda: home)

        cmds = discover_omp_commands(str(tmp_path / "proj"))
        by_name = {c.name: c for c in cmds}

        assert [c.name for c in cmds].count("alpha") == 1
        assert by_name["alpha"].source == "command"
        assert by_name["alpha"].description == "Prompt alpha"
        assert by_name["skill:alpha"].source == "skill"
        assert [c.name for c in cmds].count("beta") == 1
        assert by_name["beta"].source == "command"
        assert by_name["beta"].description == "Prompt beta"
