from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / "scripts/git-hooks/pre-push"
_HEAD = "a" * 40
_OLD = "b" * 40
_ZERO = "0" * 40

_CASES = [
    {"id": "docs-only", "files": [], "success": True},
    {"id": "source-only", "success": True},
    {"id": "policy-proposal", "policy": True, "success": True},
    {"id": "policy-only", "policy": True, "files": [".archfit.yaml"], "success": True},
    {"id": "new-policy-branch", "policy": True, "new": True, "success": True},
    {"id": "existing-policy-docs-only", "policy": True, "files": [], "success": True},
    {"id": "main-policy", "policy": True, "remote_ref": "refs/heads/main"},
    {"id": "tag-policy", "policy": True, "remote_ref": "refs/tags/v1.0.0"},
    {"id": "mixed-default-policy", "policy": True, "extra_ref": "refs/heads/main"},
    {"id": "mixed-tag-policy", "policy": True, "extra_ref": "refs/tags/v1.0.0"},
    {"id": "local-main-policy", "policy": True, "local_ref": "refs/heads/main"},
    {"id": "unknown-default", "policy": True, "unknown_default": True},
    {"id": "foreign-remote", "policy": True, "remote": "mirror"},
    {"id": "different-head", "policy": True, "different_head": True},
    {"id": "dirty-policy", "policy": True, "dirty": True},
    {"id": "untracked-source", "policy": True, "untracked": True},
    {"id": "candidate-error", "policy": True, "candidate_exit": 1},
    {"id": "candidate-unknown", "policy": True, "candidate_exit": 2},
    {"id": "guard-error", "policy": True, "guard_exit": 1},
    {"id": "protected-check-error", "check_exit": 1},
    {"id": "git-diff-error", "git_error": True},
    {"id": "branch-deletion", "delete": True, "policy": True, "success": True},
]


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_pre_push_policy_paths(tmp_path, case):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls.jsonl"
    boundary = (
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
case = json.loads(os.environ["HOOK_CASE"])
with Path(os.environ["HOOK_LOG"]).open("a") as stream:
    stream.write(json.dumps([name, args]) + "\\n")
if name == "git":
    if args[0] == "symbolic-ref":
        if case.get("unknown_default"): sys.exit(1)
        print("origin/main")
    elif args[0] == "rev-parse": print("a" * 40)
    elif args[0] == "merge-base": print("b" * 40)
    elif args[0] == "status":
        if case.get("dirty"): print(" M .archfit.yaml")
        if case.get("untracked"): print("?? src/ccgram/uncommitted.py")
    elif args[0] == "diff":
        if case.get("git_error"): sys.exit(128)
        if "--quiet" in args: sys.exit(1 if case.get("dirty") else 0)
        paths = args[args.index("--") + 1:]
        if paths == [".archfit.yaml"]:
            if case.get("policy"): print(".archfit.yaml")
        else:
            for path in case.get("files", ["src/ccgram/example.py"]): print(path)
    else: raise AssertionError(args)
elif name == "make":
    if args == ["arch-guard"]: sys.exit(case.get("guard_exit", 0))
    assert args == ["arch-check"]
    sys.exit(case.get("check_exit", 2 if case.get("policy") else 0))
elif name == "archfit":
    assert args == ["check", "--policy", ".archfit.yaml"]
    sys.exit(case.get("candidate_exit", 0))
else: raise AssertionError(name)
"""
    )
    for name in ("git", "make", "archfit"):
        executable = binaries / name
        executable.write_text(boundary)
        executable.chmod(0o700)
    local_ref = case.get("local_ref", "refs/heads/feat/test")
    remote_ref = case.get("remote_ref", "refs/heads/feat/test")
    local_sha = (
        _ZERO if case.get("delete") else _OLD if case.get("different_head") else _HEAD
    )
    remote_sha = _ZERO if case.get("new") else _OLD
    updates = f"{local_ref} {local_sha} {remote_ref} {remote_sha}\n"
    if extra := case.get("extra_ref"):
        updates += f"refs/heads/other {_HEAD} {extra} {_OLD}\n"
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        HOOK_CASE=json.dumps(case),
        HOOK_LOG=str(log),
    )

    result = subprocess.run(
        [
            "bash",
            str(_HOOK),
            case.get("remote", "origin"),
            "https://example.test/repo.git",
        ],
        cwd=tmp_path,
        input=updates,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert (result.returncode == 0) is case.get("success", False), (
        result.stdout + result.stderr
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    if (
        case.get("success")
        and case.get("policy")
        and case.get("files") != []
        and not case.get("delete")
    ):
        assert ["make", ["arch-guard"]] in calls
        assert ["archfit", ["check", "--policy", ".archfit.yaml"]] in calls
        assert "approval" in result.stdout.lower()
        assert ["make", ["arch-check"]] not in calls
