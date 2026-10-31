"""The probe's speculative writes must all be covered by the rollback.

Source audit in the style of test_no_tty_outside_backend: every cache
`_prepare_observed_generation` writes directly, plus everything its
schedule helpers write, must appear in `_rollback_probe_observation`'s
restore body. A new probe write without a matching restore silently
survives an identity flip; the docstring contract there states the why,
this gate enforces the what.
"""

import ast
from pathlib import Path

import pytest

import ccgram.transcript_reader as transcript_reader

_SOURCE = Path(transcript_reader.__file__).read_text()
_TREE = ast.parse(_SOURCE)


def _method(name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in ast.walk(_TREE):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    pytest.fail(f"method {name} not found in transcript_reader")


def _helper_names(body: ast.AST) -> set[str]:
    """self._helper(...) calls inside a body."""
    helpers = set()
    for node in ast.walk(body):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr.startswith("_")
        ):
            helpers.add(node.func.attr)
    return helpers


def _self_cache_of(expr: ast.expr) -> str | None:
    """The self._X cache an expression writes or mutates, else None."""
    if (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Attribute)
        and isinstance(expr.value.value, ast.Name)
        and expr.value.value.id == "self"
    ):
        return expr.value.attr
    if (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
    ):
        return expr.attr
    return None


def _write_targets(body: ast.AST) -> tuple[set[str], set[str]]:
    """(self-caches written or popped, tracked attributes assigned)."""
    caches: set[str] = set()
    tracked: set[str] = set()
    for node in ast.walk(body):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.target:
            targets = [node.target]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
        ):
            targets = [node.func.value]
        else:
            continue
        for tgt in targets:
            cache = _self_cache_of(tgt)
            if cache is not None:
                caches.add(cache)
            elif (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "tracked"
            ):
                tracked.add(tgt.attr)
    return caches, tracked


def _transitive_writes(body: ast.AST) -> tuple[set[str], set[str]]:
    """Direct writes plus the writes of every self._helper the body calls."""
    caches, tracked = _write_targets(body)
    seen: set[str] = set()
    pending = _helper_names(body)
    while pending - seen:
        name = (pending - seen).pop()
        seen.add(name)
        helper = _method(name)
        more_caches, more_tracked = _write_targets(helper)
        caches |= more_caches
        tracked |= more_tracked
        pending |= _helper_names(helper)
    return caches, tracked


def test_every_probe_write_has_a_rollback_restore():
    probe = _method("_prepare_observed_generation")
    rollback = _method("_rollback_probe_observation")

    probe_caches, probe_tracked = _transitive_writes(probe)
    rollback_caches, rollback_tracked = _write_targets(rollback)

    missing_caches = probe_caches - rollback_caches
    missing_tracked = probe_tracked - rollback_tracked
    assert not missing_caches, (
        f"probe writes {sorted(missing_caches)} without a restore in "
        "_rollback_probe_observation; an identity flip leaves them stale"
    )
    assert not missing_tracked, (
        f"probe writes tracked.{sorted(missing_tracked)} without a restore "
        "in _rollback_probe_observation"
    )
