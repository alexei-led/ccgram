from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "lint_lazy_imports.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lint_lazy_imports", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lint_lazy_imports"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lint_module():
    return _load_module()


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


# Sources the lint must accept: the import is excused by a `# Lazy:` marker,
# by an `if TYPE_CHECKING:` block, by a reset-for-testing function, or by
# being a module-level (eager) import.
CLEAN_SOURCES = {
    "documented_lazy_import": """
        def fn():
            # Lazy: avoid cycle with handlers.foo
            from .foo import bar
            return bar
    """,
    "documented_plain_import": """
        def fn():
            # Lazy: avoid cycle
            import foo
            return foo
    """,
    "type_checking_block": """
        from typing import TYPE_CHECKING

        def fn():
            if TYPE_CHECKING:
                from .foo import bar
            return None
    """,
    "type_checking_attribute_form": """
        import typing

        def fn():
            if typing.TYPE_CHECKING:
                from .foo import bar
            return None
    """,
    "reset_for_testing_functions": """
        def _reset_state_for_testing():
            from .foo import bar
            return bar

        def reset_for_testing():
            from .baz import qux
            return qux
    """,
    "module_level_import": """
        from .foo import bar

        def fn():
            return bar
    """,
    "documented_import_in_method": """
        class Widget:
            def fn(self):
                # Lazy: cycle with handlers.foo
                from .foo import bar
                return bar
    """,
    "documented_import_in_try_body": """
        def fn():
            try:
                # Lazy: optional dep
                from .foo import bar
                return bar
            except ImportError:
                return None
    """,
    "multiline_lazy_comment_block": """
        def fn():
            # Lazy: this annotation wraps over multiple lines because the
            # cycle reason needs more than one line of explanation to be
            # comprehensible to future readers.
            from .foo import bar
            return bar
    """,
    "lazy_above_blank_line_then_import": """
        def fn():
            # Lazy: still applies despite blank separator below.

            from .foo import bar
            return bar
    """,
    "nested_function_lazy_import": """
        def outer():
            def inner():
                # Lazy: documented inside nested function
                from .foo import bar
                return bar
            return inner
    """,
    "try_star_lazy_import": """
        def fn():
            try:
                x = 1
            except* ValueError:
                # Lazy: documented inside except* handler
                from .foo import bar
                return bar
    """,
    "nested_class_lazy_import": """
        class Outer:
            class Inner:
                def method(self):
                    # Lazy: documented inside nested class
                    from .foo import bar
                    return bar
    """,
    "documented_import_in_match_case": """
        def fn(x):
            match x:
                case 1:
                    # Lazy: optional path for specific case
                    from .foo import bar
                    return bar
                case _:
                    return None
    """,
}


# Sources the lint must reject. Every one hides exactly one undocumented
# in-function import of `.foo`, reached through a different nesting shape.
VIOLATING_SOURCES = {
    "undocumented_lazy_import": """
        def fn():
            from .foo import bar
            return bar
    """,
    "method_inside_class": """
        class Widget:
            def fn(self):
                from .foo import bar
                return bar
    """,
    "async_function": """
        async def fn():
            from .foo import bar
            return bar
    """,
    "try_body": """
        def fn():
            try:
                from .foo import bar
                return bar
            except ImportError:
                return None
    """,
    "except_body": """
        def fn():
            try:
                pass
            except ValueError:
                from .foo import bar
                return bar
    """,
    "finally_body": """
        def fn():
            try:
                pass
            finally:
                from .foo import bar
    """,
    "try_else_body": """
        def fn():
            try:
                pass
            except ValueError:
                pass
            else:
                from .foo import bar
                return bar
    """,
    "if_body": """
        def fn(flag):
            if flag:
                from .foo import bar
                return bar
            return None
    """,
    "else_body": """
        def fn(flag):
            if flag:
                return None
            else:
                from .foo import bar
                return bar
    """,
    "with_body": """
        def fn(handle):
            with handle:
                from .foo import bar
                return bar
    """,
    "async_with_body": """
        async def fn(handle):
            async with handle:
                from .foo import bar
                return bar
    """,
    "for_body": """
        def fn(items):
            for item in items:
                from .foo import bar
                return bar
    """,
    "for_else_body": """
        def fn(items):
            for item in items:
                pass
            else:
                from .foo import bar
                return bar
    """,
    "async_for_body": """
        async def fn(items):
            async for item in items:
                from .foo import bar
                return bar
    """,
    "while_body": """
        def fn(cond):
            while cond:
                from .foo import bar
                return bar
    """,
    "while_else_body": """
        def fn(cond):
            while cond:
                pass
            else:
                from .foo import bar
                return bar
    """,
    "nested_try": """
        def fn(x):
            if x:
                try:
                    from .foo import bar
                    return bar
                except OSError:
                    pass
    """,
    "nested_function": """
        def outer():
            def inner():
                from .foo import bar
                return bar
            return inner
    """,
    "try_star_handler": """
        def fn():
            try:
                x = 1
            except* ValueError:
                from .foo import bar
                return bar
    """,
    "method_inside_function_class": """
        def outer():
            class Inner:
                def method(self):
                    from .foo import bar
                    return bar
            return Inner
    """,
    "method_inside_nested_class": """
        class Outer:
            class Inner:
                def method(self):
                    from .foo import bar
                    return bar
    """,
    "method_inside_doubly_nested_function_class": """
        def outer():
            class Inner:
                class InnerInner:
                    def method(self):
                        from .foo import bar
                        return bar
            return Inner
    """,
    "match_case": """
        def fn(x):
            match x:
                case 1:
                    from .foo import bar
                    return bar
                case _:
                    return None
    """,
    "top_level_type_checking_else_branch": """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            pass
        else:
            def fn():
                from .foo import bar
                return bar
    """,
    "class_body_control_flow": """
        FLAG = True

        class Outer:
            if FLAG:
                def method(self):
                    from .foo import bar
                    return bar
    """,
    "class_body_try_block": """
        class Outer:
            try:
                pass
            except Exception:
                def method(self):
                    from .foo import bar
                    return bar
    """,
    "class_body_inside_function": """
        def outer():
            class Inner:
                from .foo import bar
            return Inner
    """,
}


class TestAcceptedSources:
    @pytest.mark.parametrize("name", sorted(CLEAN_SOURCES))
    def test_reports_no_violations(
        self, lint_module, tmp_path: Path, name: str
    ) -> None:
        path = _write(tmp_path, name, CLEAN_SOURCES[name])
        assert lint_module.find_violations(path) == []


class TestRejectedSources:
    @pytest.mark.parametrize("name", sorted(VIOLATING_SOURCES))
    def test_catches_the_undocumented_import(
        self, lint_module, tmp_path: Path, name: str
    ) -> None:
        path = _write(tmp_path, name, VIOLATING_SOURCES[name])
        violations = lint_module.find_violations(path)
        assert len(violations) == 1
        assert "from .foo import bar" in violations[0][1]

    def test_plain_import_statement_is_caught(
        self, lint_module, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            "plain_import",
            """
            def fn():
                import foo
                return foo
            """,
        )
        violations = lint_module.find_violations(path)
        assert len(violations) == 1
        assert "import foo" in violations[0][1]

    def test_reports_every_undocumented_import_in_a_file(
        self, lint_module, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            "multiple",
            """
            def one():
                from .foo import bar
                return bar

            def two():
                from .baz import qux
                return qux
            """,
        )
        violations = lint_module.find_violations(path)
        assert len(violations) == 2
        assert [v[1] for v in violations] == [
            "from .foo import bar",
            "from .baz import qux",
        ]

    def test_lazy_marker_covers_only_the_import_directly_below(
        self, lint_module, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            "broken_walk",
            """
            def fn():
                # Lazy: this annotation is for the FIRST import below.
                from .foo import bar

                from .baz import qux
                return bar, qux
            """,
        )
        violations = lint_module.find_violations(path)
        assert len(violations) == 1
        assert "from .baz import qux" in violations[0][1]

    @pytest.mark.parametrize(
        "fn_name", ["reset_thing", "reset_for_testing_helper", "for_testing_reset"]
    )
    def test_reset_exemption_does_not_leak_to_similar_names(
        self, lint_module, tmp_path: Path, fn_name: str
    ) -> None:
        path = _write(
            tmp_path,
            f"near_miss_{fn_name}",
            f"""
            def {fn_name}():
                from .foo import bar
                return bar
            """,
        )
        assert len(lint_module.find_violations(path)) == 1


class TestCli:
    def test_returns_zero_when_clean(self, lint_module, tmp_path: Path) -> None:
        _write(tmp_path, "clean", CLEAN_SOURCES["documented_lazy_import"])
        assert lint_module.main(["lint_lazy_imports.py", str(tmp_path)]) == 0

    def test_returns_one_when_violations(self, lint_module, tmp_path: Path) -> None:
        _write(tmp_path, "dirty", VIOLATING_SOURCES["undocumented_lazy_import"])
        assert lint_module.main(["lint_lazy_imports.py", str(tmp_path)]) == 1

    def test_reports_the_offending_path_and_line(
        self, lint_module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(tmp_path, "dirty", VIOLATING_SOURCES["undocumented_lazy_import"])
        lint_module.main(["lint_lazy_imports.py", str(tmp_path)])
        out = capsys.readouterr().out
        assert (
            "dirty.py:3: undocumented in-function import: from .foo import bar" in out
        )
        assert "1 undocumented in-function import(s)." in out
