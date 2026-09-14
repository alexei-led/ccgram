"""Translate HerdrManager CLI arguments to public JSON-RPC requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

__all__ = ["command_request"]

_Request = tuple[str, dict[str, object]]
_Handler = Callable[[tuple[str, ...]], _Request]

_COMMAND_PARTS = 2
_LIST_PARTS = 2
_TARGET_PARTS = 3
_FOUR_PARTS = 4
_PANE_READ_MIN_PARTS = 3
_REPORT_METADATA_PARTS = 7
_UINT32_MAX = 2**32 - 1

_LIST_METHODS: Mapping[tuple[str, str], str] = {
    ("agent", "list"): "agent.list",
    ("workspace", "list"): "workspace.list",
    ("tab", "list"): "tab.list",
    ("pane", "list"): "pane.list",
}

_TARGET_METHODS: Mapping[tuple[str, str], tuple[str, str]] = {
    ("pane", "get"): ("pane.get", "pane_id"),
    ("pane", "close"): ("pane.close", "pane_id"),
    ("tab", "close"): ("tab.close", "tab_id"),
    ("workspace", "close"): ("workspace.close", "workspace_id"),
}


def command_request(args: Sequence[str]) -> _Request:
    """Convert one manager-generated Herdr CLI invocation to JSON-RPC.

    The returned method and parameters are ready for the public Herdr socket
    API. No command is executed here; unsupported or malformed argument
    vectors raise ``ValueError``.
    """
    argv = _validate_args(args)
    if not argv:
        raise ValueError("Herdr command arguments are empty")

    if argv == ("status", "--json"):
        return "ping", {}

    if len(argv) < _COMMAND_PARTS:
        raise ValueError(f"Unsupported Herdr command: {list(argv)!r}")
    handler = _COMMAND_HANDLERS.get((argv[0], argv[1]))
    if handler is None:
        raise ValueError(f"Unsupported Herdr command: {list(argv)!r}")
    return handler(argv)


def _validate_args(args: Sequence[str]) -> tuple[str, ...]:
    if isinstance(args, (str, bytes, bytearray)) or not isinstance(args, Sequence):
        raise ValueError("Herdr command arguments must be a sequence of strings")
    argv = tuple(args)
    if not all(isinstance(value, str) for value in argv):
        raise ValueError("Herdr command arguments must be strings")
    return argv


def _require_identifier(value: str, name: str) -> str:
    if not value or value.startswith("--"):
        raise ValueError(f"Malformed Herdr {name}")
    return value


def _pane_read(argv: tuple[str, ...]) -> _Request:
    if len(argv) < _PANE_READ_MIN_PARTS:
        raise ValueError("Malformed pane read command")
    pane_id = _require_identifier(argv[2], "pane_id")
    options = _parse_options(
        argv[3:],
        value_options={"--source": "source", "--lines": "lines", "--format": "format"},
        flag_options={},
        required={"source", "format"},
    )
    source = options["source"]
    if not isinstance(source, str) or source not in {
        "visible",
        "recent",
        "recent-unwrapped",
    }:
        raise ValueError(f"Unsupported pane read source: {source!r}")
    fmt = options["format"]
    if not isinstance(fmt, str) or fmt not in {"text", "ansi"}:
        raise ValueError(f"Unsupported pane read format: {fmt!r}")
    params: dict[str, object] = {
        "pane_id": pane_id,
        "source": source.replace("-", "_"),
        "format": fmt,
        "strip_ansi": fmt != "ansi",
    }
    if "lines" in options:
        params["lines"] = _parse_uint32(options["lines"])
    return "pane.read", params


def _pane_report_metadata(argv: tuple[str, ...]) -> _Request:
    if (
        len(argv) != _REPORT_METADATA_PARTS
        or argv[3] != "--source"
        or argv[4] != "ccgram"
        or argv[5] != "--title"
    ):
        raise ValueError("Malformed pane report-metadata command")
    return "pane.report_metadata", {
        "pane_id": _require_identifier(argv[2], "pane_id"),
        "source": argv[4],
        "title": argv[6],
    }


def _creation_request(
    argv: tuple[str, ...],
    *,
    method: str,
    value_options: Mapping[str, str],
    flag_options: Mapping[str, tuple[str, bool]],
    required: set[str],
) -> _Request:
    options = _parse_options(
        argv[2:],
        value_options=value_options,
        flag_options=flag_options,
        required=required,
    )
    options.pop("_json", None)
    return method, options


def _parse_options(
    tokens: Sequence[str],
    *,
    value_options: Mapping[str, str],
    flag_options: Mapping[str, tuple[str, bool]],
    required: set[str],
) -> dict[str, object]:
    options: dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        output_key = value_options.get(token)
        if output_key is not None:
            if output_key in options:
                raise ValueError(f"Duplicate Herdr option: {token}")
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"Missing value for Herdr option: {token}")
            options[output_key] = tokens[index + 1]
            index += 2
            continue
        flag = flag_options.get(token)
        if flag is not None:
            output_key, value = flag
            if output_key in options:
                raise ValueError(f"Duplicate Herdr option: {token}")
            options[output_key] = value
            index += 1
            continue
        raise ValueError(f"Unsupported Herdr option: {token}")
    missing = required - options.keys()
    if missing:
        raise ValueError(f"Missing Herdr options: {sorted(missing)!r}")
    return options


def _parse_uint32(value: object) -> int:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdigit()
    ):
        raise ValueError(f"Invalid Herdr line count: {value!r}")
    parsed = int(value)
    if parsed > _UINT32_MAX:
        raise ValueError(f"Invalid Herdr line count: {value!r}")
    return parsed


def _list_command(argv: tuple[str, ...]) -> _Request:
    if len(argv) != _LIST_PARTS:
        raise ValueError(f"Malformed {argv[0]} list command")
    return _LIST_METHODS[(argv[0], argv[1])], {}


def _target_command(argv: tuple[str, ...]) -> _Request:
    if len(argv) != _TARGET_PARTS:
        raise ValueError(f"Malformed {argv[0]} {argv[1]} command")
    method, parameter = _TARGET_METHODS[(argv[0], argv[1])]
    return method, {parameter: _require_identifier(argv[2], parameter)}


def _tab_rename(argv: tuple[str, ...]) -> _Request:
    if len(argv) != _FOUR_PARTS:
        raise ValueError("Malformed tab rename command")
    return "tab.rename", {
        "tab_id": _require_identifier(argv[2], "tab_id"),
        "label": argv[3],
    }


def _pane_layout_or_process_info(argv: tuple[str, ...]) -> _Request:
    if len(argv) != _FOUR_PARTS or argv[2] != "--pane":
        raise ValueError(f"Malformed pane {argv[1]} command")
    method = {
        "layout": "pane.layout",
        "process-info": "pane.process_info",
    }[argv[1]]
    return method, {"pane_id": _require_identifier(argv[3], "pane_id")}


def _pane_send_text(argv: tuple[str, ...]) -> _Request:
    if len(argv) != _FOUR_PARTS:
        raise ValueError("Malformed pane send-text command")
    return "pane.send_text", {
        "pane_id": _require_identifier(argv[2], "pane_id"),
        "text": argv[3],
    }


def _pane_send_keys(argv: tuple[str, ...]) -> _Request:
    if len(argv) < _FOUR_PARTS:
        raise ValueError("Malformed pane send-keys command")
    return "pane.send_keys", {
        "pane_id": _require_identifier(argv[2], "pane_id"),
        "keys": list(argv[3:]),
    }


def _pane_run(argv: tuple[str, ...]) -> _Request:
    if len(argv) != _FOUR_PARTS:
        raise ValueError("Malformed pane run command")
    return "pane.send_input", {
        "pane_id": _require_identifier(argv[2], "pane_id"),
        "text": argv[3],
        "keys": ["enter"],
    }


_COMMAND_HANDLERS: Mapping[tuple[str, str], _Handler] = {
    **dict.fromkeys(_LIST_METHODS, _list_command),
    **dict.fromkeys(_TARGET_METHODS, _target_command),
    ("tab", "rename"): _tab_rename,
    ("pane", "layout"): _pane_layout_or_process_info,
    ("pane", "process-info"): _pane_layout_or_process_info,
    ("pane", "send-text"): _pane_send_text,
    ("pane", "send-keys"): _pane_send_keys,
    ("pane", "run"): _pane_run,
    ("pane", "read"): _pane_read,
    ("pane", "report-metadata"): _pane_report_metadata,
    ("workspace", "create"): lambda argv: _creation_request(
        argv,
        method="workspace.create",
        value_options={"--cwd": "cwd"},
        flag_options={"--focus": ("focus", True), "--no-focus": ("focus", False)},
        required={"cwd", "focus"},
    ),
    ("tab", "create"): lambda argv: _creation_request(
        argv,
        method="tab.create",
        value_options={
            "--cwd": "cwd",
            "--workspace": "workspace_id",
            "--label": "label",
        },
        flag_options={"--focus": ("focus", True), "--no-focus": ("focus", False)},
        required={"cwd", "focus", "workspace_id"},
    ),
    ("worktree", "create"): lambda argv: _creation_request(
        argv,
        method="worktree.create",
        value_options={
            "--cwd": "cwd",
            "--branch": "branch",
            "--path": "path",
            "--workspace": "workspace_id",
            "--label": "label",
        },
        flag_options={
            "--focus": ("focus", True),
            "--no-focus": ("focus", False),
            "--json": ("_json", True),
        },
        required={"cwd", "branch", "path", "focus", "_json"},
    ),
}
