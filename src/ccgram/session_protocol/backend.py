"""Backend port and production adapter of the existing Multiplexer seam."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
from subprocess import TimeoutExpired
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from ..command_catalog import CommandCatalog
from ..multiplexer import agent_status_cache
from ..multiplexer.base import Multiplexer, WindowRef
from ..providers import (
    get_provider_for_window,
    has_yolo_mode,
    resolve_launch_command,
    TerminalMenu,
)
from ..providers.process_detection import classify_provider_from_argv
from ..providers.shell_infra import KNOWN_SHELLS, match_prompt
from ..window_state_ports import identity_state
from ..utils import log_throttled
from ..worktrees import (
    check_worktree_eligibility,
    create_worktree,
    slug_for_path,
    suggest_branch_name,
    validate_branch_name,
    worktree_path_for,
    WorktreeError,
)
from .models import ProtocolError, Status, utc_now

logger = logging.getLogger(__name__)

_CREATION_UNCONFIRMED = (
    "Could not confirm session creation; inspect the terminal before retrying"
)


def _validate_workspace_id(workspace_id: Any) -> None:
    try:
        encoded = workspace_id.encode("utf-8") if isinstance(workspace_id, str) else b""
    except UnicodeError:
        encoded = b""
    if not encoded or b"\x00" in encoded:
        raise ProtocolError(
            400,
            "invalid_request",
            "workspaceId must be a nonempty UTF-8 string without NUL",
        )


@dataclass(frozen=True, slots=True)
class Target:
    id: str
    title: str
    cwd: str
    provider: str
    conversation_id: str
    transcript: Path | None
    pane_id: str
    pane_title: str
    rows: int
    cols: int
    process_id: int
    approval_mode: str
    status: Status
    aliases: tuple[str, ...] = ()
    host: str = ""
    resume_provider: str = ""
    transcript_provider: str = ""
    history_generation: int = 0
    transcript_stamp: tuple[int, int, int, int] | None = None
    transcript_digest: str | None = None
    selected_pane_command: str | None = None
    foreground_argv: tuple[str, ...] = ()
    available: bool = True

    @property
    def foreground_verified(self) -> bool:
        return self.available and bool(self.foreground_argv)

    @property
    def binding(self) -> str:
        return json.dumps(
            [
                self.id.casefold(),
                self.provider,
                self.conversation_id,
                self.pane_id,
                self.process_id,
                self.history_generation,
                self.foreground_argv,
                self.available,
            ],
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class Prompt:
    text: str
    kind: str
    menu: TerminalMenu | None = None

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CreationOptions:
    """Trusted launch options supplied by in-process Telegram workflows."""

    approval_mode: Literal["normal", "yolo"] = "normal"
    resume_id: str | None = None
    continue_session: bool = False


class SessionBackend(Protocol):
    async def targets(self) -> list[Target]: ...
    async def inspect(self, target_id: str) -> Target: ...
    async def prompt(self, target: Target) -> Prompt | None: ...
    async def terminal(self, target: Target, pane: str) -> dict[str, Any]: ...
    async def send_text(
        self, target: Target, text: str, *, enter: bool, raw: bool | None = None
    ) -> bool: ...
    async def send_key(self, target: Target, key: str) -> bool: ...
    async def send_followup(self, target: Target, text: str) -> bool: ...
    async def select_pane(self, target: Target, pane_id: str) -> Target: ...
    async def split(self, target: Target) -> str | None: ...
    async def close(self, target: Target) -> bool: ...
    async def rename(self, target: Target, title: str) -> bool: ...
    async def commands(self, target: Target) -> list[dict[str, str]]: ...
    async def create(
        self, data: dict[str, Any], *, trusted_options: CreationOptions | None = None
    ) -> Target: ...
    async def resume(self, target: Target, conversation_id: str) -> bool: ...


class MultiplexerSessionBackend:
    """No provider is launched through an SDK; every operation uses the seam."""

    def __init__(self, multiplexer: Multiplexer, command_dir: str) -> None:
        self.mux = multiplexer
        self.command_dir = command_dir
        self.catalog = CommandCatalog()
        self.host = socket.gethostname()

    async def targets(self) -> list[Target]:
        windows = await self.mux.list_windows_for_reconciliation()
        if windows is None:
            raise ProtocolError(
                503, "backend_unavailable", "Multiplexer listing could not be confirmed"
            )
        result = []
        for window in windows:
            if (
                window.topic_eligible
                or identity_state.get_identity(window.window_id) is not None
            ):
                result.append(await self._target(window, cache_optional_status=True))
        return result

    async def inspect(self, target_id: str) -> Target:
        windows = await self.mux.list_windows_for_reconciliation()
        if windows is None:
            raise ProtocolError(
                503, "backend_unavailable", "Multiplexer is unavailable"
            )
        window = next((w for w in windows if w.matches(target_id)), None)
        if window is None:
            raise ProtocolError(404, "not_found", "Session target no longer exists")
        return await self._target(window)

    async def _target(
        self, window: WindowRef, *, cache_optional_status: bool = False
    ) -> Target:
        identity = identity_state.get_identity(window.window_id)
        fg = await self.mux.foreground(window.window_id)
        provider_name = classify_provider_from_argv(fg.argv) if fg else ""
        provider_name = provider_name or (
            identity.provider_name if identity and fg is None else "unknown"
        )
        provider = get_provider_for_window(
            window.window_id, provider_name=provider_name or "shell"
        )
        resume_provider = (
            identity.provider_name
            if identity
            and provider_name == "shell"
            and fg is not None
            and len(fg.argv) == 1
            and fg.argv[0].rsplit("/", 1)[-1].lstrip("-") in KNOWN_SHELLS
            and identity.provider_name not in {"shell", "unknown"}
            else ""
        )
        if (
            resume_provider
            and not get_provider_for_window(
                window.window_id, provider_name=resume_provider
            ).capabilities.supports_resume
        ):
            resume_provider = ""
        history_provider = resume_provider or provider_name
        conversation_id = (
            identity.session_id
            if identity and identity.provider_name == history_provider
            else ""
        )
        transcript = identity.transcript_path if conversation_id and identity else None
        if history_provider not in {"claude", "codex", "gemini", "pi", "antigravity"}:
            transcript = None
        panes = await self.mux.list_panes(window.window_id)
        active = next((p for p in panes if p.active), None)
        if resume_provider:
            screen = (
                await self.mux.capture_pane_by_id(
                    active.pane_id, window_id=window.window_id
                )
                if active
                else await self.mux.capture_pane(window.window_id)
            )
            lines = (screen or "").rstrip().splitlines()
            marker = match_prompt(lines[-1]) if lines else None
            if marker is None or marker.trailing_text.strip():
                resume_provider = ""
        try:
            if cache_optional_status and self.mux.capabilities.native_agent_status:
                native = await agent_status_cache.get_status_or_probe(
                    window.window_id,
                    lambda: self.mux.agent_status(window.window_id),
                    # Push coverage is Telegram-bound, not guaranteed for every
                    # target returned by the shared core.
                    cache_positive=False,
                )
            else:
                native = await self.mux.agent_status(window.window_id)
        except Exception as exc:  # noqa: BLE001  # optional enrichment is logged and discarded
            # Native status is optional; keep building this target and let the
            # next refresh retry the probe.
            log_throttled(
                logger,
                f"session-protocol-agent-status:{window.window_id}",
                "Native agent-status probe failed for window %s: %s",
                window.window_id,
                exc,
                level="warning",
            )
            native = None
        now = utc_now()
        status = Status("unknown", "", "screen", now)
        if native and native.state in {
            "starting",
            "working",
            "blocked",
            "done",
            "idle",
            "dead",
            "unknown",
        }:
            status = Status(native.state, native.custom_status, "native", now)
        elif provider_name != "unknown":
            screen = await self.mux.capture_pane(window.window_id)
            if screen:
                title = (
                    await self.mux.get_pane_title(window.window_id)
                    if provider.capabilities.uses_pane_title
                    else ""
                )
                parsed = provider.parse_terminal_status(screen, pane_title=title)
                status = Status(
                    "working" if parsed and not parsed.is_interactive else "idle",
                    parsed.display_label if parsed else "",
                    "screen",
                    now,
                )
        return Target(
            window.window_id,
            window.window_name,
            window.cwd,
            provider_name or "unknown",
            conversation_id,
            transcript,
            active.pane_id if active else "",
            active.command if active else "",
            active.height if active else window.pane_height,
            active.width if active else window.pane_width,
            fg.pid if fg else 0,
            identity.approval_mode if identity else "normal",
            status,
            window.alias_window_ids,
            self.host,
            resume_provider,
            history_provider,
            foreground_argv=tuple(fg.argv) if fg else (),
        )

    async def prompt(self, target: Target) -> Prompt | None:
        text = (
            await self.mux.capture_pane_by_id(target.pane_id, window_id=target.id)
            if target.pane_id
            else await self.mux.capture_pane(target.id)
        )
        if not text:
            raise ProtocolError(
                503, "backend_unavailable", "Cannot verify the active pane"
            )
        provider = get_provider_for_window(target.id, provider_name=target.provider)
        title = (
            await self.mux.get_pane_title(target.id)
            if provider.capabilities.uses_pane_title
            and target.selected_pane_command is None
            else ""
        )
        status = provider.parse_terminal_status(text, pane_title=title)
        if status and status.is_interactive and not status.ui_advisory:
            return Prompt(status.raw_text, status.ui_type or "prompt", menu=status.menu)
        return None

    async def terminal(self, target: Target, pane: str) -> dict[str, Any]:
        if pane not in {"p0", "", target.pane_id}:
            raise ProtocolError(404, "not_found", "Pane not found")
        text = (
            await self.mux.capture_pane_by_id(
                target.pane_id, window_id=target.id, with_ansi=True
            )
            if target.pane_id
            else await self.mux.capture_pane(target.id, with_ansi=True)
        )
        if text is None:
            raise ProtocolError(503, "backend_unavailable", "Pane capture failed")
        return {
            "pane": "p0",
            "rows": target.rows,
            "cols": target.cols,
            "ansi": text,
            "hash": hashlib.sha256(text.encode()).hexdigest(),
        }

    async def send_text(
        self,
        target: Target,
        text: str,
        *,
        enter: bool,
        raw: bool | None = None,
    ) -> bool:
        await self._verify_selected_pane(target)
        raw_input = target.provider in {"shell", "unknown"} if raw is None else raw
        if target.pane_id:
            return await self.mux.send_to_pane(
                target.pane_id,
                text,
                enter=enter,
                literal=True,
                window_id=target.id,
                raw=raw_input,
            )
        return await self.mux.send(
            target.id,
            text,
            enter=enter,
            literal=True,
            raw=raw_input,
        )

    async def send_key(self, target: Target, key: str) -> bool:
        await self._verify_selected_pane(target)
        if target.pane_id:
            return await self.mux.send_to_pane(
                target.pane_id, key, enter=False, literal=False, window_id=target.id
            )
        return await self.mux.send(target.id, key, enter=False, literal=False, raw=True)

    async def send_followup(self, target: Target, text: str) -> bool:
        if target.provider != "pi":
            raise ProtocolError(
                422,
                "unsupported_action",
                "Follow-up input is supported for Pi sessions",
            )
        if not await self.send_text(target, text, enter=False):
            return False
        await asyncio.sleep(0.5)
        return await self.send_key(target, "M-Enter")

    async def select_pane(self, target: Target, pane_id: str) -> Target:
        panes = await self.mux.list_panes(target.id)
        pane = next(
            (candidate for candidate in panes if candidate.pane_id == pane_id), None
        )
        if pane is None:
            raise ProtocolError(
                409, "binding_conflict", "Selected pane no longer exists"
            )
        return replace(
            target,
            pane_id=pane.pane_id,
            pane_title=pane.command,
            rows=pane.height,
            cols=pane.width,
            selected_pane_command=pane.command,
        )

    async def _verify_selected_pane(self, target: Target) -> None:
        foreground = await self.mux.foreground(target.id)
        if foreground is None or not foreground.argv:
            raise ProtocolError(
                503, "backend_unavailable", "Foreground process could not be verified"
            )
        if (foreground.pid, tuple(foreground.argv)) != (
            target.process_id,
            target.foreground_argv,
        ):
            raise ProtocolError(409, "binding_conflict", "Foreground process changed")
        if target.selected_pane_command is None:
            return
        panes = await self.mux.list_panes(target.id)
        pane = next(
            (candidate for candidate in panes if candidate.pane_id == target.pane_id),
            None,
        )
        if pane is None or pane.command != target.selected_pane_command:
            raise ProtocolError(409, "binding_conflict", "Selected pane changed")

    async def split(self, target: Target) -> str | None:
        fresh = await self.inspect(target.id)
        if (fresh.process_id, fresh.pane_id, fresh.foreground_argv) != (
            target.process_id,
            target.pane_id,
            target.foreground_argv,
        ):
            raise ProtocolError(409, "binding_conflict", "Session binding changed")
        return await self.mux.split_window(target.id)

    async def close(self, target: Target) -> bool:
        return await self.mux.kill_window(target.id)

    async def rename(self, target: Target, title: str) -> bool:
        return await self.mux.rename_window(target.id, title)

    async def commands(self, target: Target) -> list[dict[str, str]]:
        provider = get_provider_for_window(target.id, provider_name=target.provider)
        commands = await asyncio.to_thread(
            self.catalog.get_provider_commands, provider, self.command_dir
        )
        return [
            {"name": c.name, "description": c.description, "source": c.source}
            for c in commands
        ]

    async def resume(self, target: Target, conversation_id: str) -> bool:
        fresh = await self.inspect(target.id)
        if (
            fresh.provider != "shell"
            or not fresh.resume_provider
            or conversation_id != fresh.conversation_id
            or (fresh.process_id, fresh.pane_id, fresh.foreground_argv)
            != (target.process_id, target.pane_id, target.foreground_argv)
        ):
            raise ProtocolError(
                422,
                "unsupported_action",
                "Resume requires a verified shell and its previous conversation",
            )
        provider = get_provider_for_window(
            target.id, provider_name=target.resume_provider
        )
        if not provider.capabilities.supports_resume:
            raise ProtocolError(
                422, "unsupported_action", "Provider does not support resume"
            )
        command = resolve_launch_command(target.resume_provider, approval_mode="normal")
        arguments = provider.make_launch_args(resume_id=conversation_id)
        return await self.send_text(
            target, f"{command} {arguments}".strip(), enter=True
        )

    async def create(
        self, data: dict[str, Any], *, trusted_options: CreationOptions | None = None
    ) -> Target:
        directory_value = data.get("directory", data.get("cwd"))
        if not isinstance(directory_value, str):
            raise ProtocolError(400, "invalid_request", "directory must be a path")
        directory = Path(directory_value).expanduser()
        if not directory.is_absolute() or not directory.is_dir():
            raise ProtocolError(
                400,
                "invalid_request",
                "directory must be an existing absolute directory",
            )
        if trusted_options is None and data.get("mode", "normal") != "normal":
            raise ProtocolError(
                422,
                "unsupported_action",
                "Remote creation supports the normal approval mode only",
            )
        if "workspaceId" in data:
            _validate_workspace_id(data["workspaceId"])
        provider = data["provider"]
        candidate = get_provider_for_window("", provider_name=provider)
        if candidate.capabilities.name != provider:
            raise ProtocolError(400, "invalid_request", "Unknown provider")
        approval_mode, command, agent_args = self._launch_options(
            provider, candidate, data, trusted_options
        )
        try:
            worktree_path = directory
            if data.get("worktree"):
                created = await self._create_worktree(directory, data, command)
                if isinstance(created, Target):
                    if approval_mode == "yolo":
                        identity_state.set_window_approval_mode(
                            created.id, approval_mode
                        )
                    return await self.inspect(created.id)
                worktree_path = created
            return await self._create_target(
                worktree_path,
                command,
                agent_args,
                approval_mode,
                data.get("workspaceId"),
            )
        except (OSError, RuntimeError, TimeoutExpired) as exc:
            raise ProtocolError(
                503, "backend_unavailable", _CREATION_UNCONFIRMED
            ) from exc

    @staticmethod
    def _launch_options(
        provider: str,
        candidate: Any,
        data: dict[str, Any],
        options: CreationOptions | None,
    ) -> tuple[str, str, str]:
        approval_mode = options.approval_mode if options else "normal"
        if approval_mode not in {"normal", "yolo"}:
            raise ProtocolError(400, "invalid_request", "Invalid approval mode")
        if approval_mode == "yolo" and not has_yolo_mode(provider):
            raise ProtocolError(
                422, "unsupported_action", "Provider does not support YOLO mode"
            )
        resume_id = options.resume_id if options else None
        continue_session = options.continue_session if options else False
        if resume_id and continue_session:
            raise ProtocolError(
                400, "invalid_request", "Choose resume_id or continue_session"
            )
        if (resume_id or continue_session) and data.get("worktree"):
            raise ProtocolError(
                422, "unsupported_action", "Cannot resume into a new worktree"
            )
        if resume_id and not candidate.capabilities.supports_resume:
            raise ProtocolError(
                422, "unsupported_action", "Provider does not support resume"
            )
        if continue_session and not candidate.capabilities.supports_continue:
            raise ProtocolError(
                422, "unsupported_action", "Provider does not support continue"
            )
        command = resolve_launch_command(provider, approval_mode=approval_mode)
        arguments = candidate.make_launch_args(
            resume_id=resume_id, use_continue=continue_session
        )
        return approval_mode, command, arguments

    async def _create_target(
        self,
        worktree_path: Path,
        command: str,
        agent_args: str,
        approval_mode: str,
        workspace_id: str | None,
    ) -> Target:
        if self.mux.capabilities.native_topic_targets:
            created = await self.mux.create_topic_target(
                str(worktree_path),
                launch_command=command,
                workspace_id=workspace_id,
                agent_args=agent_args,
            )
            target_id = created.target_id
            success = bool(target_id)
        else:
            success, _, _, target_id = await self.mux.create_window(
                str(worktree_path),
                launch_command=command,
                workspace_id=workspace_id,
                agent_args=agent_args,
            )
        if success and approval_mode == "yolo":
            identity_state.set_window_approval_mode(target_id, approval_mode)
        return await self._created_target(success, target_id)

    async def _create_worktree(
        self, directory: Path, data: dict[str, Any], command: str
    ) -> Path | Target:
        if self.mux.capabilities.native_worktrees and data.get("workspaceId"):
            raise ProtocolError(
                422,
                "unsupported_action",
                "Native worktrees cannot select an existing workspace",
            )
        eligibility = await asyncio.to_thread(check_worktree_eligibility, directory)
        if not eligibility.eligible or eligibility.repo_path is None:
            raise ProtocolError(
                400, "invalid_request", "Directory is not eligible for a worktree"
            )
        branch = (
            data["branch"]
            if "branch" in data
            else await asyncio.to_thread(
                suggest_branch_name, None, eligibility.repo_path
            )
        )
        if not isinstance(branch, str) or not await asyncio.to_thread(
            validate_branch_name, branch
        ):
            raise ProtocolError(400, "invalid_request", "Invalid worktree branch")
        worktree_path = worktree_path_for(eligibility.repo_path, slug_for_path(branch))
        if self.mux.capabilities.native_worktrees:
            ok, _, _, target_id = await self.mux.create_worktree_window(
                str(eligibility.repo_path),
                str(worktree_path),
                branch,
                launch_command=command,
            )
            return await self._created_target(ok, target_id)
        try:
            await asyncio.to_thread(
                create_worktree, eligibility.repo_path, branch, worktree_path
            )
        except WorktreeError as exc:
            raise ProtocolError(
                503, "backend_unavailable", _CREATION_UNCONFIRMED
            ) from exc
        return worktree_path

    async def _created_target(self, ok: bool, target_id: str) -> Target:
        if not ok or not target_id:
            raise ProtocolError(
                503,
                "backend_unavailable",
                _CREATION_UNCONFIRMED,
            )
        try:
            return await self.inspect(target_id)
        except ProtocolError as exc:
            raise ProtocolError(
                503, "backend_unavailable", _CREATION_UNCONFIRMED
            ) from exc
