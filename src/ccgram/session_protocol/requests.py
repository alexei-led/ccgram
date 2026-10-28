"""Open requests keyed by control session, with server-owned response evidence."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .backend import Prompt
from .events import EventHub
from .messages import tool_kind
from .models import CLOSE_REASONS, Request, RequestOption, utc_now
from .store import new_id

_HOOK_MATCH_TTL = 30
_MIN_MENU_OPTIONS = 2
_NUMBERED = re.compile(r"^\s*[❯>●]?\s*([1-8])[.)]\s+(.+?)\s*$")


def _choices(text: str) -> tuple[tuple[str, str], ...]:
    choices = []
    for line in text.splitlines():
        match = _NUMBERED.fullmatch(line)
        if match:
            choices.append((match[1], match[2]))
    if len(choices) < _MIN_MENU_OPTIONS or [k for k, _ in choices] != [
        str(i + 1) for i in range(len(choices))
    ]:
        return ()
    return tuple(choices)


@dataclass(frozen=True, slots=True)
class HookIdentity:
    conversation_id: str
    kind: str
    tool_name: str
    input_hash: str
    timestamp: float


def hook_input_hash(tool_input: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PendingHook:
    conversation_id: str
    kind: str
    tool_name: str
    tool_input: dict[str, Any]
    timestamp: float

    @property
    def identity(self) -> HookIdentity:
        return HookIdentity(
            self.conversation_id,
            self.kind,
            self.tool_name,
            hook_input_hash(self.tool_input),
            self.timestamp,
        )


@dataclass(frozen=True, slots=True)
class OpenRequest:
    request: Request
    binding_version: int
    fingerprint: str
    option_keys: dict[str, str]
    answer_keys: dict[str, str]
    hook: PendingHook | None = None
    # Unmatched evidence also invalidates an in-flight dismissal when replaced.
    observed_hook: PendingHook | None = None


def _question_label(hook: PendingHook, label: str) -> str | None:
    question = hook.tool_input["questions"][0]
    labels = [
        o["label"]
        for o in question.get("options", [])
        if isinstance(o, dict) and isinstance(o.get("label"), str)
    ]
    if len(labels) != len(set(labels)):
        return None
    return label if label in labels else None


def _response_options(
    hook: PendingHook | None, prompt: Prompt
) -> tuple[list[RequestOption], dict[str, str], dict[str, str], dict[str, Any] | None]:
    options: list[RequestOption] = []
    keys: dict[str, str] = {}
    answers: dict[str, str] = {}
    response_schema = None
    if hook:
        for key, label in _choices(prompt.text):
            lower = label.casefold()
            if any(
                word in lower
                for word in (
                    "always",
                    "session",
                    "don't ask",
                    "don’t ask",
                    "remember",
                )
            ):
                continue
            if hook.kind == "question":
                matched = _question_label(hook, label)
                if matched:
                    if matched in answers:
                        return [], {}, {}, None
                    answers[matched] = key
                    options.append(RequestOption(key, label, "allow_once"))
                    keys[key] = key
            elif lower.startswith(("yes", "allow", "approve", "accept")):
                options.append(RequestOption(key, label, "allow_once"))
                keys[key] = key
            elif lower.startswith(("no", "deny", "reject", "cancel")):
                options.append(RequestOption(key, label, "reject_once"))
                keys[key] = key
        if answers:
            response_schema = {"type": "string", "enum": list(answers)}
    return options, keys, answers, response_schema


class RequestStore:
    def __init__(self, hub: EventHub) -> None:
        self.hub = hub
        self._open: dict[str, OpenRequest] = {}
        self._hooks: dict[str, PendingHook] = {}
        self._hook_prompts: dict[str, tuple[PendingHook, str, int]] = {}
        self._consumed_hooks: list[tuple[str, HookIdentity]] = []
        self._latest_hooks: dict[str, HookIdentity] = {}
        self._deferred: dict[str, int] = {}

    def get(self, session_id: str) -> OpenRequest | None:
        return self._open.get(session_id)

    def is_current(self, session_id: str, request: OpenRequest) -> bool:
        return (
            not self.has_deferred(session_id)
            and self.get(session_id) is request
            and (self._hooks.get(session_id) == request.observed_hook)
        )

    def defer(self, session_id: str) -> None:
        self._deferred[session_id] = self._deferred.get(session_id, 0) + 1

    def resolve(self, session_id: str) -> None:
        count = self._deferred[session_id]
        if count == 1:
            self._deferred.pop(session_id)
        else:
            self._deferred[session_id] = count - 1

    def has_deferred(self, session_id: str) -> bool:
        return session_id in self._deferred

    def observe_hook(self, session_id: str, hook: PendingHook) -> None:
        if self.observe_identity(session_id, hook.identity):
            self._hooks[session_id] = hook

    def observe_identity(self, session_id: str, hook: HookIdentity) -> bool:
        if abs(time.time() - hook.timestamp) > _HOOK_MATCH_TTL or self._is_consumed(
            session_id, hook
        ):
            return False
        pending = self._latest_hooks.get(session_id)
        if pending is not None and pending != hook:
            if hook.timestamp < pending.timestamp:
                return False
            self.close(session_id, "vanished")
        self._latest_hooks[session_id] = hook
        return True

    def discard_identity(self, session_id: str, hook: HookIdentity) -> None:
        if self._latest_hooks.get(session_id) == hook:
            self.close(session_id, "vanished")

    def close(
        self,
        session_id: str,
        reason: str,
        *,
        preserve_identity: HookIdentity | None = None,
    ) -> None:
        if reason not in CLOSE_REASONS:
            raise ValueError(f"Unknown request close reason: {reason}")
        current = self._open.pop(session_id, None)
        pending = self._hooks.pop(session_id, None)
        if pending is not None and pending.identity != preserve_identity:
            self._consume(session_id, pending.identity)
        bound = self._hook_prompts.pop(session_id, None)
        if bound:
            self._consume(session_id, bound[0].identity)
        latest = self._latest_hooks.get(session_id)
        if latest is not None and latest != preserve_identity:
            self._consume(session_id, latest)
        if pending is not None and pending.identity == preserve_identity:
            self._hooks[session_id] = pending
        if current:
            self.hub.publish(
                "request.closed",
                session_id,
                {"requestId": current.request.id, "reason": reason},
            )

    def _consume(self, session_id: str, hook: HookIdentity) -> None:
        if not self._is_consumed(session_id, hook):
            self._consumed_hooks.append((session_id, hook))

    def _is_consumed(self, session_id: str, hook: HookIdentity) -> bool:
        now = time.time()
        self._consumed_hooks = [
            (sid, consumed)
            for sid, consumed in self._consumed_hooks
            if abs(now - consumed.timestamp) <= _HOOK_MATCH_TTL
        ]
        return (session_id, hook) in self._consumed_hooks

    def reconcile(
        self, session_id: str, conversation_id: str, version: int, prompt: Prompt | None
    ) -> OpenRequest | None:
        current = self.get(session_id)
        if prompt is None:
            if current:
                pending = self._latest_hooks.get(session_id)
                bound_identity = (
                    current.hook.identity if current.hook is not None else None
                )
                self.close(
                    session_id,
                    "vanished",
                    preserve_identity=pending if pending != bound_identity else None,
                )
            return None
        hook = self._hooks.get(session_id)
        if hook is not None and self._is_consumed(session_id, hook.identity):
            hook = None
        bound = self._hook_prompts.get(session_id)
        if bound and bound[0] == hook and bound[1:] != (prompt.fingerprint, version):
            self._consume(session_id, bound[0].identity)
            self._hook_prompts.pop(session_id)
            hook = None
        # Native selectors may resemble questions, but never authorize a tool answer.
        structured = prompt.menu is None and (
            (hook is not None and bound == (hook, prompt.fingerprint, version))
            or self._match(hook, conversation_id, prompt)
        )
        if (
            current
            and current.binding_version == version
            and current.fingerprint == prompt.fingerprint
            and current.hook == (hook if structured else None)
            and current.observed_hook == self._hooks.get(session_id)
        ):
            return current
        if current:
            if current.hook is not None:
                self._consume(session_id, current.hook.identity)
                self._hook_prompts.pop(session_id, None)
            # Retain a matching observation when promoting screen -> hook.
            self._open.pop(session_id)
            self.hub.publish(
                "request.closed",
                session_id,
                {"requestId": current.request.id, "reason": "vanished"},
            )
        options, keys, answers, response_schema = _response_options(
            hook if structured else None, prompt
        )
        request = Request(
            new_id("r_"),
            session_id,
            hook.kind if structured and hook else "prompt",
            "hook" if structured else "screen",
            prompt.text,
            {
                "name": hook.tool_name,
                "kind": tool_kind(hook.tool_name),
                "input": hook.tool_input,
            }
            if structured and hook
            else None,
            tuple(options),
            response_schema,
            utc_now(),
        )
        if structured and hook:
            self._hook_prompts[session_id] = (hook, prompt.fingerprint, version)
        opened = OpenRequest(
            request,
            version,
            prompt.fingerprint,
            keys,
            answers,
            hook if structured else None,
            self._hooks.get(session_id),
        )
        self._open[session_id] = opened
        self.hub.publish("request.opened", session_id, request)
        return opened

    @staticmethod
    def _match(hook: PendingHook | None, conversation_id: str, prompt: Prompt) -> bool:
        if (
            hook is None
            or hook.conversation_id != conversation_id
            or abs(time.time() - hook.timestamp) > _HOOK_MATCH_TTL
        ):
            return False
        if hook.kind == "question":
            questions = hook.tool_input.get("questions")
            if not isinstance(questions, list) or len(questions) != 1:
                return False
            question = questions[0]
            return (
                isinstance(question, dict)
                and bool(question.get("question"))
                and question.get("multiSelect", False) is False
                and not any(
                    marker in prompt.text.casefold()
                    for marker in (
                        "select all",
                        "multi-select",
                        "multiselect",
                        "[ ]",
                        "[x]",
                        "☐",
                        "☑",
                    )
                )
                and str(question["question"]).strip()
                in {line.strip() for line in prompt.text.splitlines()}
            )
        value = next(
            (
                hook.tool_input[k]
                for k in ("command", "file_path", "path")
                if isinstance(hook.tool_input.get(k), str) and hook.tool_input[k]
            ),
            None,
        )
        lines = {line.strip() for line in prompt.text.splitlines()}
        lines.update(
            line.removeprefix(hook.tool_name + ":").strip() for line in tuple(lines)
        )
        lines.update(
            line.removeprefix(hook.tool_name + " ").strip() for line in tuple(lines)
        )
        return bool(value and value.strip() in lines and _choices(prompt.text))
