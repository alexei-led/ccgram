"""Clear an agent's first-run approval screens without a human.

A fresh agent CLI can stop before its prompt — Claude asks to trust the
folder, Codex asks about newly installed hooks — and the topic then binds to
a session that never answers. Two layers, cheapest first:

1. :func:`preseed_trust` writes the record the dialog itself would write, so
   the screen never appears (Claude: ``~/.claude.json``; Codex: the
   ``projects`` table in ``config.toml``). Deterministic, no keystrokes, no
   timing.
2. :func:`answer_first_run_prompt` captures the pane and, only on the exact
   fingerprint a provider declares in its capabilities, sends the declared
   keys. Unknown screens are reported, never guessed at.

Seeding runs in the bot process, so it uses the bot's environment. A window
whose shell was started with a different profile (a reused multiplexer
session, for example) is seeded in the wrong place: the write is additive and
harmless there, and the pane fingerprint above still clears the screen once
the agent is up.

Both layers are best-effort: every failure is logged and swallowed, because a
readiness nicety must never fail a launch. Opt out globally with
``CCGRAM_AUTOSTART_TRUST=off`` or per agent with
``CCGRAM_<AGENT>_AUTOSTART=off``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tomllib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session_protocol.telegram import TelegramSessionAdapter

logger = structlog.get_logger()

# Pane polling for a declared first-run fingerprint. Bounded tight: a
# declared screen renders with the agent's first frame, so seconds are plenty
# and a longer wait only delays the launch's own readiness timeout.
FIRST_RUN_TIMEOUT_S = 8.0
FIRST_RUN_POLL_S = 0.4

_CODEX_TRUST_LINE = 'trust_level = "trusted"'


def autostart_enabled(provider_name: str) -> bool:
    """Whether readiness handling runs for *provider_name*."""
    # Lazy: Config requires the runtime environment.
    from .config import config

    if not config.autostart_trust:
        # The global switch is an off switch: a per-agent "on" cannot override
        # an operator who turned the whole layer off.
        return False
    override = os.getenv(f"CCGRAM_{provider_name.upper()}_AUTOSTART")
    if override is not None and override.strip():
        return override.strip().lower() in ("1", "true", "yes", "on")
    return True


def _write_atomic(path: Path, text: str) -> bool:
    """Replace *path* with *text* atomically; False on any filesystem error.

    A symlinked config (a dotfile-managed ``~/.codex/config.toml``) is written
    through its target: replacing the link itself would silently detach the
    file future managed updates write to. A failed write never leaves the
    temporary file behind.
    """
    target = path
    try:
        if path.is_symlink():
            target = path.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            handle.write(text)
            tmp = Path(handle.name)
    except OSError as exc:
        logger.debug("could not prepare write for %s: %s", path, exc)
        return False
    try:
        tmp.replace(target)
    except OSError as exc:
        logger.debug("could not write %s: %s", path, exc)
        tmp.unlink(missing_ok=True)
        return False
    return True


def _claude_config_file() -> Path | None:
    """Locate Claude's JSON config, or None when no candidate exists.

    A configured profile wins: with ``CLAUDE_CONFIG_DIR`` set, the agent reads
    that profile, so seeding trust into the home profile would change a file
    the launch never consults and leave the real one untrusted.
    """
    # Lazy: Config reads the environment.
    from .config import config

    home_profile = Path.home() / ".claude.json"
    config_dir = config.claude_config_dir
    if os.getenv("CLAUDE_CONFIG_DIR", "").strip():
        # The configured profile is the only profile the launch reads, so it is
        # the only candidate. A parent or home fallback would silently seed an
        # unrelated profile (home/.claude.json is the parent of ~/.claude) and
        # leave the real one untrusted.
        candidates = [config_dir / ".claude.json"]
    else:
        candidates = [
            home_profile,
            config_dir / ".claude.json",
            config_dir.parent / ".claude.json",
        ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def preseed_claude_trust(cwd: Path) -> bool:
    """Mark *cwd* as a trusted Claude project in ``~/.claude.json``."""
    config_file = _claude_config_file()
    if config_file is None:
        return False
    try:
        data = json.loads(config_file.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("claude config unreadable (%s): %s", config_file, exc)
        return False
    if not isinstance(data, dict):
        return False
    projects = data.setdefault("projects", {})
    if not isinstance(projects, dict):
        return False
    entry = projects.get(str(cwd))
    if not isinstance(entry, dict):
        entry = {}
        projects[str(cwd)] = entry
    if entry.get("hasTrustDialogAccepted") is True:
        return False
    entry["hasTrustDialogAccepted"] = True
    if not _write_atomic(config_file, json.dumps(data, indent=2) + "\n"):
        return False
    logger.info("Seeded Claude folder trust for %s in %s", cwd, config_file)
    return True


def _codex_config_file() -> Path:
    """Codex config path: ``$CODEX_HOME/config.toml`` or ``~/.codex/config.toml``."""
    home = os.getenv("CODEX_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".codex"
    return base / "config.toml"


def preseed_codex_trust(cwd: Path) -> bool:
    """Add a trusted project entry to Codex's ``config.toml``.

    Appends one ``[projects."<cwd>"]`` section and leaves the rest of the
    file byte-identical, so a hand-edited config keeps its layout. Both the
    existing file and the result are parsed with tomllib: a config this code
    cannot read is never modified, and a duplicate table (which TOML forbids)
    can never be produced by a quoting mismatch.
    """
    config_file = _codex_config_file()
    try:
        existing = (
            config_file.read_text(encoding="utf-8") if config_file.is_file() else ""
        )
    except OSError as exc:
        logger.debug("codex config unreadable: %s", exc)
        return False
    state = _codex_project_trust(existing, cwd)
    if state is None:
        logger.warning(
            "Codex config %s is not valid TOML or not a table; trust not seeded",
            config_file,
        )
        return False
    if state:
        return False
    key = _toml_quote(str(cwd))
    addition = f"\n[projects.{key}]\n{_CODEX_TRUST_LINE}\n"
    body = existing + ("" if existing.endswith("\n") or not existing else "\n")
    updated = body + addition
    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        logger.warning(
            "Refusing to write an unparseable Codex config for %s: %s", cwd, exc
        )
        return False
    if not _write_atomic(config_file, updated):
        return False
    logger.info("Seeded Codex project trust for %s in %s", cwd, config_file)
    return True


def _toml_quote(text: str) -> str:
    """Quote *text* as a TOML basic string (backslash and quote escaped)."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _codex_project_trust(existing: str, cwd: Path) -> bool | None:
    """Whether Codex already trusts *cwd*, or None when the file is unusable.

    Parsing rather than matching header text is what makes a config written
    with single quotes, extra whitespace, or an inline table count as already
    trusted instead of gaining a duplicate ``[projects...]`` table, which
    TOML rejects outright.
    """
    if not existing.strip():
        return False
    try:
        data = tomllib.loads(existing)
    except tomllib.TOMLDecodeError:
        return None
    projects = data.get("projects", {})
    if not isinstance(projects, dict):
        return None
    entry = projects.get(str(cwd))
    if entry is None:
        return False
    if not isinstance(entry, dict):
        return None
    # Any existing entry already defines the table: appending a second one is
    # invalid TOML, and an explicit non-trusted value is the operator's
    # decision, not something to rewrite behind their back.
    return True


def preseed_trust(provider_name: str, cwd: Path) -> bool:
    """Best-effort trust seeding for *provider_name* in *cwd*.

    Skipped when the provider has a launch-command override: an override can
    point the agent at its own profile (``CODEX_HOME=... codex``), which this
    process cannot resolve, so seeding the bot's own profile would write to a
    file the launch never reads.
    """
    if not autostart_enabled(provider_name):
        return False
    if os.environ.get(f"CCGRAM_{provider_name.upper()}_COMMAND", "").strip():
        logger.debug(
            "Skipping trust seeding for %s: launch command override in use",
            provider_name,
        )
        return False
    seeder = _PRESEEDERS.get(provider_name)
    if seeder is None:
        return False
    try:
        return seeder(Path(cwd))
    except Exception as exc:  # noqa: BLE001 - never fail a launch for this
        logger.warning("Trust seeding failed for %s in %s: %s", provider_name, cwd, exc)
        return False


_PRESEEDERS = {
    "claude": preseed_claude_trust,
    "codex": preseed_codex_trust,
}


def _first_run_recipe(provider_name: str):
    """The provider's declared first-run fingerprint, if any."""
    try:
        # Lazy: providers package heavy bootstrap (process detection, registry)
        # and this module is imported by the launch path only. is_known_provider
        # registers the built-ins first; the raw registry is empty until
        # something asks for a provider.
        from .providers import is_known_provider, registry

        if not is_known_provider(provider_name):
            return None
        return registry.get(provider_name).capabilities.first_run
    except Exception:  # noqa: BLE001 - never fail a launch for a recipe lookup
        return None


def _matches(pane_text: str, needles: tuple[str, ...]) -> bool:
    lowered = pane_text.lower()
    return bool(needles) and all(needle.lower() in lowered for needle in needles)


def _settled(pane_text: str, settled: tuple[str, ...]) -> bool:
    lowered = pane_text.lower()
    return any(token.lower() in lowered for token in settled)


async def _capture_pane(window_id: str) -> str | None:
    """Read the target's pane text through the active backend."""
    # Lazy: the multiplexer proxy resolves the active backend.
    from .multiplexer import multiplexer

    text = await multiplexer.capture_pane(window_id)
    return text if isinstance(text, str) else None


async def _send_answer(
    window_id: str, provider_name: str, recipe: Any, adapter: TelegramSessionAdapter
) -> bool:
    """Send the declared keys once; False when the send fails or raises."""
    try:
        ok, message = await adapter.send_keys(window_id, list(recipe.keys))
    except Exception as exc:  # noqa: BLE001 - best effort, never fail a launch
        logger.warning(
            "First-run answer for %s in %s raised: %s", provider_name, window_id, exc
        )
        return False
    if not ok:
        logger.warning(
            "First-run answer for %s in %s failed: %s",
            provider_name,
            window_id,
            message,
        )
        return False
    logger.info(
        "Answered %s first-run screen in %s (%s)",
        provider_name,
        window_id,
        recipe.needles[0],
    )
    return True


async def answer_first_run_prompt(
    window_id: str,
    provider_name: str,
    *,
    adapter: TelegramSessionAdapter,
    timeout: float = FIRST_RUN_TIMEOUT_S,
    poll: bool = True,
) -> bool:
    """Answer a declared first-run screen; True when keys were sent.

    Only the exact fingerprint the provider declares triggers a keypress.
    With ``poll=False`` the pane is read once and the call costs a single
    capture, which is what the launch path uses on its first attempt: the
    screen is either up already or the caller's own readiness wait will bring
    it back here. Never raises: the caller is mid-launch.
    """
    if not autostart_enabled(provider_name):
        return False
    recipe = _first_run_recipe(provider_name)
    if recipe is None:
        return False
    deadline = asyncio.get_running_loop().time() + (timeout if poll else 0.0)
    while True:
        try:
            pane_text = await _capture_pane(window_id)
        except Exception as exc:  # noqa: BLE001 - best effort, never fail a launch
            logger.warning(
                "First-run check for %s in %s could not read the pane: %s",
                provider_name,
                window_id,
                exc,
            )
            return False
        if pane_text and _matches(pane_text, recipe.needles):
            return await _send_answer(window_id, provider_name, recipe, adapter)
        if pane_text and _settled(pane_text, recipe.settled):
            return False
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(FIRST_RUN_POLL_S)
