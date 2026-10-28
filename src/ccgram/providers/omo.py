"""omo coding agent provider — omo-ai on top of @code-yeongyu/senpi.

omo is a senpi (pi fork) distribution: it keeps pi's JSONL v3 session format
but stores state under ``~/.omo/agent`` and reports a different TUI title.
Everything non-obvious is therefore inherited: session discovery, the
``--session``/``--continue`` resume flags and the TUI parser all come from
PiProvider; this module only overrides identity, state location and the
detection hints a window started as a JS interpreter needs.

Session path: ``~/.omo/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl``

State directory resolution mirrors omo itself
(``omo-ai/bin/lib/agent-dir.js``): ``OMO_CODING_AGENT_DIR`` →
``SENPI_CODING_AGENT_DIR`` → ``PI_CODING_AGENT_DIR`` → ``~/.omo/agent``.
``OMO_AGENT_DIR`` is *not* part of that chain — it addresses the omo daemon,
not the session store.

Detection: when omo is launched from its bin shim, tmux reports
``pane_current_command=bun`` (the CLI is JS). The basename tables therefore
cannot name it; the pane title set by omo's TUI (``OmO - <project>``) can.
``requires_pane_title_for_detection`` asks the caller to fetch the title for
interpreter processes and ``detect_from_pane_title`` performs the match.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ccgram.providers.pi import PiProvider, provider_sessions_dir

# Mirrors omo's own resolution order (omo-ai/bin/lib/agent-dir.js).
_AGENT_DIR_ENV = "OMO_CODING_AGENT_DIR"
_AGENT_DIR_FALLBACKS = ("SENPI_CODING_AGENT_DIR", "PI_CODING_AGENT_DIR")

#: Interpreters that can host a JS agent CLI.  A pane whose foreground
#: process is one of these needs the pane title to tell the agents apart.
_INTERPRETER_BASENAMES = frozenset({"bun", "bunx", "node", "npx"})

#: omo's TUI window title starts with this marker (observed: "OmO - <project>").
_TITLE_PREFIX = "omo"


class OmoProvider(PiProvider):
    """AgentProvider implementation for the omo coding agent CLI."""

    def _sessions_dir(self) -> Path:
        return provider_sessions_dir(
            (_AGENT_DIR_ENV, *_AGENT_DIR_FALLBACKS), (".omo", "agent")
        )

    _CAPS = replace(
        PiProvider._CAPS,
        name="omo",
        launch_command="omo",
        # omo ships no cc-thingz hook runner; the transcript is the only
        # status source, exactly like a hookless pi.
        supports_hook=False,
        hook_install_managed_by_ccgram=False,
        # `/model` in omo prints a list into the transcript instead of
        # opening a modal picker, so advertising it would offer a picker that
        # cannot be driven.  The owner only needs the default model anyway.
        tui_picker_commands=frozenset(
            name for name in PiProvider._CAPS.tui_picker_commands if name != "model"
        ),
    )

    # ── Detection hints ──────────────────────────────────────────────────

    @staticmethod
    def _is_interpreter(pane_current_command: str) -> bool:
        command = pane_current_command.strip().lower()
        if not command:
            return False
        basename = command.split()[0].rsplit("/", 1)[-1]
        return basename in _INTERPRETER_BASENAMES

    def requires_pane_title_for_detection(self, pane_current_command: str) -> bool:
        """A bun/node pane may be pi, omo or something else — ask the title."""
        return self._is_interpreter(pane_current_command)

    def detect_from_pane_title(
        self, pane_current_command: str, pane_title: str
    ) -> bool:
        """Match omo's TUI title (``OmO - <project>``) on interpreter panes."""
        if not self._is_interpreter(pane_current_command):
            return False
        title = " ".join(pane_title.split()).lower()
        return title.startswith(_TITLE_PREFIX)
