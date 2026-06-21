# Shell provider on herdr (tty → foreground seam)

## Overview

The shell provider mostly rides the multiplexer seam for free — sending commands, capturing output, prompt-marker setup, and exit-code detection all operate on the shell _inside_ the pane via `send_keys`/`capture_pane`, which are multiplexer-agnostic. The one hard dependency that breaks on herdr is **foreground-process detection via `ps -t <tty>`**: herdr exposes no tty (`pane process-info` has no `tty` on macOS), so `shell_infra._detect_shell_tools` and the provider auto-detection (`detect_provider_from_tty`/`detect_provider_cached`) that share `process_detection.py` both fail on herdr.

This plan makes `Multiplexer.foreground(window_id)` the single source of foreground-process truth, moving `ps -t` into the tmux backend as a private detail. On tmux the change is behavior-preserving; on herdr `foreground()` reads `pane process-info.foreground_processes[]` (pid/argv/cmdline/cwd — more than `ps -t` gives). It also clamps shell captures to herdr's 1000-line `read` limit and locks the boundary with a no-tty drift gate so shell/detection code can never reach for `tty`/`ps` again.

## Context

- Impacted components: `src/ccgram/providers/process_detection.py` (the `ps -t` helper), `src/ccgram/providers/shell_infra.py` (`_detect_shell_tools`), `src/ccgram/providers/__init__.py` (`detect_provider_from_pane`), `src/ccgram/multiplexer/herdr.py` + `multiplexer/tmux.py` (the `foreground()` backends), `src/ccgram/handlers/shell/shell_capture.py` (scrollback capture), `.archfit.yaml`, the boundary-audit test suite.
- Constraints: tmux path stays byte-for-byte behavior-compatible; no module outside the tmux backend may reference `pane_tty`/`ps -t`/`get_foreground_args`; checkboxes only inside Task sections (ralphex parses these as work items).
- Code evidence (verified by grep): `process_detection.py:130 get_foreground_args` (`ps -t`), `shell_infra.py:234,242` (`w.pane_tty` + `get_foreground_args`), `process_detection.py:172 detect_provider_from_tty` / `:186 detect_provider_cached`, `shell_capture.py:125 capture_pane_scrollback`. Send/capture paths (`shell_commands.py:183,197,259`, `shell_infra.py:297-309`) are already multiplexer-agnostic.
- Depends on the seam plan `docs/plans/20260621-multiplexer-seam.md`: the `Multiplexer.foreground()` Protocol method + tmux impl (its Tasks 1–2) and the herdr backend (its Task 7).

## Source artifact

- Design: `docs/architecture-design/2026-06-21-herdr-multiplexer-support.md`.
- Contracts used: `Multiplexer.foreground(window_id) -> ForegroundInfo`; `MultiplexerCapabilities.exposes_pane_tty` and `read_max_lines` ("The Multiplexer contract").
- Modules used: leak sites `providers/shell_infra.py`, `providers/process_detection.py` ("Module map", "Integration contracts → herdr backend: no tty on macOS, use process_info").
- Risks used: macOS `process-info` exposes no tty; herdr `read` caps at 1000 lines ("Open risks", "Minor gaps").
- Decisions used: gate on capabilities, not backend name; reuse polling (status/foreground via the backend, not the event stream).

## Success criteria

- `Multiplexer.foreground(window_id)` is the only way shell and provider-detection learn the foreground process; no module outside `multiplexer/tmux.py` references `pane_tty`, `ps -t`, or `get_foreground_args`.
- Under `CCGRAM_MULTIPLEXER=tmux` the full existing suite stays green (zero behavior change).
- Under `CCGRAM_MULTIPLEXER=herdr` a shell window works end to end: prompt-marker setup, NL→command approval, command run, output isolation, exit-code detection, `C-c`, `clear`.
- herdr `foreground()` returns pid/argv/cwd from `pane process-info`; shell-vs-agent classification uses herdr data, not a tty.
- Shell captures honor `read_max_lines`; a >1000-line command surfaces `truncated` instead of silently dropping output.
- A drift gate fails the build if shell/detection code reintroduces a direct tty/ps dependency.

## Development Approach

- Testing approach: regular; the tmux foreground migration is characterization-guarded by the existing suite.
- Complete each task fully — green verification — before the next.
- Tasks 1–2 are behavior-preserving on tmux; Tasks 3–4 are herdr-only behavior. Update this plan if scope shifts.

## Testing Strategy

- Unit tests for every code-changing task; the tmux migration is pinned by existing provider/shell tests as characterization.
- A new audit test enforces the no-tty boundary.
- herdr shell behavior covered by an integration test (skips without a herdr socket).
- Run project tests after each task before proceeding.

## Validation Commands

Whole-plan commands:

- `make check` — fmt, lint, typecheck, deptry, unit + integration tests.
- `make test` — `uv run pytest tests/ -m "not integration and not e2e" -n auto --dist=loadscope`.
- `make typecheck` — `uv run pyright src/ccgram/ tests/`.
- `make lint` — `scripts/lint_lazy_imports.py` + `uv run ruff check src/ tests/`.
- New drift gate: `uv run pytest tests/ccgram/test_no_tty_outside_backend.py -v`.
- herdr shell leg (needs a running herdr): `uv run pytest tests/integration/ -m "herdr" -v`.
- Deterministic architecture gate: `archfit check --config .archfit.yaml --full`. Note: `archfit` is not installed locally or wired into CI, so the enforced gate is the pytest drift audit; the `.archfit.yaml` rule (Task 2) is the recommended promotion.
- Impact/blast-radius: GitNexus is not available in this repo. Fallback per task: `git diff --name-only` plus the no-tty audit as the dependency-direction proxy.

## Technical Details

- `ForegroundInfo` (from the seam) carries `pid, pgid, argv, cwd` — the exact shape `process_detection` already returns and `shell_infra._detect_shell_tools` already consumes, so the call-site swap is mechanical.
- tmux `foreground()` keeps today's behavior (`pane_tty` + `ps -t <tty>`), now a private detail of `multiplexer/tmux.py`. herdr `foreground()` maps `pane process-info.foreground_processes[]` → `ForegroundInfo`; `pgid` from `foreground_process_group_id`.
- Capability gating: `exposes_pane_tty == False` must short-circuit any tty-dependent branch; shell code must never read a `pane_tty` field directly.
- Capture clamp: shell scrollback requests are clamped to `caps.read_max_lines` (herdr 1000); `CaptureResult.truncated` is surfaced to the LLM-context and isolation logic so a truncated capture is handled, not mistaken for short output.
- herdr classifies a plain shell pane as `agent_status: unknown` / `agent: null` and returns its foreground process directly, so shell-vs-agent detection can lean on herdr data instead of scraping a tty.

## Implementation Steps

### Task 1: Make `foreground()` the single foreground source (tmux-only, behavior-preserving)

- Justification: design "Integration contracts → herdr backend" (no tty on macOS, use `process_info`) and the `foreground()` contract; code evidence `shell_infra.py:234,242`, `process_detection.py:130,172,186`.
- Files: `src/ccgram/providers/process_detection.py` (becomes a private helper used only by `multiplexer/tmux.py`'s `foreground()`), `src/ccgram/providers/shell_infra.py` (`_detect_shell_tools` → `multiplexer.foreground(window_id)`), `src/ccgram/providers/__init__.py` + `process_detection.py` (`detect_provider_from_pane`/`from_tty`/`cached` consume `ForegroundInfo` from `foreground()`), `src/ccgram/handlers/recovery/transcript_discovery.py` if it calls detection directly.
- Preconditions: seam plan `20260621-multiplexer-seam.md` Tasks 1–2 merged (`Multiplexer.foreground()` + tmux impl).
- Postconditions: shell tool-detection and provider auto-detection get the foreground process via `multiplexer.foreground(window_id)`; no module outside `multiplexer/tmux.py` reads `pane_tty` or calls `get_foreground_args`/`ps -t`. tmux behavior unchanged.
- Impact: `git diff --name-only` (GitNexus unavailable); the no-tty audit (Task 2) is the dependency-direction proxy.
- Fitness gate: relies on the seam's F1; the dedicated no-tty gate lands in Task 2.
- Verification: `make check` (characterization — must stay green); `uv run pytest tests/ -k "provider_detection or shell" -v`.
- Manual checks: confirm `ForegroundInfo` field names match what `_detect_shell_tools` and detection already consume, so no logic changed.
- [x] move `ps -t`/`get_foreground_args` into the tmux backend as the private `foreground()` implementation
- [x] switch `shell_infra._detect_shell_tools` to `multiplexer.foreground(window_id)`
- [x] switch `detect_provider_from_tty`/`detect_provider_cached`/`detect_provider_from_pane` to consume `ForegroundInfo` from `foreground()`
- [x] remove all direct `pane_tty`/`get_foreground_args` reads outside `multiplexer/tmux.py`
- [x] write tests pinning foreground-based shell tool detection and provider detection (tmux)
- [x] run project tests (`make test`) - must pass before next task

### Task 2: No-tty drift gate (fitness) + archfit rule

- Justification: "keep architecture from drifting" — the seam exists to stop callers reaching past it; tty/ps is the shell-specific leak. Design "Architecture-fitness checks" pattern (F1-style audit).
- Files: new `tests/ccgram/test_no_tty_outside_backend.py` (AST/source walk modeled on `tests/ccgram/test_window_state_access_audit.py`), `.archfit.yaml` (classify `process_detection` under the tmux backend module; forbid `handlers`/`providers` shell modules from importing `process_detection` directly).
- Preconditions: Task 1 merged.
- Postconditions: the audit forbids `pane_tty`, `ps","-t`/`ps -t`, and `get_foreground_args` references outside `multiplexer/tmux.py`; `.archfit.yaml` models `process_detection` as a tmux-backend internal with a forbidden-dependency rule.
- Impact: `git diff --name-only`.
- Fitness gate: this task _is_ the gate. Before-fail/after-pass: planting `get_foreground_args(pane_tty)` in `shell_infra.py` must fail `test_no_tty_outside_backend.py`; removing it must pass. archfit rule is the recommended promotion (archfit not in CI).
- Verification: `uv run pytest tests/ccgram/test_no_tty_outside_backend.py -v`; `archfit check --config .archfit.yaml --full` if available, else record the missing-tool note.
- Manual checks: confirm the audit allow-list is exactly `multiplexer/tmux.py` (and the relocated `process_detection` if it stays a tmux-backend submodule).
- [x] add `test_no_tty_outside_backend.py` forbidding tty/ps/`get_foreground_args` outside the tmux backend
- [x] confirm the gate fails on a planted tty reference, then passes after removal
- [x] update `.archfit.yaml`: `process_detection` under the tmux backend + forbidden-dependency rule
- [x] write tests for the audit itself
- [x] run project tests (`make test`) - must pass before next task

### Task 3: herdr `foreground()` via process-info + shell capture clamp

- Justification: design "Integration contracts → herdr backend" (`process_info` foreground; `exposes_pane_tty=False`) and "Open risks" (1000-line `read` cap).
- Files: `src/ccgram/multiplexer/herdr.py` (`foreground()` from `pane process-info`; clamp `capture_scrollback` to `read_max_lines`, set `CaptureResult.truncated`), `src/ccgram/handlers/shell/shell_capture.py` (request scrollback within `caps.read_max_lines`; handle `truncated`), `src/ccgram/providers/shell_infra.py` (tool detection consumes `ForegroundInfo` on herdr unchanged from Task 1).
- Preconditions: Tasks 1–2 merged; seam plan Task 7 (herdr backend) merged.
- Postconditions: herdr `foreground()` returns pid/argv/pgid/cwd from `foreground_processes[]`; `exposes_pane_tty=False` short-circuits any tty branch; shell captures clamp to 1000 lines and surface truncation.
- Impact: `git diff --name-only`; the no-tty gate keeps the tty branch out of shell code.
- Fitness gate: Task 2 audit stays green; no new archfit rule.
- Verification: `uv run pytest tests/ccgram/test_herdr_backend.py -k "foreground or read or clamp" -v`; `make test`.
- Manual checks: with a running herdr, confirm `foreground()` resolves a running command's argv and that a >1000-line command reports `truncated`.
- [x] implement herdr `foreground()` from `pane process-info.foreground_processes[]`
- [x] clamp shell scrollback capture to `read_max_lines`; surface `CaptureResult.truncated`
- [x] ensure `exposes_pane_tty == False` short-circuits any tty-dependent path
- [x] write unit tests (process-info fixtures → ForegroundInfo; read clamp/truncation)
- [x] run project tests (`make test`) - must pass before next task

### Task 4: Verify shell end-to-end on herdr (integration)

- Justification: success criteria — a shell window must work under `CCGRAM_MULTIPLEXER=herdr`; risk that prompt-marker isolation or exit-code detection regresses on herdr captures.
- Files: new `tests/integration/test_shell_herdr.py` (marked `herdr`, skips without `$HERDR_SOCKET_PATH`).
- Preconditions: Tasks 1–3 merged.
- Postconditions: a herdr shell pane completes the full flow — PS1 marker setup, NL→command approval, run, output isolation, exit-code detection, `C-c`, `clear` — and is classified as shell (not agent).
- Impact: `git diff --name-only`.
- Fitness gate: Task 2 audit green; no new gate.
- Verification: `uv run pytest tests/integration/ -m "herdr" -v` (with herdr running).
- Manual checks: drive a real shell window over Telegram on a herdr backend; confirm command output and exit code render correctly; confirm a plain shell pane is detected as shell via herdr `agent_status`/foreground.
- [x] add a herdr-marked integration test covering PS1 setup, run, output isolation, exit code, `C-c`, `clear`
- [x] assert shell-vs-agent classification uses herdr data, not a tty
- [x] write the integration test fixtures/skips for the no-socket case
- [x] run project tests (`make test`) - must pass before next task

> Verification note: ran live against a herdr socket (3 herdr-marked tests pass). The test surfaced a real gap — `detect_pane_shell` relied on `pane_current_command`, which herdr leaves empty for a bare shell pane, so marker setup built the wrong shell's prompt. Fixed by adding a `Multiplexer.foreground()` fallback (behavior-preserving on tmux; uses herdr data, not a tty). Covered by new unit tests in `tests/ccgram/providers/test_shell.py`.

### Task 5: Verify acceptance criteria

- Justification: architecture-plan final verification/documentation/handoff.
- Files: docs only — `.claude/rules/architecture.md` (note `process_detection` is a tmux-backend internal; shell + detection get foreground via the seam; the no-tty gate); `docs/providers.md` if shell behavior on herdr needs a line.
- Preconditions: Tasks 1–4 merged.
- Postconditions: whole-plan validation green for both backends; shell-on-herdr verified; docs updated; re-review recorded.
- Impact: `git diff --name-only` for the whole branch.
- Fitness gate: the no-tty drift gate and seam F1 green.
- Verification: `make check`; `uv run pytest tests/ccgram/test_no_tty_outside_backend.py -v`; with herdr running, `uv run pytest tests/integration/ -m "herdr" -v`.
- Manual checks: run `architecture-review` scoped to the shell/foreground seam and confirm code matches the design.
- [x] verify all Overview requirements hold for both `CCGRAM_MULTIPLEXER=tmux` and `=herdr` (tmux: full suite 5374 passed; herdr: 5 `-m herdr` integration tests pass against live socket)
- [x] verify the no-tty drift gate is part of the suite and green (`test_no_tty_outside_backend.py`, 186 passed)
- [x] run the full project test suite and the drift gate
- [x] run the project linter (`make lint`, including `lint-lazy`) - all issues fixed (lint-lazy clean + ruff clean; typecheck 0 errors)
- [x] update architecture docs for the foreground seam and the no-tty gate (`.claude/rules/architecture.md`: process_detection now a tty-free classifier, tmux `foreground()` sole `ps -t` site, herdr `foreground()` via process-info, no-tty drift-gate decision; `docs/providers.md`: backend-neutral foreground detection)
- [x] run project tests (`make test`) - must pass

## Acceptance criteria

- No module outside `multiplexer/tmux.py` references `pane_tty`, `ps -t`, or `get_foreground_args`; the drift gate proves it (fails on a planted violation).
- `make check` passes with the default (tmux) backend; the existing suite is unchanged by Task 1.
- A shell window works end to end under `CCGRAM_MULTIPLEXER=herdr` (markers, run, isolation, exit code, `C-c`).
- herdr `foreground()` returns pid/argv/cwd from `process-info`; shell captures clamp to `read_max_lines` and surface `truncated`.
- `.archfit.yaml` models `process_detection` as a tmux-backend internal with a forbidden-dependency rule.

## Safety notes

- Task 1 is the widest blast radius (relocating `process_detection`, swapping shell + detection call sites) but behavior-preserving on tmux — gated by `make check` as characterization. Roll back by restoring direct `get_foreground_args(pane_tty)` calls.
- Tasks 3–4 change behavior on the herdr path only; tmux branches are gated by `exposes_pane_tty`. No data migration, no irreversible steps.
- This plan depends on the seam plan's `foreground()` method and herdr backend; do not start it before those land.
- Execution: an engineer, mutator agent, or `ralphex` runs this approved plan task by task.

## Post-Completion

Items requiring manual intervention. No checkboxes — informational only.

- Run `architecture-review` scoped to the shell/foreground seam after Task 5 to confirm code matches the design and no tty dependency remains.
- If the seam plan later wires `archfit` into CI (its F6), the `.archfit.yaml` rule from Task 2 becomes an enforced gate automatically.
- herdr upgrade watch: re-run the herdr shell integration leg after a herdr version bump, in case `process-info` fields or the `read` cap change.
