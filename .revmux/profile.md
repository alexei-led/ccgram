# ccgram review profile

## System

- Python 3.14+ async bridge between Telegram forum topics and terminal-agent sessions.
- Backends: tmux, Herdr, and agterm behind the neutral `Multiplexer` protocol.
- Persistent topic bindings, transcript cursors, provider identity, and cleanup/recovery state are safety-sensitive.

## Real failure severity

- Critical: wrong-topic or cross-user routing, transcript replay/flood, deletion of a live binding or topic, killing the wrong terminal session, or treating backend outage as confirmed absence.
- Major: supported backend sessions cannot be adopted, resumed, discovered, or controlled; partial listings prune state; ID aliases or casing strand bindings.
- Minor: local UX or documentation defects without routing, state, or cleanup impact.
- Do not report style without a concrete failure scenario.

## Deliberate contracts

- Selection listings may be filtered. Reconciliation listings must be complete and tri-state: `None` means unavailable/unconfirmed, never empty.
- Destructive actions require confirmed absence or presence as appropriate; unknown state must fail closed.
- `WindowRef.topic_eligible` is a backend-owned adoption verdict and is separate from liveness.
- Current backend IDs compare case-insensitively, while reported IDs are preserved for backend calls and stored cleanup.
- Herdr guarded targets and provider-prefixed labels stay behind the adapter boundary.
- Native agent status, workspace selection, and native topic targets are separate capabilities.

## Evidence bar

- Every actionable finding needs a changed `file:line`, a realistic execution path, and a minimal fix.
- Distinguish change-introduced defects from pre-existing issues and explicit follow-ups.
- Check failure paths, TOCTOU windows, partial backend responses, aliases, case variants, and all destructive callers.
- Verify test claims against Ruff, Pyright, unit/integration tests, and the enforced architecture guard. Local-only e2e tests requiring live tmux, Herdr, agterm, or agent CLIs are not CI evidence unless their dependencies were actually available.
- GitNexus results marked stale, partial, truncated, or UNKNOWN are coverage gaps, not proof of safety.
