# Marked topic discovery: implementation contract

Status: approved design for [issue #400](https://github.com/alexei-led/ccgram/issues/400), not a release or backend-support claim. The baseline implements automatic discovery. Marked discovery and the restore guarantees below require integrated code and verification. In particular, Herdr support and tmux restore have not yet been verified here.

## Operator model

`CCGRAM_TOPIC_DISCOVERY=auto|marked` selects automatic Telegram topic admission. `auto` remains the default and preserves existing discovery and unbound-window TTL behavior. In `marked` mode, unattended discovery admits only an eligible target with a confirmed native mark. An unknown mark is not confirmation. The local controls are `ccgram topic mark TARGET`, `ccgram topic unmark TARGET`, and `ccgram topic status [TARGET]`. Get the exact target ID from `ccgram topic status`: mutations require an explicit ID, never a name or the active GUI terminal. Status returns JSON with `backend`, `mode`, and `targets`; each target has `id`, `name`, native `marked` (true, false or null) and `markable`. CLI unmark leaves existing topics connected. CLI help and target-safety checks are implemented, but successful backend mutation remains pending integration verification.

A mark selects a terminal/window, not an agent conversation. Replacing an agent or using `/clear` does not change the selection. A newly created or duplicated worker terminal starts unmarked. A mark survives restoration only when the backend verifies that it restored the same terminal. Names, working directories and layout are not identity evidence.

This setting does not filter liveness, mobile listings, explicit binding, routing identity, or ongoing forwarding. Existing bound topics continue working after unmarking or changing modes. `/new` and `/bind` select their target before a successful binding. Creation ownership, foreground readiness and quarantine remain in force.

In marked mode, intentionally closing or unbinding the last binding clears the native mark, prevents automatic recreation, and leaves the terminal running. A deliberate detach is exempt from orphan TTL. An abandoned failed create retains its existing cleanup behavior. If clearing the mark fails, the lifecycle operation remains unfinished. A separate suppression preference must not mask the failure.

## Native storage and restoration

The tmux adapter uses a window-local `@ccgram_topic` option (`1` or `0`), ignoring inherited session and global defaults. Missing local selection is unmarked. A malformed value or failed read is unknown. The same live window retains its mark across agent respawn and CCGram reconnect. Destroying the tmux server loses its window options: a reused window ID and matching name do not restore selection. An external restorer would need to persist and reapply selection only after verifying the same-window identity. That restore path is not verified support.

The agterm adapter reads the native session `flagged` boolean for the primary only. A split reports `topic_markable=False` and `topic_marked=False`; explicit split binding must skip marker writes.

## Backend contract

The existing `topic_eligible` answers whether a backend target is eligible for unattended adoption. It remains separate from existence and visibility. The neutral interface adds `WindowRef.topic_marked: bool | None = False` and `WindowRef.topic_markable: bool = True`. `None` means the mark is unconfirmed and cannot grant marked-mode admission. `Multiplexer.set_topic_mark(window_id: str, marked: bool) -> bool` is idempotent and reports success only after verification. An unavailable listing is not an empty listing and never proves closure.

| Backend | Selection boundary and required verification | Evidence status |
| --- | --- | --- |
| tmux | Window mark, retained across agent replacement; restore only with verified same-window identity. | Native storage implemented; integrated mutation and same-window restore verification pending. |
| agterm | Native session mark selects the primary only. A split is explicitly bindable but not natively markable; binding it must not toggle the primary mark or create a primary topic. | Native adapter implemented; integrated mutation verification pending. |
| Herdr | Terminal selection must survive agent replacement and verified restoration without using transient agent or locator identity as durable proof. | Native durable support and restore pending; do not treat ephemeral tokens as durable. |

Herdr routing remains keyed by the guarded agent-session target, not by the selection mark. A durable selection mechanism must not replace its existing guards, infer identities from labels, or make a bare shell automatically topic-eligible.

### Local upstream proposal: Herdr

For discussion only. Not submitted. Provide a public, documented terminal-scoped selection value with read and idempotent set/clear operations, verifiable after mutation. Its identity must remain stable while an agent is replaced or cleared, and a newly created or duplicated terminal must not inherit the mark. For restoration, the API must attest that a restored terminal is the same terminal before restoring its selection. The API must distinguish unavailable or unconfirmed state from `false`, expose enough information to target the exact terminal safely, and document concurrency and failure behavior. CCGram must still use fresh guarded agent targets for routing. Upstream acceptance and publication are outside this proposal.

## Acceptance matrix

These are required behavior checks, not claims that tests have passed.

| Scenario | Expected result |
| --- | --- |
| Default `auto`, eligible unmarked target | Existing automatic admission and TTL behavior unchanged. |
| `marked`, eligible target with confirmed mark / false / unknown | Admit only the confirmed mark; do not adopt false or unknown. |
| Marked but ineligible target; unavailable listing | Do not adopt; unavailable listing does not prove the target dead. |
| Unmark or switch mode with an existing binding | Keep the binding and forwarding; liveness and mobile listing remain complete. |
| Agent replacement or `/clear` in selected terminal | Selection survives; newly created or duplicated terminal is unmarked. |
| Restore same terminal / unrelated terminal with matching name, cwd or layout | Restore mark only with verified same-terminal evidence; never infer from resemblance. |
| `/new` or `/bind` | Select before successful bind; preserve creation ownership, readiness and quarantine on failures. |
| agterm split explicit bind | Bind split without changing primary mark or automatically creating a primary topic. |
| Marked-mode deliberate detach, last binding / other bindings remain | Last binding clears mark and keeps terminal alive, without orphan TTL; do not clear while another binding remains. |
| Mark clear fails | Retain unfinished lifecycle operation; do not claim completed detach or suppress recreation with competing state. |
| Abandoned failed creation | Retain existing cleanup and TTL behavior. |
| Herdr agent replacement, restart and verified restore | Selection and identity guarantees hold; missing or ambiguous evidence fails closed. |
