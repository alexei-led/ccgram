# Marked topic discovery

CCGram supports opt-in marked-only Telegram topic discovery with tmux, standard Herdr, and agterm. Native selection is independent of agent conversations and routing identity. Herdr requires no fork or upstream change. Tmux selection follows its native window lifetime, not a promised cold-restore path. For setup, see the [operator guide](../guides.md#marked-only-discovery).

## Operator model

`CCGRAM_TOPIC_DISCOVERY=auto|marked` selects automatic Telegram topic admission (CLI: `--topic-discovery auto|marked`). `auto` remains the default and preserves existing discovery and unbound-window TTL behavior. In `marked`, unattended discovery requires an eligible, markable target with a confirmed true native mark and a confirmed selection owner. Unknown metadata cannot grant admission.

The local controls are `ccgram topic mark TARGET`, `ccgram topic unmark TARGET`, and `ccgram topic status [TARGET]`. Get the exact target ID from `ccgram topic status`: mutations require an explicit ID, never a name or the active GUI terminal. These commands do not require bot credentials. CLI unmark leaves existing topics connected.

Status writes JSON to stdout and diagnostics to stderr. Its fields are `backend`, `mode`, `targets`, and `pending_operations`. Each live target reports `id`, `name`, native `marked` (true, false or null), `markable`, and opaque diagnostic `owner`. An owner is not an input target or routing alias. Pending operations report original `window_id`, `owner`, and boolean `desired`, not selected-state truth. Without a target, status includes the complete journal, including references no longer in the live listing. With a live target, it filters operations by exact known owner, falling back to route only when ownership is unknown.

A read-only journal guard refuses a manual native write that conflicts with unfinished opposite intent. Repeating the same desired value is nonconflicting. The CLI neither creates nor completes bot-owned retry operations and never rewrites the journal. An unreadable or malformed journal makes status and mutation fail closed. Native update failure is not reported as success.

A mark selects a terminal/window, not an agent conversation. Replacing an agent or using `/clear` does not change that selection. New native worker targets start unmarked. Explicitly copying native selection metadata is a separate operation with the Herdr limits below. Names, working directories and layout resemblance never prove terminal identity.

The mode does not filter liveness, mobile listings, explicit binding, routing identity, or ongoing forwarding. Existing bound topics keep working after unmarking or changing modes. `/new` and picker binding attempt selection in both modes before binding. In `auto`, an unverified native write warns and persists a mark-on retry without changing historical binding success. In `marked`, it prevents binding and keeps creation quarantined until verification. Creation ownership and foreground readiness remain in force.

In marked mode, intentional close or `/unbind` of the last binding clears that terminal's native mark, prevents automatic recreation, and leaves the terminal running. Other bindings and active creation claims protect the shared selection from clearing. Deliberate detach is exempt from orphan TTL without changing terminal origin. Abandoned failed creation retains its existing cleanup behavior.

If a target is unmarked during unfinished explicit creation, admission is withdrawn. CCGram retains the exact uncertain creation claim unbound. It does not replay a stale mark-on or delete a possibly human-used topic. The operator can reselect the native target to permit final recovery. Already-bound topics remain connected.

### Unfinished or ambiguous detach

A failed clear remains unfinished lifecycle work, not a permanent suppression preference. Persisted desired-false operations survive restart and block readmission of the exact same native owner, including after conversation re-keying. When both owners are known, exact owner matching takes precedence over routing IDs. An old owner's clear does not block an independent new owner reusing that ID. Route matching is a fallback only when ownership is unknown. Topic maintenance retries unfinished operations every 60 seconds. Independent owners remain independently admissible.

If neither saved nor current verifiable ownership exists, CCGram does not guess which terminal to clear or report completed native detach. It keeps the unfinished operation and warns that ownership is unconfirmed. The operator must explicitly unmark the current target. A backend may settle an ownerless clear only if it safely proves no selected native owners remain. Merely losing the old routing target is not success.

## Native storage and restoration

### tmux

Selection lives only in the window-local `@ccgram_topic` option, ignoring inherited global/session defaults. The native value is `v1:1:<32 lowercase hex>` when selected and `v1:0:<same 32 lowercase hex>` when cleared. Clearing retains the owner as a native tombstone. Use the CCGram commands rather than writing raw option values. An absent option is unmarked. Malformed data or a failed read is unknown. A healthy unmarked window has a native witness tied to the actual server socket incarnation and window ID, not a reusable ID or PID alone. Root code never constructs or parses that witness.

A live window retains its selection across agent respawn, CCGram restart and client reconnect while the tmux server lives. Destroying a vanilla server destroys its terminals and window options. External restorers must preserve the full native option value within their own verified window snapshots. No particular plugin is required or claimed as supported. Reused window IDs, names and working directories cannot recover selection.

### agterm

The native session `flagged` boolean selects the primary only. The native session UUID is its owner. A split reports `topic_markable=False` and `topic_marked=False`: explicit binding is allowed but must skip primary flag writes and must not create a primary topic.

### Standard Herdr

Standard `pane.rename` and raw manual labels from `pane.get`/`pane.list` carry selection. Dynamic titles, terminal titles and agent names are not substitutes. The exact suffix is `[ccgram-topic-v1:<32 lowercase hex owner>]`. It follows the original manual label and one space, or stands alone when no original label exists. Ordinary labels do not select. Native snapshots preserve the explicitly written suffix with their saved terminal. This is stored state, not matching a display label heuristically. Unmarked ownership comes from the current native terminal and connection namespace. Once an assigned mark is persisted, its stored owner is authoritative across native restoration.

Marking rejects non-reversible normalization, unsafe serialized requests, reserved/nested/malformed or colliding suffixes, and overlarge labels **before writing**. The complete serialized rename request, including its framing newline, must not exceed 1 MiB. It never truncates or overwrites user data. Unmark removes only the managed suffix and restores the original label, including null when no original existed. A literal `--clear` label remains user data, not a clear command.

Marker observations join authoritative `agent.list` records by current native terminal identity. A marker-query failure leaves liveness and auto eligibility intact. Routing still uses the guarded agent-session target, never the selection owner. Bare shells remain ineligible for automatic topics but remain in the native inventory used to resolve an owned clear.

#### Explicit metadata-copy limit

Normal native spawn/split creates no assigned mark. However, `layout.export` includes manual labels and `layout.apply` copies them into new terminals, including managed selection metadata. Duplicate assigned owners are quarantined: no adoption or marker writes. Templates for new workers must omit managed selection metadata. A native operator can remove a copied suffix with pane rename, preserving the original label.

A sole copied marker is indistinguishable from an imported selection and is treated as explicitly copied selection. CCGram does **not** promise to reset selection for arbitrary marker-copy operations or infer lineage from layout. No fork, upstream publication, sidecar selection registry or label-based routing recovery is required.

## Native verification and limits

Isolated tests against stock Herdr 0.9.5 verified exact Unicode/literal-`--clear` label restoration, conversation re-key and bare-shell clearing, fresh unmarked splits, and duplicate-owner quarantine. Cold restoration preserved the stored owner despite a changed runtime terminal ID. These tests used genuine native helpers with controlled standard hook reports. They did not launch an AI CLI or prove actual provider-session resumption.

Tmux test coverage includes native token retention, full-server loss, reused window IDs, restored full tokens, duplicate-owner quarantine, and acknowledgement-loss retries. All 16 native tmux cases passed in the current integration suite. That run established an owned default namespace before imports and retained the creating socket/incarnation for teardown. This fresh evidence does not establish that earlier unguarded execution was fully isolated or had no effects. Historical default-server effects remain unknown. Agterm adapter tests and pinned upstream source establish the intended persisted primary flag and unmarked duplicates. A bounded live test verified flag on/off, a fresh adapter reading the same owner and selection, and restoration of the original flag. Its inventory was pinned to the owned test window. It does not verify application cold restore, conversation reset, or full multi-window inventory.

Native CLI read/check/write operations are not an atomic compare-and-set. Inventory, identity and readback checks detect observed changes. An external replacement or metadata change between verification and dispatch can still affect a different resource before a postcheck reports uncertainty. No atomic-write or rollback guarantee is claimed.

## Backend and policy contract

The existing `topic_eligible` remains separate from existence and visibility. `WindowRef` exposes these fields at the end for positional compatibility:

- `topic_marked: bool | None = False`: `None` is unconfirmed.
- `topic_markable: bool = True`: false for agterm splits.
- `topic_mark_owner: str | None = None`: opaque native selection-resource identity, never a routing alias.

`async Multiplexer.set_topic_mark(window_id: str, marked: bool, *, owner: str | None = None) -> bool` is idempotent and reports success only after verification. Bot and CLI callers supply an observed or saved owner. The CLI helper consumes its already captured `WindowRef` rather than resolving the routing ID again. An explicit owner permits mutation only of that exact proven resource, even when the old conversation target is gone. First mark requires a matching current unmarked witness. A missing mark-on owner cannot be guessed or recreated after restart.

For clear, complete validated owner absence can prove the old selection off. Duplicate owners, malformed owned markers, incomplete inventory or read errors cannot. Resolution includes bare shells but never falls back to a stale locator, another backend, a display name or a guessed current terminal. An unavailable listing is not an empty listing and never proves closure.

`topic_discovery.py` owns the stdlib-only mode constants and validator. Configuration exposes `config.topic_discovery_mode`. `should_adopt_window(window, *, mode: str, detach_pending: bool) -> bool` requires both keywords. Every automatic site supplies `lifecycle_state.is_topic_detach_pending(window.window_id, owner=window.topic_mark_owner)`, with `owner` required. Pending state gates marked admission only. The existing `is_agent_topic_window` keeps its baseline eligibility semantics.

## Lifecycle ownership

`WindowState` stores observed `topic_mark_owner` and intentional `topic_detached` metadata. Unfinished native effects live in the owner-keyed `topic_mark_operations` journal within the existing `state.json`, not in a per-window pending flag. Each operation records its original `window_id`, opaque `owner` (or null when unknown), and boolean `desired`. Known owners key operations independently of routing IDs. Unknown-owner operations retain their original route as a quarantine key. Native metadata remains the only selected-state truth: this journal authorizes retries, not routing or a competing selection preference.

The credential-free journal reader is read-only. A missing state file or journal field means no operations. Malformed, duplicate or unreadable records fail closed rather than hiding unfinished work. Opposite desired intent on a matching operation conflicts with manual CLI mutation. Both known owners compare exactly, with route fallback only for unknown ownership. Root code never generates backend-specific owner IDs or writes native storage.

Explicit selection observes a healthy current owner and persists owner plus desired-on before mutation. Unknown ownership produces a recoverable diagnostic, not a candidate identity. Automatic admission records the actual observed owner before binding **without marking** automatically discovered targets, including unmarked targets in auto mode. Already-bound, verifiable rows refresh observed ownership only when no pending operation owns that state. Unadmitted workers do not acquire window state merely for being listed.

Before last detach, capture current verifiable ownership while the binding remains intact. Synchronously unbind, record owner plus desired-off/detached, and flush before awaiting native clear. Retry by saved owner without requiring the old routing target to resolve. Lock by owner when known. Recheck remaining bindings and creation claims under that lock before clearing. If another consumer now protects the resource, complete the clear operation without changing native selection. Preserve journal operations through cancellation, stale pruning and checkpoint failure. Observing a new owner must not retarget or discard an old owner's operation. Split exceptions never touch the primary.

Fresh automatic admission checks run inside the actual creation-ownership lock and again before commit or recovery. A stale pre-lock verdict must not bypass a newly pending owner-clear. Unmarking alone still leaves admitted topics connected. Settling a native operation, including a protected no-op, does not remove detached-lifetime TTL protection. Only successful actual readmission clears that protection, not observing a true native flag.

## Acceptance matrix

This matrix summarizes the behavior covered by the policy, lifecycle and native-storage checks. Native execution limits are listed above.

| Scenario | Expected result |
| --- | --- |
| Default `auto`, eligible unmarked or metadata-unknown target | Preserve automatic admission and TTL. Selection metadata does not filter auto. |
| `marked`, true mark with confirmed owner / missing owner / false / unknown | Admit only the confirmed true mark with an owner, eligibility and markability. |
| Ineligible target or unavailable listing | Do not adopt ineligible targets. Unavailable listing never proves death. |
| Unmark or switch mode with an existing binding | Keep binding, forwarding, liveness and mobile listing. |
| Agent replacement or `/clear` | Keep native selection independent of conversation identity. |
| Normal new target; reused tmux window ID after server loss | Start unmarked. No name/cwd/layout recovery. |
| Native snapshot restore with full owned marker | Preserve selection only through verified native metadata, not transient tokens. |
| Ordinary Herdr label / exact managed suffix | Ordinary label does not select. Only valid owned metadata can select. |
| Herdr normalization, nested suffix, excessive size, literal `--clear` | Reject unsafe mutation without data loss. Preserve literal label data. |
| Herdr copied layout, duplicate owner / sole copied marker | Quarantine duplicates. Sole copy is explicitly copied selection, not reset automatically. |
| `/new` or `/bind`, native mark failure | Auto warns and retains binding with retry. Marked blocks binding until verification. |
| Unmark during unfinished explicit creation | Retain exact uncertain claim unbound, without stale mark-on or topic deletion. Native reselection permits recovery. |
| agterm explicit split binding | Bind split without flagging or auto-adopting primary. |
| Last marked-mode detach / another binding remains | Clear only after last binding. Keep terminal alive and exempt intentional detach from TTL. |
| Lost mark acknowledgement, conversation re-key or bare shell | Retry the exact saved owner, not the old conversation locator. |
| Clear fails, local checkpoint fails or current owner changes | Retain original owner-keyed operation across restart without retargeting it. No competing suppression preference. |
| Manual CLI write conflicts with journal | Reject opposite matching intent. Same desired value is nonconflicting but does not settle the journal. |
| Known new owner reuses routing ID with old-owner clear pending | Keep old operation, but admit independent new owner if otherwise eligible. |
| Owner unknown, duplicate or stale | Fail closed without clearing another terminal. Unknown historical ownership requires operator unmark. |
| Clear owner absent in complete validated native inventory | Backend may verify old selection off without mutating another resource. |
| Pending clear appears after pre-lock admission check | Recheck under creation ownership before bind/recovery. |
| Abandoned failed creation | Preserve existing cleanup and TTL. |
