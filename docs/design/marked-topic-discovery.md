# Marked topic discovery

CCGram supports opt-in marked-only Telegram topic discovery with tmux, standard Herdr, and agterm. Native selection is independent of agent conversations and routing identity. Herdr requires no fork or upstream change. Tmux selection follows its native window lifetime, not a promised cold-restore path. For setup, see the [operator guide](../guides.md#marked-only-discovery).

## Operator model

`CCGRAM_TOPIC_DISCOVERY=auto|marked` selects automatic Telegram topic admission (CLI: `--topic-discovery auto|marked`). `auto` remains the default and preserves ordinary discovery and unbound-window TTL behavior. In `marked`, unattended discovery requires an eligible, markable target with a confirmed true native mark and a confirmed selection owner. Unknown metadata cannot grant marked admission. An unresolved native write blocks new admission for its affected resource in **both** modes.

The local controls are `ccgram topic mark TARGET`, `ccgram topic unmark TARGET`, and `ccgram topic status [TARGET]`. Get the exact target ID from `ccgram topic status`: mutations require an explicit ID, never a name or the active GUI terminal. These commands do not require bot credentials. CLI unmark leaves existing topics connected.

Status writes JSON to stdout and diagnostics to stderr. Its fields are `backend`, `mode`, `targets`, and `pending_operations`. Each live target reports `id`, `name`, native `marked` (true, false or null), `markable`, and opaque diagnostic `owner`. An owner is not an input target or routing alias. Pending operations report original `window_id`, `owner`, latest boolean `desired`, and `attempt`, not selected-state truth. A null attempt is prepared retryable work. A non-null attempt records its dispatched desired value and native write witness. Without a target, status includes the complete journal, including references no longer in the live listing. With a live target, it filters operations by exact known owner, falling back to route only when ownership is unknown.

Every CLI mutation uses the shared durable journal. With the bot running, the CLI submits the request to the matching state/backend/native-endpoint owner's private local socket. Backend type and copied owner/target IDs do not establish endpoint identity. Without a running writer, the CLI acquires the same exclusive state lease and preserves unrelated JSON state. A busy lease or unavailable matching daemon never permits a direct-write fallback. Status remains read-only and credential-free.

Opposite prepared intent conflicts with a manual write. Same-desired prepared work can complete under the state owner's journal protocol. Neither same nor opposite intent can bypass an armed attempt. An unreadable, incompatible or malformed journal makes status and mutation fail closed. Native update failure is not reported as success.

A mark selects a terminal/window, not an agent conversation. Replacing an agent or using `/clear` does not change that selection. New native worker targets start unmarked. Explicitly copying native selection metadata is a separate operation with the Herdr limits below. Names, working directories and layout resemblance never prove terminal identity.

Selection does not filter liveness, mobile listings, routing identity, or ongoing forwarding. Existing bound topics keep working after unmarking or changing modes. `/new` and picker binding attempt selection in both modes before binding. In `auto`, a failed write that was not armed retains ordinary binding behavior for a stable native owner. An armed attempt instead blocks new binding and admission until recovery. An initially ownerless target remains admissible in auto mode, but losing a previously confirmed owner blocks unfinished admission until that exact owner is verifiable again. In `marked`, unverified selection prevents binding. Creation ownership and foreground readiness remain in force.

In marked mode, intentional close or `/unbind` of the last binding clears that terminal's native mark, prevents automatic recreation, and leaves the terminal running. Other bindings and active creation claims protect the shared selection from clearing. Deliberate detach is exempt from orphan TTL without changing terminal origin. Abandoned failed creation retains its existing cleanup behavior.

If a target is unmarked during unfinished explicit creation, admission is withdrawn. CCGram retains the exact uncertain creation claim unbound. It does not replay a stale mark-on or delete a possibly human-used topic. If no armed attempt remains, the operator can reselect the native target to permit final recovery. Reselection cannot resolve dispatch ambiguity. Already-bound topics remain connected.

### Unfinished or ambiguous detach

A failed clear remains unfinished lifecycle work, not a permanent suppression preference. Persisted desired-false operations survive restart and block readmission of the exact same native owner, including after conversation re-keying. When both owners are known, exact owner matching takes precedence over routing IDs. An old owner's clear does not block an independent new owner reusing that ID. Route matching is a fallback only when ownership is unknown. Topic maintenance checks unfinished operations every 60 seconds. It retries prepared work, but must resolve an armed attempt before another write or settlement. Independent, verifiable owners remain independently admissible.

If neither saved nor current verifiable ownership exists, CCGram does not guess which terminal to clear or report completed native detach. It keeps the unfinished operation and warns that ownership is unconfirmed. Explicitly unmark the current target to withdraw selection there. This does not itself settle an ownerless journal entry: historical ambiguity can remain quarantined, and the CLI does not rewrite that evidence. Merely losing the old routing target is not success.

### Dispatch quarantine and recovery

Before any possible mutating send or subprocess spawn, CCGram durably arms an exact native write attempt. A timeout, EOF, malformed response, cancellation or crash can leave that write able to execute later. The attempt remains quarantined across CLI and bot restart, including a crash after arming but before send. New admission, retry and settlement for the affected resource remain blocked. Existing bindings and liveness continue.

A current flag, absent owner, completed client process, elapsed timer or healthy snapshot cannot prove that an old write will never apply. Protected no-ops cannot settle an armed attempt either. Opposing intent updates the latest desired value without erasing the old attempt or its original owner.

Normal completion requires a correlated synchronous native success and safe cleanup of the exact owned socket pin, where used. The terminal callback removes only its matching dispatch attempt. It does not establish the latest desired selection: readback and journal settlement remain separate checks.

For uncertain completion, existing maintenance verifies retirement of the exact supported native process and absence of a serving successor. It also safely cleans only the recorded pin. Recovery never kills or restarts a native process. PID reuse, socket replacement or unlink alone is not retirement proof. Unknown provenance or cleanup failure keeps quarantine in place.

After verified retirement, maintenance clears only that attempt and performs fresh native-owner and latest-intent checks. Offline CLI mutation uses the same retirement helper under its exclusive lease and owner lock before attempting current intent. Any required write becomes a new journaled attempt. A changed or unknown owner cannot inherit the old operation. There is no separate recovery command, acknowledgement switch or host-reboot-only default. See the [operator procedure](../guides.md#native-write-quarantine-and-recovery).

## Native storage and restoration

### tmux

Selection lives only in the window-local `@ccgram_topic` option, ignoring inherited global/session defaults. The native value is `v1:1:<32 lowercase hex>` when selected and `v1:0:<same 32 lowercase hex>` when cleared. Clearing retains the owner as a native tombstone. Use the CCGram commands rather than writing raw option values. An absent option is unmarked. Malformed data or a failed read is unknown. A healthy unmarked window has an owner tied to the proven native server process incarnation and native window ID. A reusable ID, pathname or PID alone is insufficient. Creating or removing a dispatch socket pin does not change that owner. Root code never constructs or parses backend owner identities.

A live window retains its selection across agent respawn, CCGram restart and client reconnect while the tmux server lives. Destroying a vanilla server destroys its terminals and window options. External restorers must preserve the full native option value within their own verified window snapshots. No particular plugin is required or claimed as supported. Reused window IDs, names and working directories cannot recover selection.

### agterm

The native session `flagged` boolean selects the primary only. The native session UUID is its owner. A split reports `topic_markable=False` and `topic_marked=False`: explicit binding is allowed but must skip primary flag writes and must not create a primary topic.

### Standard Herdr

Standard `pane.rename` and raw manual labels from `pane.get`/`pane.list` carry selection. Dynamic titles, terminal titles and agent names are not substitutes. The exact suffix is `[ccgram-topic-v1:<32 lowercase hex owner>]`. It follows the original manual label and one space, or stands alone when no original label exists. Ordinary labels do not select. Native snapshots preserve the explicitly written suffix with their saved terminal. This is stored state, not matching a display label heuristically. Unmarked ownership comes from the current native terminal and connection namespace. Once an assigned mark is persisted, its stored owner is authoritative across native restoration.

Marking rejects non-reversible normalization, unsafe serialized requests, reserved/nested/malformed or colliding suffixes, and overlarge labels **before writing**. The complete serialized rename request, including its framing newline, must not exceed 1 MiB. It never truncates labels. Without concurrent manual-label edits, unmark removes only the managed suffix and restores the original label, including null when no original existed. A literal `--clear` label remains user data, not a clear command.

Marker observations join authoritative `agent.list` records by current native terminal identity. A marker-query failure leaves liveness and ordinary auto eligibility intact. Existing dispatch quarantine still blocks new admission. Routing still uses the guarded agent-session target, never the selection owner. Bare shells remain ineligible for automatic topics but remain in the native inventory used to resolve an owned clear.

#### Explicit metadata-copy limit

Normal native spawn/split creates no assigned mark. However, `layout.export` includes manual labels and `layout.apply` copies them into new terminals, including managed selection metadata. Duplicate assigned owners are quarantined: no adoption or marker writes. Templates for new workers must omit managed selection metadata. A native operator can remove a copied suffix with pane rename, preserving the original label.

A sole copied marker is indistinguishable from an imported selection and is treated as explicitly copied selection. CCGram does **not** promise to reset selection for arbitrary marker-copy operations or infer lineage from layout. No fork, upstream publication, sidecar selection registry or label-based routing recovery is required.

## Native verification and limits

Isolated tests against stock Herdr 0.9.5 verified exact Unicode/literal-`--clear` label restoration, conversation re-key and bare-shell clearing, fresh unmarked splits, and duplicate-owner quarantine. Cold restoration preserved the stored owner despite a changed runtime terminal ID. These tests used genuine native helpers with controlled standard hook reports. They did not launch an AI CLI or prove actual provider-session resumption.

On stock tmux 3.8, the guarded native suite passed 18 cases, including token retention, server loss and reused IDs, restored tokens, duplicate-owner quarantine, acknowledgement-loss fencing, and exact endpoint-scope equality through hardlink aliases. Same-boot recovery required verified original-executor retirement and preserved a replacement server. These tests established an owned private namespace before imports and used captured socket/process ownership for teardown. This evidence does not establish that earlier unguarded execution was isolated or harmless. Historical default-server effects remain unknown. Agterm adapter tests and pinned upstream source establish the intended persisted primary flag and unmarked duplicates. An earlier bounded live test verified flag on/off, a fresh adapter reading the same owner and selection, and restoration of the original flag. Its inventory was pinned to the owned test window. That test predates the guarded-write contract and does not verify a positive guarded write, application cold restore, conversation reset, or full multi-window inventory.

Current agterm native metadata validation refused with macOS `-67034` (`errSecCSStaticCodeChanged`): the installed App and running image differ. Normal CLI output can instead report `Native marker endpoint is unconfirmed`. This is a fail-closed diagnostic, not a reason to bypass provenance. An operator may need to save work and restart agterm normally before retrying. CCGram does not restart the application. No positive live guarded agterm write or local Linux-native execution is claimed.

Native CLI read/check/write operations are not an atomic compare-and-set. Inventory, identity and readback checks detect observed changes, but concurrent external metadata edits can still race those checks. No atomic metadata-write or rollback guarantee is claimed. This limit does not waive CCGram's own dispatch-quarantine obligations.

## Backend and policy contract

The existing `topic_eligible` remains separate from existence and visibility. `WindowRef` exposes these fields at the end for positional compatibility:

- `topic_marked: bool | None = False`: `None` is unconfirmed.
- `topic_markable: bool = True`: false for agterm splits.
- `topic_mark_owner: str | None = None`: opaque native selection-resource identity, never a routing alias.

`async Multiplexer.set_topic_mark(window_id, marked, *, owner=None, write_guard=None) -> bool` reports verified native selection, not transport terminality. An actual mutation requires a `TopicMarkWriteGuard`. Without a guard, only an already-desired no-op can succeed after the shared quarantine check. Bot and CLI callers supply an observed or saved owner. An explicit owner permits mutation only of that exact proven resource, even when the old conversation target is gone. First mark requires a matching current unmarked witness. A missing mark-on owner cannot be guessed or recreated after restart.

The stdlib-only `topic_mark_types.py` defines immutable `TopicMarkWriteWitness` and the guard protocol. The witness contains exactly `dispatch_id`, `backend`, `host_id`, `boot_id`, `namespace_id`, `native_pid`, `native_start`, `code_id`, `socket_dev`, `socket_ino`, and `pin_path`. Its read-only `endpoint_scope` property hashes canonical native-peer/process/socket identity through the neutral leaf's shared recipe. Dispatch IDs, pin paths, ctime, nlink, owners, targets and configured path strings do not affect endpoint scope. Hardlink aliases of the same actual endpoint match. The serialized eleven fields remain unchanged. The witness records dispatch provenance, not raw labels, selected-state truth or routing aliases. `arm(witness)` must durably succeed before dispatch. `terminal(dispatch_id)` can remove only its own attempt after native completion and owned-pin cleanup.

`async Multiplexer.topic_mark_endpoint_scope() -> str | None` observes the actual resolved native endpoint. Unknown scope is not an empty or global identity. The CLI requires equal known scopes before and after inventory. The daemon rechecks scope before prepare/no-op acknowledgement, after awaited owner reads, at commit and in the final arm. `arm` compares the witness's scope before any newly runnable manual intent, journal arm or native dispatch. A no-op must also freshly verify endpoint scope. Scope mismatch preserves preexisting intent/fences and any newer OFF, without creating new runnable manual intent.

Herdr mutation stays on its proven connected transport. Mutating agtermctl/tmux commands use an exclusive, never-reused same-inode socket pin and an explicit endpoint. Tmux uses `-N` to prevent autostart. Guarded native writes require the reviewed executor versions: agterm 0.42.3, Herdr 0.9.5, or tmux 3.8. Other versions refuse before mutation. This restriction is separate from ordinary backend protocol compatibility. Native code identity, process birth, host/boot/namespace and reviewed process ownership must be proven before dispatch. Forwarded, inherited or otherwise unproved endpoints fail closed without canonical-socket fallback.

`async Multiplexer.recover_topic_mark_write(witness) -> bool | None` returns true only after verified native retirement/no serving successor and safe owned-pin cleanup. False means that native incarnation remains alive. None means unknown or unproved recovery. None and false both retain quarantine. Threaded adapters execute guard callbacks in the actual state-owner context before dispatch. Native subprocess waits and cancellation use bounded, cooperative paths. Synchronous provenance, fsync or process creation can exceed an asynchronous deadline. No hard wall-clock shutdown guarantee is claimed.

With no armed attempt, complete validated owner absence can prove the old selection off. Duplicate owners, malformed owned markers, incomplete inventory or read errors cannot. Resolution includes bare shells but never falls back to a stale locator, another backend, a display name or a guessed current terminal. An unavailable listing is not an empty listing and never proves closure.

`topic_discovery.py` owns the stdlib-only mode constants and validator. Configuration exposes `config.topic_discovery_mode`. `should_adopt_window(window, *, mode: str, detach_pending: bool, quarantined: bool) -> bool` requires all three keywords. Every automatic site supplies current detach and quarantine state with an explicit observed owner. `is_topic_mark_quarantined(window_id, *, owner)` blocks new admission in both modes. Ordinary detach-pending state retains its marked-mode meaning. The existing `is_agent_topic_window` keeps its baseline eligibility semantics.

## Lifecycle ownership

`WindowState` stores observed `topic_mark_owner` and intentional `topic_detached` metadata. Unfinished native effects live in the owner-keyed `topic_mark_operations` journal within the existing `state.json`, not in a per-window pending flag. Each operation has exactly four serialized keys: original `window_id`, opaque `owner` (or null when unknown), latest boolean `desired`, and `attempt`. A prepared operation has `attempt: null`. An armed attempt contains its frozen `desired` and `witness`. Updating latest intent preserves the existing attempt and original identity. Known owners key operations independently of routing IDs. Unknown-owner operations retain their original route as a quarantine key. Native metadata remains the only selected-state truth. The journal retains retry authority and safety evidence, not routing or a competing selection preference.

Each provisioning claim also retains its original confirmed native owner in the existing state file. Bound-window observations may change independently, but cannot retarget an unfinished claim. New launches capture and checkpoint ownership with selection intent before readiness can time out; final binding still requires fresh native ownership and provider readiness. Automatic recovery reapplies current backend admission eligibility, while explicit targets keep their explicit authority.

A missing state file or journal field means no operations, including ordinary v4.22 state. Unpublished three-field development rows are incompatible. Missing attempt fields, malformed witnesses, duplicate identities, corrupt JSON and unreadable state fail closed without migration or evidence deletion. Both known owners compare exactly, with route fallback only for unknown ownership. Root code never generates backend-specific owner IDs or writes native storage.

One exclusive state-writer lease is acquired before loading SessionManager and retained for the writer's process lifetime. Offline CLI mutation obtains that same lease and preserves unrelated raw JSON. Live credential-free CLI mutation uses the matching state/backend/native-endpoint daemon's private, bounded Unix RPC. There is no lease stealing or unjournaled fallback.

Live mutation uses same-connection prepare/commit under the native-owner lock. Prepare changes no durable intent and authorizes no retry. Its required `endpoint_scope` binds the request to the caller's actual native origin. Its token belongs only to that bounded handler and cannot be replayed on another connection. EOF/deadline closes the handler before releasing its lock. Commit rechecks current endpoint, owner and intent before durable arm. New manual journal insertion is deferred until authorized arm. Unknown or mismatching scope cannot leave newly runnable manual work behind. Disconnect after dispatch cannot erase the armed attempt.

`begin_topic_mark_attempt(operation, witness)` records exact dispatch authority. The state-owner guard must flush it durably before dispatch. `finish_topic_mark_attempt(window_id, *, owner, dispatch_id)` removes only the current matching attempt and retains any newer desired value. Its caller must durably checkpoint completion. Failed arm sends nothing. Failed terminal checkpoint restores the blocking fence. Completion A cannot clear attempt B, including same-value retries.

Explicit selection observes a healthy current owner and persists owner plus desired-on before mutation. Unknown ownership produces a recoverable diagnostic, not a candidate identity. Automatic admission records the actual observed owner before binding **without marking** automatically discovered targets, including unmarked targets in auto mode. Already-bound, verifiable rows refresh current observed ownership independently of pending operations. Those immutable operations retain their original captured owner; observation does not retarget or discard them. Unadmitted workers do not acquire window state merely for being listed.

Before last detach, capture current verifiable ownership while the binding remains intact. A saved owner is authoritative for that detach. Neither unavailable metadata nor a different live owner may substitute another resource. Capture current ownership only if no saved owner exists; the backend must verify the captured owner on retry. Synchronously unbind, record owner plus desired-off/detached, and flush before awaiting native clear. Retry by saved owner without requiring the old routing target to resolve. Lock by owner when known. Recheck remaining bindings and creation claims under that lock before clearing. If another consumer now protects the resource and no armed attempt remains, complete the clear operation without changing native selection. Preserve journal operations through cancellation, stale pruning and checkpoint failure. Observing a new owner must not retarget or discard an old owner's operation. Split exceptions never touch the primary.

Fresh automatic admission checks run inside the actual creation-ownership lock and again before commit or recovery. A stale pre-lock verdict must not bypass a newly pending owner-clear or armed attempt. Explicit picker/launch and recovery commits also recheck quarantine before a new binding. Unmarking alone still leaves admitted topics connected. Settling a native operation, including a protected no-op, does not remove detached-lifetime TTL protection. Only successful actual readmission clears that protection, not observing a true native flag.

## Acceptance matrix

This matrix summarizes the policy, lifecycle and native-storage contract. Native execution limits are listed above.

| Scenario | Expected result |
| --- | --- |
| Default `auto`, eligible unmarked or metadata-unknown target, no armed attempt | Preserve automatic admission and TTL. Selection metadata does not filter ordinary auto. |
| `marked`, true mark with confirmed owner / missing owner / false / unknown | Admit only the confirmed true mark with an owner, eligibility and markability. |
| Ineligible target or unavailable listing | Do not adopt ineligible targets. Unavailable listing never proves death. |
| Unmark or switch mode with an existing binding | Keep binding, forwarding, liveness and mobile listing. |
| Agent replacement or `/clear` | Keep native selection independent of conversation identity. |
| Normal new target; reused tmux window ID after server loss | Start unmarked. No name/cwd/layout recovery. |
| Native snapshot restore with full owned marker | Preserve selection only through verified native metadata, not transient tokens. |
| Ordinary Herdr label / exact managed suffix | Ordinary label does not select. Only valid owned metadata can select. |
| Herdr normalization, nested suffix, excessive size, literal `--clear` | Reject unsafe mutation without data loss. Preserve literal label data. |
| Herdr copied layout, duplicate owner / sole copied marker | Quarantine duplicates. Sole copy is explicitly copied selection, not reset automatically. |
| `/new` or `/bind`, native mark failure | Prepared failure retains ordinary auto behavior. Armed uncertainty blocks new binding in both modes. Marked also requires selection verification. |
| Unmark during unfinished explicit creation | Retain exact uncertain claim unbound, without stale mark-on or topic deletion. Native reselection permits recovery only after dispatch quarantine is resolved. |
| agterm explicit split binding | Bind split without flagging or auto-adopting primary. |
| Last marked-mode detach / another binding remains | Clear only after last binding. Keep terminal alive and exempt intentional detach from TTL. |
| Lost mark acknowledgement | Retain armed attempt. No retry or settlement until exact native terminality or verified retirement. |
| Conversation re-key or bare shell, no armed attempt | Retry the exact saved owner, not the old conversation locator. |
| Clear fails, local checkpoint fails or current owner changes | Retain original owner-keyed operation across restart without retargeting it. No competing suppression preference. |
| Manual CLI write conflicts with journal | Reject opposite prepared intent. Same-desired prepared work uses the state owner's protocol. Any armed attempt blocks mutation. |
| Known new owner reuses routing ID with old-owner clear pending | Keep old operation, but admit independent new owner if otherwise eligible. |
| Owner unknown, duplicate or stale | Fail closed without clearing another terminal. Operator unmark can withdraw current selection, but cannot erase unknown ownership or an armed attempt. |
| Clear owner absent in complete validated native inventory | Without an armed attempt, backend may verify old selection off without mutating another resource. |
| Pending clear or quarantine appears after pre-lock admission check | Recheck under creation ownership before bind/recovery. |
| Crash before/after send, timeout, EOF or cancellation | Keep durable armed uncertainty, even if a later snapshot matches desired state. |
| Same-boot native retirement verified | Clean only exact owned pin, remove matching attempt, then freshly check owner/latest intent. No native kills or restarts. |
| Wrong/unknown endpoint, proxied executor or installed/running code mismatch | Refuse mutation. Preserve liveness and existing bound delivery. |
| Ordinary v4.22 state / unpublished three-field development journal | Missing journal key is valid. Incompatible development rows fail closed without dropping evidence. |
| Abandoned failed creation | Preserve existing cleanup and TTL. |
