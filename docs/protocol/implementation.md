# Implementation and verification

Telegram and the mobile API use one in-process session core around existing
terminal sessions. Neither client launches an agent SDK or wrapper runtime.
Telegram calls the core directly. HTTP and WebSocket expose its wire contract.

| Module | Responsibility |
| --- | --- |
| `session_protocol/models.py` | Dataclasses, wire serialization, and protocol errors. |
| `session_protocol/backend.py` | Consumer-owned backend Protocol and its Multiplexer adapter. |
| `session_protocol/service.py` | Session discovery, snapshots, history, live replacements, and delivery correlations. |
| `session_protocol/transcript_source.py` | Stat-first, append-only observation of one transcript: identity, generation, and byte-range reads of spilled values. |
| `session_protocol/messages.py` | Canonical provider normalization, structured records, and the history/live message builder. |
| `transcript_reader.py` | Delivery cursor reads and Telegram rendering of canonical records. |
| `session_protocol/telegram.py` | In-process Telegram adapter and application-owned service access. |
| `session_protocol/runtime.py` | Always-on core, process lock, recovery, hook queue, and polling lifecycle. |
| `session_protocol/actions.py` | Schema boundary checks, persistent deduplication, binding checks, and input execution. |
| `session_protocol/input_lock.py` | Task-reentrant per-target locks shared with Telegram text and interactive input. |
| `session_protocol/requests.py` | Open requests, hook evidence, supported options, and close events. |
| `session_protocol/events.py` | Epochs, durable replay, subscription filtering, and overflow detection. |
| `session_protocol/store.py` | SQLite identity, device, action, source-generation, and delivery metadata. |
| `session_protocol/uploads.py` | Private uploaded files and opaque references. |
| `api/server.py` | Host/Origin checks, bearer auth, rate limits, HTTP routes, and WebSocket lifecycle. |
| `api/runtime.py` | Optional HTTP listener and its cleanup. |

`bootstrap.py` owns creation and teardown. It creates the core even when
`CCGRAM_API_PORT` is unset and injects the service into the Telegram application.
Only the HTTP listener depends on that setting.

Protocol session IDs and native target IDs are separate. Telegram creation returns
`CreatedSession.target_id` for topic claims and multiplexer lookups, alongside
`CreatedSession.session.id` for protocol operations. A topic must never bind the
opaque protocol ID as its native window ID.
The API and core have no imports from Telegram or its handlers. The worktree helpers
now live in `worktrees.py`, where both session creation and the Telegram directory
flow can use them without a handler dependency.

## Compatibility and migration boundary

History uses only the transcript binding already established by ccgram. It does
not guess a conversation from the latest transcript in the same directory. Claude,
Codex, Gemini, Pi, and Antigravity have structured history. Other or unbound transcripts use the
terminal view.
Persisted terminal identities resolve case-insensitively. Backend calls retain the
backend-reported spelling. Gemini uses `displayContent` when its primary content
has no displayable parts, while preserving primary tool results.

The previous `transcript_parser.py`, provider parse methods, and direct window-send
wrappers are removed. Provider records have one interpretation: `parse_records` validates the native
conversation envelope and role before producing structured records used by both
`build_messages` and Telegram delivery. Provider prompts, diagnostic request
snapshots, and unknown record kinds are not conversation. Unknown providers have
no generic chat fallback. Codex injected context can carry a user role; its
reserved context envelopes are excluded without stripping ordinary mentions or
explicit tool output. Resume and completion summaries and Claude task extraction
also use this classification. They cannot interpret internal records as user
text, final answers or tasks. Telegram rendering
keeps formatting, tool batching, topics, chunking, edits, and queue retries in the
adapter. Codex streaming completion and phase survive normalization. Tool results
can complete calls observed in an earlier delivery batch.
Claude local-command replies and Pi shell-execution records become notices.
Claude command names survive across parser batches until their reply arrives.
Protocol message construction projects records into the declared `Part` fields.
Telegram rendering retains its command metadata. Telegram history includes an
`ExitPlanMode` plan before the corresponding tool and result rows.

History and live mobile events read a verified transcript observation. Each
observation starts with one stat. An unchanged stamp (device, inode, size, mtime,
and ctime) reads nothing. Otherwise one open handle checks the previous
observation's identity, parses only the complete records appended since the last
line boundary, and digests the new tail. The shared `MessageBuilder` in
`messages.py` builds the message index only when something reads it. Message events
are published for a session while an EventHub subscriber covers it, and for 300
seconds after its last subscriber leaves (`SUBSCRIBER_GRACE_SECONDS`). The index is
resident in that window and for 60 seconds after an on-demand history read
(`INDEX_IDLE_SECONDS`). When publishing stops, the hub records a gap
(`EventHub.mark_gap`), so any resume at or before it fails and the client reloads.
Otherwise the source keeps only its cursor, and nothing is parsed except appended
bytes and bounded windows: delivery correlation covers the appended range and the
newest windows until the send is matched, and hook matching searches the newest 16 MiB
(`HOOK_SCAN_BYTES`). A session that gains a subscriber gets its current messages as
upserts from its first publish.

The resident index keeps message IDs, revisions, source offsets, and small
parts. A text, tool input, or diff over 4 KiB, and a tool output over 4 KiB, stay in
the transcript. The index keeps their record offset and slot, and materialization
re-reads that one record by byte range through a per-target LRU capped at 512 KiB.
Publishing pins one handle per batch. A replaced transcript raises
`SourceChangedError` and the source is observed again. Batch builds keep every value.

Identity is the inode, the observed size, and a digest of the last 4 KiB before that
size. A replacement, a truncation, or a changed tail increments the persisted
generation and rebuilds from byte 0. A rewrite that keeps the inode, size, and tail
is not detected; transcripts are append-only, so this is accepted. Gemini chats are
JSONL (`session-*.jsonl`) and take the same incremental path. A failed read is retried when the stamp changes or
after a backoff of 1 s doubling to 30 s.

A source replaced or truncated during a read is retried. A final source check after
publication prevents action preflight from accepting a source that changed during
observation. An unstable source becomes unavailable and exposes no input actions.

Telegram delivery uses a separate byte cursor because its receipt checkpoint must
survive queue failures and restart. This is a delivery policy, not another provider
parser. The shared event hub dispatches normalized delivery batches to durable
in-process consumers. Telegram renders them under delivery receipts. Only settled
receipts advance its checkpoint. Consumed input with no visible records still
produces a checkpoint-only batch. A tool-call fence stays in place until delivery
through its result checkpoint settles. Later unrelated output does not hold that
fence. A completed backlog skip retires pending calls and tool fences for the
skipped range. Canceled skips leave the durable checkpoint unchanged and rewind
parsing if their receipts retire. Startup, completed skips, and session identity
adoption rebuild parser context from committed source bytes. The rebuild runs on
the first poll that has new entries to parse, so an idle restored session costs
nothing until it produces output. Source replacement
retires the old generation's receipts and parser context. It emits a
checkpoint-only batch at the replacement's EOF. Receipt settlement persists that
boundary.
Skip fences retain their source boundary.
Mobile replay subscriptions keep their bounded queues and never hold Telegram's
checkpoint. Telegram does not replay the mobile history ring at startup.

All Telegram input goes through the shared executor/backend. Handlers retain
ownership checks and presentation, then call the in-process adapter. The executor
owns reservation, binding checks, prompt checks, and the per-target input lock.
A handler that authorizes shell input against a resolved snapshot passes that
binding version through the adapter. A changed foreground or pane rejects input.
Shell submission is one atomic paste-and-submit operation. Shell prompt setup is
one internal executor action: interruption, setup text, and optional clearing hold
the same target lock throughout. Providers generate setup commands and receive the
adapter explicitly. Setup writes recheck the captured pane and foreground.
Explicit sibling-pane
controls retain pane ownership checks. The multiplexer does not expose a sibling
foreground PID. Those controls therefore do not claim the active pane's PID guarantee.
Explicit sibling-pane controls are available on tmux. Herdr and agterm use their
active target because they expose no pane inventory through this seam.
Local creation options preserve Telegram's configured launch/resume behavior.
Remote creation remains restricted to normal approval mode.

Live Telegram history, including byte-range unread history, delegates to the same
service observation. Session discovery and archive summaries can read a named
transcript through the canonical reader. They do not select a live source.

The handler boundary test rejects direct multiplexer input and lifecycle mutations.
There is no exception baseline or fallback to the former Telegram send helpers.
APNs, the native iOS UI, and a Telegram-free process remain later work.

## Contract corrections

The implementation makes these clarifications to the proposed design:

- Persistent state uses `protocol.sqlite3`, separate from delayed `state.json` saves.
  Full commits reserve an action before any input and preserve an unknown outcome
  after a crash. Device and action storage use the Python standard library.
  `ActionResult.data` carries optional result metadata, including the created pane
  for an in-process split retry. `error` is reserved for failures.
- `bindingVersion` remains a persisted increasing integer. The active pane,
  foreground PID and arguments, source availability, and transcript generation
  also participate in the binding. agterm uses its observed arguments when no PID
  is available. Unknown foreground state blocks input actions. Closing remains
  available subject to native target and binding checks.
- Message revision counts repeated native records and result records. Adding an
  action correlation also increases the revision, so a client replaces an entry
  published before send acknowledgement. The association and revision survive
  restart.
- `historyGeneration` marks source replacement. Clients clear the old conversation
  cache when it changes. A restart retains the generation. Continuity checks hash
  the full previously observed prefix in a worker thread. Upgrading the earlier
  first-page fingerprint format invalidates the cache once.
- A resume at the current sequence succeeds even with an empty ring. `welcome`
  precedes replay. Subscription queues are installed before reading replay.
- Cursors use opaque conversation/generation/message-boundary tokens. The server
  reads and pairs the source transcript before paging. It has no persistent message
  index or message database.
- Hook events retain a tool-input digest instead of raw input. Matching reads the
  authoritative input from the transcript. Hooks remain observations and return no
  permission decision.

Raw hook intake and transcript resolution use the same `HookIdentity`: conversation,
kind, tool name, input digest, and timestamp. `RequestStore` checks freshness,
ordering, and replay before accepting either form. A distinct identity retires the
previous request synchronously, before queueing or reading the transcript. Deferred
tickets block approvals and messages until processing finishes. Expiry in the queue
cannot restore the retired request.
An unmatched transcript lookup retires only its own identity. A newer identity
received during that read remains available for later matching.

Refresh, current-session inspection, creation, and hook transcript resolution share
one lock to serialize authoritative observations. An unavailable
transcript or pane capture degrades only that session to unknown status and blocks
input actions. Its identity stays present, closing remains available subject to
native target and binding checks, and healthy sessions continue to refresh.
Successful delivery registration shares that lock and invalidates the transcript
stamp, so the next refresh can correlate a user entry observed before send acknowledgement.

tmux message submission uses the existing vim recovery, command-mode handling, and
delayed Enter while pinning every capture and key to the verified pane. Raw paste
and terminal keys retain their direct input behavior. Shell launches and resume
also use raw input, so cached agent vim state cannot add editing keys to a command.

The current history builder reads the transcript before paging. Large conversations
therefore cost a full parse when content changes or history is requested. Add an
incremental source index if profiling shows that cost dominates API response time.
The single builder and identity rules must remain unchanged.

## Tests

Run the focused conformance and integration checks:

```console
uv run pytest tests/ccgram/session_protocol -q
uv run pytest tests/integration/test_session_protocol.py -q
```

The unit suite covers message identity, revisions, source replacement, pagination,
and partial JSONL writes. Action tests cover crash reservations, retries, binding
changes, request safety, and uncertain delivery. API tests cover uploads, pairing,
revocation, Host/Origin guards, rate limits, replay, queue overflow, and schemas.
Regression checks cover PID-less foreground identity, safe resume, pinned tmux
input, exact question labels, stale hook reuse, and multi-select fallback. They
also cover per-session source failures, binding races, and transcript rewrites
beyond the first page. Schema checks validate nonempty history and live
message frames from Claude, Codex, Gemini, and Pi, including internal Codex role
filtering. A backend regression checks Antigravity bound history through the core.
Additional regressions cover concurrent source replacement and equal-length
rewrites during history, live refresh, and action preflight. Continuously changing
sources fail closed. They also cover callback lock timeout acknowledgement.
Other regressions cover Gemini display fallback, creation errors and failed
post-launch inspection, persisted ID casing, delivery before acknowledgement, and
terminal resizes without content changes.

Golden payloads live in `protocol/v1/fixtures/`. `_manifest.json` maps each fixture
to its shared JSON Schema. The server's builder output is compared with the golden
message fixture. The iOS package can decode the same payload files. Update schemas
and affected fixtures together.

The integration test starts an isolated tmux server and a deterministic terminal
program, then uses real capture/input through the Multiplexer adapter and real
HTTP/WebSocket connections. It checks list, history, send, verified hook response,
resume, and durable replay. It does not call an LLM or require a paid agent account.
It skips only when tmux is unavailable.

Run `make check` for formatting, lint, types, dependency checks, unit tests, and
integration tests. APNs and the iOS UI require their own later implementation and
verification.

## Native relay smoke test

From an agterm session, run:

```bash
uv run --python 3.14 python scripts/smoke_agent_relay.py --provider claude --cwd "$PWD"
```

Use `pi` or `codex` for another installed CLI. Lifecycle hooks must already be
installed for that provider; the harness does not change global agent settings or
guess identity from the newest transcript. It fails without sending input when
no ready, attested hook mapping appears.

This starts a dedicated agent in a separate agterm workspace and makes three real
model requests: initial, follow-up, and after restarting the session service and
monitor. It uses the normal input executor, transcript normalization, Telegram
routing and queue, and receipt checkpoints. `FakeTelegramClient` records output
and acknowledges it locally; no request goes to the production Telegram bot.
The account's normal model usage charges still apply.

State and `report.json` stay in a private temporary directory printed by the
command. The harness closes only its create-only target and deletes its workspace
only after confirming it is empty. It does not resume another bridge, restart
existing agents, edit raw transcripts, or change delivered watermarks manually.
