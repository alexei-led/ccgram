# Implement a ccgram/1 client

Use [`openapi.yaml`](../../protocol/v1/openapi.yaml) for HTTP models and
[`asyncapi.yaml`](../../protocol/v1/asyncapi.yaml) for stream frames. JSON is UTF-8.
Ignore unknown response fields and unknown server event types. Check capabilities
before offering an operation.

## Authenticate

Scan the payload from `ccgram pair`, verify its HTTPS origin and fingerprint against
the Mac, then exchange the code:

```http
POST /v1/devices/pair
Content-Type: application/json

{"code":"<one-time code>","name":"My iPhone"}
```

The `201` response contains `deviceId`, `deviceToken`, and `fingerprint`. Every other
HTTP request and the WebSocket upgrade needs `Authorization: Bearer <deviceToken>`.
Tokens in query parameters are not accepted. Never log the pairing payload or token.

## Initial sync and reconnect

1. Open `/v1/stream` and send `hello`. Include the last persisted epoch and sequence
   when reconnecting.
2. Read `welcome`. Its `data.seq` is the replay boundary. When `resumed` is true,
   the server sends missed durable events through that boundary before live events.
3. Send `subscribe` before loading snapshots or history. A fresh connection initially
   observes all sessions, including during replay, so there is no unsubscribed gap.
   A resume succeeds only if no message events were lost since the cursor. Message
   events are published to a session's subscribers for 300 seconds after their last
   subscriber leaves. A reconnect within that window replays every message event
   exactly, as long as the replay ring still holds them: it keeps at most 1,000
   durable events (shared by all sessions) and at most 600 seconds of events. A
   longer backlog or gap returns `resumed: false`, even inside the grace window.
   After the grace window, the server stops publishing messages for that session and
   marks the gap. Any resume at or before the gap returns `resumed: false`. Subscribing
   again afterwards sends each current message as an upsert, so no message is missing.
4. Load `/v1/sessions` and the open conversation's history on a fresh connection, a
   new epoch, or a failed resume. Refresh snapshots on every app foreground to remove
   requests that closed while the app was suspended.
5. Merge messages by opaque `id` and `revision`. A higher revision replaces the full
   message. Ignore equal or lower revisions. Remove a request for every close reason.

```json
{"type":"hello","data":{"protocol":"ccgram/1","resume":{"epoch":"e_example","seq":42}}}
```

```json
{"type":"subscribe","data":{"sessionIds":"*"}}
```

Durable sequence numbers are global within a server epoch. Filtering subscriptions
can leave sequence gaps. A gap in a filtered stream does not itself mean data loss.
Use the largest received durable sequence as the next resume cursor. Do not persist
`welcome.data.seq` until all replay frames through that boundary have been applied.
If no replay frames are needed, the boundary is already applied.

The server retains at most 1,000 durable events and at most ten minutes of events.
An up-to-date cursor resumes even when the ring is empty. An unknown epoch, future
cursor, or expired replay range sets `resumed` to false. Each connection has a bounded
queue. Overflow closes the socket with code `4009` (`resync_required`). Discard its
resume cursor and reload snapshots/history. Authentication revocation closes with
`4401`. Malformed client frames close with `1008`.

HTTP JSON responses carry `X-CCGram-Epoch` and `X-CCGram-Seq` for diagnostics.
These headers describe response emission, not an atomic transcript snapshot. They
do not replace the subscribe-before-history merge rule.

## History and identity

`GET /v1/sessions/{id}/history?limit=50` returns messages oldest first, `conversation`,
`historyGeneration`, `prevCursor`, and `historyLimit`. Limits are 1–200. Pass
`before=<prevCursor>` to load older messages. An optional `conversation` must match
that session's currently bound conversation. Cursors are opaque, scoped to both
conversation and transcript generation, and invalid after source replacement.

Message upserts are sent only to subscribers. A client that has not subscribed to a
session gets its messages from history reads, which build the index on demand; a
client that subscribes later receives the current messages as upserts. Do not rely
on the replay buffer for messages published before you subscribed.

Native transcript entry IDs are retained when first observed. Missing native IDs
use UTF-8 byte offsets. A streamed Codex message keeps its first visible ID when
the final response arrives.
A tool result replaces the tool part in its original message and increases the
revision. Adding `actionId` after send acknowledgement also advances the revision.
Result-only entries make no separate bubble. Tool pairing happens before
pagination, including results beyond a page boundary. Repeated Gemini entry IDs
produce higher revisions when the provider appends another version of the record.
`complete` reflects streaming completion. Optional `phase` preserves provider phase
metadata, such as Codex `final_answer`. Clients can ignore phases they do not know.

A transcript replacement or truncation increments persisted `historyGeneration`.
A change that keeps the file identity, size, and last 4 KiB is not detected; the
transcript is append-only in practice.
Nonzero generations also change message IDs. On a changed `conversation.id` or
`conversation.historyGeneration`, clear that conversation's cached messages and
load its history again. `bindingVersion` also increases, invalidating queued input.

Sessions without a transcript report `historyLimit: "scrollback"` and an empty
message history. Use the terminal view for their available screen content.
Transcript images currently appear as notices. Their local paths and embedded data
are not exposed as public image URLs.

## Execute actions

Persist a UUID and the complete action body before sending it:

```json
{
  "id":"00000000-0000-4000-8000-000000000001",
  "bindingVersion":12,
  "kind":"message.send",
  "data":{"text":"Run the tests again."}
}
```

Send it to `POST /v1/sessions/{id}/actions`. Retry transport failures with exactly
that ID and body. Reusing an ID for a different body or session is an error.

| Outcome or error | Client behavior |
| --- | --- |
| `sent` | Show a sent mark. This confirms terminal delivery, not that the agent has processed the input. |
| `action.updated` with `messageId` | Show delivered. Matching uses conversation, normalized whitespace, and a transcript timestamp within 60 seconds of sending. History also carries the correlated `actionId`. |
| `unknown` | Stop retries and show “Delivery unknown. Check the terminal.” A crash or a partial composite input can cause this result. |
| `409 binding_conflict` | Read the current snapshot in `error.details`, keep the draft, and require a new user intent with the new binding version. |
| `409 request_open` | Refresh requests and wait for queued hook processing. If a prompt remains, confirm dismissal before a new `message.send` with `dismissRequest: true`. |
| `409 request_closed` | Refresh the requests. A replaced or closed request requires a new user intent. |
| `429 rate_limited` | Wait `error.details.retryAfterMs`, then retry the same action. |

A known action ID returns its original result even when its session is now gone.
Binding checks include the foreground process and active pane. Input stays pinned
to that pane. A focus change cannot redirect the remaining keys to a sibling.
agterm supplies foreground arguments without a PID. Those arguments participate
in its binding. A restart with identical arguments in the same pane cannot be
distinguished through that backend. Unknown foreground observations block input
actions. `session.close` remains available subject to native target and binding checks.
A process can still exit between verification and input. The multiplexer has no
atomic compare-and-send operation.

`message.send` dismisses only after explicit `dismissRequest: true`, then confirms
that the prompt is gone before sending text. `terminal.paste` sends literal text
without Enter. Use `terminal.key` with `Enter` to submit terminal input.

Session resume requires a bare shell, an idle ccgram prompt marker, and a tracked
previous agent conversation. It excludes scripts, commands, and shells waiting for
input. Unmarked shells use terminal controls. The server checks the shell and marker
again before injecting the launch command. The conversation ID must name that
previous conversation. A running agent must first return to its shell.
Session creation is a separate HTTP operation
and has no action ID: do not automatically retry an ambiguous creation failure.
Backend creation failures, including a failed inspection after launch, return
`503 backend_unavailable`. Check the session list and terminal first.
Native worktree creation cannot select an existing workspace. Combining `worktree`
with `workspaceId` on that backend returns `422` before creating anything.
For worktrees, supply `branch` or let the server choose an unused `ccg/agent-N`
name. Undeclared creation fields, including `title`, are ignored.

## Requests

A `screen` request has no structured response controls. Show the terminal and key
bar. A `hook` request can expose supported once-only options after its tool input
digest matches the transcript and its prompt matches the screen. The executor reads
the screen again before answering and verifies that the prompt closed afterward.
If another observation replaces or closes the request during that screen read,
the server rejects the response before sending input. New hook evidence invalidates
the old response immediately, even before the next screen refresh creates a new
request. Finishing an earlier response or dismissal preserves that new evidence.
An absent prompt retires its request and preserves evidence for a newer prompt
that has not rendered yet. Explicit turn or session invalidation clears that evidence.
An observation does not consume itself. Replacement or explicit invalidation
consumes pending evidence even if it has not bound to a matching prompt.
Consumed or explicitly invalidated hooks stay blocked for their full matching
window, including after a different hook binds. Duplicate events cannot reopen them.
Consumed and expired observations are ignored before they can displace newer
controls. A continuously active request keeps its bound evidence until it retires.
A distinct hook supersedes pending evidence immediately, before screen
reconciliation. Replays cannot restore the superseded request's controls.
Hook intake blocks approvals and messages for the affected session until queued
observations finish processing. A fresh unmatched digest removes old controls and uses
terminal fallback. Queue overflow or processing failure keeps affected approvals
blocked. See [operator recovery](operations.md#hook-processing-recovery).
If polling closes the original request during dismissal, the server also stops
before forwarding the message. Escape was already sent, so the outcome is
`unknown`. Refresh the session and ask for a new user intent.
Hook evidence belongs to one prompt fingerprint and binding. A changed prompt needs
fresh evidence. Permission commands and question labels must match exactly.
Ambiguous or decorated choices use terminal controls.

The current responder supports numbered single-choice menus, including a single
`AskUserQuestion` question whose choices match the observed menu. Its
`responseSchema` enumerates accepted answer labels. Free text, multi-select,
multiple-question forms, plan approvals, and “allow always” use terminal controls.
Unsupported structured responses return `422`. Do not infer options from a screen
request. Show the full tool input before an approval and use device authentication
for allow actions, as required by the iOS design.

## Terminal and files

`GET /v1/sessions/{id}/terminal?pane=p0` returns a full ANSI frame with dimensions
and a content hash. `terminal.watch` sends a frame when its content hash or dimensions
change, at intervals from 500 to 30,000 ms. These frames are ephemeral and have no
sequence number. One active-pane view (`p0`) is exposed per control session.
A new watch replaces the previous watch
on that connection. `terminal.unwatch` stops it.

Upload bytes with `POST /v1/uploads`, the media type in `Content-Type`, and a plain
filename in `X-Filename`. The limit is 20 MiB. The response is a file reference
(`id`, `url`, `name`, `size`, `mime`). Add `{"id":"<upload ID>"}` to
`message.send.data.files`. The service appends the server-owned upload paths to the
agent message. A client cannot supply an arbitrary server path. Download URLs require
the same bearer authentication.

## Smoke client

The repository includes a small test client:

```console
export CCGRAM_DEVICE_TOKEN='<paired device token>'
uv run python scripts/session_protocol_client.py --url https://mac.example-tailnet.ts.net sessions
uv run python scripts/session_protocol_client.py --url https://mac.example-tailnet.ts.net history s_EXAMPLE
uv run python scripts/session_protocol_client.py --url https://mac.example-tailnet.ts.net stream --count 1
```

For an action, save its complete JSON to a file and use `action s_EXAMPLE action.json`.
Reuse that file for a retry. Tokens are read from the environment, never command-line
arguments. The client also permits plain HTTP on loopback for local conformance runs.
