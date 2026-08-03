# Plan: Route CCGram topics to guarded Herdr agent sessions

## Quick guide

This is the only plan for this change.

This plan changes CCGram only. It does not change Herdr source code.

Run the tasks in order. Each task is one small delivery slice.

Do not start the next task after a failed check.

Stop at each stop condition. Record the reason in the task report.

## User result

Each Telegram topic maps to one Herdr agent session.

If the session moves between panes, tabs, workspaces, and terminals, the topic follows that session.

The topic does not follow a pane slot or a focused pane.

CCGram does not claim atomic delivery. Current Herdr actions accept terminal, pane, or name targets.

A session can change after CCGram reads `agent.list` and before Herdr receives an action.

CCGram reduces this risk with a fresh guard before every action. CCGram records this remaining risk.

## Terms

| Term | Meaning |
| --- | --- |
| session composite | The Herdr values `source`, `agent`, `kind`, and `value`. |
| session target ID | A versioned digest of one complete session composite. |
| live record | One current `agent.list` record for a session target ID. |
| live locator | Current `terminal_id`, `pane_id`, `tab_id`, and `workspace_id`. |
| sessionless agent | An agent record without `agent_session`. CCGram creates no topic for it. |
| unresolved target | A target with zero current live records. |
| ambiguous target | A target with two or more current live records. |
| legacy binding | An old Herdr tab or pane binding. CCGram blocks its actions. |
| guarded action | An action that uses one fresh live record for one dispatch. |

## Architecture

```text
Telegram topic
    ↓ opaque session target ID
Topic and session state
    ↓ neutral multiplexer contract
Herdr adapter
    ↓ agent.list and one guarded action
Current Herdr terminal or pane
```

### Module ownership

| Module | Owns | Does not own |
| --- | --- | --- |
| `multiplexer/base.py` | Neutral types, target creation contract, capabilities, and method contracts. | Herdr fields or RPC details. |
| `multiplexer/herdr.py` | Herdr protocol calls, session parsing, digest, guard, live locator use, tab creation, agent launch, and session wait. | Telegram state or generic recovery rules. |
| `multiplexer/topic_mapping.py` | Sessionful target eligibility and labels. | Herdr discovery or topic persistence. |
| `multiplexer/self_identify.py` | Hook input to a neutral self-identity result. | Herdr fallback identity. |
| `hook.py` | Hook I/O and event writing. | Herdr target selection rules. |
| `session_map.py` | Opaque binding map persistence. | Herdr pane or tab interpretation. |
| `window_resolver.py` | Generic stale-target outcomes. | Display-name or pane-slot recovery. |
| `session.py` and `thread_router.py` | Binding and routing state. | Herdr discovery. |
| Handlers | User actions and user messages. | Concrete Herdr imports. |
| Tests | Public behavior, contract, race, and live evidence. | Private call-count assertions. |

## Coupling design

Use the Balanced Coupling model for each boundary.

| Boundary | Strength | Distance | Volatility | Decision |
| --- | --- | --- | --- | --- |
| CCGram generic code → `Multiplexer` | Contract | High | Low | Keep the neutral protocol. |
| Herdr adapter → Herdr `agent.list` payload | Contract | Low | Medium | Keep parsing inside the adapter. |
| Herdr adapter → Herdr terminal or pane action | Functional | Low | Medium | Keep locator use inside one adapter method. |
| Topic state → session target ID | Contract | Medium | Low | Persist only the opaque digest. |
| Hook → Herdr adapter | Contract | High | Medium | Use an injected neutral query. |
| Handlers → Herdr adapter | Intrusive | High | Medium | Prohibit direct imports. |
| Topic launch → neutral target creation | Contract | Medium | Medium | Pass selected workspace as an opaque value. |
| Generic recovery → display or layout data | Functional | High | Medium | Delete this coupling. |

The handler boundary has high strength and high distance. Keep it neutral to prevent a costly cascade.

The Herdr adapter has high strength and low distance. Keep Herdr details there.

The session target ID is a narrow contract. Do not expose the session composite above the adapter.

The old display and layout recovery is high strength and high distance. Delete it.

## Architecture fitness

The repository already runs `make arch-guard` and `make arch-check`.

Add these failing checks:

- Generic modules cannot import `multiplexer.herdr`.
- Handlers cannot import concrete multiplexer backends.
- Herdr identity code cannot use `_active_pane` or `_representative_pane`.
- Herdr target persistence cannot use tab IDs, pane slots, names, titles, directories, or focus.
- Only the Herdr adapter can parse `agent_session`.
- Only the Herdr adapter can use a Herdr live locator.
- Current docs cannot describe tab or pane identity as the Herdr topic identity.
- One canonical digest function exists for session target IDs.
- Topic launch can pass an opaque selected workspace ID. Handler code cannot call Herdr tab APIs.

These checks are new gates. Each gate must fail before its rule is implemented.
Each gate must pass after its rule is implemented.

Name the checks as follows:

- `tests/ccgram/test_herdr_boundary_audit.py` checks generic imports, handler imports, Herdr field parsing, and live-locator ownership.
- `tests/ccgram/test_herdr_identity_audit.py` checks one digest function and forbidden identity fallbacks.
- `tests/ccgram/test_herdr_legacy_model_audit.py` checks old code, tests, configuration, and current documentation.

Run all three tests in `make arch-guard`.

## Source evidence

- GitHub issue #137 requests independent agent topics inside one Herdr tab.
- Herdr `AgentInfo` contains `agent_session`, `terminal_id`, `pane_id`, `tab_id`, and `workspace_id`.
- Herdr `agent.list` reports agent records across tabs and workspaces.
- `src/ccgram/multiplexer/self_identify.py` currently maps `HERDR_PANE_ID` to a Herdr tab.
- `src/ccgram/multiplexer/herdr.py` currently uses `_active_pane` and tab identity.
- `src/ccgram/session_map.py` currently stores backend-specific tab keys.
- `tests/integration/test_herdr_contract.py` currently asserts tab identity and focused-pane behavior.
- `docs/architecture.md` currently says that each topic maps to one multiplexer window.
- `README.md` currently describes one topic per multiplexer window.

## Design decisions

- **D1: One identity source.** Use Herdr `agent.list` only.
- **D2: One durable identity.** Persist a versioned digest of the complete session composite.
- **D3: Fresh guard.** Read a fresh `agent.list` snapshot before every action.
- **D4: Short-lived locator.** Use the matched terminal or pane only for that action.
- **D5: Fail closed.** Block zero, duplicate, malformed, sessionless, and pre-guard changed targets.
- **D6: No container target.** A Herdr tab is a location, not a Telegram topic.
- **D7: No implicit migration.** Archive legacy bindings and require explicit rebind.
- **D8: One refresh path.** Refresh from `agent.list` after reconnect, event loss, or action error.
- **D9: Known race.** A post-guard session change can cause misdelivery. Record and document it.

## Target contract

Create the target ID from canonical UTF-8 JSON.

Use these keys in this order: `source`, `agent`, `kind`, `value`.

Use JSON string escaping and no extra spaces.

Prefix the bytes with `ccgram-herdr-session-v1\0`.

Compute SHA-256 over the prefix and JSON bytes.

Use `herdr-session-v1-<hex>` as the target ID.

Add fixed digest vectors to `tests/ccgram/test_herdr_backend.py`.

Do not expose raw session values in target IDs, labels, logs, or fixtures.

## Action contract

Every Herdr action must use this sequence:

1. Read a fresh `agent.list` snapshot.
2. Parse all complete session composites.
3. Find records with the stored target ID.
4. Require exactly one record.
5. Use its current terminal or pane locator.
6. Perform the action.
7. Read a fresh snapshot after an action error.
8. Report the result and the known race limit.

A sessionless record never becomes a topic target.

A zero-match target becomes unresolved.

A multi-match target becomes ambiguous.

A pre-guard change blocks the action.

A post-guard change can cause misdelivery. Tests must separate these two cases.

## Telegram topic creation contract

A user creates a Telegram forum topic and selects a Herdr workspace in the existing workspace picker.

The picker stores `PENDING_WORKSPACE_ID`. The launch service passes this opaque value to the neutral target creation contract.

Herdr topic creation requires this value. If the user skips workspace selection, CCGram returns to the workspace picker.

The Herdr adapter performs this sequence:

1. Confirm that the selected workspace exists in `workspace list`.
2. Create one tab in that exact workspace with `tab create --workspace <workspace-id> --cwd <selected-path> --no-focus`.
3. Save the new tab ID and root pane ID.
4. Start the configured provider command in the root pane.
5. Poll `agent.list` until one sessionful record matches the new workspace, tab, and root pane.
6. Derive the session target ID from this record.
7. Return the target ID, display label, tab ID, and root pane ID.

The generic launch service registers the returned target ID as pending before it awaits topic binding.

The launch service binds the Telegram topic to the returned session target ID. It does not bind the topic to the new tab ID.

If the selected workspace is missing, the adapter stops before tab creation.

If tab creation or agent launch fails, the adapter closes the new tab.

If session reporting times out or returns multiple records, the adapter closes the new tab.

The adapter does not close the selected workspace. The workspace existed before this request.

The launch service clears pending state and leaves the Telegram topic unbound. It shows the failure and retry action.

Do not fall back to the active workspace, a matching directory workspace, a tab, a pane, or a shell record.

## Test contract

Unit tests must cover pure digest and guard rules.

Adapter tests must cover all Herdr operations that accept `window_id`.

Hook tests must cover valid, missing, stale, and duplicate locator cases.

State tests must cover move, restart, reconnect, event loss, unresolved, and ambiguous targets.

Race tests must cover changes before the guard and after the guard.

Inject a fake Herdr transport with a barrier after `agent.list` returns.

Change the matching record at that barrier.

Assert that CCGram reports the post-guard race as indeterminate or possible misdelivery.

Do not assert that CCGram can prove correct delivery after that barrier.

Real-session tests must use two disposable sessions with official Herdr integration.

Real-session tests must cover topic creation in a selected workspace, focus, move, restart, replacement, duplicate, reconnect, split, archive, and close.

Creation tests must cover a missing workspace, agent launch failure, session-report timeout, and multiple matching reports.

Creation failure tests must prove that the new tab closes, the selected workspace remains open, pending state clears, and the Telegram topic stays unbound.

Real-session tests must store redacted evidence.

## Validation Commands

Run these commands after Task 5:

- `uv run pytest -q tests/ccgram/test_herdr_backend.py tests/ccgram/test_self_identify.py tests/ccgram/test_window_resolver.py tests/ccgram/test_session_map_backend.py tests/ccgram/test_herdr_boundary_audit.py tests/ccgram/test_herdr_identity_audit.py tests/ccgram/test_herdr_legacy_model_audit.py`
- `uv run pytest -q tests/ccgram/handlers/topics/test_window_launch_service.py tests/ccgram/handlers/topics/test_topic_lifecycle.py tests/ccgram/handlers/test_sync_command.py tests/ccgram/handlers/test_split_command.py`
- `uv run pytest -q tests/integration/test_herdr_contract.py -m herdr -v --setup-show`
- `uv run pytest -q tests/integration/test_herdr_contract.py -m herdr -v --junitxml=artifacts/herdr-session-junit.xml`
- `make arch-guard`
- `make arch-check`
- `uv run ruff check src/ tests/`
- `uv run pyright src/ccgram/ tests/`
- `make check`
- `git diff --check`
- `gitnexus_detect_changes({scope: "all", repo: "ccgram"})`

## Implementation Steps

### Task 1: Add the guarded session target seam

**Goal**

Add the neutral target contract and one Herdr guard.

**Why**

The current Herdr code mixes tab identity, pane identity, and agent identity.

**Files**

- If the current contract requires neutral target metadata, `src/ccgram/multiplexer/base.py` adds it.
- `src/ccgram/multiplexer/herdr.py` adds parsing, digest, snapshot, and guard code.
- `tests/ccgram/test_herdr_backend.py` adds digest and guard tests.
- `tests/ccgram/test_multiplexer_contract.py` adds neutral contract tests.

**Depends on**

- Current Herdr unit tests pass.
- Synthetic `agent.list` fixtures contain two sessionful records and one sessionless record.

**Implementation shape**

1. Define the complete session composite.
2. Define canonical bytes and the versioned digest.
3. Read a fresh `agent.list` snapshot.
4. Match records by the digest.
5. Return one record, unresolved, or ambiguous.
6. Keep live locators inside the returned record.

**Done when**

- Complete equal composites have equal IDs.
- Different composite fields have different IDs.
- IDs contain no raw session value.
- Sessionless records have no ID.
- The guard does not read focus, title, name, directory, screen, or layout.

**Checks**

Run the commands below.

**Fitness gate**

Keep session parsing and live locator use inside `multiplexer/herdr.py`.

**Impact commands**

- `gitnexus_impact({target: "Method:src/ccgram/multiplexer/herdr.py:HerdrManager.list_windows#0", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "WindowRef", direction: "upstream", depth: 2, include_tests: true, repo: "ccgram"})`
- `gitnexus_detect_changes({scope: "all", repo: "ccgram"})`

**Verification commands**

- `uv run pytest -q tests/ccgram/test_herdr_backend.py tests/ccgram/test_multiplexer_contract.py`
- `uv run ruff check src/ccgram/multiplexer/herdr.py tests/ccgram/test_herdr_backend.py`
- `uv run pyright src/ccgram/multiplexer/herdr.py tests/ccgram/test_herdr_backend.py`
- `make arch-guard`

**Manual checks**

- Read a redacted `agent.list` fixture.
- Make sure that two composites produce two IDs.
- Make sure that a sessionless record produces no ID.
- Make sure that duplicate IDs block the guard.

**Steps**

- [x] Before editing a listed symbol, run the impact commands.
- [x] Add canonical composite bytes and the versioned digest.
- [x] Add the fresh `agent.list` guard.
- [x] Add unresolved, ambiguous, malformed, and sessionless errors.
- [x] Add fixed digest vectors and privacy tests.
- [x] Add guard tests that reject layout and focus data.
- [x] Run the verification commands.
- [x] Before committing this task, run `gitnexus_detect_changes`.

**Stop condition**

If a guard selects an agent without a complete composite, stop. Remove that path.

### Task 2: Route every Herdr action through the seam

**Goal**

Route every Herdr target action through one fresh session guard.

**Why**

The current adapter uses focused-pane and tab behavior.

**Files**

- `src/ccgram/multiplexer/base.py` adds a neutral `create_topic_target` contract and result type.
- `src/ccgram/multiplexer/herdr.py` routes all target actions through the guard and creates session targets in selected workspaces.
- `src/ccgram/multiplexer/tmux.py` implements the neutral creation contract without behavior change.
- `src/ccgram/handlers/topics/window_launch_service.py` uses the neutral creation result and binds the returned target ID.
- `src/ccgram/handlers/topics/provider_mode_callbacks.py` supplies the selected provider and approval mode.
- `src/ccgram/handlers/topics/topic_orchestration.py` suppresses duplicate topic creation for a returned session target ID.
- `src/ccgram/handlers/split_command.py` uses session-target split operations and does not send to a caller-supplied pane ID.
- `src/ccgram/multiplexer/topic_mapping.py` lists sessionful agents only.
- `src/ccgram/multiplexer/self_identify.py` resolves hook identity from a unique locator match.
- `src/ccgram/hook.py` writes guarded session target IDs.
- `tests/ccgram/test_herdr_backend.py` adds action, creation, and race tests.
- `tests/ccgram/test_topic_mapping.py` adds session target tests.
- `tests/ccgram/test_self_identify.py` adds hook tests.
- `tests/ccgram/handlers/topics/test_window_launch_service.py` adds selected-workspace creation and rollback tests.
- `tests/ccgram/handlers/topics/test_provider_mode_callbacks.py` adds normal and YOLO provider callback tests.
- `tests/ccgram/handlers/test_split_command.py` proves that raw pane IDs cannot bypass the guard.

**Depends on**

- Task 1 is complete and committed.
- The user selected one Herdr workspace through the existing workspace picker.
- The selected provider reports `agent_session` through an official Herdr integration.
- Fixtures contain two sessionful agents in one tab.
- Fixtures contain a replacement session.

**Implementation shape**

1. Add a neutral `create_topic_target` contract that returns a target ID and creation locator.
2. If a native-worktree request cannot carry the selected workspace ID, reject it.
3. In Herdr, validate the selected workspace ID before tab creation.
4. Create one tab in the selected workspace and save its root pane ID.
5. Start the configured agent in that root pane.
6. Poll `agent.list` for one sessionful record in the new workspace, tab, and root pane.
7. Return its session target ID to the generic launch service.
8. Register the target ID as pending before topic binding.
9. List only sessionful agent targets.
10. Guard `find_window_by_id`, send, read, capture, dimensions, foreground, title, status, rename, split, and close.
11. Guard `send_to_pane`, `send_keys_to_pane`, and `capture_pane_by_id` with a session-derived locator.
12. Reject caller-supplied raw pane IDs from handlers.
13. Use the matched terminal for agent actions.
14. Use the matched pane for pane actions.
15. Refresh after action errors.
16. Block changed, missing, and duplicate guard results.
17. Record the post-guard race result.

**Done when**

- If a native-worktree request cannot carry the selected workspace ID, it stops.
- `create_topic_target` creates one tab in the selected workspace.
- A missing selected workspace returns to the workspace picker.
- A new topic binds to the returned session target ID, not the tab ID.
- A creation timeout or ambiguous report closes the new tab and leaves the topic unbound.
- Creation cleanup does not close the selected workspace.
- `list_windows()` returns session targets for Herdr.
- Every Herdr action starts with a fresh guard.
- The adapter does not call `_active_pane` or `_representative_pane`.
- The adapter does not use `pane.list` for agent identity.
- The adapter creates no Herdr tab topic.
- An action error causes one fresh guard read.

**Checks**

Run the commands below.

**Fitness gate**

Keep locators private to one guarded action or one creation result. Never persist a locator in a topic binding.

Keep workspace selection in the generic launch request as an opaque ID. Handler code cannot call Herdr workspace or tab APIs.

Keep raw pane APIs inside the adapter. Handlers cannot pass a pane ID to Herdr.

**Impact commands**

- `gitnexus_impact({target: "Method:src/ccgram/multiplexer/herdr.py:HerdrManager.create_window#6", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "_create_topic_window", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "find_window_by_id", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "send_to_pane", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "send_keys_to_pane", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "capture_pane_by_id", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "send", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "capture_scrollback", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "kill_window", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_detect_changes({scope: "all", repo: "ccgram"})`

**Verification commands**

- `uv run pytest -q tests/ccgram/test_herdr_backend.py tests/ccgram/test_topic_mapping.py tests/ccgram/test_self_identify.py tests/ccgram/handlers/topics/test_window_launch_service.py tests/ccgram/handlers/topics/test_provider_mode_callbacks.py tests/ccgram/handlers/test_split_command.py`
- `uv run pytest -q tests/ccgram/test_multiplexer_contract.py tests/ccgram/test_multiplexer_boundary.py tests/ccgram/test_no_tty_outside_backend.py`
- `make arch-guard`
- `uv run ruff check src/ccgram/multiplexer/ src/ccgram/hook.py tests/ccgram/test_herdr_backend.py`
- `uv run pyright src/ccgram/ tests/ccgram/test_herdr_backend.py`

**Manual checks**

- Select one Herdr workspace in the Telegram workspace picker.
- Create one Telegram topic and select a provider.
- If the provider has a mode picker, select `normal`.
- Make sure that Herdr creates one tab in the selected workspace.
- Make sure that the new topic binds to one reported session target ID.
- Make sure that timeout cleanup closes the new tab and leaves the topic unbound.
- Start a native-worktree request with a selected workspace.
- Make sure that the request stops or carries the workspace explicitly.
- Make sure that a raw pane action from `/split` cannot bypass the guard.
- Start two sessionful agents in one tab.
- Change focus before each action.
- Make sure that each action uses the selected session target.
- Replace one agent before the next action.
- Make sure that the old target blocks the action.

**Steps**

- [ ] Before editing a listed symbol, run all impact commands.
- [ ] Add the neutral `create_topic_target` contract and result type.
- [ ] Reject or pass the selected workspace through native-worktree creation.
- [ ] Create a Herdr tab in the exact selected workspace.
- [ ] Call `_handle_mode_select` for providers with a mode picker.
- [ ] Launch the configured agent in the new root pane.
- [ ] Wait for exactly one sessionful `agent.list` record in the new root pane.
- [ ] Bind the Telegram topic to the returned target ID.
- [ ] Close the new tab and clear pending state after any creation failure.
- [ ] Replace Herdr discovery with sessionful `agent.list` records.
- [ ] Guard every Herdr action with a fresh snapshot.
- [ ] Keep locators inside the guarded action.
- [ ] Guard raw pane operations or reject caller-supplied pane IDs.
- [ ] Add selected-workspace, skipped-picker, missing-workspace, launch-failure, timeout, ambiguous-report, and rollback tests.
- [ ] Add focus, replacement, missing, duplicate, and race tests.
- [ ] Document the post-guard dispatch race.
- [ ] Run the verification commands.
- [ ] Before committing this task, run `gitnexus_detect_changes`.

**Stop condition**

If one target action bypasses the guard, stop. Route it through the guard.

### Task 3: Preserve guarded targets through hooks and recovery

**Goal**

Keep each topic bound to the same session target through moves, restart, reconnect, and event loss.

**Why**

A tab ID can point to a different agent after a layout or process change.

**Files**

- `src/ccgram/multiplexer/self_identify.py` matches `(workspace_id, pane_id)` to one live record.
- `src/ccgram/session_map.py` stores opaque target IDs.
- `src/ccgram/window_resolver.py` preserves unresolved and ambiguous targets.
- `src/ccgram/session.py` refreshes targets from `agent.list`.
- `src/ccgram/multiplexer/herdr_events.py` requests refresh after Herdr events.
- `src/ccgram/session_monitor.py` refreshes after reconnect or event loss.
- `src/ccgram/thread_router.py` indexes bindings by target ID.
- `tests/ccgram/test_self_identify.py` adds zero and duplicate locator tests.
- `tests/ccgram/test_session_map_backend.py` adds opaque target tests.
- `tests/ccgram/test_window_resolver.py` adds unresolved and ambiguous tests.
- `tests/ccgram/test_session.py` adds restart and reconnect tests.
- `tests/integration/test_herdr_contract.py` adds live move and restore tests.

**Depends on**

- Tasks 1 and 2 are complete and committed.
- Herdr events or refresh polling are available.
- The test server supports moves and native restore.

**Implementation shape**

1. Store opaque target IDs only.
2. Resolve a hook pane through the unique `(workspace_id, pane_id)` match.
3. If zero or multiple records match, block the hook.
4. Refresh all targets after reconnect, event loss, and action errors.
5. Mark zero session matches unresolved.
6. Mark multiple session matches ambiguous.
7. Delete tab-ID and display-name recovery.

**Done when**

- A moved session keeps the same target ID.
- A supported restored session keeps the same target ID.
- A replacement session does not receive the old binding.
- A missing session remains unresolved.
- A duplicate session remains ambiguous.
- A hook blocks zero and multiple locator matches.
- Recovery uses no title, name, directory, focus, pane, tab, or terminal identity.

**Checks**

Run the commands below.

**Fitness gate**

Keep Herdr parsing inside the adapter. Keep generic state code independent from Herdr fields.

**Impact commands**

- `gitnexus_impact({target: "resolve_self_identity", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "Function:src/ccgram/window_resolver.py:resolve_stale_ids", direction: "upstream", depth: 2, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "session_map_prefix_for", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_detect_changes({scope: "all", repo: "ccgram"})`

**Verification commands**

- `uv run pytest -q tests/ccgram/test_self_identify.py tests/ccgram/test_session_map_backend.py tests/ccgram/test_window_resolver.py tests/ccgram/test_session.py`
- `uv run pytest -q tests/integration/test_herdr_contract.py -m herdr -v`
- `make arch-guard`
- `make arch-check`
- `uv run pyright src/ccgram/ tests/ccgram/test_window_resolver.py`

**Manual checks**

- Bind a topic to one sessionful agent.
- Move the agent across a pane, tab, and workspace.
- Restart Herdr with native session restore.
- Make sure that the same topic keeps the same target ID.
- Replace the old pane agent.
- Make sure that the old target is unresolved.
- Create a duplicate locator in a disposable test.
- Make sure that the hook blocks the duplicate.

**Steps**

- [ ] Before editing a listed symbol, run all impact commands.
- [ ] Store only opaque target IDs in bindings and maps.
- [ ] Resolve hook identity through one unique locator match.
- [ ] Block zero and multiple hook locator matches.
- [ ] Refresh `agent.list` after reconnect, event loss, and action errors.
- [ ] Preserve unresolved and ambiguous targets without retargeting.
- [ ] Delete tab-ID, display-name, and suffix recovery.
- [ ] Add move, restart, replacement, reconnect, and duplicate tests.
- [ ] Run the verification commands.
- [ ] Before committing this task, run `gitnexus_detect_changes`.

**Stop condition**

If recovery uses a display or layout value, stop. Remove that path.

### Task 4: Delete the old model and align all documentation

**Goal**

Delete old Herdr identity code, old tests, old configuration, and stale current documentation.

**Why**

Dead fallback code can return. Stale documentation can teach the unsafe model.

**Files**

- `src/ccgram/multiplexer/herdr.py` removes tab and focused-pane identity helpers.
- `src/ccgram/multiplexer/self_identify.py` removes pane-to-tab identity.
- `src/ccgram/multiplexer/topic_mapping.py` removes Herdr tab eligibility.
- `src/ccgram/window_resolver.py` removes Herdr tab-ID remapping.
- `src/ccgram/session.py` removes the old Herdr restart path.
- `src/ccgram/session_map.py` removes Herdr tab prefixes and suffix matching.
- `src/ccgram/config.py` removes old Herdr topic-scope configuration.
- Legacy unit tests are deleted or rewritten for guarded targets.
- `tests/ccgram/test_herdr_legacy_model_audit.py` checks active code, tests, and current docs.
- `tests/ccgram/handlers/topics/test_topic_lifecycle.py` tests legacy detection, blocked actions, archive, rollback, and explicit rebind.
- `tests/ccgram/handlers/test_sync_command.py` tests the legacy user message and available session targets.
- `tests/ccgram/test_window_state_store.py` tests the `legacy_herdr` state transition and rollback.
- `tests/ccgram/test_herdr_boundary_audit.py` checks import and ownership rules.
- `tests/ccgram/test_herdr_identity_audit.py` checks one digest owner and forbidden fallback calls.
- `README.md` explains guarded session targets and the dispatch race.
- `docs/guides.md` explains setup, errors, migration, rollback, and live tests.
- `docs/architecture.md` describes the guarded-session architecture.
- `docs/ai-agents/architecture-map.md` removes active tab identity guidance.
- `docs/ai-agents/codebase-index.md` lists guarded-session paths.
- Historical plans get a historical notice.

**Depends on**

- Tasks 1 through 3 are complete and committed.
- The documentation inventory is complete.
- A migration message has review approval.

**Implementation shape**

1. When a Herdr target is not a `herdr-session-v1-` ID, mark its binding as legacy.
2. Add `legacy_herdr` state to the existing binding record.
3. Block all actions for a `legacy_herdr` record.
4. Show an archive and explicit rebind message to the user.
5. Archive the record without closing its tab or pane.
6. Keep the archived record with state `legacy_herdr` until the user confirms removal.
7. On rollback, restore the old record and keep its actions blocked.
8. Require explicit rebind to create a new session target record.
9. Delete `_active_pane` and `_representative_pane` from Herdr identity paths.
7. Delete Herdr tab-ID `WindowRef` construction for topics.
8. Delete Herdr display-name and suffix recovery.
9. Delete Herdr tab and pane configuration branches.
10. Delete tests that assert tab identity or focus routing.
11. Add the legacy-model audit test.
12. Update current user and agent documentation.
13. Mark old design text as historical.

**Done when**

- No active code maps a Herdr topic to a tab or pane.
- No active code uses focus to select a Herdr agent.
- No active code recovers by name, suffix, or layout value.
- Legacy records are detected by the exact predicate above.
- Legacy records block actions and preserve their Herdr target.
- Archive removes only the CCGram binding.
- The archived record remains available for rollback.
- Rollback restores the record and keeps its actions blocked.
- Explicit rebind creates a new session target record.
- No current test asserts the old model.
- Current documentation describes guarded targets only.
- The audit allows locators only inside guarded actions.

**Checks**

Run the commands below.

**Fitness gate**

Add `tests/ccgram/test_herdr_legacy_model_audit.py` as an architectural fitness test.

The test scans these paths:

- `src/ccgram/multiplexer/`
- `src/ccgram/hook.py`
- `src/ccgram/session.py`
- `src/ccgram/session_map.py`
- `src/ccgram/window_resolver.py`
- `tests/ccgram/`
- `tests/integration/test_herdr_contract.py`
- `README.md`
- `docs/guides.md`
- `docs/architecture.md`
- `docs/ai-agents/architecture-map.md`
- `docs/ai-agents/codebase-index.md`

The test rejects `_active_pane`, `_representative_pane`, `CCGRAM_HERDR_TOPIC_SCOPE`, Herdr tab-ID recovery, display-name recovery, and session-map suffix matching.

The test allows live locators inside one guarded action and historical notices in completed plans.

**Impact commands**

- `gitnexus_impact({target: "_active_pane", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "_representative_pane", direction: "upstream", depth: 3, include_tests: true, repo: "ccgram"})`
- `gitnexus_impact({target: "Function:src/ccgram/window_resolver.py:resolve_stale_ids", direction: "upstream", depth: 2, include_tests: true, repo: "ccgram"})`
- `gitnexus_detect_changes({scope: "all", repo: "ccgram"})`

**Verification commands**

- `uv run pytest -q tests/ccgram/test_herdr_legacy_model_audit.py tests/ccgram/test_herdr_backend.py tests/ccgram/test_self_identify.py tests/ccgram/test_window_resolver.py tests/ccgram/test_session_map_backend.py`
- `make arch-guard`
- `make arch-check`
- `rg -n '_active_pane|_representative_pane|HERDR_PANE_ID.*tab|window_id == tab_id|CCGRAM_HERDR_TOPIC_SCOPE|live_window_session_ids|_resolve_by_session_id|session_map_prefix_for' src tests README.md docs --glob '!docs/plans/completed/**'`
- `git diff --check`

**Manual checks**

- Read every current Herdr section in the README, guides, architecture docs, and agent maps.
- Make sure that each section names `agent.list` as the only identity source.
- Make sure that no current example binds a topic to a tab or pane.
- Make sure that migration text does not guess a session.
- Make sure that legacy records block actions before archive.
- Make sure that archive leaves the Herdr tab and pane open.
- Make sure that explicit rebind uses a listed session target.
- Make sure that the dispatch race appears in the limits section.

**Steps**

- [ ] Before editing a listed symbol, run all impact commands.
- [ ] Detect legacy Herdr records with the exact target-ID predicate.
- [ ] Add the `legacy_herdr` state and persist it through the existing state schema.
- [ ] Block actions for legacy records.
- [ ] Add archive-only behavior and explicit rebind behavior.
- [ ] Persist `legacy_herdr` until explicit removal.
- [ ] Add rollback that restores the record but keeps actions blocked.
- [ ] Delete old tab, pane, focus, and display fallback code.
- [ ] Delete old Herdr tab-ID recovery and session-map code.
- [ ] Delete or rewrite tests that assert the old model.
- [ ] Add the three named architecture audit tests.
- [ ] Update README, guides, architecture docs, and agent maps.
- [ ] Mark old design text as historical.
- [ ] Run the verification commands.
- [ ] Before committing this task, run `gitnexus_detect_changes`.

**Stop condition**

If another backend uses a shared helper, stop. Split the Herdr behavior first.

### Task 5: Run unit, contract, race, and real-session tests

**Goal**

Prove guarded routing with fast tests and real Herdr sessions.

**Why**

Mocks cannot prove process integration, real session reports, pane moves, restore, or input isolation.

**Files**

- `tests/ccgram/test_herdr_backend.py` adds the unit and race matrix.
- `tests/ccgram/test_self_identify.py` adds hook failures.
- `tests/ccgram/test_window_resolver.py` adds fail-closed recovery.
- `tests/ccgram/test_session_map_backend.py` adds opaque-target persistence.
- `tests/ccgram/handlers/topics/test_topic_lifecycle.py` adds archive and legacy cases.
- `tests/ccgram/handlers/test_sync_command.py` adds migration messages.
- `tests/ccgram/handlers/test_split_command.py` adds guarded split cases.
- `tests/ccgram/test_herdr_boundary_audit.py` checks import and ownership rules.
- `tests/ccgram/test_herdr_identity_audit.py` checks one digest owner and fallback deletion.
- `tests/ccgram/test_herdr_legacy_model_audit.py` checks old model deletion.
- `tests/integration/test_herdr_contract.py` adds the real-session suite.
- `tests/integration/herdr_session_fixture.py` creates and cleans a disposable live test workspace.
- `tests/integration/herdr_session_evidence.py` writes redacted evidence.
- `docs/guides.md` documents live setup and redacted evidence.

**Depends on**

- Tasks 1 through 4 are complete and committed.
- A disposable Herdr workspace is available.
- Two real agents use official Herdr integration.
- `HERDR_SOCKET_PATH` points to the disposable server.
- The `herdr` executable is available in `PATH`.
- Session values and credentials are disposable.
- The fixture can create and close a disposable workspace.

**Implementation shape**

1. Add unit tests for each guard and action result.
2. Add table tests for missing, duplicate, sessionless, replacement, and malformed records.
3. Add deterministic pre-guard race tests.
4. Add a documented post-guard dispatch-race test with a transport barrier after `agent.list`.
5. Create a disposable workspace with `herdr workspace create --cwd <temp-dir> --no-focus`.
6. Select this workspace through the Telegram workspace picker.
7. Create a Telegram topic and complete provider selection.
8. Confirm one new tab and one sessionful record in the selected workspace.
9. Read the returned target ID through the test socket.
10. Run missing-workspace, launch-failure, timeout, and ambiguous-report rollback cases.
11. Run focus, move, restore, replacement, duplicate, reconnect, split, archive, and close cases.
12. Close the fixture workspace in a `finally` cleanup block.
13. Write redacted evidence to `artifacts/herdr-session-evidence.json`.
14. Record the race result.

**Done when**

- Unit tests cover every success and failure result.
- Contract tests cover every guarded action and topic creation result.
- Telegram creation places one new tab and one agent session in the selected workspace.
- Creation failure closes only the new tab and leaves the Telegram topic unbound.
- Pre-guard changes block actions.
- The barrier test reports `dispatch_race_possible` after a post-guard change.
- The barrier test does not report atomic delivery.
- Two real sessions receive only intended input markers in the tested window.
- Focus changes do not change guard selection.
- Move and restore keep the same target ID.
- Replacement, missing, and duplicate sessions block actions during the guard.
- Archive leaves the real session open.
- Close affects only the selected real session.
- The live report records version, protocol, agents, commands, and evidence.

**Checks**

Run the commands below.

**Fitness gate**

Keep real-session tests under `herdr` and `integration` markers.

Keep real-session tests outside the fast unit-test path.

Do not replace public-contract assertions with private call-count assertions.

**Impact commands**

- `gitnexus_detect_changes({scope: "all", repo: "ccgram"})`

**Verification commands**

- `uv run pytest -q tests/ccgram/test_herdr_backend.py tests/ccgram/test_self_identify.py tests/ccgram/test_window_resolver.py tests/ccgram/test_session_map_backend.py tests/ccgram/test_herdr_boundary_audit.py tests/ccgram/test_herdr_identity_audit.py tests/ccgram/test_herdr_legacy_model_audit.py`
- `uv run pytest -q tests/ccgram/handlers/topics/test_topic_lifecycle.py tests/ccgram/handlers/test_sync_command.py tests/ccgram/handlers/test_split_command.py`
- `uv run pytest -q tests/integration/test_herdr_contract.py -m herdr -v --setup-show`
- `uv run pytest -q tests/integration/test_herdr_contract.py -m herdr -v --junitxml=artifacts/herdr-session-junit.xml`
- `make arch-guard`
- `make arch-check`
- `make check`
- `git diff --check`

**Manual checks**

1. Create a disposable Herdr workspace.
2. Select that workspace through the Telegram picker.
3. Create a Telegram topic and select a sessionful provider.
4. Make sure that one new Herdr tab appears in the selected workspace.
5. Make sure that the topic binds to one reported target ID.
6. Run launch-failure, timeout, and ambiguous-report creation tests.
7. Make sure that each failed creation closes only its new tab.
8. Start two real sessionful agents in one tab.
9. Bind one topic to each target ID.
10. Send unique markers after each focus change.
11. Make sure that each pane shows only its marker.
12. Move one session to another pane, tab, and workspace.
13. Restart Herdr with native restore.
14. Replace the old pane agent with a new session.
15. Create a duplicate session report in the disposable server.
16. Disconnect and reconnect the CCGram event stream.
17. Split beside one guarded session.
18. Archive one topic and close one selected session.
19. Make sure that each result matches the selected target ID.

**Steps**

- [ ] Add the complete unit and contract matrix.
- [ ] Add the three named architecture audit tests.
- [ ] Add pre-guard replacement and action-race tests.
- [ ] Add the documented post-guard race test.
- [ ] Add `tests/integration/herdr_session_fixture.py` with `finally` cleanup.
- [ ] Add `tests/integration/herdr_session_evidence.py` with redaction.
- [ ] Run the Telegram selected-workspace creation flow.
- [ ] Run missing-workspace, launch-failure, timeout, and ambiguous-report cleanup cases.
- [ ] Run two real sessions in one tab.
- [ ] Run focus, move, restart, replacement, duplicate, and reconnect cases.
- [ ] Run split, archive, and close cases with real sessions.
- [ ] Run the full verification commands.
- [ ] Record live server, protocol, agent count, routing, and race-limit evidence in `artifacts/herdr-session-evidence.json`.
- [ ] Before committing this task, run `gitnexus_detect_changes`.

**Stop condition**

If any tested input appears in the wrong session, stop. Preserve logs and pane output. Do not continue rollout.

## Live fixture contract

Create `tests/integration/herdr_session_fixture.py`.

The fixture must:

- Create a temporary directory with `tmp_path`.
- Confirm the server with `herdr workspace list`.
- Create one workspace with `herdr workspace create --cwd "$TMPDIR" --no-focus`.
- Parse the JSON response for workspace, tab, and root pane IDs.
- Set `CCGRAM_HERDR_ITEST_PROVIDER=pi` by default.
- Let an explicit `CCGRAM_HERDR_ITEST_PROVIDER` value replace this default.
- Export `HERDR_SOCKET_PATH` for CCGram tests.
- Build a Telegram callback context with the existing test doubles.
- Call `_handle_workspace_callback` with `CB_WS_SELECT + str(index)`.
- Call `_handle_provider_select` with `CB_PROV_SELECT + provider_name`.
- If the provider has a mode picker, call `_handle_mode_select` with `CB_MODE_SELECT + provider_name + ":normal"`.
- Let `launch_window` resolve the provider command with `approval_mode="normal"`.
- Do not resolve or start the provider command outside `launch_window`.
- Poll `herdr agent list` until one sessionful record appears in the new tab.
- Repeat the Telegram callback flow to create the second session.
- Return the workspace ID, created tab IDs, pane IDs, and redacted agent records.
- Close each created tab with `herdr tab close "$TAB_ID"` in a `finally` block.
- Close the workspace with `herdr workspace close "$WORKSPACE_ID"` in the same block.
- Run `herdr workspace list` after cleanup.
- If the workspace still exists, fail cleanup.

If the resolved provider does not report `agent_session`, fail the fixture with the provider name and command basename.

Use `herdr wait agent-status "$PANE_ID" --status idle --timeout 30000` for readiness.

Use `herdr agent list` after each create, move, replacement, reconnect, and restore step.

Create markers with `uuid.uuid4().hex`, for example `CCGRAM_ITEST_A_<hex>` and `CCGRAM_ITEST_B_<hex>`.

Create `tests/integration/herdr_session_evidence.py`.

Write only these fields to `artifacts/herdr-session-evidence.json`:

- Herdr version.
- Herdr protocol.
- Test timestamp.
- Workspace ID.
- Agent count.
- Redacted session target IDs.
- Action name.
- Expected agent label.
- Observed marker result.
- Guard result.
- Dispatch-race result.
- Cleanup result.

Do not write session paths, raw session values, credentials, or full pane output.

## Acceptance criteria

- `agent.list` is the only Herdr identity source.
- CCGram stores opaque session target IDs only.
- Each action uses one fresh session guard.
- Missing, duplicate, sessionless, malformed, and pre-guard changed targets block actions.
- The post-guard dispatch race is documented and tested.
- A Telegram topic creates one Herdr tab and one agent session in the selected workspace.
- A failed creation closes only its new tab and leaves the Telegram topic unbound.
- CCGram has no layout or display fallback.
- Old tab, pane, focus, configuration, and recovery code is deleted.
- Current docs describe the guarded-session architecture.
- Unit, contract, race, and real-session tests pass.
- Tmux behavior does not change.
- All validation commands pass.

## Safety notes

CAUTION: Use disposable Herdr servers, workspaces, agents, and session files.

A fault can send input to the wrong agent or close the wrong pane.

CAUTION: Current Herdr has a post-guard dispatch race.

Do not claim atomic session delivery.

Do not create a target from a screen label, pane, terminal, tab, directory, title, process, or focus state.

Do not act after an unresolved or ambiguous guard result.

Do not create a Herdr topic without a selected workspace or a sessionful provider report.

Do not migrate a legacy binding automatically. Archive it and bind a listed target explicitly.

Keep raw session values out of labels, logs, fixtures, and public target IDs.

Use `/exec start docs/plans/herdr-agent-endpoint-adaptation.md` in an isolated CCGram worktree.

## Re-review

Run a scoped `architecture-review` after implementation.

Review these areas:

- Guarded `agent.list` use in `src/ccgram/multiplexer/herdr.py`.
- Hooks, bindings, events, maps, and recovery paths.
- Legacy cleanup and lifecycle handlers.
- Current documentation and the legacy-model audit.
- Telegram selected-workspace creation and rollback evidence.
- Unit, race, and real-session evidence.

Accept the implementation only after these facts are true:

- `agent.list` is the only Herdr identity source.
- CCGram has no layout or display fallback.
- Missing and duplicate targets fail closed.
- Current documentation states the dispatch race.
- Coupling remains neutral above the adapter.
- All architecture and validation commands pass.
