---
artifact: architecture-report
schema_version: 2
rubric_version: 1
report_id: ccgram-full-2026-05-23
date: 2026-05-23

target:
  repo: ccgram
  scope: full
  out_of_scope: []

comparability:
  scope: full
  rubric_version: 1
  tool_coverage_level: standard

interview_context:
  system_goal: Telegram control plane for tmux-hosted AI coding agents; one Telegram Forum topic controls one tmux window and its agent session.
  quality_goals:
    - Maintainability of high-change Telegram UX and provider code.
    - Change locality around topic/window/session routing.
    - Provider extensibility through explicit capabilities and transcript/status contracts.
    - Testability through narrow Telegram and state-query seams.
    - Safe Telegram formatting and send-layer splitting.
  intended_units:
    - PTB bot factory and bootstrap lifecycle.
    - Feature handler subpackages.
    - Provider abstraction and provider implementations.
    - Session monitor and transcript/event readers.
    - Session/window state stores plus read-only query layer.
    - Tmux manager infrastructure.
    - Messaging pipeline and mailbox.
    - Optional Mini App HTTP/WebSocket surface.
  domains:
    core:
      - Telegram UX handlers.
      - Provider abstraction and status/transcript parsing.
      - Polling and status detection.
      - Inter-agent messaging.
      - Mini App.
    supporting:
      - Session monitoring.
      - State persistence and query projections.
      - Hook event ingestion.
    generic:
      - tmux integration.
      - Telegram Bot API adapter.
      - LLM/Whisper HTTP integrations.
  volatile_areas:
    - Handler UX flows.
    - Provider capability matrix.
    - Polling/status detection.
    - Window/session state.
    - Mini App.
  team_ownership:
    - Unknown; no CODEOWNERS or team ownership file found.
  known_pain:
    - Prior modularity reviews called out singleton/proxy state, lazy imports/cycles, oversized handlers, and mixed state access paths.
    - Current docs claim several earlier findings were fixed and now enforced.
  review_scope: full
  out_of_scope: []

system_map:
  languages:
    - Python 3.14
    - JavaScript assets for Mini App
    - YAML workflows
  package_managers:
    - uv
    - hatchling/hatch-vcs
  units:
    - ccgram CLI entrypoint
    - Telegram bot application
    - Claude/Codex/Gemini/Pi/Shell provider implementations
    - hook CLI/event writer
    - session monitor
    - status polling loop
    - tmux manager
    - inter-agent mailbox and broker
    - optional aiohttp Mini App
  deploy_units:
    - Python package/wheel with ccgram console script
    - local Telegram bot process
    - hook subprocesses invoked by agent CLIs
    - optional local Mini App aiohttp server
  public_interfaces:
    - ccgram console script
    - Telegram commands and callback handlers
    - ccgram hook CLI
    - ccgram msg CLI
    - provider protocol and capabilities
    - TelegramClient protocol
    - Mini App HTTP/WebSocket routes
    - state files under ~/.ccgram
  declared_modules:
    - handlers feature subpackages
    - providers package
    - llm/whisper adapters
    - miniapp package
    - session/window/thread state modules
    - polling observe/decide/apply split
  observed_modules:
    - Handler layer imports the TelegramClient seam and state query layer for most reads.
    - Provider layer centers on AgentProvider and ProviderCapabilities.
    - Polling is split into pure types, stateful strategies, and window_tick observe/decide/apply.
    - State remains centered on SessionManager plus module-level store proxies.
    - tmux_manager is the infrastructure hub for terminal I/O.
  high_risk_entrypoints:
    - handlers/topics/directory_callbacks.py
    - handlers/polling/polling_state.py
    - handlers/polling/window_tick/apply.py
    - window_state_store.py
    - session.py
    - tmux_manager.py
    - providers/codex.py
    - providers/gemini.py
  missing_evidence:
    - GitNexus index is fresh for symbol/process/impact analysis; historical co-change still uses git-log fallback because the exposed runtime tools did not return rename-aware co-change pairs.
    - No import-linter configuration found; dependency contracts are enforced by custom tests, not import-linter.
    - Team ownership unavailable.

scores:
  boundary_integrity:
    value: 78
    band: serviceable
    confidence: high
    evidence_refs: [E3, E6, E7, E8, E17]
    gaps:
      - Write-path dependency injection is not fully enforced; handler writes still use SessionManager allow-list.
  coupling_balance:
    value: 68
    band: serviceable
    confidence: high
    evidence_refs: [E16, E18, E19, E20, E23, E24, E25]
    gaps:
      - Historical co-change evidence is still a git-log fallback; GitNexus contributed fresh static/process blast-radius evidence.
  dependency_graph_health:
    value: 79
    band: serviceable
    confidence: high
    evidence_refs: [E15, E16, E17, E22, E23, E24, E25]
    gaps:
      - Codegraph affected output did not expose useful reverse-dependency paths for sampled files.
  cohesion_modularity:
    value: 62
    band: serviceable
    confidence: medium
    evidence_refs: [E12, E13, E19, E20, E25, E26]
    gaps:
      - LOC/function scans are structural evidence; they do not prove runtime pain by themselves.
  change_locality:
    value: 58
    band: mixed
    confidence: medium
    evidence_refs: [E18, E23, E24, E27]
    gaps:
      - Rename-aware historical co-change remains unavailable from the exposed GitNexus runtime tools.
  architecture_fitness:
    value: 86
    band: strong
    confidence: high
    evidence_refs: [E5, E6, E7, E8, E9, E17]
    gaps:
      - No import-linter contract; architectural rules are custom tests/scripts.
  analysis_confidence:
    value: 82
    band: strong
    confidence: high
    evidence_refs: [E1, E2, E3, E15, E17, E22, E23, E24, E25, E26, E27]
    gaps:
      - Review context reconstructed from docs and confirmed by user, not from a full stakeholder interview.
      - Team ownership evidence is unavailable.

findings:
  - id: F1
    title: Polling state remains a mutable strategy hub
    severity: medium
    dimension: cohesion_modularity
    evidence_refs: [E12, E16, E19, E25]
    narrative:
      problem: The pure polling contract is now isolated, but the mutable polling implementation is still concentrated in one 1017-line module with five singleton strategy instances.
      knowledge_or_boundary_leakage: Status parsing, screen-buffer cache, interactive UI state, topic lifecycle timers, and pane scanning all share the same polling_state module and are imported by window_tick plus shell, text, commands, lifecycle, and hook-event code.
      complexity_impact: A developer changing pane lifecycle, RC debounce, or status parsing must reason about multiple runtime caches and callback paths in one module; GitNexus reports HIGH upstream impact for PaneStatusStrategy with 22 direct import dependents and 93 total impacted symbols/files.
      cascading_change_scenarios:
        - Adding a new pane lifecycle state can touch PaneStatusStrategy, window_tick/apply, topic lifecycle cleanup, status UI, and tests.
        - Changing RC or screen-buffer caching can affect status polling, remote-control probes, screenshots, and provider status parsing.
      recommended_improvement: Split polling_state by runtime concern, or inject the specific state strategy into window_tick/apply instead of importing the whole singleton hub.
      tradeoffs: Keeping one module makes shared polling state easy to locate; splitting too aggressively could create more cross-file plumbing than it saves.
    recommended_action: Split or DI-bound the mutable polling strategies after preserving current purity tests.
  - id: F2
    title: Topic creation callback flow is still too broad
    severity: medium
    dimension: cohesion_modularity
    evidence_refs: [E13, E19, E20, E26]
    narrative:
      problem: directory_callbacks.py handles directory browsing, favorites, worktree selection, provider picking, launch mode, YOLO setup, window creation, topic binding, and pending-message forwarding in one 1086-line callback dispatcher.
      knowledge_or_boundary_leakage: Telegram callback state, git-worktree rules, provider launch behavior, tmux window creation, and shell readiness all share one file-level workflow.
      complexity_impact: A change to any topic-creation step forces reviewers to load unrelated state-machine branches; stale callback handling and cleanup rules are easy to break because they sit beside provider/mode code. GitNexus keeps the static upstream impact local to Topics, so this is a cohesion and reviewability risk more than a repo-wide blast-radius risk.
      cascading_change_scenarios:
        - Adding a new provider launch option would touch provider picker, mode picker, window creation, and forwarding logic in the same file.
        - Changing worktree branch validation risks affecting ordinary directory-confirm and provider-select callbacks.
      recommended_improvement: Split into directory navigation, worktree step, provider/mode selection, and window-binding modules with one orchestration surface.
      tradeoffs: The current file keeps the sequential user flow visible; splitting must preserve callback-data invariants and cleanup order.
    recommended_action: Extract topic-creation subflows behind a small orchestrator and keep the existing integration tests as the safety net.
  - id: F3
    title: Window state is a high-fan-in, high-churn state hub
    severity: high
    dimension: change_locality
    evidence_refs: [E14, E16, E18, E23, E24]
    narrative:
      problem: The read path is guarded, but window_state_store remains a global proxy-backed state hub with broad imports, recent cross-module co-change, and CRITICAL GitNexus upstream impact.
      knowledge_or_boundary_leakage: Per-window provider, approval, batching, tool-call visibility, panes, origin, transcript, and worktree metadata all live behind one store; handlers and session modules still import the store or SessionManager for mutations/admin flows.
      complexity_impact: Feature-specific state changes can ripple through session, status bubble, tool batching, window query tests, and persistence tests because they share one storage model; GitNexus reports 42 direct dependents and 172 total impacted symbols/files for WindowStateStore.
      cascading_change_scenarios:
        - Adding pane-state fields can force window_state_store serialization changes plus status, pane callback, and window query tests.
        - Changing provider/window metadata can co-change session.py, window_query.py, session_map.py, and multiple handler tests.
      recommended_improvement: Keep the existing read-query rule, then carve stable feature-specific projections or accessors for panes, batching/tool visibility, worktrees, and provider metadata.
      tradeoffs: A single store simplifies persistence and migration; splitting storage prematurely could make state versioning uglier. Start with projections before physical split.
    recommended_action: Reduce the write surface by adding feature-specific state accessors and monitoring co-change after each extraction.

evidence:
  - id: E1
    type: file
    ref: README.md:11-25
    summary: README states ccgram bridges Telegram to tmux and keeps tmux as the source of truth.
  - id: E2
    type: file
    ref: docs/architecture.md:5-93
    summary: Generated architecture doc maps Telegram topics to tmux windows and lists module layers, including handlers, query layer, state, infrastructure, providers, and monitor.
  - id: E3
    type: file
    ref: docs/ai-agents/architecture-map.md:86-102
    summary: Design constraints require topic-window mapping, centralized tmux operations, TelegramClient seam, state mutation rules, query-layer reads, polling purity, and lazy-import annotations.
  - id: E4
    type: file
    ref: pyproject.toml:21-45
    summary: Python package declares PTB, libtmux, aiohttp, pyte, and console script ccgram.
  - id: E5
    type: file
    ref: .github/workflows/ci.yml:25-36
    summary: CI gates format, ruff lint, pyright, deptry, unit tests, and integration tests.
  - id: E6
    type: file
    ref: tests/ccgram/test_query_layer_only_for_handlers.py:1-112
    summary: AST test enforces handler SessionManager access is restricted to write/admin allow-list; reads must use window_query/session_query.
  - id: E7
    type: file
    ref: tests/ccgram/handlers/polling/test_polling_types_purity.py:1-93
    summary: Tests enforce polling_types import-level purity and forbid pulling in polling_state.
  - id: E8
    type: file
    ref: tests/integration/test_import_no_cycles.py:1-53
    summary: Integration test imports every ccgram module in a clean interpreter to catch import cycles.
  - id: E9
    type: file
    ref: scripts/lint_lazy_imports.py:1-245
    summary: Custom linter enforces every in-function import has Lazy reason, TYPE_CHECKING guard, or reset-for-testing exception.
  - id: E10
    type: file
    ref: src/ccgram/providers/base.py:104-345
    summary: ProviderCapabilities and AgentProvider define the provider contract for launch, transcript parsing, status parsing, discovery, commands, snapshots, and task tracking.
  - id: E11
    type: file
    ref: src/ccgram/telegram_client.py:1-192
    summary: TelegramClient Protocol and PTBTelegramClient adapter define the Telegram outbound seam used by handlers and tests.
  - id: E12
    type: file
    ref: src/ccgram/handlers/polling/polling_state.py:1-16
    summary: Module docstring states polling_state owns mutable strategy classes and five module-level singletons; pure types live elsewhere.
  - id: E13
    type: file
    ref: src/ccgram/handlers/topics/directory_callbacks.py:1-14
    summary: Module docstring and dispatcher describe the directory/worktree/provider/mode callback flow.
  - id: E14
    type: file
    ref: src/ccgram/window_state_store.py:597-653
    summary: WindowStateStore exposes module-level wiring helpers and a proxy-backed window_store singleton.
  - id: E15
    type: command
    command: codegraph status
    summary: Codegraph index was up to date for /Users/alexei/Workspace/ccgram with 391 files, 11194 nodes, and 24204 edges.
  - id: E16
    type: command
    command: "python AST internal-import analysis over src/ccgram"
    summary: Top-level internal import graph had zero SCCs; all-import hubs included config, thread_router, TelegramClient, message_sender, tmux_manager, window_state_store, and polling_state; polling_state had 12 inbound internal imports.
  - id: E17
    type: command
    command: "RUFF_CACHE_DIR=$(mktemp -d) uv run ruff check src/ tests/ && uv run pyright src/ccgram tests && uv run deptry src && uv run python scripts/lint_lazy_imports.py src/ccgram && uv run pytest tests/ccgram/test_query_layer_only_for_handlers.py tests/ccgram/handlers/polling/test_polling_types_purity.py tests/integration/test_import_no_cycles.py -q"
    summary: Ruff passed, pyright reported 0 errors, deptry found no dependency issues, lint-lazy found no undocumented in-function imports, and 266 architecture/import tests passed.
  - id: E18
    type: command
    command: "git log --since=2026-05-01 --name-only over src/ccgram and tests, grouped by commit"
    summary: 26 commits since 2026-05-01; top current-path hotspot was window_state_store.py with 6 commits; high co-change pairs included session.py with window_state_store.py and window_query.py with window_state_store.py.
  - id: E19
    type: command
    command: "python LOC/function scan for src/ccgram plus rg function list for selected hotspots"
    summary: Largest files include tmux_manager.py 1189 LOC, hook.py 1143, directory_callbacks.py 1086, polling_state.py 1017, gemini.py 908, codex.py 813, session.py 696, window_state_store.py 653, and window_tick/apply.py 579.
  - id: E20
    type: command
    command: "uvx radon cc -s src/ccgram -n C"
    summary: Complexity hotspots include TranscriptParser.parse_entries F(72), SessionManager.audit_state D(28), screenshot._apply_ansi_codes D(27), hook processing/install functions, and several handler/provider functions; polling_state and window_tick/apply have C-grade methods.
  - id: E21
    type: command
    command: "uvx --from import-linter lint-imports"
    summary: Import-linter could not read configuration; no import-linter contracts are configured.
  - id: E22
    type: command
    command: "npx gitnexus status"
    summary: GitNexus index was up to date for commit 3d9f04d, matching the current commit.
  - id: E23
    type: graph-query
    query: "gitnexus_list_repos(repo='ccgram')"
    summary: GitNexus listed ccgram with 430 files, 16681 symbols, 38134 relationships, 720 communities, and 300 execution flows indexed.
  - id: E24
    type: graph-query
    query: "gitnexus_context(name='WindowStateStore', repo='ccgram'); gitnexus_impact(target='WindowStateStore', direction='upstream', depth=3, include_tests=true, repo='ccgram')"
    summary: WindowStateStore had 42 direct upstream dependents and 172 total impacted symbols/files, with CRITICAL risk; dependents include session, session_map, session_lifecycle, transcript_reader, window_query, miniapp terminal API, topic lifecycle, directory callbacks, and many tests.
  - id: E25
    type: graph-query
    query: "gitnexus_context(name='PaneStatusStrategy', repo='ccgram'); gitnexus_impact(target='PaneStatusStrategy', direction='upstream', depth=3, include_tests=true, repo='ccgram')"
    summary: PaneStatusStrategy had 22 direct upstream import dependents and 93 total impacted symbols/files, with HIGH risk; dependents span text, topic lifecycle, status emoji, RC probe, shell commands, recovery, live screenshots, command forwarding, window tick, and polling tests.
  - id: E26
    type: graph-query
    query: "gitnexus_context(name='handle_directory_callback', repo='ccgram'); gitnexus_context(name='_create_window_and_bind', repo='ccgram'); gitnexus_impact(target='_create_window_and_bind', direction='upstream', depth=3, include_tests=true, repo='ccgram')"
    summary: handle_directory_callback dispatches 11 directory/worktree/provider/mode/cancel subhandlers; _create_window_and_bind calls provider resolution, tmux window creation, SessionManager writes, thread routing, worktree persistence, shell setup, messaging-skill install, and pending-message forwarding. Upstream impact for _create_window_and_bind is LOW and local to Topics with 14 total impacted symbols/files.
  - id: E27
    type: graph-query
    query: "gitnexus_detect_changes(scope='all', repo='ccgram')"
    summary: GitNexus detected only CLAUDE.md documentation changes among indexed source/doc changes at review time, with zero affected execution processes and LOW risk.

tool_coverage:
  - dimension: discovery
    tools_used:
      - fd
      - rg
      - targeted file reads
      - git ls/status/log
    tools_skipped: []
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: structural
    tools_used:
      - Python AST import scan
      - codegraph query
      - GitNexus context/impact queries
      - rg line/function scans
      - radon complexity scan
    tools_skipped:
      - ast-grep pattern results; exploratory patterns did not add evidence beyond AST tests and rg scans
    tools_missing: []
    tools_failed: []
    confidence_impact: low
  - dimension: semantic
    tools_used:
      - pyright
      - ruff
      - pytest architecture tests
    tools_skipped: []
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: dependency
    tools_used:
      - deptry
      - codegraph status/query
      - GitNexus context/impact queries
      - pydeps cycle check
      - clean-interpreter import-cycle test
    tools_skipped: []
    tools_missing: []
    tools_failed:
      - import-linter had no configuration to read
    confidence_impact: low
  - dimension: change
    tools_used:
      - GitNexus status/list/context/impact/detect_changes runtime tools
      - git log current-path churn/co-change fallback
    tools_skipped:
      - Rename-aware GitNexus historical co-change query; exposed tools returned symbol/process impact but not historical co-change pairs.
    tools_missing: []
    tools_failed: []
    confidence_impact: low
  - dimension: operational
    tools_used:
      - GitHub Actions workflow inspection
      - pyproject/Makefile inspection
    tools_skipped:
      - Docker/Kubernetes/Terraform scans; no deploy manifests found in reviewed scope
    tools_missing: []
    tools_failed: []
    confidence_impact: low
  - dimension: security
    tools_used:
      - Send/security docs and dependency checks were included in discovery/dependency coverage
    tools_skipped:
      - dedicated SAST/secrets scan
    tools_missing: []
    tools_failed: []
    confidence_impact: medium
  - dimension: report
    tools_used:
      - architect-validate-report
    tools_skipped:
      - Mermaid rendering
      - link checker
      - spell checker
    tools_missing: []
    tools_failed: []
    confidence_impact: none
---

<!-- markdownlint-disable MD025 -->

# Architecture report: ccgram

## Executive summary

ccgram has serviceable-to-strong architecture bones. The intended topic → tmux window → agent session model is explicit, the provider and Telegram seams are real contracts, and CI enforces several architecture rules that most small Python tools never bother to encode. Miracles remain rationed: the remaining risk lives in stateful hubs and broad callback workflows, not in missing layers.

Dominant risks:

1. `polling_state.py` concentrates mutable polling strategies and singleton state.
2. `directory_callbacks.py` still carries too many topic-creation subflows.
3. `window_state_store.py` is a high-fan-in, high-churn state hub despite the query-layer guardrails.

Overall confidence: high for static structure, dependency graph, semantic checks, and CI-enforced fitness. Change-history confidence remains medium because GitNexus is fresh for symbol/process impact, but rename-aware historical co-change still falls back to git-log analysis.

## Interview context

Context was reconstructed from README, architecture docs, AI-agent architecture map, pyproject, CI, and prior modularity-review docs, then confirmed by the user for full-repo scope.

The system goal is a Telegram control plane for local tmux-hosted AI coding agents. Core domain areas are Telegram UX, provider abstraction, polling/status detection, inter-agent messaging, and Mini App. Supporting areas are session monitoring, hooks, and state persistence. Generic infrastructure is tmux, Telegram API adaptation, and LLM/Whisper HTTP integration.

Quality goals are maintainability, change locality, provider extensibility, testability, and safe Telegram formatting.

Unknown: team ownership.

## System map

Declared architecture:

- `bot.py` and `bootstrap.py`: application factory and lifecycle wiring.
- `handlers/`: Telegram feature handlers.
- `providers/`: `AgentProvider` contract and provider implementations.
- `session_monitor.py`, `event_reader.py`, `transcript_reader.py`: inbound transcript/event monitoring.
- `session.py`, `window_state_store.py`, `thread_router.py`, `window_query.py`, `session_query.py`: state and read projections.
- `tmux_manager.py`: terminal infrastructure.
- `handlers/messaging_pipeline/` and `mailbox.py`: outbound queue and inter-agent messaging.
- `miniapp/`: optional aiohttp HTTP/WebSocket dashboard.

Observed architecture matches the declared shape in the important places:

- Provider behavior is capability-gated by `ProviderCapabilities` and implemented through `AgentProvider`.
- Telegram send/edit/upload behavior sits behind `TelegramClient` and `PTBTelegramClient`.
- Handler state reads are enforced through query projections by an AST test.
- Polling has a pure `polling_types.py` contract and pure `window_tick/decide.py` decision kernel.
- Stateful polling and topic creation remain larger than their surrounding architecture suggests.

## Intended architecture

Intent sources, in source order:

1. README: tmux is the source of truth; each Forum topic binds to one tmux window.
2. `docs/architecture.md`: module layers, provider protocol, state flow, and polling split.
3. `docs/ai-agents/architecture-map.md`: design constraints to preserve, including TelegramClient seam, query-layer reads, centralized tmux operations, polling purity, lazy import contract, and bootstrap ordering.
4. CI/tests: rules are partly executable, not just prose.

Docs are treated as intent, not proof. Current tests and command evidence were used to check whether the implementation still matches them.

## Observed architecture

What is healthy:

- No top-level internal import SCCs in AST import scan.
- Clean-interpreter import-cycle test passed across the module tree.
- Query-layer handler access rule passed.
- Polling type-purity test passed.
- Ruff, pyright, deptry, lazy-import linter, and targeted architecture tests passed.
- Codegraph index was fresh.
- GitNexus index was fresh at the current commit and exposed symbol/process/impact data for the main hotspots.

What is not free:

- Stateful polling is still a shared hub imported across volatile handler paths.
- Topic creation is still concentrated in a 1086-line callback module.
- Window state is still the storage model most feature state orbits around.

## Score map

- `boundary_integrity`: 78 / serviceable / high confidence. The main boundaries are explicit and enforced by tests, but write paths still use direct SessionManager/global-store mutation APIs.
- `coupling_balance`: 68 / serviceable / high confidence. High-distance seams use contracts; GitNexus confirms the remaining unbalanced edges are mutable state and polling hubs with high blast radius.
- `dependency_graph_health`: 79 / serviceable / high confidence. Fresh codegraph, fresh GitNexus, no top-level SCCs, and import-cycle tests are good. Hubs remain expected but real.
- `cohesion_modularity`: 62 / serviceable / medium confidence. Feature subpackages exist; several large modules still bundle multiple reasons to change.
- `change_locality`: 58 / mixed / medium confidence. Recent current-path history shows state-store co-change across session/status/tooling tests; GitNexus static impact confirms the blast radius but does not replace rename-aware co-change mining.
- `architecture_fitness`: 86 / strong / high confidence. CI runs several real architecture fitness checks.
- `analysis_confidence`: 82 / strong / high confidence. Broad coverage now includes fresh GitNexus evidence; ownership and stakeholder interview depth remain the main gaps.

## Key findings

### F1 — Polling state remains a mutable strategy hub

- Problem: The pure polling contract is isolated, but mutable polling implementation remains concentrated in one 1017-line module.
- Knowledge/boundary leakage: status parsing, screen-buffer cache, interactive UI, topic lifecycle timers, and pane scanning share `polling_state.py` and singleton instances.
- Complexity impact: status and pane changes require understanding multiple caches and lifecycle strategies together; GitNexus reports HIGH upstream impact for `PaneStatusStrategy` with 22 direct import dependents and 93 total impacted symbols/files.
- Cascading-change scenario: adding a pane state or RC-status rule can touch `polling_state.py`, `window_tick/apply.py`, status UI, topic lifecycle, and tests.
- Recommendation: split the mutable strategies or inject only the needed strategy into the side-effecting window_tick path.
- Trade-off: one module makes shared runtime state visible; over-splitting would add plumbing.

### F2 — Topic creation callback flow is still too broad

- Problem: `directory_callbacks.py` carries directory browsing, favorites, worktree flow, provider/mode selection, tmux launch, shell readiness, and binding.
- Knowledge/boundary leakage: Telegram callback state, git worktree rules, provider setup, and tmux window creation are in one state machine.
- Complexity impact: unrelated topic-creation changes share failure/cleanup paths. GitNexus keeps `_create_window_and_bind` impact local to Topics, so the problem is module cohesion and callback reviewability more than repo-wide blast radius.
- Cascading-change scenario: adding a provider launch option risks edits to provider picker, mode picker, window creation, and pending-message forwarding.
- Recommendation: split navigation, worktree, provider/mode, and window-binding subflows behind a small orchestrator.
- Trade-off: keep the end-to-end flow visible through tests and a single orchestration surface.

### F3 — Window state is a high-fan-in, high-churn state hub

- Problem: query-layer reads are enforced, but `window_state_store.py` remains a broad proxy-backed state hub with CRITICAL GitNexus upstream impact.
- Knowledge/boundary leakage: panes, provider metadata, approval modes, batching/tool visibility, transcript/worktree fields, and origin share one storage model.
- Complexity impact: feature-specific state changes can ripple through persistence, session, status, tool batching, and query tests; GitNexus reports 42 direct dependents and 172 total impacted symbols/files for `WindowStateStore`.
- Cascading-change scenario: adding pane metadata can touch serialization, status bubble, pane callbacks, and window-query tests.
- Recommendation: keep one persistence model for now, but add feature-specific projections/accessors and monitor co-change before physical splitting.
- Trade-off: a single store simplifies migrations; splitting storage too early would make persistence uglier.

## Coupling review

Balanced relationships:

- Provider implementations ↔ provider protocol: contract strength, package distance, high volatility. Balanced by explicit `AgentProvider`/`ProviderCapabilities`.
- Handlers ↔ Telegram Bot API: contract strength through `TelegramClient`, third-party distance, low API volatility. Balanced.
- `window_tick/decide.py` ↔ polling inputs: contract strength through `TickContext`, low runtime distance, enforced by purity tests. Balanced.

Unbalanced relationships:

- Polling features ↔ `polling_state.py`: model/shared-state strength, handler-subpackage distance, high volatility. Medium risk.
- Topic creation subflows ↔ `directory_callbacks.py`: functional strength packed into one module, low distance but low cohesion, high volatility. Medium risk.
- Features ↔ `window_state_store.py`: model/shared-state strength, cross-feature distance, medium/high volatility. High risk due to churn.

## Boundary violations

No hard boundary violation was found in the enforced areas checked:

- Handler read access rule passed.
- Lazy import rule passed.
- Polling pure-types rule passed.
- Clean import cycle guard passed.

Residual boundary weakness is not a direct violation: write/admin mutation paths still use direct SessionManager and store APIs. That is an accepted escape hatch, but it keeps coupling stronger than the read path.

## Change locality and hotspots

Recent current-path history since 2026-05-01 shows `window_state_store.py` as the hottest source file among current paths in the sampled window. Co-change pairs tie it to `session.py`, `window_query.py`, tool-batch tests, and status-bubble tests.

GitNexus adds static/process blast-radius confirmation: `WindowStateStore` has CRITICAL upstream impact with 42 direct dependents and 172 total impacted symbols/files. `PaneStatusStrategy` has HIGH upstream impact with 22 direct dependents and 93 total impacted symbols/files. `_create_window_and_bind` is broad but local: LOW upstream impact, 14 impacted symbols/files in Topics.

LOC/complexity hotspots:

- `tmux_manager.py`: 1189 LOC.
- `hook.py`: 1143 LOC.
- `handlers/topics/directory_callbacks.py`: 1086 LOC.
- `handlers/polling/polling_state.py`: 1017 LOC.
- `providers/gemini.py`: 908 LOC.
- `providers/codex.py`: 813 LOC.
- `session.py`: 696 LOC.
- `window_state_store.py`: 653 LOC.
- `handlers/polling/window_tick/apply.py`: 579 LOC.

Not all large modules are bad. `tmux_manager.py` is generic/low-volatility infrastructure and centralization is intended. The risky ones combine size with core-domain volatility or recent co-change.

## Recommendations

1. Treat F3 first. Add feature-specific state projections/accessors around window state. Do not split persistence until projections prove useful.
2. Split `directory_callbacks.py` by subflow. Keep one small callback orchestrator and reuse current integration tests.
3. Split or inject polling mutable strategies. Preserve existing polling purity tests and add a new test that side-effecting window tick receives a narrow strategy set.
4. Add change-locality fitness if churn remains painful: a small script can flag commits that touch `window_state_store.py` plus unrelated handler packages.
5. If this becomes refactoring work, use `architecture-plan`. Source edits belong to a mutator/engineer, not this review.

## Evidence appendix

See frontmatter `evidence` entries E1–E27. Important command results:

- `codegraph status`: fresh index, 391 files, 11194 nodes, 24204 edges.
- `npx gitnexus status`: indexed commit 3d9f04d matches current commit.
- `gitnexus_list_repos`: 16681 symbols, 38134 relationships, 300 execution flows.
- `gitnexus_impact(WindowStateStore)`: CRITICAL risk, 42 direct dependents, 172 total impacted symbols/files.
- `gitnexus_impact(PaneStatusStrategy)`: HIGH risk, 22 direct dependents, 93 total impacted symbols/files.
- `gitnexus_impact(_create_window_and_bind)`: LOW risk, 4 direct dependents, 14 total impacted symbols/files, local to Topics.
- Quality/tool gate: ruff, pyright, deptry, lazy-import lint, and targeted architecture tests passed.
- Architecture tests: 266 passed.
- Import graph script: zero top-level SCCs; polling_state and window_state_store remain hubs.
- Git log fallback: window_state_store is the top current-path churn hotspot since 2026-05-01.

## Tool coverage and gaps

Covered: docs/manifests, source reads, codegraph status/query, GitNexus status/query/context/impact/detect-changes, AST import scan, pyright, ruff, deptry, radon, pydeps, targeted pytest, CI inspection, and git-log churn fallback.

Gaps:

- GitNexus is fresh for symbol/process impact, but exposed tools did not return rename-aware historical co-change pairs; git-log remains the history fallback.
- No import-linter config, so boundary contracts are custom tests rather than standard import contracts.
- No ownership file.
- Dedicated SAST/secrets scan skipped; this review targeted architecture dimensions.
