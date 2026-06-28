---
artifact: architecture-report
schema_version: 2
rubric_version: 1
report_id: ccgram-archfit-coupling-modularity-2026-06-27
date: 2026-06-27

target:
  repo: ccgram
  scope: full
  out_of_scope: [miniapp-frontend-assets, tts]

comparability:
  scope: full
  rubric_version: 1
  tool_coverage_level: deep   # scip-python(4240f,0 unresolved)+grimp+gitnexus+loc; lizard/jscpd gaps

interview_context:
  system_goal: >
    Telegram bot that drives terminal coding agents (Claude Code, Codex, Gemini, Pi, shell)
    running in a terminal multiplexer (tmux/herdr); 1 Telegram topic = 1 mux window = 1 agent session.
  quality_goals: [change-locality, enforced-boundaries, backend-neutrality, single-read-path, runtime-import-safety]
  intended_units: [core, adapter]   # two-layer model in .archfit.yaml
  domains:
    core: [handlers-orchestration, session/window-state, transcript-parsing]
    supporting: [providers (AgentProvider seam), multiplexer (Multiplexer seam), llm/whisper/tts]
    generic: [config/utils, multiplexer backends, provider-detection]
  volatile_areas: [herdr backend (young CLI, protocol v14), event-stream, worktrees, provider transcript formats]
  team_ownership: [single-owner solo project (alexei-led)]
  known_pain: [handler-layer tangle managed via lazy-import contract]
  review_scope: full
  out_of_scope: [miniapp frontend, tts]

system_map:
  languages: [Python 3.14]
  package_managers: [uv]
  units: [ccgram CLI/bot (single process), claude-code hook subprocess, optional miniapp]
  deploy_units: [one pip/uv-installed process; hook runs as a separate short-lived CC subprocess]
  public_interfaces:
    - Multiplexer Protocol (multiplexer/base.py)
    - AgentProvider Protocol (providers/base.py)
    - TelegramClient Protocol (telegram_client.py)
    - file contract session_map.json + events.jsonl (hook <-> monitor)
    - herdr unix-socket JSON protocol (HERDR_PROTOCOL_VERSION=14)
    - external CLI JSONL transcript formats (claude/codex/gemini/pi)
  declared_modules: [22 modules in .archfit.yaml across core/adapter layers]
  observed_modules: [185 py files, 47,128 LOC, avg ~255 LOC/file; handlers split into 14 feature subpackages]
  high_risk_entrypoints: [bootstrap.py, main.py, hook.py, session_monitor.py, handlers/registry.py]
  missing_evidence: [complexity (lizard not enabled), clone/symmetric coupling (jscpd disabled), fresh change-coupling (gitnexus index stale Jun 24 vs HEAD Jun 27)]

module_volatility:
  - module: handlers
    classification: core
    volatility: high
    source: architect-inferred
    evidence_refs: [E-LLM-SUBDOMAIN]
    confidence: medium
    notes: Orchestration layer; drives most cross-boundary edges. High churn expected.
  - module: domain_config (config/utils)
    classification: generic
    volatility: low
    source: architect-inferred
    evidence_refs: [E-FANIN]
    confidence: high
    notes: config fan-in 72, tiny outbound footprint — stable shared infra. Undeclared volatility is the #1 BC false-alarm driver.
  - module: multiplexer (base/proxy)
    classification: generic
    volatility: low
    source: docs
    evidence_refs: [E-CONFIG]
    confidence: high
    notes: declared volatility:low in .archfit.yaml; functional low, implementation medium (two backends).
  - module: multiplexer_backends (tmux/herdr)
    classification: generic
    volatility: medium
    source: docs
    evidence_refs: [E-CONFIG]
    confidence: high
    notes: herdr young (v0.7), protocol v14 may bump — declared volatility:medium.
  - module: providers (AgentProvider + parsers)
    classification: supporting
    volatility: medium
    source: architect-inferred
    evidence_refs: [E-FORMATS]
    confidence: medium
    notes: transcript parsers track external CLI formats — implementation volatility from upstream tools.

scores:
  boundary_integrity:
    value: 78
    band: serviceable
    confidence: high
    evidence_refs: [E-AUDITS, E-CI, E-FORBIDDEN, E-INTRUSIVE]
    gaps: [archfit forbidden-dep rules are gate:warn; pytest audits are the enforced gate]
  coupling_balance:
    value: 52
    band: mixed
    confidence: medium
    evidence_refs: [E-EDGES, E-CHEAPEST, E-EXPERIMENT, E-RESIDUAL, E-SEAMS, E-FORMATS, E-HERDR]
    gaps: [after honest volatility config 62 functional+high-volatility edges remain on the live-session core (genuine high-strength/high-distance/high-volatility cluster); LLM strength labels all draft so strength is SCIP-defaulted]
  dependency_graph_health:
    value: 52
    band: mixed
    confidence: high
    evidence_refs: [E-CYCLES, E-LAZY, E-CYCLETEST, E-HUBS]
    gaps: [cycles are lazy-mediated/runtime-safe; 28-node handler SCC is real static tangle]
  cohesion_modularity:
    value: 68
    band: serviceable
    confidence: high
    evidence_refs: [E-LOC, E-DECOMP, E-CLONES, E-COMPLEXITY]
    gaps: [archfit cohesion=5/critical is a raw clone-pair-count artifact; measured duplication is only 1.44% and top clones are non-code (font licenses) + frontend JS]
  change_locality:
    value: 58
    band: mixed
    confidence: medium
    evidence_refs: [E-GITNEXUS, E-STALE]
    gaps: [gitnexus index stale ~3 days/several commits behind HEAD]
  architecture_fitness:
    value: 85
    band: strong
    confidence: high
    evidence_refs: [E-AUDITS, E-PLANTED, E-CI, E-LAZYLINT]
    gaps: [archfit check itself not yet wired into CI (F6 pending); its boundary rules stay gate:warn]
  analysis_confidence:
    value: 85
    band: strong
    confidence: high
    evidence_refs: []
    gaps: [complexity + clone gaps closed via scratchpad-config experiment; only gitnexus change-index remains stale; coupling inputs config-defaulted but the before/after was measured directly]

findings:
  - id: F1
    title: archfit's "distributed-monolith" coupling (mean 2.9/10, 387 critical edges) is an UNDECLARED-VOLATILITY artifact — proven by experiment; owner declaration is inert
    severity: medium
    dimension: coupling_balance
    evidence_refs: [E-EDGES, E-CHEAPEST, E-EXPERIMENT]
    narrative:
      problem: >
        archfit scores coupling_balance 21/100 (poor) with 387 worst-case "critical" cross-boundary edges
        and labels it distributed-monolith risk. ccgram is a single-process, single-owner monolith. I
        tested the cause on a scratchpad copy of the config (repo config untouched).
      knowledge_or_boundary_leakage: >
        430 of 489 cross-module edges have undeclared volatility -> scored worst-case high. 443 of 470
        default to functional strength via the SCIP heuristic (the 56 LLM strength labels are all
        status:draft, never promoted). In the BALANCE formula max(|strength-distance|, 10-volatility)+1,
        the 10-volatility term pins balance to ~2 when volatility is worst-case.
      complexity_impact: >
        MEASURED: declaring subdomain+volatility on every module moved mean_balance 2.9 -> 6.0 and critical
        edges 387 -> 62 (-84%), with zero code change. CRUCIALLY, also declaring `owner` on every module
        left by_distance UNCHANGED ({cross_module_different_owner: 489} before and after) — archfit's
        distance is code_structure-based, so owner declaration is INERT. Volatility is the entire lever;
        the earlier "declare owner to fix distance" intuition is wrong.
      cascading_change_scenarios:
        - A reader takes archfit's 21 at face value and "decouples" cohesive co-located modules, adding distance and unbalancing genuinely healthy coupling.
      recommended_improvement: >
        Declare `subdomain` + `volatility` on the ~14 modules that omit them (proven: mean 2.9->6.0,
        critical 387->62). Do NOT bother declaring `owner` — it does not move the score. archfit's own
        cheapest_move was declare_volatility on 56/61 advisories; after declaring it shifts to
        lower_volatility (26) / reduce_strength (10), pointing at the real residual (F6).
      tradeoffs: Subdomain/volatility labels are domain judgments; get the map right once (LLM review's subdomain suggestions are a starting point) rather than rubber-stamping.
    recommended_action: Declare subdomain+volatility (not owner) in .archfit.yaml; re-run check.
  - id: F6
    title: Once volatility is declared, archfit correctly surfaces the real coupling hotspot — 62 functional+high-volatility edges into the live-session core
    severity: medium
    dimension: coupling_balance
    evidence_refs: [E-RESIDUAL, E-EXPERIMENT]
    narrative:
      problem: >
        With volatility declared, 62 critical-band edges survive (down from 387). Every one is `functional`
        strength into a `high`-volatility target, concentrated on session_monitor, session, claude_task_state,
        monitor_state, message_routing (the live-session core). This is archfit working as intended: the
        false 387 collapsed and the genuine hotspot remained.
      knowledge_or_boundary_leakage: >
        Examples: bootstrap->session_monitor, providers.claude->claude_task_state, tool_batch->claude_task_state,
        bot->session, transcript_reader->monitor_state, miniapp.api.transcript->session_query. These modules
        share business knowledge and must co-evolve with the volatile session lifecycle.
      complexity_impact: >
        score_value 2 implies archfit assigns these high distance (|functional(8) - distance| <= 1 => distance
        ~7-9 from code_structure package separation) on top of functional strength and high volatility — the
        critical corner of the Balanced Coupling model. Whether you accept archfit's high distance or argue
        moderate distance (single deployable, single owner, in-process), functional strength + high volatility
        means changes DO cascade across these edges. This is the genuine coupling risk, not cohesion to ignore.
      cascading_change_scenarios:
        - A claude_task_state schema change ripples to providers.claude, tool_batch, and status rendering at once.
        - A session-lifecycle change touches session_monitor, monitor_state, and the miniapp transcript reader together.
      recommended_improvement: >
        Lower strength on the hottest fan-in (the ~10 with cheapest_move=reduce_strength, esp. the
        claude_task_state cluster): introduce a narrow, stable read-contract so consumers depend on an
        interface, not the volatile internal shape. That converts functional strength toward contract strength
        and cuts cascade.
      tradeoffs: Don't over-contract — co-located core modules that genuinely change together can stay cohesive; only the highest-fan-in volatile targets justify a contract.
    recommended_action: Introduce a read-contract for the highest-fan-in volatile core target (claude_task_state).
  - id: F2
    title: archfit cycle gate fails (verdict=fail) on lazy-import edges the runtime proves safe
    severity: medium
    dimension: dependency_graph_health
    evidence_refs: [E-CYCLES, E-LAZY, E-CYCLETEST]
    narrative:
      problem: >
        archfit no-import-cycles gate fires twice (a 6-node bootstrap<->transport cycle and a 28-node
        handlers SCC), driving verdict=fail. The repo's own test_import_no_cycles (clean-interpreter
        import of every module) PASSES and runs in CI.
      knowledge_or_boundary_leakage: >
        The cycle edges are deliberately deferred in-function imports. upgrade.py:103
        `from .. import main` carries the comment "Lazy: main -> bot -> handlers/upgrade. Hoisting forms
        a hard cycle"; bootstrap.py has matching "Lazy: bootstrap <-> main cycle" markers. archfit's SCIP
        graph counts in-function imports as cycle edges; at module load none of them fire.
      complexity_impact: >
        The 28-node handler SCC is a real static tangle, but it is RUNTIME-SAFE and managed by an enforced
        lazy-import contract (lint_lazy_imports.py + test_lint_lazy_imports + clean-interpreter cycle test).
        archfit cannot model the contract, so it conflates managed deferral with a load-time cycle.
      cascading_change_scenarios:
        - Wiring archfit check into CI as-is would block every PR on a "failure" that the runtime and the existing cycle test both say is safe.
        - Conversely, if a lazy edge were ever accidentally hoisted to module level, the real cycle would bite — the latent fragility is genuine, just currently contained.
      recommended_improvement: >
        Treat the cycle gate as latent-fragility signal, not a failure. Either configure archfit to
        exclude in-function imports from cycle detection, or keep its cycle rule advisory and rely on the
        clean-interpreter test as the runtime gate (which it already is).
      tradeoffs: Excluding lazy edges hides the tangle; keeping it advisory keeps visibility without false CI failures. Prefer advisory.
    recommended_action: Keep archfit cycle rule advisory; runtime gate stays test_import_no_cycles.
  - id: F3
    title: The architecturally load-bearing couplings are invisible to archfit (no import edges)
    severity: high
    dimension: coupling_balance
    evidence_refs: [E-HOOKFILE, E-HERDR, E-FORMATS]
    narrative:
      problem: >
        archfit's SCIP classifier only scores import-edge coupling. ccgram's highest-distance, most
        volatile couplings cross process/socket/format boundaries with no import edge, so they are
        entirely absent from the 470 scored edges.
      knowledge_or_boundary_leakage: >
        (1) hook.py (a separate Claude-Code subprocess) writes session_map.json + events.jsonl; ~10
        modules incl. session_monitor.py read them. hook.py and session_monitor.py share ZERO imports —
        the contract is the file/JSONL schema. (2) herdr.py + herdr_events.py couple to an external young
        (v0.7) herdr daemon over a unix socket with a pinned HERDR_PROTOCOL_VERSION=14 (HerdrProtocolError
        on mismatch). (3) provider parsers (codex.py 12x json.loads, gemini.py 5x, pi/claude) track
        external CLI JSONL formats — functional coupling to formats archfit cannot index.
      complexity_impact: >
        These are the real high-distance edges. A change to the events.jsonl schema, the herdr protocol,
        or a provider's transcript format cascades across a process/socket/tool boundary that no static
        graph captures — the exact implicit/functional coupling the Balanced Coupling model warns about.
      cascading_change_scenarios:
        - herdr bumps its protocol past 14 -> HerdrProtocolError refuses to start (handled), but new fields silently ignored until the ACL is updated.
        - A provider CLI changes its JSONL shape -> the parser drops or mis-pairs tool_use/tool_result with no compile-time signal.
      recommended_improvement: >
        These are already well-mitigated by anti-corruption layers (herdr ACL, Multiplexer/AgentProvider
        Protocols, _jsonl.py shared base) — that is good design, low strength against high distance.
        Make the contracts executable: version the session_map/events schema and add a contract test;
        the herdr protocol pin is already a runtime check (consider a unit test asserting the pinned value).
      tradeoffs: Schema-version tests add maintenance; worth it for the cross-process backbone, optional for stable internal files.
    recommended_action: Add schema/contract tests for the file + socket + format seams archfit can't see.
  - id: F4
    title: Enforced architecture fitness is the repo's strongest property and archfit under-credits it
    severity: low
    dimension: architecture_fitness
    evidence_refs: [E-AUDITS, E-PLANTED, E-CI]
    narrative:
      problem: >
        archfit scores architecture_fitness 67 (2 of 3 enforcement signals; it name-matched 3 arch_tests
        + the CI workflow). The real fitness suite is far larger and stronger.
      knowledge_or_boundary_leakage: >
        F1-F6 audits enforce every declared boundary: test_multiplexer_boundary (no caller imports a
        concrete backend), test_no_tty_outside_backend, test_query_layer_only_for_handlers,
        test_window_state_access_audit, test_window_store_import_boundary, test_lint_lazy_imports,
        test_import_no_cycles, test_multiplexer_contract, test_handler_layering_invariants. Each carries
        PLANTED-violation negative tests (test_audit_flags_planted_*) that prove the gate actually bites,
        walks >50 files, and self-checks its own allow-list. All run in make check AND .github/workflows/ci.yml.
      complexity_impact: Intent is genuinely executable and CI-gated, not aspirational prose — the single biggest reason the lazy-import tangle and backend seam stay safe under change.
      cascading_change_scenarios:
        - A refactor that imports a concrete backend, reads a tty outside the backend, or bypasses the query layer fails CI immediately with a named audit.
      recommended_improvement: >
        Keep doing this. To close archfit's blind spot, promote archfit check to a CI advisory step (F6)
        AFTER fixing F1 (config labels) and F2 (cycle-rule advisory) so it adds signal without false failures.
      tradeoffs: None material; archfit-in-CI is additive once the false alarms are addressed.
    recommended_action: Promote archfit check to advisory CI step after F1/F2 are resolved.
  - id: F5
    title: archfit's LLM review amplifies the config-gap mismeasurement into a false big-ball-of-mud verdict
    severity: low
    dimension: analysis_confidence
    evidence_refs: [E-LLMREVIEW]
    narrative:
      problem: >
        `archfit review` (anthropic/claude-opus-4-8) narrates the deterministic numbers as
        "distributed-monolith risk" / "big-ball-of-mud shape" and recommends routing handlers "through
        the window_state_ports/contract layer" — which the repo ALREADY enforces (test_query_layer_only_for_handlers).
      knowledge_or_boundary_leakage: >
        The narrative inherits archfit's inflated distance/volatility, doesn't recognize the cycles are
        lazy-mediated, and recommends an already-implemented fix. It even states "lazy imports create
        strength the static graph can't see" yet still scores coupling poor.
      complexity_impact: Confirms the tool's own guidance: LLM narration is advisory and must never be source-of-truth. Its genuine value here is the subdomain suggestions (config->generic, providers.base->supporting, session->core).
      cascading_change_scenarios:
        - Acting on the LLM verdict would mean re-decoupling already-balanced, already-enforced boundaries.
      recommended_improvement: Use the LLM review only for its subdomain/label hypotheses; feed those into the F1 config fix after human confirmation.
      tradeoffs: None; treat as hypotheses.
    recommended_action: Adopt the LLM subdomain suggestions as labels_to_confirm, ignore its scores.

archfit_calibration:
  source_commands:
    - "archfit check --config .archfit.yaml --full --advisory --report --format json"
    - "archfit check --config .archfit.yaml --full --advisory --report --format scorecard"
    - "archfit score --config .archfit.yaml --full"
    - "archfit review --config .archfit.yaml"
    - "archfit doctor"
  artifacts: [scratchpad/archfit-out/check.json, check.scorecard, score.txt, review.md]
  confirmed:
    - EXPERIMENT: declaring subdomain+volatility moved mean_balance 2.9->6.0 and critical edges 387->62 (measured, scratchpad config)
    - EXPERIMENT: declaring owner left by_distance unchanged ({cross_module_different_owner:489}) -> owner is INERT for distance
    - EXPERIMENT: enabling jscpd revealed 46 clone pairs BUT total duplication is only 1.44% (697 lines); top clones are font-license text + frontend JS
    - EXPERIMENT: enabling lizard found 29 functions CCN>15, headline parse_entries CCN 72 (transcript_parser.py:401) — a real complexity hotspot
    - 62 residual critical edges are all functional+high-volatility into the session/task-state core (real coupling pressure)
    - 28-node handler SCC and 6-node bootstrap<->transport SCC exist as static import tangles
    - 95 blast-radius hubs, 77 modules with instability I>0.7, propagation_cost 0.26 (graph shape real)
    - 3 intrusive edges exist (tmux->vim_state private _vim_state names; upgrade->main; bootstrap->shell_capture)
    - config fan-in 72, providers.base fan-in 70 (hub concentration real)
    - tool extraction trustworthy (scip-python 4240 files, 0 unresolved; grimp 184; gitnexus 104; loc 831)
  severity_adjusted:
    - coupling_balance 21 -> 60: volatility worst-case by undeclared config (PROVEN lever); strength SCIP-defaulted (LLM labels draft); residual is mostly cohesion-in-core
    - 2 cycle gate findings (verdict=fail) -> managed latent fragility, NOT failure (lazy-mediated; clean-interpreter test passes in CI)
    - cohesion_modularity 5 -> 68: archfit counts raw clone PAIRS (46) incl. non-code; actual duplication 1.44% on a finely decomposed repo
    - 3 intrusive edges -> minor (1 intentional/balanced tmux->vim_state score 8; 2 are lazy-cycle edges)
    - architecture_fitness 67 -> 85: full F1-F6 audit suite with planted negative tests, CI-enforced, under-detected by name heuristic
  false_positive_or_noise:
    - "owner-declaration fixes the distance" intuition — DISPROVEN by experiment (by_distance unchanged). Volatility is the only lever.
    - cohesion 5/critical from raw clone-pair count: includes 89-line font-LICENSE text pair + frontend panes.js/terminal.js JS; real Python duplication is ~1.44%
    - "distributed-monolith / 387 critical edges" framing for a single-process single-owner monolith (config artifact)
    - cohesion "god files 87" LOC-skew on a finely decomposed repo (185 files, avg 255 LOC; only ~12 files >700 LOC, all naturally large)
    - .gitnexus/ and .archfit-cache/ generated tool state under the scanned root (measurement contamination risk if not excluded)
  missed_by_archfit:
    - hook.py <-> session_monitor.py cross-process coupling via session_map.json + events.jsonl (no import edge; ~10 module schema sharers)
    - herdr unix-socket JSON contract + HERDR_PROTOCOL_VERSION=14 pin to external young CLI (external, no import edge)
    - provider parsers <-> external CLI JSONL transcript formats (functional coupling to unindexable formats)
    - LLM prompt/response contracts (summarizer, shell NL->command) and shared state.json schema
  config_changes:
    - declare subdomain + volatility on the ~14 modules that omit them -> PROVEN: mean 2.9->6.0, critical 387->62
    - do NOT add owner -> EXPERIMENT showed it leaves distance unchanged (archfit ignores it for code_structure distance)
    - enable tools.complexity (lizard) and tools.clones (jscpd) BUT exclude fonts/ and miniapp/static (non-Python) from clone scan -> avoids 5/critical false cohesion verdict
    - refresh gitnexus index (node .gitnexus/run.cjs analyze --index-only) -> trustworthy change_locality
    - ensure .gitnexus/ and .archfit-cache/ are gitignored / excluded from scan root
  archfit_tool_gaps:
    - distance never uses ownership even when owner is declared (top-level owner field rejected; module owner accepted but inert) -> distance is code_structure-only
    - cohesion scoring uses raw clone-PAIR count, not duplication %, and includes non-code files -> 1.44% dup yields cohesion 5/critical
    - cycle detection counts in-function (lazy) imports as cycle edges -> verdict=fail vs a clean-interpreter test that passes
    - coupling_balance dimension curve is harsh: mean_balance 6.0/10 still maps to 40/100 (poor)
    - fitness detection is name-heuristic (matched 3 tests) -> misses planted-violation audit suites
  new_fitness_checks:
    - promote archfit check to advisory CI step (F6) after F1/F2
    - schema-version + contract test for session_map.json / events.jsonl
    - unit test asserting the pinned HERDR_PROTOCOL_VERSION value
  labels_to_confirm:
    - 56 draft strength labels (16 contract, 32 functional, 8 model) — stale (Jun 13, pre event-stream/worktree); approve or delete
    - LLM subdomain suggestions: config=generic, handlers=core, providers.base=supporting, multiplexer.base=supporting, session=core
  confidence_impact: medium

evidence:
  - id: E-CONFIG
    type: file
    ref: .archfit.yaml:124-171
    summary: Module layering + declared subdomain/volatility for multiplexer modules; most modules omit owner/subdomain/volatility.
  - id: E-EDGES
    type: command
    command: "jq '.classified_edges' check.json"
    summary: 937 total / 470 scored / 19 abstained; mean_balance 2.9; by_distance all cross_module_different_owner; by_volatility undeclared 430/489; by_strength functional 443.
  - id: E-CHEAPEST
    type: command
    command: "jq -r '.findings[]|select(.rule_id==\"bc/imbalanced_coupling\")|.matched_by.cheapest_move' check.json | sort | uniq -c"
    summary: 56/61 BC advisories cheapest_move=declare_volatility; distance_basis=code_structure on all 61.
  - id: E-CYCLES
    type: command
    command: "jq -c '.findings[]|select(.rule_id==\"no-import-cycles\")' check.json"
    summary: 6-node (bootstrap->bot->cli->registry->upgrade->main) and 28-node handlers SCC; verdict=fail.
  - id: E-LAZY
    type: file
    ref: src/ccgram/handlers/upgrade.py:100-103
    summary: "# Lazy: main -> bot -> handlers/upgrade. Hoisting forms a hard cycle" then in-function `from .. import main`; bootstrap.py mirrors with "Lazy: bootstrap <-> main cycle".
  - id: E-CYCLETEST
    type: file
    ref: tests/integration/test_import_no_cycles.py:48-97
    summary: Imports every module in a clean subprocess; asserts returncode 0; proves no runtime cycle. Runs in CI integration step.
  - id: E-AUDITS
    type: file
    ref: tests/ccgram/test_multiplexer_boundary.py:32-137
    summary: Forbidden-prefix import audit with allow-list; walks src; planted-violation tests assert it flags ccgram.multiplexer.tmux/libtmux/legacy imports.
  - id: E-PLANTED
    type: file
    ref: tests/ccgram/test_no_tty_outside_backend.py:113-138
    summary: test_audit_actually_walks_files (>50) + planted pane_tty/get_foreground_args/ps-t tests prove the gate bites.
  - id: E-CI
    type: file
    ref: .github/workflows/ci.yml:25-36
    summary: CI runs ruff format/check, pyright, deptry, pytest unit (all boundary audits) + pytest integration (cycle + herdr contract).
  - id: E-LAZYLINT
    type: command
    command: "grep -nE '^(check|lint|lint-lazy):' Makefile"
    summary: make check = fmt lint typecheck deptry test test-integration; lint includes lint-lazy (lazy-import contract).
  - id: E-FORBIDDEN
    type: command
    command: "jq -r '.findings[]|.rule_id' check.json | sort | uniq -c"
    summary: Only rule_ids are bc/imbalanced_coupling (61) and no-import-cycles (2). Zero forbidden_dependency violations.
  - id: E-INTRUSIVE
    type: command
    command: "jq -c '.findings[]|select(.matched_by.strength==\"intrusive\")' check.json"
    summary: 3 intrusive edges: bootstrap->shell_capture(4), upgrade->main(4), tmux->vim_state(8,low).
  - id: E-HOOKFILE
    type: command
    command: "grep -nE 'session_monitor' src/ccgram/hook.py ; grep -nE 'import.*hook' src/ccgram/session_monitor.py"
    summary: No import edge either direction; coupling is via session_map.json + events.jsonl shared by ~10 modules.
  - id: E-HERDR
    type: file
    ref: src/ccgram/multiplexer/herdr.py:78-372
    summary: HERDR_PROTOCOL_VERSION=14; raises HerdrProtocolError when server protocol != pinned value.
  - id: E-FORMATS
    type: command
    command: "for f in providers/*.py; do echo $f $(grep -cE 'json.loads|\\.jsonl' $f); done"
    summary: codex.py 12, gemini.py 5, pi/_jsonl 2 — external CLI JSONL format parsing (functional coupling).
  - id: E-HUBS
    type: command
    command: "jq -r '.metrics[]|select(.name|test(\"blast|instability|propagation\"))' check.json"
    summary: blast_radius 95, instability 77 (I>0.7), propagation_cost 0.26.
  - id: E-LOC
    type: command
    command: "find src/ccgram -name '*.py' | xargs wc -l | sort -rn | head"
    summary: 185 files, 47,128 LOC, avg ~255; largest hook.py 1307, directory_callbacks 1216, tmux 1166, herdr 1118 — ~12 files >700.
  - id: E-DECOMP
    type: file
    ref: .claude/rules/architecture.md:1-1
    summary: Deliberate decomposition — handlers in 14 feature subpackages, multiplexer/providers seams, window_state_ports, pure decision kernels.
  - id: E-GITNEXUS
    type: command
    command: "archfit score --full (change_locality line)"
    summary: change_locality 66 serviceable; 5 co-changing module pairs, 1 amplifying hub.
  - id: E-STALE
    type: command
    command: "ls -la .gitnexus/ ; git log -1"
    summary: .gitnexus dir Jun 24 23:32; HEAD Jun 27 — index misses event-stream/worktree/PR#122 commits.
  - id: E-FANIN
    type: file
    ref: scratchpad/archfit-out/review.md:31-33
    summary: config fan-in 72, providers.base fan-in 70 (from archfit evidence surfaced by LLM review).
  - id: E-LLMREVIEW
    type: file
    ref: scratchpad/archfit-out/review.md:7-35
    summary: LLM narrative calls it distributed-monolith/big-ball-of-mud; recommends already-implemented query-layer routing; useful subdomain suggestions.
  - id: E-EXPERIMENT
    type: command
    command: "archfit check --config <scratchpad>/.archfit.yaml --root <repo> --full (with subdomain+volatility+owner declared)"
    summary: "BEFORE mean 2.9/critical 387/vol{undeclared:430}; AFTER mean 6.0/critical 62/vol{high:69,low:227,med:193}. by_distance unchanged {cross_module_different_owner:489} => owner inert."
  - id: E-RESIDUAL
    type: command
    command: "jq '.findings[]|select(.matched_by.score_band==\"critical\")|{from,to,str,vol}' check2.json"
    summary: All 62 residual critical edges are functional+high-volatility into session_monitor/session/claude_task_state/monitor_state (the live-session core).
  - id: E-CLONES
    type: command
    command: "jscpd src/ccgram --reporters json --min-tokens 50"
    summary: 46 clones / 697 lines / 1.44% duplication. Top: fonts/LICENSE (89L, non-code), miniapp JS (48L), handlers cleanup.py boilerplate (18-28L), providers/_jsonl<->claude (23L).
  - id: E-COMPLEXITY
    type: command
    command: "archfit metrics.complexity (lizard)"
    summary: 29 functions CCN>15; parse_entries CCN 72 (transcript_parser.py:401), format_tool_use_summary 28, audit_state 28 (session.py:358), _apply_ansi_codes 27.
  - id: E-SEAMS
    type: file
    ref: .claude/rules/architecture.md:1-1
    summary: Multiplexer/AgentProvider/TelegramClient Protocols are the contract-strength seams at every volatile/external boundary.

tool_coverage:
  - dimension: discovery
    tools_used: [fd, rg, git, archfit doctor]
    tools_skipped: []
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: structural
    tools_used: [archfit(loc), wc, ast-grep(via archfit), lizard(via experiment — 29 CCN>15)]
    tools_skipped: []
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: semantic
    tools_used: [scip-python(4240 files,0 unresolved), grimp(184)]
    tools_skipped: []
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: dependency
    tools_used: [scip, grimp, archfit classified_edges, jscpd(via experiment — 46 clones/1.44%)]
    tools_skipped: [dependency-cruiser(absent, TS off)]
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: change
    tools_used: [gitnexus(104 files)]
    tools_skipped: []
    tools_missing: []
    tools_failed: [gitnexus index stale ~3 days]
    confidence_impact: medium
  - dimension: operational
    tools_used: []
    tools_skipped: [single-process app; no k8s/terraform]
    tools_missing: []
    tools_failed: []
    confidence_impact: none
  - dimension: report
    tools_used: [archfit scorecard/json, jq]
    tools_skipped: []
    tools_missing: []
    tools_failed: []
    confidence_impact: none
---

# Architecture report: ccgram

> **Status update (2026-06-27, archfit v0.12.1):** re-run on the latest book-verbatim
> scorer with LLM assist, config corrected, and the drift gate implemented. The body
> below is the original v0.10 review (calibrated architect scores still hold); see
> **[Appendix A: v0.12.1 re-review, release delta, and governance](#appendix-a)** at the
> end for the current numbers, the v4.1.0→v4.2.0 delta, the config-coverage fix, and the
> implemented gate.

## Executive summary

ccgram is a **cohesive, single-process, single-owner monolith with exemplary enforced boundaries**. Its
strongest property — a suite of F1–F6 fitness functions that gate every declared boundary in CI, each with
planted-violation negative tests — is the thing archfit under-credits most.

archfit's headline (Overall **41/100 mixed**, with coupling_balance **21** and dependency_graph_health **24**
"poor") reads as near-distributed-monolith. That verdict is **not credible for this repo**, and the reasons
are instructive for the tool:

1. **Coupling looks critical only because volatility is under-declared — proven by experiment.** On a scratchpad
   copy of the config (repo untouched), declaring `subdomain`+`volatility` moved mean balance **2.9 → 6.0** and
   critical edges **387 → 62** with zero code change. Declaring `owner` too changed **nothing** (`by_distance`
   identical) — archfit's distance is `code_structure`-based, so owner is inert. Volatility is the whole lever.
2. **The cycle gate fails on lazy imports the runtime proves safe.** The two "critical" cycles are deferred
   in-function imports (commented `# Lazy: ... Hoisting forms a hard cycle`). The repo's own
   `test_import_no_cycles` imports every module in a clean interpreter and passes in CI.
3. **The couplings that actually carry risk are invisible to archfit** because they have no import edge: the
   hook↔monitor file contract, the herdr socket/protocol-14 contract, and provider↔external-format coupling.
4. **Cohesion=5/critical is a clone-count artifact.** Enabling jscpd surfaced 46 clone _pairs_, but measured
   duplication is only **1.44%** and the top clones are a font-LICENSE text pair + frontend JS. Real Python
   duplication is minor boilerplate. archfit scores raw pair count, not duplication ratio.

Calibrated scores (mine, with the swings explained below): boundary 78, coupling_balance 52,
dependency_graph_health 52, cohesion 68, change_locality 58, **architecture_fitness 85**, analysis_confidence 85.
Overall a **serviceable** architecture with a strong fitness/boundary story. The genuine coupling residual —
which archfit surfaces correctly _once volatility is declared_ — is 62 functional + high-volatility edges into
the live-session core (claude_task_state, session, session_monitor); the ~10 highest-fan-in are real
contract candidates. The other real risks are the external-contract seams archfit can't see (already
well-mitigated by anti-corruption layers).

## Where archfit succeeded

- **Trustworthy extraction.** scip-python indexed 4240 files with 0 unresolved; grimp, gitnexus, and loc all ran.
  The dependency graph it built is real.
- **Real graph shape.** It correctly surfaced the 28-node handler SCC, 95 blast-radius hubs, 77 unstable
  modules (I>0.7), propagation_cost 0.26, and fan-in hubs (config 72, providers.base 70). These are worth
  attention even though the cycles are lazy-mediated.
- **The 3 intrusive edges** (tmux→vim_state private-name import, upgrade→main, bootstrap→shell_capture) — real,
  if minor.
- **Book-aligned math, and it works once fed honest inputs.** The `bc_score.v3` equation is the correct
  Balanced Coupling formula. The failure is garbage-in (config-defaulted volatility), not the scorer — and the
  experiment proved it: with volatility declared, the 387 false criticals collapsed to exactly the 62 edges that
  matter, all clustered on the genuinely volatile live-session core. Configured properly, archfit found the
  real hotspot.
- **Honest about its own gaps** — it reports the lizard coverage gap, caps analysis_confidence at 90 for the
  unmeasured dimension, and its cohesion description explicitly says high-strength/low-distance cohesion should
  not be penalized.

## Where archfit failed (the user's question)

1. **Volatility is the only coupling lever, but it is undeclared → false distributed-monolith verdict.**
   Experiment: declaring volatility moved mean balance 2.9→6.0 / critical 387→62. **Owner is inert** — declaring
   it left `by_distance` byte-identical; archfit's distance is `code_structure`-only and never consumes
   ownership (the top-level `owner` field is even rejected). So the natural "declare owner to fix distance"
   reflex does nothing; only volatility/subdomain helps.
2. **Cohesion scoring uses raw clone-PAIR count, not duplication %.** 46 pairs (incl. an 89-line font-LICENSE
   text pair and frontend JS) → cohesion 5/critical, when actual duplication is 1.44%. A ratio-based or
   code-only metric would not call this critical.
3. **Cycle detection conflates lazy/deferred imports with load-time cycles** → `verdict=fail` against a repo
   whose clean-interpreter cycle test passes in CI. archfit can't model the lazy-import contract.
4. **Structurally blind to non-import coupling** — the file-schema (session_map/events), socket (herdr proto 14),
   and external-format (provider JSONL) contracts are exactly the high-distance edges that matter, and none have
   import edges. This is the canonical implicit/functional-coupling blind spot.
5. **Under-credits enforced fitness** — name-matched 3 tests + CI; missed the full F1–F6 suite with planted
   negative tests. The repo's best property is nearly invisible.
6. **The coupling_balance curve is harsh** — even after honest config, mean_balance 6.0/10 maps to 40/100
   (poor). The number under-credits a genuinely cohesive monolith; read the mean + critical-edge delta instead.
7. **LLM enrichment is drafted-but-dormant** — all 56 strength labels are `status:draft` (stale Jun 13), so the
   deterministic gate runs purely on SCIP and the model/contract refinements are unused.
8. **`archfit review` amplifies the mismeasurement** into a "big-ball-of-mud" narrative and recommends a fix the
   repo already enforces (route handlers through the query layer) — a concrete demonstration of why LLM
   narration is advisory only.

## Coupling review (focus dimension)

| Relationship                                                | Strength                                  | Distance (code/own/runtime/deploy)                                  | Volatility                  | Balance                                            | Severity       | Move                                                        |
| ----------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------- | --------------------------- | -------------------------------------------------- | -------------- | ----------------------------------------------------------- |
| handlers → core query layer / window_state_ports            | functional (intra-app, contract-mediated) | code med / own low / runtime in-proc / deploy same                  | low–med                     | balanced (high strength + low distance = cohesion) | low            | leave alone                                                 |
| bootstrap → modules (wiring)                                | functional                                | code med / own low / in-proc / same                                 | low                         | balanced (composition root)                        | low            | leave alone                                                 |
| caller → Multiplexer/AgentProvider/TelegramClient Protocols | **contract**                              | code high / own low / in-proc / same                                | low fn, med impl            | well-balanced (low strength vs high distance)      | low            | leave alone — textbook ACL                                  |
| herdr.py ↔ herdr daemon (socket + proto 14)                 | contract (ACL)                            | code high / own high (external) / **socket** / **separate process** | **medium-high** (young CLI) | balanced by the ACL + protocol pin                 | medium (watch) | keep ACL; add proto-pin test                                |
| hook.py ↔ session_monitor (session_map/events)              | functional/contract via file schema       | code high / own low / **cross-process** / separate                  | low–med                     | acceptable; stable schema                          | medium (watch) | version the schema + contract test                          |
| provider parsers ↔ external CLI JSONL                       | **functional**                            | code high / own high (external) / file / external                   | medium                      | least-balanced of the seams                        | medium         | shared _jsonl base already helps; add format contract tests |
| tmux ↔ vim_state (private `_vim_state`)                     | intrusive                                 | code low (same module) / own low / in-proc                          | low                         | balanced (score 8)                                 | low            | leave alone — documented neutral-cache seam                 |

The one cluster that _does_ approach the critical corner is **internal**, not at the seams: the 62 residual
edges into the live-session core (claude_task_state, session, session_monitor) are functional strength +
high volatility + high code_structure distance (F6). archfit surfaces these correctly once volatility is
declared. The external seams, by contrast, are well-balanced — held at contract strength by anti-corruption
layers (herdr ACL, the three Protocols), so their high distance is offset by low strength. The external-format
provider coupling is the closest seam watch-item (functional strength to an external format), mitigated by the
shared `_jsonl.py` base.

## Recommendations for ccgram (prioritized, incremental)

**A. Make archfit honest about this repo (config, no code change) — highest ROI, measured**

1. Add `subdomain:` + `volatility:` to the ~14 modules that omit them (the LLM review's subdomain map is a
   starting point: config=generic, handlers=core, providers.base/multiplexer.base=supporting, session=core).
   **Measured effect: mean balance 2.9→6.0, critical edges 387→62.**
2. **Do not bother with `owner:`** — the experiment proved it does not change distance (archfit's distance is
   `code_structure`-only). Skipping it avoids a misleading config change.
3. Enable `tools.complexity` (lizard) and `tools.clones` (jscpd), **but exclude `fonts/` and `miniapp/static/`
   from the clone scan** so the cohesion score reflects Python duplication (1.44%), not font-license text.
   Refresh the gitnexus index and ensure `.gitnexus/`/`.archfit-cache/` are gitignored.

**B. Address the real residual coupling (the 62 surviving edges)** 4. The residual is functional+high-volatility edges into the live-session core. Most are healthy cohesion;
the ~10 with `cheapest_move=reduce_strength` (several into `claude_task_state`) are contract candidates —
introduce a narrow read-interface for the highest-fan-in volatile targets. Leave the rest alone.

**C. Resolve the false cycle failure** 5. Keep archfit's `no-import-cycles` rule **advisory**; the runtime gate is already `test_import_no_cycles`.
The 28-node handler SCC is real latent fragility — keep it visible, don't let it fail CI.

**D. Tackle the genuine complexity hotspot + minor DRY** 6. Refactor `parse_entries` (CCN 72, transcript_parser.py:401) — by far the worst complexity outlier; also
`audit_state` (28) and `_apply_ansi_codes` (27). DRY the small cleanup.py boilerplate repeated across ~5
handlers if cheap; the rest of the 1.44% duplication is not worth chasing.

**E. Make the invisible contracts executable (small code/test additions)** 7. Add a schema-version constant + contract test for `session_map.json` / `events.jsonl`; a unit test asserting
the pinned `HERDR_PROTOCOL_VERSION`; and lightweight format-contract tests for the provider JSONL parsers.

**F. Promote the gate (after A & C)** 8. Wire `archfit check` into CI as an **advisory** step (the pending F6), now that false alarms are addressed. 9. Decide on the 56 draft LLM labels: review+approve the ones you trust (they then override SCIP and sharpen
coupling scoring), or delete them so they don't rot.

## Plan summary

This is a read-only review. Recommended single next step: **architecture-design** — to define (a) the corrected
`.archfit.yaml` subdomain/volatility model, (b) the read-contract for the live-session core (F6), and (c) the
schema/protocol contracts for the seams archfit can't see (F3) as target boundaries with fitness checks. Only
after that design is approved would **architecture-plan** sequence it. No big-bang refactor is warranted — the
architecture is sound; the gaps are measurement honesty, one core read-contract, and a few executable contracts.

---

<a name="appendix-a"></a>

## Appendix A: v0.12.1 re-review, release delta, and governance

Re-run on **archfit v0.12.1** (book-verbatim `bc_score.v3` scorer) with the Anthropic LLM
assist, after correcting the config. The original review above used v0.10; my calibrated
architect scores (boundary 78, coupling 52, dep_health 52, cohesion 68, change_locality 58,
fitness 85, analysis_confidence 85) still hold — the v0.12.1 deltas below are deterministic
inputs to those, not replacements. For future scans, use `archfit check --config .archfit.yaml --full`
for the full scan and `archfit check --config .archfit.yaml --base <ref>` for delta scans.

### A.1 Config corrections (the design model is now right)

- **Forbidden-dep rules were silent no-ops (archfit bug/gotcha).** `forbidden_dependency`
  `from`/`to` take **path globs** (`ccgram.handlers**`), not module-group names (`handlers`).
  The repo's rules used names → matched nothing → never fired. Fixed to globs and **proven**:
  planted `from ccgram.multiplexer import herdr` in a handler → `verdict: fail`. (Prior memory
  obs 23467 half-spotted this on Jun 21.)
- **Coverage drift fixed.** The config used explicit submodule path lists, so the
  push-event-stream feature's 4 new modules were silently **unmapped** (vanished from edge
  classification, no error): `event_stream_monitor`→`session_state`,
  `agent_status_cache`+`topic_mapping`→`multiplexer`, `herdr_events`→`multiplexer_backends`.
  Now mapped (edge coverage 471→**484**). Re-check coverage after any feature work.
- **`volatility_cascade_enabled: true`** added (book Ch9; archfit self-config enables it).
- **Owner is a hygiene label, not a coupling lever.** The current config keeps `owner: alexei-led`
  on every module for completeness, but the calibrated evidence still says only `subdomain` +
  `volatility` move coupling. Treat owner as documentation, not score input.

### A.2 v0.12.1 deterministic scorecard (HEAD, corrected config)

- Overall 43 (mixed); boundary 58, coupling_balance 40, dependency_graph_health 24,
  cohesion 5, change_locality 66, fitness 67, analysis_confidence 90.
- classified edges: 484 scored, mean book balance **4.68/10**, **218 critical-band** edges,
  **0 gate findings** (green), 2 cycles (lazy-mediated, advisory).
- The cascade lifted high-volatility edges to ~248 (from 69 pre-cascade): ccgram's large,
  central, actively-developed core (handlers + session_state) propagates volatility widely.
  Read coupling 40/poor as **"big volatile core with wide reach"** (cohesion + blast radius),
  NOT distributed-monolith (single process). cohesion 5 remains a clone-pair-count artifact
  (real duplication 1.4%).

### A.3 Release delta — v4.1.0 → v4.2.0 (same config, apples-to-apples)

v4.1.0's code analyzed under the current config (throwaway worktree):

| Dimension               | v4.1.0 | HEAD | Δ       |
| ----------------------- | ------ | ---- | ------- |
| Overall                 | 48     | 43   | −5      |
| change_locality         | 84     | 66   | **−18** |
| cohesion_modularity     | 11     | 5    | −6      |
| boundary_integrity      | 60     | 58   | −2      |
| coupling_balance        | 40     | 40   | 0       |
| dependency_graph_health | 24     | 24   | 0       |
| architecture_fitness    | 67     | 67   | 0       |

The v4.2.0 release (push-event-stream + worktrees: 10 files, +723 lines, 3 new modules) was a
**cross-cutting seam addition** — it threaded an event-streaming concern through
multiplexer → event monitor → agent_status_cache → polling → death notification. That shows up
as a change_locality drop (−18, co-changed many modules at once) and a small cohesion dip
(new clone pairs). It added **no new import cycles, no boundary violations, no gate findings** —
a clean cross-cutting feature that stayed within the seam contracts.

### A.4 Architecture governance (implemented)

The architecture intent is now executable as a fast gate plus an advisory whole-graph layer:

- **`make arch-guard`** — fast (~3s) reliable pytest F1–F6 boundary audits (planted-violation
  tests prove they bite). The authoritative drift gate.
- **`make arch-check`** — archfit whole-graph gate (forbidden-dep + cycle + coupling), ~45s;
  blocks only on gate findings, warnings pass.
- **`make arch-score` / `make arch-review`** — banded scorecard / LLM narrative for tracking
  health and improvement over time.
- **Pre-push hook** (`scripts/git-hooks/pre-push`, `core.hooksPath=scripts/git-hooks`) runs
  arch-guard + arch-check, scoped to src/.archfit changes so doc pushes stay instant.
- **CI advisory job** (`.github/workflows/ci.yml` `architecture`, `continue-on-error`) installs
  archfit + analyzers, runs the gate, and uploads the scorecard artifact.

### A.5 Where archfit failed (updated)

The v0.10 list still holds (lazy-cycle false-fail, non-import coupling blind spot,
under-credited fitness, cohesion clone-count artifact, harsh coupling curve). Add, from this
round:

- **`forbidden_dependency` rules silently match nothing when `from`/`to` are module names
  instead of path globs** — the gate the user intended (F6) would have passed everything. The
  single most important fix; now corrected and verified.
- **Config exclusions don't reach the loc-based metrics** (`file_structural_weight` counted a
  stray `pkg/mod/` Go cache until it was deleted).
- **`archfit update`'s auto-discovery flattens semantic module groupings** to raw package
  structure — do not `--apply` it; the hand-authored architecture model is more correct.
