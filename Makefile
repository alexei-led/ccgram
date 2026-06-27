.PHONY: fmt lint lint-lazy test test-integration test-integration-llm test-e2e test-all typecheck deptry check install dev build clean arch-guard arch-check arch-score arch-review

fmt:
	uv run ruff format src/ tests/

lint: lint-lazy
	uv run ruff check src/ tests/

lint-lazy:
	uv run python scripts/lint_lazy_imports.py

typecheck:
	uv run pyright src/ccgram/ tests/

deptry:
	uv run deptry src

test:
	uv run pytest tests/ -m "not integration and not e2e" -n auto --dist=worksteal

test-serial:
	uv run pytest tests/ -m "not integration and not e2e"

test-integration:
	uv run pytest tests/integration/ -m "not llm" -n auto --dist=worksteal -v

test-integration-llm:
	uv run pytest tests/integration/ -m "llm" -v

test-e2e:
	uv run pytest tests/e2e/ -v --timeout=300

test-all:
	uv run pytest tests/ -n auto --dist=worksteal -v -m "not e2e"

check: fmt lint typecheck deptry test test-integration

# ── Architecture (archfit + fitness audits) ──────────────────────────────────
# arch-guard: fast (~3s), comprehensive, RELIABLE drift gate — the pytest F1-F6
# boundary audits (each has planted-violation tests proving it bites). Excludes
# the subprocess-heavy import-cycle test (runs in test-integration). This is the
# authoritative boundary gate; archfit is the whole-graph + coupling mirror.
arch-guard:
	uv run python scripts/lint_lazy_imports.py
	uv run pytest -q \
	  tests/ccgram/test_multiplexer_boundary.py \
	  tests/ccgram/test_no_tty_outside_backend.py \
	  tests/ccgram/test_query_layer_only_for_handlers.py \
	  tests/ccgram/test_window_state_access_audit.py \
	  tests/ccgram/test_window_store_import_boundary.py \
	  tests/ccgram/test_multiplexer_contract.py \
	  tests/ccgram/test_handler_layering_invariants.py

# arch-check: archfit whole-graph drift gate (forbidden-dep + cycle + coupling).
# ~45s (scip indexing). Blocks only on gate findings (exit 1); warnings (exit 2:
# advisory cycle, BC advisories, coupling scorecard) do NOT block. Run by the
# pre-push hook (scoped to src changes) and suitable for CI.
arch-check:
	@command -v archfit >/dev/null 2>&1 || { echo "archfit not installed — skipping (see github.com/alexei-led/archfit)"; exit 0; }
	@archfit check --config .archfit.yaml --full; ec=$$?; \
	if [ $$ec -eq 0 ] || [ $$ec -eq 2 ]; then exit 0; fi; \
	echo "make: archfit drift gate FAILED (exit $$ec)"; exit $$ec

# arch-score: banded architecture scorecard (coupling / cohesion / fitness).
# Report-only — use to track architecture health and improvement over time.
arch-score:
	@archfit score --config .archfit.yaml --full

# arch-review: off-gate LLM narrative (advisory; never gates). Needs
# ANTHROPIC_API_KEY in the environment.
arch-review:
	@archfit review --config .archfit.yaml

install:
	uv sync

dev:
	uv sync --extra dev

build:
	uv build

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
