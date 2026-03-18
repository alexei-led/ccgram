.PHONY: fmt lint test test-integration test-e2e test-all typecheck deptry check install dev build clean

fmt:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run pyright src/ccgram/

deptry:
	uv run deptry src

test:
	uv run pytest tests/ -m "not integration and not e2e"

test-integration:
	uv run pytest tests/integration/ -v

test-e2e:
	uv run pytest tests/e2e/ -v --timeout=300

test-all:
	uv run pytest tests/ -v -m "not e2e"

check: fmt lint typecheck deptry test test-integration

install:
	uv sync

dev:
	uv sync --extra dev

build:
	uv build

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
