.PHONY: install install-dev run demo test test-unit test-integration test-agent evals lint typecheck clean help

# ── Installation ──────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ── Running ───────────────────────────────────────────────────────────────────
run:
	python -m app.main

demo:
	python -m app.main --scenario demo

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-agent:
	pytest tests/agent/ -v

test-cov:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# ── Evals ─────────────────────────────────────────────────────────────────────
evals:
	python -m evals.runner

evals-verbose:
	python -m evals.runner --verbose

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check app/ tests/ evals/

lint-fix:
	ruff check --fix app/ tests/ evals/

typecheck:
	mypy app/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "AI Calling Agent — Available Commands"
	@echo "────────────────────────────────────────"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install dev dependencies"
	@echo "  make run            Launch interactive CLI simulator"
	@echo "  make demo           Run the built-in demo scenario"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-agent     Run agent behaviour tests"
	@echo "  make evals          Run evaluation suite (9 scenarios)"
	@echo "  make lint           Check code style"
	@echo "  make clean          Remove build artifacts"
	@echo ""
