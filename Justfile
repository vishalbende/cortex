# Cortex — uv-powered task runner
# Install just: https://just.systems/man/en/
# Or use the commands directly with `uv ...`

default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────

# Install all deps (project + dev)
install:
    uv sync --dev

# Install cortex as a global CLI command
install-global:
    uv tool install .

# Uninstall global CLI
uninstall-global:
    uv tool uninstall cortex

# ── Run ──────────────────────────────────────────────────────────────

# Run an intent headless
run *ARGS:
    uv run cortex run {{ARGS}}

# Launch interactive REPL
repl:
    uv run cortex interactive

# Launch the TUI dashboard
tui *ARGS:
    uv run cortex tui {{ARGS}}

# Run with tmux panes per agent
tmux *ARGS:
    uv run cortex run --tmux {{ARGS}}

# Index a document into RAG
index *FILES:
    uv run cortex index {{FILES}}

# List registered agents
agents:
    uv run cortex agents

# Show engine status
status:
    uv run cortex status

# ── Dev ──────────────────────────────────────────────────────────────

# Run the test suite
test *ARGS:
    uv run pytest {{ARGS}}

# Run tests with verbose output
test-v:
    uv run pytest -v

# Type check
typecheck:
    uv run mypy cortex/

# Lint
lint:
    uv run ruff check cortex/ tests/

# Lint and auto-fix
lint-fix:
    uv run ruff check --fix cortex/ tests/

# Format
fmt:
    uv run ruff format cortex/ tests/

# Run all checks (lint + typecheck + test)
check: lint typecheck test

# ── Packaging ────────────────────────────────────────────────────────

# Build the wheel
build:
    uv build

# Show project info
info:
    uv run cortex version
    @echo "Python: $(uv run python --version)"
    @echo "uv:     $(uv --version)"

# Lock dependencies
lock:
    uv lock

# Update all dependencies
update:
    uv lock --upgrade
    uv sync --dev
