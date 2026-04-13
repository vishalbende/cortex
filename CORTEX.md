# Project conventions

## Commands
- Build: `uv run build`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`

## Stack
- Python with type hints
- Cortex agent orchestration

## Rules
- Follow SOLID principles
- All agents must implement BaseAgent contract
- Tests live next to source: `foo.py` → `test_foo.py`
- Every agent ships with at least 3 test cases
