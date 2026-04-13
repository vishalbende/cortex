<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Claude_Code-Local_CLI-6366F1?style=for-the-badge" alt="Claude Code"/>
  <img src="https://img.shields.io/badge/No_API_Keys-Required-10B981?style=for-the-badge" alt="No API Keys"/>
  <img src="https://img.shields.io/badge/tests-72_passing-22C55E?style=for-the-badge" alt="72 tests passing"/>
  <img src="https://img.shields.io/badge/version-0.1.0-F59E0B?style=for-the-badge" alt="v0.1.0"/>
</p>

<h1 align="center">
  <br>
  <code>cortex</code>
  <br>
  <sub>Intelligent Agent Orchestration System</sub>
</h1>

<p align="center">
  <b>Decompose. Plan. Execute. Learn.</b><br>
  A local-first agent orchestrator powered by Claude Code CLI.<br>
  No API keys. No cloud dependency. Full observability.
</p>

---

## What is Cortex?

Cortex is an intelligent agent orchestration system that turns natural-language intents into structured, dependency-ordered execution plans and runs them through specialized domain agents. It runs entirely on the **local Claude Code CLI** — no API keys, no cloud services, no data leaving your machine.

```
You: "Design a login component with tests and accessibility review"

Cortex:
  step_1  [design_agent]      Generate component spec with design tokens     ✓ (2.1s)
  step_2  [ai_agent]          Implement React component from spec            ✓ (4.3s)
  step_3  [test_writer_agent] Generate pytest + jest test suite              ✓ (1.8s)
  step_4  [design_agent]      Accessibility audit (WCAG 2.1 AA)             ✓ (1.2s)

Done. 4/4 steps completed. 0 mistakes. Session saved.
```

---

## Key Features

**Plan-Decompose-Execute Loop** — Every intent is broken into atomic, dependency-ordered steps with confidence scoring. Low-confidence steps pause for user clarification instead of guessing.

**Domain Agents** — Stateless, swappable agents (AI reasoning, UI/UX design, test generation) registered at runtime. Add your own with a single class.

**Mistake Tracking & Learning** — Every error is recorded with type, description, correction, and lessons learned. Mistake pages are never evicted from context, so the system learns as it goes.

**Session Persistence** — Full state capture (plan, pages, mistakes, events, results) saved to JSON. Resume any session exactly where it left off.

**Permission Validation** — Semantic similarity matching against granted scopes. Destructive actions always require explicit confirmation.

**Real-Time TUI Dashboard** — Textual-based terminal UI with live plan status, agent activity, context pages, mistake log, and streaming output. Detects git repo, branches, and remotes automatically.

**No API Keys** — Runs entirely on the local Claude Code CLI. Your data stays on your machine.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and on your PATH

### Install

```bash
# Clone the repo
git clone https://github.com/yourusername/cortex.git
cd cortex

# Install globally as a CLI tool
uv tool install .

# Or install in development mode
uv pip install -e ".[dev]"
```

### First Run

```bash
# Scaffold the .cortex/ directory in your project
cortex init

# Run your first intent
cortex run "Explain the architecture of this codebase"

# Or launch the interactive REPL
cortex interactive

# Or launch the TUI dashboard
cortex tui
```

---

## Architecture

Cortex follows a three-layer execution model with strict separation of concerns:

```
                          ┌─────────────────────┐
                          │    User Intent       │
                          │  "Build a login..."  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                    ┌─────│    CortexEngine      │─────┐
                    │     │  (Orchestrator)       │     │
                    │     └──────────┬──────────┘     │
                    │                │                  │
              ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
              │ PageStore  │  │  Planner    │  │  Sessions   │
              │ (Context)  │  │ (Decompose) │  │ (Persist)   │
              └─────┬─────┘  └──────┬──────┘  └─────────────┘
                    │               │
                    │     ┌─────────▼─────────┐
                    │     │  Plan Execution    │
                    │     │  (step by step)    │
                    │     └──┬──────┬──────┬──┘
                    │        │      │      │
              ┌─────▼──┐ ┌──▼───┐ ┌▼────┐ ┌▼──────────┐
              │ RAG    │ │ AI   │ │Design│ │TestWriter │
              │ Bridge │ │Agent │ │Agent │ │Agent      │
              └────────┘ └──┬───┘ └──┬──┘ └──┬────────┘
                            │        │       │
                      ┌─────▼────────▼───────▼─────┐
                      │    Claude Code CLI          │
                      │    (local subprocess)       │
                      └────────────────────────────┘
```

### Design Principles

- **Interface Segregation** — `BaseAgent` has a minimal contract: `async run(AgentInput) -> AgentOutput`
- **Dependency Inversion** — Planner depends on `AgentRegistry` abstraction, not concrete agents
- **Open-Closed** — New agents registered at runtime without modifying existing code
- **Stateless Agents** — All state lives in the `PageStore`; agents are pure functions
- **Mistake Preservation** — Mistake pages are never evicted from the context window

---

## CLI Reference

```
cortex setup [DIR]                              One-command install + scaffold
cortex init [DIR] [--minimal] [--force]         Scaffold .cortex/ directory
cortex run INTENT [--tui] [--tmux]              Execute a single intent
cortex interactive                              Launch interactive REPL
cortex tui [INTENT]                             Launch TUI dashboard
cortex index FILES...                           Index documents into RAG
cortex agents                                   List registered agents
cortex status                                   Show engine status
cortex sessions [--filter STATUS] [--limit N]   List saved sessions
cortex resume SESSION_ID                        Resume a previous session
cortex tree [DIR]                               Show .cortex/ structure
cortex version                                  Print version
```

**Aliases:** `interactive` = `i` / `repl`, `sessions` = `ss`, `session-show` = `show`

**Global flags:**

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable debug logging |
| `--model {haiku\|sonnet\|opus}` | Select Claude model tier |

---

## TUI Dashboard

Launch with `cortex tui` for a real-time terminal dashboard:

```
┌─ Cortex — Agent Orchestration ─── myproject (main) — ~/dev/myproject ─┐
│                                                                        │
│ ┌─── PLAN ──────────────┐ ┌─── PAGES LOADED ───────┐                 │
│ │ ✓ step_1  ai_agent    │ │ [PLAN] page:plan:abc    │                 │
│ │ ⟳ step_2  design_agent│ │ [RAG]  page:rag:login   │                 │
│ │ ○ step_3  test_writer │ │ [TOOL] page:result:001  │                 │
│ └───────────────────────┘ └─────────────────────────┘                 │
│ ┌─── ACTIVE AGENTS ────┐ ┌─── MISTAKE LOG ─────────┐                 │
│ │ ▶ design_agent        │ │   No mistakes recorded. │                 │
│ └───────────────────────┘ └─────────────────────────┘                 │
│ ┌─── OUTPUT LOG ────────────────────────────────────────────────────┐ │
│ │ Repo: myproject  Branch: main  Path: ~/dev/myproject             │ │
│ │ Branches: main → origin/main, feature/login → origin/feature/... │ │
│ │ ⟳ Decomposing intent into plan...                                │ │
│ │   ✓ step_1 [ai_agent] Analyze codebase structure (2.1s)         │ │
│ │   ⟳ step_2 [design_agent] Generate component specification...   │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│ cortex> Type your instruction and press Enter...                      │
└───────────────────────────────────────────────────────────────────────┘
```

**TUI Keybindings:**

| Key | Action |
|-----|--------|
| `Enter` | Submit instruction |
| `q` | Quit |
| `Escape` | Cancel running task |
| `p` | Show current plan |
| `m` | Show mistake log |
| `c` | Show context pages |
| `s` | List saved sessions |
| `r` | Re-run last intent |

**TUI Commands:** `/status`, `/agents`, `/sessions`, `/resume <id>`, `/show <id>`, `/help`

---

## Execution Flow

```
Intent ─→ RAG Query ─→ Plan Decomposition ─→ Step Execution ─→ Result
                              │                      │
                              │              ┌───────▼───────┐
                              │              │ Per-step loop: │
                              │              │  1. Check deps │
                              │              │  2. Validate   │
                              │              │  3. Execute    │
                              │              │  4. Track      │
                              │              │  5. Learn      │
                              │              └───────┬───────┘
                              │                      │
                              │ on failure ──→ Re-plan (not retry)
                              │                      │
                              └──────────────────────▼
                                            Session Saved
```

1. **Context Loading** — RAG retrieves relevant pages for the intent
2. **Decomposition** — Claude Code CLI breaks the intent into atomic steps with dependencies
3. **Confidence Check** — Steps below 0.6 confidence pause for user clarification
4. **Execution** — Steps run sequentially or in parallel, respecting dependency order
5. **Quality Assessment** — Each step output is rated OK / WARNING / ERROR
6. **Mistake Recording** — Errors are logged with corrections and lessons learned
7. **Re-planning** — On failure, the planner re-plans around completed steps (not blind retry)
8. **Session Save** — Full state persisted for later resumption

---

## Project Structure

```
cortex/
├── cortex/
│   ├── cli.py                    # CLI argument parser and subcommands
│   ├── engine.py                 # Top-level orchestrator
│   ├── config.py                 # Centralized configuration
│   ├── models.py                 # Core data models (dataclasses)
│   ├── claude_code.py            # Bridge to local Claude Code CLI
│   ├── tmux_runner.py            # Tmux-based parallel execution
│   │
│   ├── agents/                   # Domain agent implementations
│   │   ├── base.py               #   Abstract BaseAgent interface
│   │   ├── registry.py           #   Runtime agent registration
│   │   ├── ai_agent.py           #   General-purpose LLM reasoning
│   │   ├── design_agent.py       #   Figma-aware UI/UX design
│   │   └── test_writer_agent.py  #   Test generation
│   │
│   ├── planner/                  # Intent decomposition & execution
│   │   └── planner.py            #   Plan-decompose-execute engine
│   │
│   ├── context/                  # Context management
│   │   ├── page_store.py         #   In-memory context with eviction
│   │   └── rag_bridge.py         #   Vectorless RAG integration
│   │
│   ├── mistakes/                 # Error tracking & learning
│   │   └── tracker.py            #   MistakeTracker
│   │
│   ├── permissions/              # Permission validation
│   │   └── resolver.py           #   Semantic permission matching
│   │
│   ├── sessions/                 # Session persistence
│   │   ├── model.py              #   Session & SessionEvent models
│   │   └── manager.py            #   JSON-based session storage
│   │
│   ├── tui/                      # Terminal UI dashboard
│   │   └── app.py                #   Textual-based real-time TUI
│   │
│   └── scaffold/                 # Project scaffolding
│       ├── init.py               #   .cortex/ directory generator
│       └── templates.py          #   File templates
│
├── tests/                        # 72 tests across 7 files
├── pyproject.toml
└── README.md
```

---

## Adding Custom Agents

Create a new agent by extending `BaseAgent`:

```python
from cortex.agents.base import BaseAgent
from cortex.models import AgentInput, AgentOutput, Quality

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Does something specific"
    required_permissions = ["read", "write"]

    async def run(self, input: AgentInput) -> AgentOutput:
        result = await self.do_work(input.intent)
        return AgentOutput(result=result, quality=Quality.OK)
```

Register it at runtime:

```python
engine.registry.register(MyAgent())
```

---

## Configuration

All tunables are centralized in `CortexConfig`:

| Setting | Default | Description |
|---------|---------|-------------|
| `default_model` | `"sonnet"` | Claude model tier (haiku / sonnet / opus) |
| `max_context_tokens` | `80,000` | Context window budget |
| `confidence_threshold` | `0.6` | Minimum confidence to execute a step |
| `max_retries` | `1` | Re-plan attempts on failure |
| `similarity_threshold` | `0.75` | Permission matching sensitivity |
| `tui_refresh_rate` | `0.5s` | TUI update interval |
| `use_tmux` | `true` | Enable tmux-based visual isolation |

---

## Development

```bash
# Run tests
pytest

# Run tests with verbose output
pytest -v

# Type checking
mypy cortex/

# Linting & formatting
ruff check cortex/ tests/
ruff format cortex/ tests/
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `textual` | Terminal UI framework |
| `python-dotenv` | Environment variable management |

**Dev dependencies:** `pytest`, `pytest-asyncio`, `mypy`, `ruff`

**Runtime requirement:** [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed locally

---

<p align="center">
  <sub>Built with Claude Code. No API keys required.</sub>
</p>
