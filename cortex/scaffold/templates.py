"""
Template content for every file in the .cortex/ directory structure.

Mirrors the Claude Code .claude/ directory spec but uses Cortex naming:
  .claude/       → .cortex/
  CLAUDE.md      → CORTEX.md
  claude         → cortex  (in all references)

Each template is a (relative_path, content, badge) tuple where badge is
one of: "committed", "gitignored", "local".
"""

# ── CORTEX.md (project root) ────────────────────────────────────────

CORTEX_MD = """\
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
"""

# ── .cortex/settings.json ────────────────────────────────────────────

SETTINGS_JSON = """\
{
  "permissions": {
    "allow": [
      "Bash(uv run *)",
      "Bash(pytest *)",
      "Bash(ruff *)"
    ],
    "deny": [
      "Bash(rm -rf *)"
    ]
  },
  "hooks": {},
  "env": {}
}
"""

# ── .cortex/settings.local.json ─────────────────────────────────────

SETTINGS_LOCAL_JSON = """\
{
  "permissions": {
    "allow": []
  }
}
"""

# ── .cortex/rules/ ──────────────────────────────────────────────────

RULE_TESTING_MD = """\
---
paths:
  - "**/*test*.py"
  - "tests/**/*.py"
---

# Testing Rules

- Use descriptive test names: "test_should_[expected]_when_[condition]"
- Mock external dependencies, not internal modules
- Clean up side effects in teardown
- Every agent contract must ship with at least 3 test cases:
  happy path, empty/null input, and permission-denied path
- Mocks must be explicit — never rely on real external services in unit tests
- Test file naming: test_<module>_<feature>.py
"""

RULE_AGENTS_MD = """\
---
paths:
  - "cortex/agents/**/*.py"
  - "**/agents/**/*.py"
---

# Agent Design Rules

- Agents are stateless — all state lives in the PageIndex
- Every agent must implement the BaseAgent.execute() contract
- Input: AgentInput (intent, relevant_pages, permissions, constraints, active_skills)
- Output: AgentOutput (result, pages_to_add, confidence, quality, mistakes)
- Never pass raw user input directly to a sub-agent without sanitizing
- New capabilities = new agents, not edited existing ones (Open-Closed Principle)
- Any agent can be swapped for another of the same type (Liskov Substitution)
"""

RULE_API_MD = """\
---
paths:
  - "src/api/**/*.py"
  - "**/api/**/*.py"
---

# API Design Rules

- All endpoints must validate input with Pydantic schemas
- Return shape: { "data": T } | { "error": str }
- Rate limit all public endpoints
- Log all requests with correlation IDs
"""

# ── .cortex/skills/ ─────────────────────────────────────────────────

SKILL_SECURITY_REVIEW_MD = """\
---
description: Reviews code changes for security vulnerabilities, authentication gaps, and injection risks
disable-model-invocation: true
argument-hint: <branch-or-path>
---

## Diff to review

!`git diff $ARGUMENTS`

Audit the changes above for:

1. Injection vulnerabilities (SQL, XSS, command injection)
2. Authentication and authorization gaps
3. Hardcoded secrets or credentials
4. Permission boundary violations

Use checklist.md in this skill directory for the full review checklist.

Report findings with severity ratings and remediation steps.
"""

SKILL_SECURITY_CHECKLIST_MD = """\
# Security Review Checklist

## Input Validation
- [ ] All user input sanitized before DB queries
- [ ] File upload MIME types validated
- [ ] Path traversal prevented on file operations

## Authentication
- [ ] JWT tokens expire after 24 hours
- [ ] API keys stored in environment variables
- [ ] Passwords hashed with bcrypt or argon2

## Agent Security
- [ ] Intent is sanitized before passing to sub-agents
- [ ] Permission resolver checks all agent actions
- [ ] Destructive actions require explicit user confirmation
- [ ] Mistake records logged for all permission violations
"""

SKILL_PLAN_REVIEW_MD = """\
---
description: Reviews an execution plan for completeness, confidence, and parallelizability
argument-hint: <plan-description>
---

Review the following plan for:

1. **Completeness** — are all steps needed to fulfill the intent present?
2. **Dependencies** — are depends_on edges correct? Can any steps run in parallel?
3. **Confidence** — flag any step with confidence < 0.6
4. **Agent assignment** — is each step assigned to the right agent?
5. **Testability** — can each step's output be verified?

Plan: $ARGUMENTS
"""

# ── .cortex/commands/ ────────────────────────────────────────────────

COMMAND_FIX_ISSUE_MD = """\
---
argument-hint: <issue-number>
---

!`gh issue view $ARGUMENTS`

Investigate and fix the issue above.

1. Trace the bug to its root cause
2. Implement the fix
3. Write or update tests
4. Summarize what you changed and why
"""

COMMAND_STATUS_MD = """\
---
argument-hint:
---

Show the current Cortex engine status:

1. List all registered agents and their health
2. Show loaded context pages and token usage
3. Display the mistake log (if any)
4. Show current permissions
5. Report RAG index status
"""

# ── .cortex/agents/ ─────────────────────────────────────────────────

AGENT_CODE_REVIEWER_MD = """\
---
name: code-reviewer
description: Reviews code for correctness, security, and maintainability
tools: Read, Grep, Glob
---

You are a senior code reviewer working within the Cortex agent framework.
Review for:

1. Correctness: logic errors, edge cases, null handling
2. Security: injection, auth bypass, data exposure
3. Maintainability: naming, complexity, duplication
4. SOLID violations: single responsibility, interface segregation, dependency inversion

Every finding must include a concrete fix.
Tag each finding with quality: ok | warning | error.
"""

AGENT_PLANNER_MD = """\
---
name: plan-architect
description: Decomposes complex intents into dependency-ordered execution plans
tools: Read, Grep, Glob, Bash
---

You are the Cortex Plan Architect. Given a user intent:

1. Decompose into atomic, dependency-ordered steps
2. Identify parallelizable steps (mark parallel: true)
3. Assign each step to the correct domain agent
4. Estimate confidence per step (0.0-1.0)
5. If any step confidence < 0.6, flag for clarification

Output a structured plan as JSON.
"""

# ── .cortex/output-styles/ ──────────────────────────────────────────

OUTPUT_STYLE_TEACHING_MD = """\
---
description: Explains reasoning and asks you to implement small pieces
keep-coding-instructions: true
---

After completing each task, add a brief "Why this approach" note
explaining the key design decision.

When a change is under 10 lines, ask the user to implement it
themselves by leaving a TODO(human) marker instead of writing it.
"""

OUTPUT_STYLE_CONCISE_MD = """\
---
description: Minimal output, no explanations unless asked
keep-coding-instructions: true
---

Be extremely concise. No preamble, no summary, no explanation.
Just code and results. If the user wants reasoning, they will ask.
"""

# ── .cortex/agent-memory/ ───────────────────────────────────────────

AGENT_MEMORY_INDEX_MD = """\
# Agent Memory Index

This directory stores persistent memory for subagents.
Each agent with `memory: project` gets a subdirectory here.

Cortex agents write and maintain these files automatically.
You do not need to edit them, but you can.
"""

# ── .cortex/.gitignore ──────────────────────────────────────────────

DOTCORTEX_GITIGNORE = """\
# Cortex local-only files (not committed)
settings.local.json
agent-memory-local/
*.log
"""

# ── Root .gitignore additions ────────────────────────────────────────

ROOT_GITIGNORE_ADDITIONS = """\

# Cortex local files
.cortex/settings.local.json
.cortex/agent-memory-local/
"""
