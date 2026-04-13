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
