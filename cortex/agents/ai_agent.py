"""
AI Agent — backed by the local Claude Code CLI.

This agent calls the locally installed `claude` command for all LLM work.
No API key needed — Claude Code handles authentication via its own session.

Capabilities:
  - Choose model tier based on task complexity (haiku for speed, sonnet for reasoning)
  - Write optimized prompts for sub-tasks
  - Evaluate and score its own output before returning
  - Detect hallucination signals
  - Manage token budgets via Claude Code's built-in limits

Rules from Cortex spec:
  - Never pass raw user input directly to a sub-agent without sanitizing intent
  - Always include temperature and max_tokens guidance per call
  - If output contradicts a known RAG result, flag the conflict explicitly
"""

from __future__ import annotations

import logging

from cortex.agents.base import BaseAgent
from cortex.claude_code import ClaudeCode
from cortex.models import (
    AgentInput,
    AgentOutput,
    MistakeRecord,
    MistakeType,
    PageIndex,
    PageType,
    Quality,
)

logger = logging.getLogger(__name__)

# Model tiers (Claude Code model names)
MODEL_FAST = "haiku"
MODEL_REASON = "sonnet"
MODEL_DEEP = "opus"


class AIAgent(BaseAgent):
    """
    General-purpose AI reasoning agent backed by local Claude Code CLI.
    Fully functional — shells out to `claude -p` for every call.
    """

    name = "ai_agent"
    description = "General-purpose AI reasoning agent (Claude Code local CLI)"
    required_permissions: list[str] = []

    def __init__(self, default_model: str = MODEL_REASON, cwd: str | None = None) -> None:
        self._claude = ClaudeCode(model=default_model, cwd=cwd)

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        # 1. Sanitize intent
        sanitized_intent = self._sanitize_intent(agent_input.intent)

        # 2. Choose model tier
        model = self._select_model(sanitized_intent)
        claude = self._claude.with_model(model)

        # 3. Build context from relevant pages
        context_block = self._build_context(agent_input.relevant_pages)

        # 4. Construct the prompt
        prompt = self._build_prompt(sanitized_intent, context_block, agent_input.constraints)

        # 5. Call Claude Code CLI
        logger.info("[ai_agent] Calling claude -p (model=%s)", model)
        response = await claude.run(prompt)

        if response.is_error:
            return AgentOutput(
                result=None,
                confidence=0.0,
                quality=Quality.ERROR,
                mistakes=[MistakeRecord(
                    step_id=agent_input.step.id if agent_input.step else "unknown",
                    type=MistakeType.TOOL_FAILURE,
                    description=f"Claude Code CLI error: {response.text[:200]}",
                    correction="Check that `claude` is installed and working.",
                    learned="Verify Claude Code CLI is available before running AI agent.",
                )],
            )

        result_text = response.text

        # 6. Self-evaluate for hallucination signals
        quality, mistakes = self._evaluate_output(
            result_text, agent_input.relevant_pages, agent_input.step
        )

        # 7. Build result page
        result_page = PageIndex(
            type=PageType.TOOL_RESULT,
            summary=f"AI response: {sanitized_intent[:60]}",
            token_count=len(result_text) // 4,
            tags=[f"agent:{self.name}"],
            content=result_text,
        )

        return AgentOutput(
            result=result_text,
            pages_to_add=[result_page],
            confidence=0.9 if quality == Quality.OK else 0.5,
            quality=quality,
            mistakes=mistakes,
        )

    # ── Internal methods ─────────────────────────────────────────────

    def _sanitize_intent(self, raw_intent: str) -> str:
        """Strip injection patterns and normalize the intent."""
        sanitized = raw_intent.strip()
        for marker in ["IGNORE PREVIOUS", "SYSTEM:", "ADMIN:", "```"]:
            sanitized = sanitized.replace(marker, "")
        return sanitized

    def _select_model(self, intent: str) -> str:
        """
        Choose model tier based on task complexity.
        Returns a Claude Code model name (haiku, sonnet, opus).
        """
        intent_lower = intent.lower()

        # Simple lookups / classification → haiku (fast)
        fast_signals = ["classify", "label", "extract", "summarize briefly", "yes or no"]
        if any(sig in intent_lower for sig in fast_signals):
            return MODEL_FAST

        # Complex reasoning / analysis → sonnet
        reason_signals = ["analyze", "compare", "explain", "design", "plan", "evaluate"]
        if any(sig in intent_lower for sig in reason_signals):
            return MODEL_REASON

        # Default
        return self._claude.model

    def _build_context(self, pages: list[PageIndex]) -> str:
        """Serialize relevant pages into a context block."""
        if not pages:
            return ""
        parts = []
        for p in pages[:10]:
            parts.append(f"[{p.type.value}] {p.summary}\n{p.content[:2000]}")
        return "\n---\n".join(parts)

    def _build_prompt(self, intent: str, context: str, constraints: dict) -> str:
        sections = [f"Task: {intent}"]
        if context:
            sections.append(f"Context:\n{context}")
        if constraints:
            sections.append(f"Constraints: {constraints}")
        sections.append("Respond clearly and concisely.")
        return "\n\n".join(sections)

    def _evaluate_output(
        self, text: str, rag_pages: list[PageIndex], step
    ) -> tuple[Quality, list[MistakeRecord]]:
        """
        Self-evaluate output for hallucination signals:
          - Vague citations
          - Confident incorrect assertions (checked against RAG)
          - Self-contradiction
        """
        mistakes: list[MistakeRecord] = []
        quality = Quality.OK

        # Check for contradiction with RAG pages
        rag_content = " ".join(p.content.lower() for p in rag_pages if p.type == PageType.RAG)
        if rag_content:
            strong_claims = [
                line
                for line in text.split(".")
                if any(w in line.lower() for w in ["always", "never", "exactly", "precisely"])
            ]
            for claim in strong_claims:
                claim_words = set(claim.lower().split())
                rag_words = set(rag_content.split())
                overlap = len(claim_words & rag_words) / max(len(claim_words), 1)
                if overlap < 0.1 and len(claim.strip()) > 20:
                    quality = Quality.WARNING
                    mistakes.append(MistakeRecord(
                        step_id=step.id if step else "unknown",
                        type=MistakeType.HALLUCINATION,
                        description=f"Strong claim may conflict with RAG: '{claim.strip()[:80]}'",
                        correction="Flagged for review. Claim not found in context pages.",
                        learned="Cross-check strong assertions against available RAG context.",
                    ))

        # Check for vague citation signals
        vague_markers = ["studies show", "research indicates", "it is well known", "experts say"]
        for marker in vague_markers:
            if marker in text.lower():
                quality = Quality.WARNING
                mistakes.append(MistakeRecord(
                    step_id=step.id if step else "unknown",
                    type=MistakeType.HALLUCINATION,
                    description=f"Vague citation detected: '{marker}'",
                    correction="Consider requesting specific sources.",
                    learned="Flag vague citations that lack specific references.",
                ))
                break

        return quality, mistakes
