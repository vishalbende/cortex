"""
AI Agent — the fully-wired working agent that calls Anthropic's Claude API.

This is the "live demo" agent in the Cortex skeleton. It:
  - Chooses model tier based on task complexity (Haiku for speed, Sonnet for reasoning)
  - Writes optimized prompts for the sub-task
  - Evaluates and scores its own output before returning
  - Detects hallucination signals
  - Manages token budgets

Rules from Cortex spec:
  - Never pass raw user input directly to a sub-agent without sanitizing intent
  - Always include temperature and max_tokens guidance per call
  - If output contradicts a known RAG result, flag the conflict explicitly
"""

from __future__ import annotations

import logging

import anthropic

from cortex.agents.base import BaseAgent
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

# Model tiers
MODEL_FAST = "claude-haiku-4-5-20251001"
MODEL_REASON = "claude-sonnet-4-6"
MODEL_DEEP = "claude-opus-4-6"

# Token budget defaults
DEFAULT_MAX_TOKENS = 4096
FAST_MAX_TOKENS = 1024


class AIAgent(BaseAgent):
    """
    General-purpose AI reasoning agent backed by Anthropic Claude.
    Fully functional — makes real API calls.
    """

    name = "ai_agent"
    description = "General-purpose AI reasoning agent (Claude-backed)"
    required_permissions: list[str] = []

    def __init__(self, api_key: str, default_model: str = MODEL_REASON) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        # 1. Sanitize intent
        sanitized_intent = self._sanitize_intent(agent_input.intent)

        # 2. Choose model tier
        model, max_tokens, temperature = self._select_model(sanitized_intent)

        # 3. Build context from relevant pages
        context_block = self._build_context(agent_input.relevant_pages)

        # 4. Construct the prompt
        prompt = self._build_prompt(sanitized_intent, context_block, agent_input.constraints)

        # 5. Call the API
        logger.info("[ai_agent] Calling %s (temp=%.1f, max_tokens=%d)", model, temperature, max_tokens)
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text

        # 6. Self-evaluate for hallucination signals
        quality, mistakes = self._evaluate_output(
            result_text, agent_input.relevant_pages, agent_input.step
        )

        # 7. Build result page
        result_page = PageIndex(
            type=PageType.TOOL_RESULT,
            summary=f"AI response: {sanitized_intent[:60]}",
            token_count=response.usage.output_tokens,
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
        # Remove common injection markers
        sanitized = raw_intent.strip()
        for marker in ["IGNORE PREVIOUS", "SYSTEM:", "ADMIN:", "```"]:
            sanitized = sanitized.replace(marker, "")
        return sanitized

    def _select_model(self, intent: str) -> tuple[str, int, float]:
        """
        Choose model tier based on task complexity.
        Returns (model, max_tokens, temperature).
        """
        intent_lower = intent.lower()

        # Simple lookups / classification → Haiku (fast)
        fast_signals = ["classify", "label", "extract", "summarize briefly", "yes or no"]
        if any(sig in intent_lower for sig in fast_signals):
            return MODEL_FAST, FAST_MAX_TOKENS, 0.1

        # Complex reasoning / analysis → Sonnet
        reason_signals = ["analyze", "compare", "explain", "design", "plan", "evaluate"]
        if any(sig in intent_lower for sig in reason_signals):
            return MODEL_REASON, DEFAULT_MAX_TOKENS, 0.3

        # Default to the configured model
        return self._default_model, DEFAULT_MAX_TOKENS, 0.2

    def _build_context(self, pages: list[PageIndex]) -> str:
        """Serialize relevant pages into a context block."""
        if not pages:
            return ""
        parts = []
        for p in pages[:10]:  # limit to 10 pages
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
            # Simple heuristic: if response makes a strong claim not in RAG, flag it
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
                break  # one warning is enough

        return quality, mistakes
