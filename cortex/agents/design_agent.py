"""
Design Agent — Figma-aware UI/UX reasoning agent.

Produces structured component specs, design tokens, layout descriptions,
and accessibility critiques. Output follows Figma's Auto Layout / Frame /
Component model.

This is a skeleton agent — swap in real LLM calls by extending AIAgent
or injecting an Anthropic client.
"""

from __future__ import annotations

import json
import logging

from cortex.agents.base import BaseAgent
from cortex.models import (
    AgentInput,
    AgentOutput,
    PageIndex,
    PageType,
    Quality,
)

logger = logging.getLogger(__name__)


class DesignAgent(BaseAgent):
    """Figma-aware design reasoning agent."""

    name = "design_agent"
    description = "UI/UX design agent — produces Figma-compatible specs and tokens"
    required_permissions: list[str] = ["design:read", "design:write"]

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        intent = agent_input.intent
        logger.info("[design_agent] Processing: %s", intent[:80])

        # Generate a structured component spec (skeleton output)
        spec = {
            "component": self._extract_component_name(intent),
            "variants": ["default", "hover", "disabled"],
            "tokens": {
                "color": {
                    "primary": "#6366F1",
                    "secondary": "#A5B4FC",
                    "background": "#0F172A",
                    "text": "#F8FAFC",
                },
                "spacing": {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32},
                "typography": {
                    "heading": {"family": "Inter", "size": 24, "weight": 700},
                    "body": {"family": "Inter", "size": 14, "weight": 400},
                },
            },
            "layout": "auto-layout",
            "accessibility": {
                "contrast_ratio": "7.2:1 (AAA)",
                "aria_role": "button",
                "keyboard_nav": True,
            },
            "figma_notes": (
                f"Component generated from intent: '{intent[:60]}'. "
                "Uses Auto Layout with 16px padding. All variants share the same "
                "token set. Ensure contrast ratio meets WCAG AA minimum (4.5:1)."
            ),
        }

        result_text = json.dumps(spec, indent=2)

        result_page = PageIndex(
            type=PageType.TOOL_RESULT,
            summary=f"Design spec: {spec['component']}",
            token_count=len(result_text) // 4,
            tags=["agent:design", f"component:{spec['component']}"],
            content=result_text,
        )

        return AgentOutput(
            result=spec,
            pages_to_add=[result_page],
            confidence=0.85,
            quality=Quality.OK,
        )

    @staticmethod
    def _extract_component_name(intent: str) -> str:
        """Best-effort extraction of the component name from the intent."""
        keywords = ["button", "card", "modal", "nav", "header", "footer",
                     "sidebar", "form", "input", "table", "list", "badge",
                     "tooltip", "dropdown", "menu", "tabs", "dialog"]
        intent_lower = intent.lower()
        for kw in keywords:
            if kw in intent_lower:
                return kw.capitalize()
        return "Component"
