"""
Planner — decomposes user intents into dependency-ordered execution plans
and drives step-by-step execution through the agent registry.

Planning rules (from Cortex spec):
  - Decompose every intent into atomic, dependency-ordered steps
  - Identify parallelizable steps and mark them with parallel=True
  - Assign each step to a specific agent with expected inputs/outputs
  - Estimate confidence per step (0.0–1.0) before executing
  - If confidence of any step < 0.6, pause and clarify with the user
  - Re-plan on failure, not just retry
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

import anthropic

from cortex.models import (
    AgentInput,
    AgentOutput,
    MistakeRecord,
    MistakeType,
    PageIndex,
    PageType,
    Plan,
    PlanStep,
    Quality,
    StepStatus,
)
from cortex.agents.registry import AgentRegistry
from cortex.context.page_store import PageStore
from cortex.mistakes.tracker import MistakeTracker
from cortex.permissions.resolver import PermissionResolver

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6

# System prompt for LLM-based plan decomposition
DECOMPOSITION_PROMPT = """\
You are the Cortex Planner. Given a user intent and a list of available agents,
decompose the intent into atomic, dependency-ordered steps.

Available agents: {agents}

Rules:
- Each step must have: id, agent, action, depends_on, parallel, confidence, inputs, expected_output
- Mark steps that CAN run concurrently as parallel: true
- Estimate your confidence (0.0-1.0) for each step
- If a step requires output from a previous step, list it in depends_on
- Return valid JSON array of steps only, no markdown fences

User intent: {intent}
"""


class Planner:
    """
    Core planner that decomposes intents into Plans and executes them
    step-by-step through registered agents.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        page_store: PageStore,
        mistake_tracker: MistakeTracker,
        permissions: PermissionResolver,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.registry = registry
        self.page_store = page_store
        self.mistakes = mistake_tracker
        self.permissions = permissions
        self.model = model
        self._client: anthropic.AsyncAnthropic | None = None
        if api_key:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)

        # Callbacks the TUI / engine can hook into
        self.on_step_start: Callable[[PlanStep], Awaitable[None]] | None = None
        self.on_step_done: Callable[[PlanStep, AgentOutput], Awaitable[None]] | None = None
        self.on_replan: Callable[[Plan], Awaitable[None]] | None = None

    # ── Plan decomposition ───────────────────────────────────────────

    async def decompose(self, intent: str) -> Plan:
        """
        Turn a user intent into a structured Plan.
        Uses LLM if a client is available; otherwise uses a single-step fallback.
        """
        if self._client:
            return await self._decompose_with_llm(intent)
        return self._decompose_fallback(intent)

    async def _decompose_with_llm(self, intent: str) -> Plan:
        """Use Claude to decompose the intent into steps."""
        agents_desc = ", ".join(
            f"{name}" for name in self.registry.list_agents()
        )
        prompt = DECOMPOSITION_PROMPT.format(agents=agents_desc, intent=intent)

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            raw_steps = json.loads(text)
            steps = []
            for i, raw in enumerate(raw_steps):
                step = PlanStep(
                    id=raw.get("id", f"step_{i+1}"),
                    agent=raw.get("agent", ""),
                    action=raw.get("action", ""),
                    depends_on=raw.get("depends_on", []),
                    parallel=raw.get("parallel", False),
                    confidence=raw.get("confidence", 0.9),
                    inputs=raw.get("inputs", {}),
                    expected_output=raw.get("expected_output", ""),
                )
                steps.append(step)

            plan = Plan(intent=intent, steps=steps)
            logger.info("Decomposed intent into %d steps", len(steps))
            return plan

        except Exception as e:
            logger.warning("LLM decomposition failed: %s. Using fallback.", e)
            return self._decompose_fallback(intent)

    def _decompose_fallback(self, intent: str) -> Plan:
        """Fallback: single step assigned to the first available agent."""
        agents = self.registry.list_agents()
        agent_name = agents[0] if agents else "ai_agent"
        step = PlanStep(
            id="step_1",
            agent=agent_name,
            action=intent,
            confidence=0.7,
            expected_output="Direct response to user intent.",
        )
        return Plan(intent=intent, steps=[step])

    # ── Plan execution ───────────────────────────────────────────────

    async def execute(self, plan: Plan, max_retries: int = 1) -> list[AgentOutput]:
        """
        Execute a plan step-by-step, respecting dependencies.
        Parallelizable steps with all deps met run concurrently.
        Re-plans on failure instead of blind retry.
        """
        outputs: list[AgentOutput] = []
        retries = 0

        # Store plan as a context page
        plan_page = PageIndex(
            id=f"page:plan:{plan.plan_id}",
            type=PageType.PLAN,
            summary=f"Plan: {plan.intent[:60]}",
            token_count=200,
            tags=["system:plan"],
            content=json.dumps(plan.to_dict(), indent=2),
        )
        self.page_store.add(plan_page)

        while not plan.is_complete():
            runnable = plan.runnable_steps()
            if not runnable:
                if plan.has_failures() and retries < max_retries:
                    retries += 1
                    logger.info("Re-planning after failure (attempt %d)", retries)
                    plan = await self._replan(plan)
                    if self.on_replan:
                        await self.on_replan(plan)
                    continue
                else:
                    logger.error("No runnable steps and no retries left. Aborting.")
                    break

            # Check confidence — pause if any step is below threshold
            low_conf = [s for s in runnable if s.confidence < CONFIDENCE_THRESHOLD]
            if low_conf:
                for s in low_conf:
                    self.mistakes.record(MistakeRecord(
                        step_id=s.id,
                        type=MistakeType.LOW_CONFIDENCE,
                        description=f"Step '{s.action}' has low confidence ({s.confidence:.2f})",
                        correction="Flagged for user clarification.",
                        learned="Pause on low-confidence steps before executing.",
                    ))
                    s.status = StepStatus.SKIPPED

                runnable = [s for s in runnable if s not in low_conf]
                if not runnable:
                    continue

            # Execute runnable steps (parallel where possible)
            parallel_group = [s for s in runnable if s.parallel]
            sequential = [s for s in runnable if not s.parallel]

            if parallel_group:
                results = await asyncio.gather(
                    *[self._execute_step(s, plan) for s in parallel_group]
                )
                outputs.extend(results)

            for step in sequential:
                result = await self._execute_step(step, plan)
                outputs.append(result)

        # Sync mistake log to page store
        if self.mistakes.count() > 0:
            self.page_store.add(self.mistakes.to_page())

        return outputs

    async def _execute_step(self, step: PlanStep, plan: Plan) -> AgentOutput:
        """Execute a single plan step through its assigned agent."""
        step.status = StepStatus.RUNNING
        if self.on_step_start:
            await self.on_step_start(step)

        agent = self.registry.get(step.agent)
        if not agent:
            step.status = StepStatus.FAILED
            step.quality = Quality.ERROR
            mistake = MistakeRecord(
                step_id=step.id,
                type=MistakeType.PLAN_ERROR,
                description=f"Agent '{step.agent}' not found in registry.",
                correction="Step failed. Re-plan needed.",
                learned=f"Verify agent '{step.agent}' is registered before planning.",
            )
            self.mistakes.record(mistake)
            return AgentOutput(quality=Quality.ERROR, mistakes=[mistake])

        # Permission check
        if agent.required_permissions:
            if not self.permissions.check_agent(agent.name, agent.required_permissions, step.id):
                step.status = StepStatus.FAILED
                step.quality = Quality.ERROR
                denied = self.permissions.denied[-1] if self.permissions.denied else None
                if denied:
                    self.mistakes.record(denied)
                return AgentOutput(
                    quality=Quality.ERROR,
                    mistakes=self.permissions.denied[-1:] if self.permissions.denied else [],
                )

        # Build agent input
        context_pages = self.page_store.select_for_turn(
            intent_tags=[f"intent:{step.action[:20]}"]
        )
        agent_input = AgentInput(
            intent=step.action,
            relevant_pages=context_pages,
            permissions=self.permissions.granted_permissions,
            constraints=step.inputs,
            active_skills=[],
            step=step,
        )

        # Run
        output = await agent.run(agent_input)

        # Process output
        step.result = output.result
        step.quality = output.quality

        if output.quality == Quality.ERROR:
            step.status = StepStatus.FAILED
        else:
            step.status = StepStatus.DONE

        # Record mistakes
        for m in output.mistakes:
            self.mistakes.record(m)

        # Add new pages
        for page in output.pages_to_add:
            self.page_store.add(page)

        if self.on_step_done:
            await self.on_step_done(step, output)

        return output

    async def _replan(self, failed_plan: Plan) -> Plan:
        """
        Re-plan after a failure: skip done steps, rebuild around the failure.
        """
        remaining_intent = (
            f"Continue: {failed_plan.intent}. "
            f"Completed: {[s.id for s in failed_plan.steps if s.status == StepStatus.DONE]}. "
            f"Failed: {[s.id for s in failed_plan.steps if s.status == StepStatus.FAILED]}."
        )
        new_plan = await self.decompose(remaining_intent)
        # Carry over completed steps
        for old_step in failed_plan.steps:
            if old_step.status == StepStatus.DONE:
                old_step_copy = PlanStep(
                    id=old_step.id,
                    agent=old_step.agent,
                    action=old_step.action,
                    status=StepStatus.DONE,
                    quality=old_step.quality,
                    result=old_step.result,
                )
                new_plan.steps.insert(0, old_step_copy)
        return new_plan
