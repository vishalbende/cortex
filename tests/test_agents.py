"""Tests for agent base class, registry, and domain agents."""

import pytest
from cortex.agents.base import BaseAgent
from cortex.agents.registry import AgentRegistry
from cortex.agents.design_agent import DesignAgent
from cortex.agents.test_writer_agent import TestWriterAgent
from cortex.models import AgentInput, AgentOutput, PlanStep, Quality


# ── Concrete test agent ──────────────────────────────────────────────

class EchoAgent(BaseAgent):
    name = "echo_agent"
    description = "Echoes the intent back"

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(result=f"Echo: {agent_input.intent}", quality=Quality.OK)


class FailAgent(BaseAgent):
    name = "fail_agent"
    description = "Always fails"

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        raise RuntimeError("Intentional failure")


# ── Tests ────────────────────────────────────────────────────────────

class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        agent = EchoAgent()
        inp = AgentInput(intent="hello", step=PlanStep(id="s1"))
        out = await agent.run(inp)
        assert out.result == "Echo: hello"
        assert out.quality == Quality.OK

    @pytest.mark.asyncio
    async def test_empty_intent_fails_validation(self):
        agent = EchoAgent()
        inp = AgentInput(intent="", step=PlanStep(id="s1"))
        out = await agent.run(inp)
        assert out.quality == Quality.ERROR
        assert len(out.mistakes) == 1

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        agent = FailAgent()
        inp = AgentInput(intent="do something", step=PlanStep(id="s1"))
        out = await agent.run(inp)
        assert out.quality == Quality.ERROR
        assert out.confidence == 0.0
        assert len(out.mistakes) == 1
        assert "Intentional failure" in out.mistakes[0].description


class TestAgentRegistry:
    def test_register_and_get(self):
        reg = AgentRegistry()
        agent = EchoAgent()
        reg.register(agent)
        assert reg.get("echo_agent") is agent
        assert reg.has("echo_agent")
        assert "echo_agent" in reg.list_agents()

    def test_get_missing(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register(EchoAgent())
        reg.unregister("echo_agent")
        assert not reg.has("echo_agent")


class TestDesignAgent:
    @pytest.mark.asyncio
    async def test_generates_spec(self):
        agent = DesignAgent()
        inp = AgentInput(intent="Design a login button", step=PlanStep(id="s1"))
        out = await agent.run(inp)
        assert out.quality == Quality.OK
        assert isinstance(out.result, dict)
        assert out.result["component"] == "Button"
        assert "tokens" in out.result

    @pytest.mark.asyncio
    async def test_unknown_component(self):
        agent = DesignAgent()
        inp = AgentInput(intent="Design something abstract", step=PlanStep(id="s1"))
        out = await agent.run(inp)
        assert out.result["component"] == "Component"


class TestTestWriterAgent:
    @pytest.mark.asyncio
    async def test_generates_tests(self):
        agent = TestWriterAgent()
        inp = AgentInput(
            intent="Write tests for auth module",
            constraints={"module": "auth", "feature": "login"},
            step=PlanStep(id="s1"),
        )
        out = await agent.run(inp)
        assert out.quality == Quality.OK
        assert "test_happy_path" in out.result
        assert "test_empty_input" in out.result
        assert "test_permission_denied" in out.result
