"""
Cortex — Intelligent Agent Orchestration System
Built on Anthropic's Claude.

Three-layer execution model:
  1. PLANNER       — intent decomposition, plan building, context management
  2. DOMAIN AGENTS — specialized agents registered per capability at runtime
  3. TOOLS / MCP   — permission-gated tools via a central MCP server
"""

__version__ = "0.1.0"
