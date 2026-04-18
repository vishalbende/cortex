# Context Engine SDK — Full Research Document

## 1. Problem Statement

Teams deploying AI agents in production face an unsolved context orchestration problem. Every LLM call requires assembling: system prompts, tool definitions, memory/entity context, conversation history, and retrieved knowledge — all competing for a finite token budget. Today every team hand-rolls this assembly, resulting in wasted tokens, poor tool selection, ballooning costs, and no visibility into what's actually in the context window.

**The problem in one line:** Nobody jointly optimizes tools + memory + history as a single token budget, and nobody provides observability into context composition.

---

## 2. Foundational References

### 2.1 Manus Blog — Context Engineering for AI Agents (July 2025)
- URL: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- Key findings:
  - **KV-cache hit rate is the single most important metric** for production agents. It directly affects both latency and cost.
  - **Do NOT dynamically add/remove tools mid-iteration.** Tool definitions sit near the front of context; any change invalidates the KV-cache for all subsequent actions.
  - Manus uses a **context-aware state machine** to manage tool availability via **logit masking during decoding** — constraining tool selection without modifying tool definitions.
  - Three modes of function calling: Auto (model may call), Required (must call, unconstrained), Specified (must call from subset).
  - Recommendation: if you allow user-configurable tools (MCP), someone will plug hundreds of tools into your action space, making the agent dumber.

### 2.2 Anthropic — Effective Context Engineering for Agents (September 2025)
- URL: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Key findings:
  - Context engineering = finding the smallest possible set of high-signal tokens that maximize likelihood of desired outcome.
  - System prompts should be at the "right altitude" — not too specific (brittle), not too vague (unreliable).
  - Context engineering is iterative — curation happens each time we decide what to pass to the model.
  - LLMs are constrained by finite attention budget; every token competes.

### 2.3 Google ADK — Context-Aware Multi-Agent Framework (December 2025)
- URL: https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/
- Key findings:
  - **Separate storage from presentation**: "Session" (durable state) vs "Working Context" (per-call view).
  - **Explicit transformations**: Context built through named, ordered processors — not ad-hoc string concatenation. Makes the "compilation" step observable and testable.
  - **Scope by default**: Every model call sees minimum context required. Agents must reach for more explicitly via tools.
  - **Prefix caching optimization**: Divide context into stable prefixes (system instructions, agent identity) and variable suffixes (latest user turn, new tool outputs).
  - **Context Compaction**: When threshold reached, LLM summarizes older events over a sliding window. Prunes raw events that were summarized.
  - **Artifacts**: Large data (CSVs, PDFs, API responses) treated as named, versioned objects in an ArtifactService — not stuffed into chat history.
  - **Static instruction primitive**: Guarantees immutability for system prompts, ensuring cache prefix remains valid across invocations.

### 2.4 LangChain — Context Engineering for Agents (July 2025)
- URL: https://blog.langchain.com/context-engineering-for-agents/
- Key findings:
  - Four patterns: Writing (saving outside window), Selecting (pulling in), Compressing (retaining only needed tokens), Isolating (splitting up).
  - LangGraph Bigtool for semantic search over tool descriptions.
  - Memory selection is challenging — unexpected/undesired retrieval can make users feel the context window "no longer belongs to them."
  - Agents use tools but become overloaded with too many. Tool descriptions often overlap, causing model confusion.

### 2.5 State of Context Engineering in 2026 (March 2026)
- URL: https://www.newsletter.swirlai.com/p/state-of-context-engineering-in-2026
- Key findings:
  - Context routing: classifies query and directs to right context source before anything enters context window.
  - LLM-powered routing (more accurate, adds latency/cost) vs hierarchical routing (lead agent triages to sub-agents).
  - MCP solves the connection problem. The context cost problem remains unsolved.
  - Tool description quality matters: MCP authors write descriptions for humans, not models.

### 2.6 Towards Data Science — Context Engineering Deep Dive (April 2026)
- URL: https://towardsdatascience.com/deep-dive-into-context-engineering-for-ai-agents/
- Key findings:
  - The "harness" = everything around the model deciding how context is assembled. Many "model failures" are actually harness failures.
  - Agent forgot → nothing persisted the right state. Chose wrong tool → harness overloaded the action space.
  - A good harness = deterministic shell wrapped around a stochastic core.
  - Tools need to be well-understood with minimal overlap. Bloated tool sets lead to unclear decision-making.
  - Durable memory should contain things that continue to constrain future reasoning. Storing too much → context pollution, only persistent.
  - Memory without revision is a trap — needs conflict resolution, deletion, demotion.

---

## 3. Research Papers

### 3.1 ITR: Dynamic System Instructions and Tool Exposure (arxiv 2602.17046, Dec 2025)
- **Most directly relevant paper for our product.**
- Proposes Instruction-Tool Retrieval (ITR): per step, retrieves only minimal system-prompt fragments and smallest necessary subset of tools.
- Results: 95% reduction in per-step context tokens, 32% improvement in correct tool routing, 70% cost reduction vs monolithic baseline.
- Enables agents to run 2-20x more loops within context limits.
- Savings compound with number of agent steps — particularly valuable for long-running autonomous agents.

### 3.2 AutoTool: Dynamic Tool Selection and Integration (arxiv 2512.13278, Dec 2025)
- Two-phase approach using RL to train models for dynamic tool selection.
- Uses Plackett-Luce Ranking to frame tool selection as a sequence ranking problem.
- Results: 6.4% gain in math/science, 4.5% in search QA, 7.7% in code gen, 6.9% in multimodal.
- Addresses realistic scenario where agent must select from domain-diverse toolset and handle unseen tools at inference.

### 3.3 AutoTool: Efficient Tool Selection (arxiv 2511.14650, Nov 2025)
- Graph-based framework exploiting "tool usage inertia" — tendency of tool invocations to follow predictable sequential patterns.
- Constructs directed graph from historical trajectories (nodes=tools, edges=transition probabilities).
- Reduces inference costs by up to 30% while maintaining competitive completion rates.
- Key insight: tool selection patterns are statistically predictable.

### 3.4 Semantic Tool Discovery for MCP (arxiv 2603.20313, March 2026)
- Vector-based approach to MCP tool selection using dense embeddings.
- Results: 99.6% reduction in tool-related token consumption, 97.1% hit rate at K=3, MRR of 0.91.
- Tested across 140 queries, 121 tools from 5 MCP servers.
- Sub-100ms retrieval latency across all configurations.

### 3.5 AdaptOrch: Task-Adaptive Multi-Agent Orchestration (arxiv 2602.16873, Feb 2026)
- When model capabilities converge, the dominant optimization variable becomes structural topology of agent coordination.
- Static frameworks (MCP, LangGraph, CrewAI) define fixed topologies regardless of task demands.
- Topology selection problem remains unsolved at algorithmic level.

### 3.6 From Static Templates to Dynamic Runtime Graphs (arxiv 2603.22386, March 2026)
- Survey of workflow optimization for LLM agents as agentic computation graphs (ACGs).
- Distinguishes static methods (fixed scaffold before deployment) from dynamic methods (select/generate/revise per run).
- Three dimensions: when structure determined, what optimized, which evaluation signals guide optimization.

### 3.7 PASTE: Act While Thinking (arxiv 2603.18897, March 2026)
- Speculative tool execution to reduce agent latency.
- Predicting next tool is highly challenging — unlike CPU branch prediction, agent requests are diverse and unstructured.
- Tool initialization typically <20% of overall latency; the real bottleneck is elsewhere.

### 3.8 DICE: Dynamic In-Context Example Selection (arxiv 2507.23554, July 2025)
- Framework for dynamically selecting relevant demonstrations at each reasoning step.
- Decomposes demonstration knowledge into transferable and non-transferable components.
- Framework-agnostic, plug-in module for existing agentic frameworks.

---

## 4. Competitive Landscape

### 4.1 Tool Selection / Routing

| Solution | What It Does | Strengths | Weaknesses |
|----------|-------------|-----------|------------|
| **LangGraph BigTool** | Semantic search over tool descriptions using embeddings in InMemoryStore | Open source, simple API, supports custom retrieval functions | Locked to LangGraph. No intent classification, no budget awareness, no domain grouping. Simple embedding similarity only. |
| **Manus state machine** | Logit masking to constrain tool selection without modifying definitions. Preserves KV-cache. | Most performant approach proven in production. Cache-friendly. | Proprietary, requires self-hosted inference, not available as product. |
| **ITR (paper)** | Per-step RAG retrieval of minimal system prompt + tool subset | 95% token reduction, 32% better routing, 70% cost cut | Research paper only, no production implementation. |
| **AutoTool (paper)** | Graph-based tool transition prediction from historical trajectories | 30% cost reduction | Requires trajectory data, research-stage. |
| **MCP tool routing (paper)** | Vector-based MCP tool selection | 99.6% token reduction, 97.1% hit rate, sub-100ms | Research paper, narrow scope (MCP-only). |

### 4.2 Memory Systems

| Solution | Approach | Strengths | Weaknesses |
|----------|---------|-----------|------------|
| **Mem0** | Managed memory API, graph at $249/mo Pro tier | Drop-in, cloud-hosted, largest ecosystem | 49% on LongMemEval benchmark. Graph features gated behind expensive Pro tier. No tool output ingestion. |
| **Zep / Graphiti** | Temporal knowledge graph | Tracks how facts change over time. Hybrid vector+graph. | Requires Neo4j. Cloud-only for advanced features. Heavy infrastructure. |
| **Letta (MemGPT)** | Full agent runtime with self-editing memory. OS-inspired tiers: core (RAM), archival (disk), recall (history). | Agents control their own memory. Genuinely innovative architecture. | Full runtime lock-in — not a drop-in memory layer. Python-only SDK. Memory quality depends on model's judgment. |
| **OMEGA** | Local-first, SQLite + ONNX embeddings | 95.4% LongMemEval, zero cloud dependency, AES-256 encryption | Early stage, small community, limited production validation. |
| **LangMem** | Memory within LangGraph. Episodic, semantic, procedural. | Native LangGraph integration | Requires LangGraph's StateGraph. Framework-locked. |
| **Cognee** | Knowledge graph from raw docs | Turns unstructured data into structured graph | More RAG preprocessing than runtime memory. |
| **Kronvex** | Context injection API with semantic compression | Conversations distilled to facts/preferences, not raw dialogue | Early stage, limited features. |

### 4.3 Observability / Telemetry

| Solution | What It Does | Gap |
|----------|-------------|-----|
| **Langfuse** | Open-source tracing, cost tracking, prompt versioning, OpenTelemetry-native | Traces LLM calls, not context composition. Doesn't analyze what's in context or whether tools/memory were useful. |
| **Braintrust** | Granular cost analytics per request/user/feature. Evaluation-first. | Evaluation-focused, not context-composition-aware. |
| **LangSmith** | Trace + eval platform for LangChain | LangChain ecosystem lock-in. |
| **Helicone** | Proxy-based observability. Token monitoring, cost analytics across providers. | Request-level visibility, not agent decision analysis. |
| **Arize / Phoenix** | ML observability, embedding drift detection | Focused on ML models, not agent context specifically. |
| **Maxim AI** | End-to-end simulation + evaluation + monitoring | Full platform, not context-specific. |
| **Galileo** | AI reliability with Luna-2 SLM evaluators. Sub-200ms. | Safety/evaluation focused, not context composition. |
| **OpenTelemetry GenAI SIG** | Standard gen_ai.* attribute names for LLM instrumentation | Standard/spec, not a product. Doesn't track context composition. |
| **Datadog** | Infrastructure + application + LLM signal correlation | System-wide, not agent reasoning. 900+ integrations but generic. |

### 4.4 Context Engineering Frameworks / Repos

| Solution | What It Is | Status |
|----------|-----------|--------|
| **contextenginehq/context-engine** (GitHub) | Open-source, Rust-based, deterministic token-aware context selection. MCP server exposure. | Early stage. Focused on document context, not tool/memory orchestration. |
| **Context-Engine-AI/Context-Engine** (GitHub) | MCP-based agentic context compression. Search routing, symbol graph, batch queries, memory store. | Focused on code/repo context, not general agent context. |
| **kayba-ai/agentic-context-engine** (GitHub) | Based on ACE paper (Stanford/SambaNova). Learn from experience traces. PydanticAI-backed. | Learning/adaptation focus, not runtime context assembly. |
| **Denis2054 Context-Engine** (GitHub) | Multi-agent system with token analytics, memory, semantic blueprints. Book companion. | Educational, not production SDK. |
| **Agent-Skills-for-Context-Engineering** (GitHub) | Skill collection for context engineering principles across platforms. | Knowledge base, not runnable software. |
| **OpenDev** | Runtime system prompt assembly with conditional loading — sections factored into independent markdown files with condition predicates and priorities. | Coding agent framework, not general-purpose. |
| **Late** | Deterministic coding agent. Lead architect + ephemeral subagents with fresh context. Zero prompt bloat. | Coding-only. License restricts commercial infrastructure use. |
| **mcp-agent (lastmile-ai)** (GitHub) | MCP workflow composition with token counter, routers, orchestrators, Temporal-backed durability. | MCP orchestration framework, not context optimization product. |

### 4.5 Agent Frameworks with Context Features

| Framework | Context Features | Gap |
|-----------|-----------------|-----|
| **Google ADK** | Session/Working Context separation, context compaction, artifact handling, prefix caching, static instruction primitive | Google-ecosystem, framework not standalone product |
| **OpenAI Agents SDK** | RunContextWrapper with usage tracking, tool approval/rejection, session management | Basic, OpenAI-only |
| **LangGraph** | Bigtool for tool selection, LangMem for memory, StateGraph for flow | Framework lock-in, not a middleware layer |

---

## 5. Genuine Gaps — What Nobody Has Built

### Gap 1: Unified Context Budget Management
Nobody jointly optimizes tools + memory + history as a single token budget. Every solution handles one slice — BigTool does tools, Mem0 does memory, Langfuse does tracing. No product manages the combined token budget as a single optimization problem, making tradeoffs like "this turn needs 5 tools and minimal memory" vs "this turn needs deep memory and only 2 tools."

### Gap 2: Tool Output → Memory Writeback
Zero products automatically extract facts from tool results and update entity memory. Tool calls to Salesforce, Stripe, etc. return structured data that should update the memory layer's entity facts. Currently 100% hand-wired per tool by every team.

### Gap 3: Context Composition Observability
Langfuse/Braintrust track "how many tokens did this call use." Nobody tracks "what was IN those tokens" — which tools were loaded, which memory was included, which was actually referenced by the model, and which was wasted. The composition of the context window is completely invisible.

### Gap 4: Framework-Agnostic Context Middleware
Every solution is framework-locked — BigTool needs LangGraph, ADK needs Google, Letta is its own runtime. A framework-agnostic SDK that works with raw Anthropic/OpenAI SDKs doesn't exist. The "Switzerland of context orchestration."

### Gap 5: KV-Cache-Aware Context Assembly
Manus proved that dynamically changing tool definitions invalidates KV-cache, destroying performance. Nobody has built tooling that assembles context to maximize cache hits — keeping stable prefixes static and only varying the suffix. The ITR paper addresses this conceptually but isn't productized.

### Gap 6: Permission-Scoped Memory Views
Memory systems offer "shared" memory but it's all-or-nothing. Nobody does "support agent can see complaint history but not margin data, while sales agent sees margins but not internal escalation notes." Permission-scoped views over the same memory store don't exist as a product.

---

## 6. Architecture Decisions (Informed by Research)

### 6.1 Tool Management Strategy
Based on Manus findings:
- **Select tools once per session start** (or on topic shift), NOT per turn
- Use **domain-based grouping** + **embedding similarity** for initial selection
- **Never dynamically add/remove tools mid-iteration** — invalidates KV-cache
- Consider **logit masking** for advanced implementations
- Implement **progressive tool accumulation** within a conversation (new domains add tools, never remove)

### 6.2 Context Assembly Pipeline
Based on Google ADK + ITR paper:
```
1. Intent classification (~100ms, cheap model) → identify relevant domains
2. Tool selection → load domain-specific tool subset (NOT all tools)
3. Memory assembly → entity facts + relevant episodes, scoped by agent role
4. History management → relevance-based truncation, not arbitrary N messages
5. Budget allocation → distribute token budget across all sources
6. Prefix optimization → keep stable content (system prompt, tool defs) at front for KV-cache hits
7. Assemble → single optimized context ready for LLM call
```

### 6.3 Memory Architecture (Three-Layer)
- **Working Memory (L0)**: In-prompt context — current conversation + pre-assembled memory block. Loaded once at session start, refreshed on topic shift.
- **Episodic Memory (L1)**: Redis — hot entity facts, recent events, behavioral cues. Sub-millisecond access.
- **Semantic Memory (L2)**: Pinecone — embedded events for similarity search. 20-40ms access.
- **Cold Storage (L3)**: Postgres — full history, all fact versions, audit trail. Source of truth.

### 6.4 Memory Write Pipeline
After every session/turn (async, non-blocking via Celery):
1. Extract EVENTS (what happened) — append-only, timestamped
2. Extract/Update FACTS (what's true now) — versioned, latest wins
3. Extract RELATIONSHIPS (entity graph)
4. Generate SESSION SUMMARY (lossy but cheap)
5. **Tool output extraction** — structured outputs mapped directly, unstructured via LLM extraction

### 6.5 Telemetry Design
Based on OpenTelemetry GenAI SIG + identified gaps:
- Emit standard `gen_ai.*` spans for compatibility with existing backends
- Add custom attributes for **context composition**: tools loaded, tools used, memory tokens, history tokens, wasted tokens
- Track per-turn: token budget utilization, tool selection accuracy (tool loaded but never called = waste), memory retrieval relevance
- Dashboard showing: cost breakdown by context source, waste detection, cache hit rates

---

## 7. SDK Design — Two Function Calls

```python
from contextengine import ContextEngine

engine = ContextEngine(
    memory_store="redis://localhost:6379",
    vector_store="pinecone://my-index",
    tool_registry="./tools.yaml",  # or MCP server URLs
    model="claude-sonnet-4-20250514",
    max_context_tokens=80_000,
)

# CALL 1: Assemble optimized context
ctx = await engine.assemble(
    entity_id="customer_123",
    agent_role="sales",
    message="I want to reorder last month's shipment",
    session_id="sess_abc",
)

# Pass to LLM (framework-agnostic)
response = await claude.messages.create(
    messages=ctx.messages,
    tools=ctx.tools,
    system=ctx.system_prompt,
)

# CALL 2: Process turn (async — writes memory + telemetry)
await engine.process_turn(
    response=response,
    session_id="sess_abc",
    entity_id="customer_123",
)
```

### Core Modules:
1. **IntentClassifier** — message → relevant domains (cheap model or embedding match, ~100ms)
2. **ToolRouter** — domains → tool subset (domain groups + embedding similarity, token-budget-aware)
3. **MemoryAssembler** — entity_id + agent_role → scoped context (facts, events, behavioral cues)
4. **BudgetManager** — allocates token budget across tools/memory/history, optimizes for KV-cache hits
5. **MemoryWriter** — post-turn async pipeline: extract facts/events from conversation + tool outputs
6. **TelemetryCollector** — logs context composition per turn for dashboard analytics

---

## 8. Business Model

### Open Source SDK (Free)
- Context assembly, tool routing, basic memory integration, local analytics

### Pro ($99-299/month)
- Hosted dashboard with context analytics
- Token cost tracking per agent, per session, per entity
- Memory health monitoring
- Waste detection (tools loaded but never used, memory never referenced)
- Alerting (budget exceeded, tool selection accuracy dropping)

### Enterprise ($1000+/month)
- Permission-scoped memory views
- Multi-agent context coordination
- Audit trails / compliance (GDPR memory deletion)
- Custom extraction pipelines
- SLA, dedicated support

---

## 9. GTM Strategy

1. **Open source SDK on GitHub** — killer README showing before/after (hand-rolled mess → two function calls)
2. **Deep technical blog posts** — "How we reduced agent token costs by 60% with dynamic tool routing", "Why your agent forgets: the memory assembly problem"
3. **Framework-agnostic positioning** — Day 1: raw Anthropic/OpenAI SDK. Week 2: LangChain integration. Week 4: CrewAI integration.
4. **Target 10-50 engineer teams** deploying agents in production. Find them in Discord, YC alumni, AI engineering meetups.
5. **Content distribution** — HN, Reddit r/LocalLLaMA, r/MachineLearning, dev.to

---

## 10. Key Risks

1. **Anthropic/OpenAI build it** — they haven't shown interest in tool routing / context orchestration. Focused on making models better, not managing context. Even if they build something, it'll be model-locked. We're model-agnostic.
2. **Memory players (Mem0, Zep) add tool routing** — they're going deeper into memory (graph, temporal, benchmarks), not expanding into tool management. Different direction. Partnership opportunity, not competition.
3. **Context windows keep growing** — 1M+ tokens reduces urgency of budget management. BUT: cost still scales linearly with tokens, attention still degrades with length, and "lost in the middle" doesn't go away.
4. **Market is small right now** — few teams running 30+ tool agents in production. Market is growing fast but you'd be early.
5. **Middleware is hard to sell** — invisible layer, requires customer to understand they have a problem first.

---

## 11. Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| SDK | Python (FastAPI-compatible) | Core library |
| Hot entity facts | Redis Hash | <1ms entity context |
| Event timeline | Redis Sorted Set | Time-ordered events per entity |
| Session context cache | Redis String | Assembled context for active sessions |
| Semantic event search | Pinecone | Similar past episodes retrieval (20-40ms) |
| Entity relationships | Postgres (+ Redis cache) | Graph traversal (5-50ms) |
| Full history / audit | Postgres | Source of truth, all versions |
| Extraction pipeline | Celery + Claude Haiku | Session → structured memory (1-3s async) |
| Embedding | Voyage / OpenAI | Event text → vectors (async) |
| Telemetry | OpenTelemetry + custom attributes | Context composition tracking |
| Dashboard | React + Recharts | Analytics UI |

---

## 12. Implementation Priority

### Month 1-3: SDK + Basic Dashboard
- ContextEngine class with assemble() and process_turn()
- Intent classifier (embedding-based domain detection)
- Tool router (domain groups + similarity, budget-aware)
- Basic memory integration (Redis facts + Pinecone episodes)
- Telemetry collector (log context composition per turn)
- Simple Streamlit/React dashboard showing token breakdown

### Month 3-6: Production Features
- Memory writer with tool output → fact extraction
- Permission-scoped memory views per agent role
- KV-cache-aware context ordering
- Cost analytics and waste detection
- LangChain / CrewAI integrations

### Month 6-12: Enterprise
- Multi-agent context coordination
- GDPR-compliant memory deletion with audit trails
- Advanced analytics (retrieval quality, memory health)
- Hosted platform with managed infrastructure
