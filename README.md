<div align="center">

```
         ▄▀▀▀▄       ▄▀▀▀▄
        ▐     ▌     ▐     ▌
         ▀▄ ▄▀       ▀▄ ▄▀
           ▀           ▀
        ───────────────────
         c o n t e x t
         e n g i n e
```

### The middle layer for MCP agents

**Plug in N MCP servers → we budget-pack the right tool subset per turn,
assemble role-scoped memory, and emit full context-composition telemetry.
Framework-agnostic. Vectorless. KV-cache-aware.**

<sub>Anthropic · OpenAI · any MCP server · Python 3.10+</sub>

![status](https://img.shields.io/badge/status-alpha-yellow)
![tests](https://img.shields.io/badge/tests-87%20passing-green)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

---

## Why this exists

Every LLM call competes for a finite token budget — system prompt, tool
definitions, memory, history, retrieved knowledge. Today every team
hand-rolls that assembly. Nobody jointly optimizes tools + memory + history.
Nobody tracks what's actually *in* the context window.

Plug 100 MCP servers into one agent and you get 100+ tool definitions in
every prompt. The agent goes dumber, costs scale linearly, and you fly blind.

**contextengine** sits between your agent and your MCPs and solves this as
one optimization problem.

## The flow

```
  ╭──────────╮
  │ MCP #1   │──╮
  ╰──────────╯  │        ╭──────────────────────────────────╮
  ╭──────────╮  │        │                                  │
  │ MCP #2   │──┤        │  1. route tools (Haiku 2-pass)   │
  ╰──────────╯  │        │  2. assemble memory (role-scoped)│   system
  ╭──────────╮  ├───────▶│  3. compact older turns if long  │──▶ tools
  │ MCP #3   │──┤        │  4. greedy-pack token budget     │   messages
  ╰──────────╯  │        │  5. cache-friendly ordering      │
        ...     │        │  6. emit telemetry               │
  ╭──────────╮  │        │                                  │
  │ MCP #N   │──╯        ╰──────────────────────────────────╯
  ╰──────────╯                      contextengine
```

## Install

```bash
pip install contextengine              # core (Anthropic router)
pip install contextengine[openai]      # + GPT / o-series as router model
pip install contextengine[tokenizer]   # + tiktoken for accurate counts
pip install contextengine[dev]         # + pytest, mypy, ruff, all extras
```

## 60-second example

```python
import anthropic
from contextengine import ContextEngine, MCPServer

client = anthropic.AsyncAnthropic()

engine = ContextEngine(
    mcps=[
        MCPServer(name="linear", command=["npx", "@linear/mcp"]),
        MCPServer(name="stripe", url="https://mcp.stripe.com/sse"),
        MCPServer(name="github", command=["npx", "@github/mcp"]),
    ],
    model="claude-sonnet-4-5",
    router_model="claude-haiku-4-5",
    budget=80_000,
    system_prompt="You are a concise support agent.",
    anthropic_client=client,
)

await engine.start()    # connect, enumerate, categorize, cache

# Per turn — the engine routes + packs, your app calls the LLM:
ctx = await engine.assemble(
    message="refund order O-123 and open a Linear ticket",
    entity_id="customer_42",
    role="support",
)

response = await client.messages.create(
    model=engine.model,
    max_tokens=2048,
    system=ctx.system,     # includes role-scoped memory block
    messages=ctx.messages,
    tools=ctx.tools,       # only the ~2-3 MCP tools this turn needs
)

# Tool proxy back to the owning MCP:
if response.stop_reason == "tool_use":
    tool_block = response.content[-1]
    result = await engine.execute(tool_block)

# Post-turn memory writeback (async-safe):
await engine.process_turn(
    entity_id="customer_42",
    user_message="refund order O-123 ...",
    assistant_response=response.content[0].text,
    tool_results=[{"name": tool_block.name, "result": result}],
    role="support",
)

await engine.close()
```

## What makes it different

| The six gaps nobody solved | How contextengine answers |
|---|---|
| Unified budget across tools + memory + history | `engine.assemble()` — one greedy pack, one call |
| Tool-output → memory writeback | `engine.process_turn()` extracts facts + events via a cheap model |
| Context composition observability | `TraceRecorder` with `gen_ai.*` OTel-compatible attributes + JSONL sink |
| Framework-agnostic | `AssembleResult.to_anthropic()` / `.to_openai()` |
| KV-cache-aware assembly | stable prefix (system + memory + tools) + `cache_control: ephemeral` on router prompts |
| Permission-scoped memory views | `Fact.visibility` tuples — sales sees margins, support sees escalations |

## Pick your router: Claude or GPT

The internal LLM calls (routing, categorization, memory writeback,
compaction) work with either provider — the engine auto-detects from
the model string.

```python
# Claude everywhere (default)
engine = ContextEngine(
    mcps=[...],
    model="claude-sonnet-4-5",
    router_model="claude-haiku-4-5",
)

# GPT for routing + memory
engine = ContextEngine(
    mcps=[...],
    model="gpt-4o",
    router_model="gpt-4o-mini",
)

# Mixed — GPT router, Claude for memory writeback
engine = ContextEngine(
    mcps=[...],
    model="claude-sonnet-4-5",
    router_model="gpt-4o-mini",
    memory_model="claude-haiku-4-5",
)

# Or bring your own client implementing LLMClient
from contextengine import LLMClient, LLMResponse

class MyCustomLLM:
    async def complete(self, *, model, system, user, max_tokens, stable_prefix=None, json_mode=False):
        ...
        return LLMResponse(text=...)

engine = ContextEngine(mcps=[...], model="...", llm_client=MyCustomLLM())
```

Both paths go through the same `LLMClient` interface:
- `AnthropicClient` marks `stable_prefix` with `cache_control: ephemeral`
  for explicit prompt caching.
- `OpenAIClient` concatenates `stable_prefix` into the system message
  so it sits in OpenAI's automatic prefix cache position.
- `json_mode=True` enables structured-JSON output on OpenAI; Anthropic
  relies on the catalog-prompt prefix + `extract_json` tolerance.

## Vectorless tool routing

Instead of embedding every tool into a vector DB, contextengine builds a
**hierarchical catalog** (MCP → category → tools) at startup via one cheap
Haiku call per MCP, then uses a two-pass LLM traversal per turn:

```
  Pass 1 (MCP-level):   "given these MCP summaries, which are relevant?"
  Pass 2 (tool-level):  "within each selected MCP, which tools fit?"
```

Decisions memoize on `(message_hash, catalog_hash)`. Router cost
collapses to near-zero across a session via prompt caching on the stable
catalog prefix.

## API surface

```python
engine = ContextEngine(
    mcps=[...],                    # list[MCPServer]
    model="claude-sonnet-4-5",     # your agent model (engine doesn't call it)
    router_model="claude-haiku-4-5",
    memory_model=None,             # defaults to router_model
    budget=80_000,
    reserved_output=4096,
    memory_budget=4_000,
    system_prompt="",
    cache_dir=".contextengine",    # catalog cache + memory (if JSONStore)
    anthropic_client=None,         # provider SDK override; auto when model starts with "claude"
    openai_client=None,            # provider SDK override; auto when model starts with "gpt"/"o1"/"o3"/"o4"
    llm_client=None,               # bring-your-own LLMClient — overrides both
    tokenizer=None,                # auto: tiktoken if available, else estimator
    memory_store=None,             # InMemoryStore by default
    telemetry_sinks=[],            # StdoutSink, FileSink, …
    compaction_threshold=40,       # turns before rolling summary kicks in
    compaction_keep_recent=10,
)

await engine.start()
await engine.assemble(message, history=[], memory="", entity_id=None, role="", required_tools=())
await engine.execute(tool_use)         # proxy tool_use block → owning MCP
await engine.process_turn(entity_id=..., user_message=..., assistant_response=..., tool_results=None, role="")
await engine.add_mcp(server)           # hot-add
await engine.remove_mcp(name)          # hot-remove
await engine.close()
```

## CLI

```bash
# assemble context for one message and dump JSON
contextengine run --mcp fs='npx -y @modelcontextprotocol/server-filesystem /tmp' \
                  "list files in /tmp"

# print the hierarchical MCP catalog
contextengine catalog --mcp linear='npx @linear/mcp' \
                      --mcp stripe=https://mcp.stripe.com/sse

# with per-call telemetry
contextengine -v run --mcp ... "..."
```

## Examples

- [`examples/demo.py`](examples/demo.py) — minimal end-to-end assemble + stats
- [`examples/demo_memory.py`](examples/demo_memory.py) — full loop with `JSONStore` + stdout telemetry + `process_turn` writeback
- [`examples/demo_openai.py`](examples/demo_openai.py) — Claude-routed context, OpenAI execution via `to_openai()`

## Layout

```
contextengine/
  engine.py              top-level orchestration (assemble / execute / process_turn)
  router.py              vectorless two-pass LLM tool selection
  catalog.py             hierarchical catalog builder + disk cache
  budget.py              greedy token packer (tools prefix + newest history)
  compaction.py          rolling-summary compactor
  tokenize.py            tiktoken-backed + char-estimate fallback
  llm/
    base.py              LLMClient protocol + LLMResponse
    anthropic.py         AnthropicClient (cache_control: ephemeral)
    openai.py            OpenAIClient (response_format json_object, prefix cache)
    registry.py          auto-detect provider from model string
  mcp/
    connector.py         stdio + SSE/HTTP via official `mcp` SDK
    pool.py              lifecycle + duplicate-name guard
    schema.py            raw MCP tool → internal Tool
  memory/
    types.py             Fact, Event, EntityMemory
    store.py             MemoryStore protocol, InMemoryStore, JSONStore
    assembler.py         role-scoped, budget-aware memory block
    writer.py            post-turn LLM extraction
  telemetry/
    recorder.py          TraceRecord (gen_ai.*) + per-phase spans
    sinks.py             StdoutSink, FileSink (JSONL)
  adapters/
    openai.py            AssembleResult → OpenAI chat.completions shape
  cli.py                 `contextengine run` / `catalog`
```

## Status

Alpha. 87 tests pass. API surface is frozen-ish; wire format of cached catalogs
may change before 1.0. Not yet published to PyPI — install from source:

```bash
git clone https://github.com/vishalbende/cortex
cd cortex
pip install -e '.[dev,tokenizer]'
pytest
```

## Roadmap

- [ ] Multi-agent context coordination (shared memory, handoff protocol)
- [ ] Permission-scoped memory *writes* (not just reads)
- [ ] Streaming assemble + partial tool-set refinement mid-response
- [ ] Native Anthropic server-side token counting
- [ ] LangChain / LangGraph adapters
- [ ] Hosted dashboard consuming the JSONL telemetry
- [ ] Framework-agnostic memory query API (GDPR deletion, fact versioning UI)

## License

MIT
