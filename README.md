# rlm-runtime

Minimal runtime for **Recursive Language Models (RLMs)** inspired by the MIT CSAIL paper
"Recursive Language Models". The core idea: the long prompt lives in a persistent environment
(Python REPL), and the LLM only sees metadata plus REPL outputs. The LLM writes code to
inspect the context and can make **subcalls** over small snippets.

## What this is
- A lightweight runtime loop: root LLM <-> REPL
- Deterministic `Context` helpers (slice/find/chunk)
- A safe-ish Python REPL with minimal builtins
- Provider-agnostic adapters + a FakeAdapter for tests
- Tracing + simple cache for replay

## What this is not
- A full agents framework
- A RAG system or vector database
- A production sandbox (this is an MVP)

## Quickstart
```bash
uv run python examples/minimal.py
```

## Demo: RLM vs Baseline Comparison

The `rlm_vs_baseline.py` example demonstrates the core advantage of RLMs: maintaining accuracy as context grows beyond the LLM's window, while a naive baseline fails due to truncation.

### Running the Demo

```bash
# Quick demo (5 and 30 documents)
RLM_CONTEXT_SIZES=5,30 uv run python examples/rlm_vs_baseline.py

# Full benchmark showing crossover point (5, 20, 50, 120 documents)
RLM_CONTEXT_SIZES=5,20,50,120 uv run python examples/rlm_vs_baseline.py

# Show detailed RLM execution trajectory
SHOW_TRAJECTORY=1 RLM_CONTEXT_SIZES=5,30 uv run python examples/rlm_vs_baseline.py
```

### What the Demo Shows

This benchmark implements a **needle-in-haystack** task (similar to the MIT paper's S-NIAH):
- The context contains N documents, with one containing a hidden key term
- The query asks: "What is the key term?"
- **Baseline approach**: Sends entire context directly to LLM (truncates if too large)
- **RLM approach**: Context lives in REPL, LLM writes code to search and make subcalls

### Expected Results

The demo visualizes the **crossover point** where RLM starts outperforming baseline:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROSSOVER ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Plot 1: Success Rate vs Context Size
────────────────────────────────────
  5 docs │ B (baseline OK)
 20 docs │ B (baseline OK)
 50 docs │ b R (baseline FAIL, RLM OK) ← CROSSOVER POINT
120 docs │ b R (baseline FAIL, RLM OK)

Legend: B=baseline success, b=baseline fail, R=RLM success, r=RLM fail


Plot 2: Token Usage Comparison
───────────────────────────────
  5 docs │ baseline: ████░░░░░░ (8.8K)  🏆
         │      rlm: ████████░░ (17.3K)

 20 docs │ baseline: ████████░░ (18.5K) 🏆
         │      rlm: ████████░░ (18.0K)

 50 docs │ baseline: FAIL (truncated)
         │      rlm: █████████░ (20.9K) 🏆

120 docs │ baseline: FAIL (truncated)
         │      rlm: ██████████ (23.5K) 🏆

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULTS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detailed Comparison:
┌─────────┬──────────┬────────┬───────┬────────┬────────────┬─────────┐
│   Docs  │  Tokens  │  Time  │ OK?   │ Answer │   Method   │ Winner  │
├─────────┼──────────┼────────┼───────┼────────┼────────────┼─────────┤
│    5    │   8,831  │  1.2s  │  ✓    │  ✓     │  baseline  │ 🏆 base │
│         │  17,298  │  2.8s  │  ✓    │  ✓     │     rlm    │         │
├─────────┼──────────┼────────┼───────┼────────┼────────────┼─────────┤
│   20    │  18,454  │  2.1s  │  ✓    │  ✓     │  baseline  │ 🏆 base │
│         │  18,039  │  3.1s  │  ✓    │  ✓     │     rlm    │         │
├─────────┼──────────┼────────┼───────┼────────┼────────────┼─────────┤
│   50    │  TRUNCATED - Answer lost in truncation                    │
│         │  20,866  │  3.8s  │  ✓    │  ✓     │     rlm    │ 🏆 rlm  │
├─────────┼──────────┼────────┼───────┼────────┼────────────┼─────────┤
│  120    │  TRUNCATED - Answer lost in truncation                    │
│         │  23,489  │  4.5s  │  ✓    │  ✓     │     rlm    │ 🏆 rlm  │
└─────────┴──────────┴────────┴───────┴────────┴────────────┴─────────┘

Summary Statistics:
  • Baseline wins: 2 (at small context sizes)
  • RLM wins: 2 (at large context sizes where baseline truncates)
  • Crossover point: ~50 documents (baseline starts truncating)

RLM Efficiency Metrics:
  • Avg subcalls per task: 1-3 (uses Phase 0 deterministic extraction first)
  • Cache hit rate: 60-80% (reuses subcall results)
  • Token overhead: 2-3x at small contexts, but maintains correctness at large contexts
```

### Key Insights

1. **Small contexts (5-20 docs)**: Baseline is more efficient (fewer tokens, faster)
2. **Large contexts (50+ docs)**: Baseline fails due to truncation, RLM succeeds by:
   - Using Phase 0 deterministic extraction (`extract_after`)
   - Making targeted subcalls on document chunks only when needed
   - Leveraging caching to reduce redundant LLM calls
3. **Crossover point**: Around 50 documents, where context exceeds the LLM's window

This aligns with Figure 1 of the MIT paper: RLMs maintain performance as context grows, while baseline approaches degrade.

## Example (FakeAdapter)
```python
from rlm_runtime import Context, RLM
from rlm_runtime.adapters import FakeAdapter

adapter = FakeAdapter(
    script=[
        "snippet = peek(80)\nsummary = llm_query(f'Summarize: {snippet}')\nanswer = summary",
        "FINAL_VAR: answer",
    ]
)
adapter.add_rule("You are a sub-LLM", "fake summary")

context = Context.from_text("RLMs treat long prompts as environment state.")
output, trace = RLM(adapter=adapter).run("Summarize.", context)
print(output)
```

## Design (paper-aligned)
- **Environment**: prompt lives as `P` in a REPL; helpers (`peek`, `tail`, `lenP`) are provided.
- **Context**: safe inspection (`slice`, `find`, `chunk`).
- **Policy**: step/subcall/token budgets.
- **Tracing**: structured steps + JSON export; subcalls record input/output hashes.

## Adapters
- `FakeAdapter` for tests/examples.
- `GenericChatAdapter` for schema-configurable chat endpoints.
- `OpenAICompatAdapter` for OpenAI-compatible endpoints (including Llama servers):
  - Uses `LLM_API_KEY` (preferred) or `OPENAI_API_KEY`.
  - Uses `LLM_BASE_URL` (preferred) or `OPENAI_BASE_URL`.
  - If no key is set, it sends no auth header (works for local servers).

### Llama notes
Local Llama models can be less compliant with the REPL protocol. Use the stricter
`LLAMA_SYSTEM_PROMPT` and set `require_repl_before_final=True` to force at least one REPL step.

## Roadmap
- Async subcalls
- Stronger sandboxing
- Optional tool calling

## Dev / Quality gates
```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest
```
