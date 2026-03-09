"""
Live Rich trace demo for the RLM loop.

Run with:
    uv run python examples/rich_repl_demo.py

Optional real model:
    LLM_MODEL=gpt-4 uv run python examples/rich_repl_demo.py --live
"""

from __future__ import annotations

import os
import sys

from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import FakeAdapter, OpenAICompatAdapter
from pyrlm_runtime.rich_trace import RichTraceListener


def run_fake_demo() -> None:
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(60)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "print(summary)",
                    "answer = f'Summary -> {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "[fake] short summary")
    runtime = RLM(adapter=adapter, event_listener=RichTraceListener())
    output, _trace = runtime.run(
        "Give a short summary.",
        Context.from_text(
            "RLMs treat long prompts as environment state and inspect them via code."
        ),
    )
    print(f"\nFinal answer: {output}")


def run_live_demo() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4")
    adapter = OpenAICompatAdapter(model=model)
    runtime = RLM(adapter=adapter, event_listener=RichTraceListener())
    output, _trace = runtime.run(
        "What are the main themes in this document?",
        Context.from_text(
            "Recursive Language Models analyze long contexts by writing code in a REPL. "
            "They can inspect, chunk, and summarize the data with subcalls when needed."
        ),
    )
    print(f"\nFinal answer: {output}")


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_demo()
    else:
        run_fake_demo()
