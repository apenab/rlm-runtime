"""Tests for SmartRouter routing logic and configuration."""

from __future__ import annotations

from pyrlm_runtime import Context, RouterConfig, SmartRouter
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Routing decision tests
# ---------------------------------------------------------------------------


def test_router_baseline_below_threshold() -> None:
    """Contexts smaller than the threshold must use baseline."""
    adapter = FakeAdapter(script=["baseline answer"])

    router = SmartRouter(
        adapter,
        config=RouterConfig(baseline_threshold=8000),
    )

    # Context well below threshold
    context = Context.from_text("small context")
    result = router.run("Q?", context)

    assert result.method == "baseline"
    assert result.output == "baseline answer"


def test_router_rlm_above_threshold() -> None:
    """Contexts larger than the threshold must use RLM."""
    adapter = FakeAdapter(
        script=[
            # Step 1: REPL code
            'answer = "rlm_answer"',
            # Step 2: finalize
            "FINAL_VAR: answer",
        ]
    )

    router = SmartRouter(
        adapter,
        config=RouterConfig(baseline_threshold=100),
    )

    # Context well above threshold (> 100 chars)
    context = Context.from_text("x" * 200)
    result = router.run("Q?", context)

    assert result.method == "rlm"
    assert result.output == "rlm_answer"


# ---------------------------------------------------------------------------
# Baseline configuration tests
# ---------------------------------------------------------------------------


def test_router_baseline_uses_config_max_tokens() -> None:
    """baseline_max_tokens from RouterConfig must reach the adapter."""
    adapter = FakeAdapter(script=["ok"])

    router = SmartRouter(
        adapter,
        config=RouterConfig(baseline_max_tokens=2048),
    )

    context = Context.from_text("tiny")
    router.run("Q?", context)

    # FakeAdapter.call_log records every call.  Inspect what the adapter saw.
    # We can't directly inspect max_tokens on FakeAdapter, so patch complete.
    # Re-run with a patched complete to capture max_tokens.
    adapter2 = FakeAdapter(script=["ok"])
    router2 = SmartRouter(
        adapter2,
        config=RouterConfig(baseline_max_tokens=2048),
    )
    context2 = Context.from_text("tiny")

    captured_kwargs: dict = {}
    original_complete = adapter2.complete

    def capturing_complete(messages, *, max_tokens=512, temperature=0.0):
        captured_kwargs["max_tokens"] = max_tokens
        return original_complete(messages, max_tokens=max_tokens, temperature=temperature)

    adapter2.complete = capturing_complete  # type: ignore[assignment]
    router2.run("Q?", context2)

    assert captured_kwargs["max_tokens"] == 2048


def test_router_baseline_uses_system_prompt() -> None:
    """Baseline should use the router's system_prompt, not the generic default."""
    adapter = FakeAdapter(script=["answer"])

    custom_prompt = "You are a 23F document analyst."
    router = SmartRouter(
        adapter,
        system_prompt=custom_prompt,
    )

    context = Context.from_text("tiny")
    router.run("Q?", context)

    # The first call should have our system prompt
    call = adapter.call_log[0]
    system_msg = call[0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == custom_prompt


def test_router_baseline_system_prompt_cascade() -> None:
    """baseline_system_prompt in config takes precedence over router system_prompt."""
    adapter = FakeAdapter(script=["answer"])

    router = SmartRouter(
        adapter,
        system_prompt="router-level prompt",
        config=RouterConfig(baseline_system_prompt="config-level prompt"),
    )

    context = Context.from_text("tiny")
    router.run("Q?", context)

    call = adapter.call_log[0]
    system_msg = call[0]
    assert system_msg["content"] == "config-level prompt"


# ---------------------------------------------------------------------------
# rlm_extra_kwargs tests
# ---------------------------------------------------------------------------


def test_router_rlm_extra_kwargs_merged() -> None:
    """rlm_extra_kwargs must be forwarded to the RLM instance."""
    adapter = FakeAdapter(
        script=[
            'answer = "done"',
            "FINAL_VAR: answer",
        ]
    )

    router = SmartRouter(
        adapter,
        config=RouterConfig(baseline_threshold=10),
        rlm_extra_kwargs={
            "min_steps": 0,
            "auto_finalize_min_length": 0,
        },
    )

    context = Context.from_text("x" * 100)
    result = router.run("Q?", context)

    assert result.method == "rlm"
    assert result.output == "done"


def test_router_rlm_uses_base_system_prompt_by_default() -> None:
    """Router RLM path should default to BASE_SYSTEM_PROMPT."""
    adapter = FakeAdapter(
        script=[
            'answer = "done"',
            "FINAL_VAR: answer",
        ]
    )
    router = SmartRouter(
        adapter,
        config=RouterConfig(baseline_threshold=10),
    )

    context = Context.from_text("x" * 100)
    result = router.run("Q?", context)

    assert result.method == "rlm"
    first_call = adapter.call_log[0]
    assert first_call[0]["role"] == "system"
    assert first_call[0]["content"] == BASE_SYSTEM_PROMPT


def test_router_rlm_respects_system_prompt_override() -> None:
    """Router-level system_prompt should override BASE_SYSTEM_PROMPT for RLM."""
    adapter = FakeAdapter(
        script=[
            'answer = "done"',
            "FINAL_VAR: answer",
        ]
    )
    custom_prompt = "custom router rlm prompt"
    router = SmartRouter(
        adapter,
        system_prompt=custom_prompt,
        config=RouterConfig(baseline_threshold=10),
    )

    context = Context.from_text("x" * 100)
    result = router.run("Q?", context)

    assert result.method == "rlm"
    first_call = adapter.call_log[0]
    assert first_call[0]["role"] == "system"
    assert first_call[0]["content"] == custom_prompt
