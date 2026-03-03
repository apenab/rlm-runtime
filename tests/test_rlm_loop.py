from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import FakeAdapter


def test_rlm_loop_runs_to_final() -> None:
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(20)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "answer = f'Result: {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "ok")

    context = Context.from_text("RLMs inspect prompts via code.")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Return a result.", context)

    assert output == "Result: ok"
    kinds = [step.kind for step in trace.steps]
    assert "root_call" in kinds
    assert "repl_exec" in kinds
    assert "subcall" in kinds


def test_rlm_with_subcall_adapter() -> None:
    """Test using a different adapter for subcalls."""
    root_adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(20)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "answer = f'Got: {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )

    # Subcall adapter returns different response
    subcall_adapter = FakeAdapter(script=[])
    subcall_adapter.add_rule("You are a sub-LLM", "subcall-response")

    context = Context.from_text("Test context.")
    runtime = RLM(adapter=root_adapter, subcall_adapter=subcall_adapter)
    output, trace = runtime.run("Test query.", context)

    assert output == "Got: subcall-response"


def test_rlm_with_parallel_subcalls() -> None:
    """Test parallel subcall execution."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "chunks = ctx.chunk(5)",
                    "answers = ask_chunks('Q?', chunks, parallel=True)",
                    "result = ','.join(answers)",
                ]
            ),
            "FINAL_VAR: result",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "ans")

    context = Context.from_text("abcdefghij")
    runtime = RLM(adapter=adapter, parallel_subcalls=True)
    output, trace = runtime.run("Test.", context)

    # Should have multiple answers joined
    assert "ans" in output


def test_recursive_subcall_uses_configured_repl_backend(tmp_path) -> None:
    """Regression: recursive subcalls must use the configured REPL backend,
    not a hardcoded PythonREPL."""
    from unittest.mock import patch

    from pyrlm_runtime.trace import Trace

    adapter = FakeAdapter(
        script=[
            "result = llm_query('summarize')\nanswer = result",
            "FINAL_VAR: answer",
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        recursive_subcalls=True,
        max_recursion_depth=2,
        cache_dir=tmp_path / "cache",
    )

    with patch("pyrlm_runtime.rlm._run_recursive_subcall") as mock_rrs:
        mock_rrs.return_value = ("mocked-answer", Trace(steps=[]))
        output, trace = runtime.run("Test", context)

        # The key assertion: create_repl must be passed (not None)
        assert mock_rrs.called, "_run_recursive_subcall was never called"
        call_kwargs = mock_rrs.call_args.kwargs
        assert "create_repl" in call_kwargs, "create_repl was not passed to _run_recursive_subcall"
        assert call_kwargs["create_repl"] is not None, "create_repl should not be None"
        assert call_kwargs["create_repl"] == runtime._create_repl, (
            "create_repl should be RLM._create_repl"
        )

    assert output == "mocked-answer"


def test_rlm_with_document_context() -> None:
    """Test RLM with document list context."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "n = ctx.num_documents()",
                    "doc0 = ctx.get_document(0)",
                    "answer = f'Docs: {n}, First: {doc0[:10]}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )

    docs = ["First document content", "Second document content"]
    context = Context.from_documents(docs)
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Count docs.", context)

    assert "Docs: 2" in output
    assert "First: First docu" in output


# ---------------------------------------------------------------------------
# Conversation history tests
# ---------------------------------------------------------------------------


def test_conversation_history_messages_grow() -> None:
    """Verify the messages list grows across iterations in history mode."""
    adapter = FakeAdapter(
        script=[
            "x = 1\nprint(x)",
            "y = x + 1\nprint(y)",
            "FINAL_VAR: y",
        ]
    )
    context = Context.from_text("test data")
    runtime = RLM(adapter=adapter, conversation_history=True)
    output, trace = runtime.run("Compute something.", context)

    assert output == "2"
    # Call 1: [system, user_initial] = 2 messages
    # Call 2: + [assistant, user_repl] = 4 messages
    # Call 3: + [assistant, user_repl] = 6 messages
    assert len(adapter.call_log) == 3
    assert len(adapter.call_log[0]) == 2
    assert len(adapter.call_log[1]) == 4
    assert len(adapter.call_log[2]) == 6


def test_conversation_history_contains_previous_output() -> None:
    """Verify the LLM sees its own previous code and stdout in the history."""
    adapter = FakeAdapter(
        script=[
            'print("hello_marker")',
            "FINAL: done",
        ]
    )
    context = Context.from_text("test")
    runtime = RLM(adapter=adapter, conversation_history=True)
    output, trace = runtime.run("Test.", context)

    assert output == "done"
    # The second call should contain the first assistant response and REPL output
    second_call_msgs = adapter.call_log[1]
    # Message at index 2: assistant with the code
    assert second_call_msgs[2]["role"] == "assistant"
    assert 'print("hello_marker")' in second_call_msgs[2]["content"]
    # Message at index 3: user with REPL result containing stdout
    assert second_call_msgs[3]["role"] == "user"
    assert "hello_marker" in second_call_msgs[3]["content"]


def test_conversation_history_trimming() -> None:
    """Verify that history trimming keeps system + initial user + recent turns."""
    adapter = FakeAdapter(
        script=[
            "a = 1\nprint(a)",
            "b = 2\nprint(b)",
            "c = 3\nprint(c)",
            "FINAL: done",
        ]
    )
    context = Context.from_text("test")
    # Very small budget to force trimming
    runtime = RLM(
        adapter=adapter,
        conversation_history=True,
        max_history_tokens=200,
    )
    output, trace = runtime.run("Test.", context)

    assert output == "done"
    # System message should always be present in every call
    for call_msgs in adapter.call_log:
        assert call_msgs[0]["role"] == "system"
        if len(call_msgs) > 1:
            assert call_msgs[1]["role"] == "user"
    # Later calls should have fewer messages than unbounded growth would give
    # Without trimming call 4 would have 8 messages; with trimming it should be less
    assert len(adapter.call_log[-1]) < 8


def test_conversation_history_disabled_backward_compat() -> None:
    """When conversation_history=False, every call has exactly 2 messages."""
    adapter = FakeAdapter(
        script=[
            "x = 42\nprint(x)",
            "FINAL_VAR: x",
        ]
    )
    context = Context.from_text("test context")
    runtime = RLM(adapter=adapter, conversation_history=False)
    output, trace = runtime.run("Get x.", context)

    assert output == "42"
    # In stateless mode, every call must have exactly [system, user]
    for call_msgs in adapter.call_log:
        assert len(call_msgs) == 2
        assert call_msgs[0]["role"] == "system"
        assert call_msgs[1]["role"] == "user"
        # The user message should contain the query each time
        assert "Get x." in call_msgs[1]["content"]


def test_recursive_subcall_accepts_conversation_history_params() -> None:
    """Verify _run_recursive_subcall accepts conversation_history kwargs."""
    import inspect

    from pyrlm_runtime.rlm import _run_recursive_subcall

    sig = inspect.signature(_run_recursive_subcall)
    assert "conversation_history" in sig.parameters, (
        "_run_recursive_subcall must accept conversation_history"
    )
    assert "max_history_tokens" in sig.parameters, (
        "_run_recursive_subcall must accept max_history_tokens"
    )
    # Verify defaults
    assert sig.parameters["conversation_history"].default is True
    assert sig.parameters["max_history_tokens"].default == 0


def test_trim_history_preserves_role_alternation() -> None:
    """Regression: _trim_history must keep (assistant, user) pairs so role
    alternation is never broken by partial trimming."""
    from pyrlm_runtime.rlm import _trim_history

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "initial user message"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u3"},
    ]
    # Budget that fits head + only the last pair, not all of tail
    # estimate_tokens counts chars/4; head ≈ (1+20)/4 ≈ 5 tokens
    # Each tail message ≈ 1 token → pair ≈ 2 tokens
    # Set budget so only 1 pair from tail fits after head
    trimmed = _trim_history(msgs, max_tokens=8)

    # Must always start with system, user
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["role"] == "user"

    # Verify strict role alternation after the head
    for i in range(2, len(trimmed)):
        expected = "assistant" if i % 2 == 0 else "user"
        assert trimmed[i]["role"] == expected, (
            f"Role alternation broken at index {i}: expected {expected}, got {trimmed[i]['role']}"
        )


# ---------------------------------------------------------------------------
# min_steps regression tests
# ---------------------------------------------------------------------------


def test_min_steps_blocks_early_auto_finalize() -> None:
    """Auto-finalize must NOT trigger before min_steps is reached."""
    # The adapter produces code that sets final_answer on every step,
    # but min_steps=3 should keep the RLM iterating until step 3.
    adapter = FakeAdapter(
        script=[
            'final_answer = "step1_answer"',
            'final_answer = "step2_answer"',
            'final_answer = "step3_answer"',
            'final_answer = "step4_answer"',
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        auto_finalize_var="final_answer",
        min_steps=3,
    )
    output, trace = runtime.run("Q?", context)

    # Should have run at least 3 steps (the first two are blocked by min_steps)
    repl_steps = [s for s in trace.steps if s.kind == "repl_exec"]
    assert len(repl_steps) >= 3, f"Expected ≥3 REPL steps, got {len(repl_steps)}"
    assert output == "step3_answer"


def test_min_steps_blocks_explicit_final() -> None:
    """FINAL: must be rejected before min_steps is reached."""
    adapter = FakeAdapter(
        script=[
            # Step 1: try to finalize immediately — should be blocked
            "FINAL: early_answer",
            # Step 2: produce code after the guard blocks us
            'x = "working"',
            # Step 3: now we can finalize
            "FINAL: late_answer",
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        min_steps=2,
        require_repl_before_final=False,
    )
    output, trace = runtime.run("Q?", context)

    assert output == "late_answer"
    # Verify we went through more than one root call
    root_calls = [s for s in trace.steps if s.kind == "root_call"]
    assert len(root_calls) >= 2


def test_min_steps_zero_allows_immediate_finalize() -> None:
    """With min_steps=0 (default), FINAL on step 1 works normally."""
    adapter = FakeAdapter(
        script=[
            "FINAL: immediate",
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        min_steps=0,
        require_repl_before_final=False,
    )
    output, trace = runtime.run("Q?", context)

    assert output == "immediate"


# ---------------------------------------------------------------------------
# auto_finalize_reject_patterns regression tests
# ---------------------------------------------------------------------------


def test_auto_finalize_rejects_meta_reference() -> None:
    """Regression: GPT-5.2 writes '[La respuesta anterior...]' as final_answer.

    When auto_finalize_reject_patterns is configured, answers matching any
    pattern must be rejected so the RLM continues iterating.
    """
    adapter = FakeAdapter(
        script=[
            'final_answer = "[La respuesta anterior completa constituye la respuesta]"',
            (
                'final_answer = "Respuesta real con contenido completo que tiene '
                "más de cien caracteres de texto para superar cualquier limite mínimo "
                'de longitud configurado."'
            ),
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        auto_finalize_var="final_answer",
        auto_finalize_reject_patterns=[
            r"respuesta anterior",
            r"see above",
            r"the previous response",
        ],
    )
    output, trace = runtime.run("test query", context)

    assert "Respuesta real" in output
    # First answer was rejected, so we must have at least 2 REPL steps
    repl_steps = [s for s in trace.steps if s.kind == "repl_exec"]
    assert len(repl_steps) >= 2


def test_auto_finalize_reject_patterns_none_allows_anything() -> None:
    """Default: no reject patterns → any string accepted."""
    adapter = FakeAdapter(
        script=[
            'final_answer = "[La respuesta anterior completa]"',
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        auto_finalize_var="final_answer",
    )
    output, trace = runtime.run("test", context)

    assert "respuesta anterior" in output.lower()


def test_min_steps_does_not_block_at_max_steps_exhaustion() -> None:
    """When max_steps is exhausted, min_steps must NOT prevent returning
    whatever value is available in auto_finalize_var."""
    adapter = FakeAdapter(
        script=[
            'final_answer = "partial_result"',
            'final_answer = "still_working"',
            'final_answer = "almost_done"',
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        policy=Policy(max_steps=3),
        auto_finalize_var="final_answer",
        min_steps=10,  # Much higher than max_steps
    )
    output, trace = runtime.run("Q?", context)

    # MaxStepsExceeded handler should bypass min_steps and return the value
    assert "partial_result" in output or "still_working" in output or "almost_done" in output
