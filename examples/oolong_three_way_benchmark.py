#!/usr/bin/env python3
"""
Three-way Oolong benchmark: baseline vs rlm-minimal vs pyrlm-runtime.

This script reuses Oolong datasets ("oolongbench/oolong-synth" or "oolongbench/oolong-real")
and mirrors Oolong-style scoring while evaluating three execution strategies:
1) baseline single-shot prompt
2) rlm-minimal REPL loop
3) pyrlm-runtime agentic loop

Usage:
  uv run python examples/oolong_three_way_benchmark.py \
    --model qwen2.5-coder:14b-instruct \
    --dataset synth \
    --base-url http://127.0.0.1:11434/v1 \
    --max-examples 30

Standard reproducible run:
  cd /home/apenab/projects/rlm-runtime && \
  HF_TOKEN=hf_tu_token \
  uv run python examples/oolong_three_way_benchmark.py \
    --model qwen2.5-coder:14b-instruct \
    --dataset synth \
    --sample-strategy stratified \
    --seed 42 \
    --max-examples 100 \
    --baseline-max-tokens 1024 \
    --agent-max-tokens 2048 \
    --agent-subcall-max-tokens 2048 \
    --base-url http://127.0.0.1:11434/v1
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import GenericChatAdapter


def usage_total(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "total_tokens", 0) or 0)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def context_bucket(context_len: int) -> str:
    if context_len < 8_000:
        return "S(<8k)"
    if context_len < 32_000:
        return "M(8k-32k)"
    if context_len < 128_000:
        return "L(32k-128k)"
    return "XL(>=128k)"


def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        return answer.split()[-1], parse_confidence

    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")
    candidate_answer = candidate_answer.replace("[", "").replace("]", "")
    parse_confidence = "med"

    if (
        "User:" in answer
        or "Answer:" in answer
        or "Date:" in answer
        or "Label" in answer
    ):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    score = 0.0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )

    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1.0
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            score = 1.0
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            trimmed_output_int = int(trimmed_output)
            gold_int = int(gold)
            score = float(0.75 ** abs(gold_int - trimmed_output_int))
        except Exception:
            parse_confidence = "low"
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            parsed = dateutil.parser.parse(trimmed_output)
            score = float(parsed == gold)
        except Exception:
            parse_confidence = "low"

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "dataset": datapoint["dataset"],
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }


def dnd_parse_answer(answer: str) -> int | str | list[str]:
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple[Any, str]:
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", answer
    )
    if match:
        answer = match.group(1)
    else:
        return answer, "low"
    return dnd_parse_answer(answer), "high"


def dnd_process_response(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    gold = dnd_parse_answer(datapoint["answer"])
    trimmed_output, parse_confidence = dnd_parse_response(output)
    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        score = float(0.75 ** abs(gold - trimmed_output))
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        score = float(gold.strip().lower() == trimmed_output.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        score = float(len(overlap) / len(gold)) if gold else 0.0

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "model": model,
        "attempted_parse": trimmed_output,
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": gold,
    }


def openrouter_payload_builder(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    model: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        payload["model"] = model
    return payload


def run_baseline(
    adapter: GenericChatAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, float, str | None]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise solver. Return only the final answer. "
                "No explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\nQuestion:\n{question}\n\n"
                "Return only the answer token/string."
            ),
        },
    ]
    start = time.time()
    try:
        resp = adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return resp.text or "", usage_total(resp), time.time() - start, None
    except Exception as exc:
        return "", 0, time.time() - start, str(exc)


def run_rlm_minimal(
    root_adapter: GenericChatAdapter,
    sub_adapter: GenericChatAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    subcall_tokens: int,
    temperature: float,
    max_iterations: int,
) -> tuple[str, int, float, str | None, list[dict[str, Any]]]:
    safe_builtins = {
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "bool": bool,
        "type": type,
        "isinstance": isinstance,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "repr": repr,
        "range": range,
        "iter": iter,
        "next": next,
        "__import__": __import__,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
    }
    repl_globals: dict[str, Any] = {"__builtins__": safe_builtins}
    repl_locals: dict[str, Any] = {"context": context_text}
    total_tokens = 0
    debug_events: list[dict[str, Any]] = []

    def llm_query(prompt: str) -> str:
        nonlocal total_tokens
        msgs = [{"role": "user", "content": prompt}]
        resp = sub_adapter.complete(msgs, max_tokens=subcall_tokens, temperature=temperature)
        used = usage_total(resp)
        total_tokens += used
        debug_events.append(
            {
                "type": "subcall",
                "prompt_chars": len(prompt),
                "response_chars": len(resp.text),
                "tokens": used,
            }
        )
        return resp.text

    repl_globals["llm_query"] = llm_query

    def _exec_code(code: str) -> tuple[str, str | None]:
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        err: str | None = None
        try:
            ns = {**repl_globals, **repl_locals}
            exec(code, ns, ns)  # noqa: S102
            for k, v in ns.items():
                if k not in repl_globals:
                    repl_locals[k] = v
        except Exception as exc:
            err = str(exc)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue(), err

    def _find_code_blocks(text: str) -> list[str]:
        blocks: list[str] = []
        blocks.extend(re.findall(r"```repl\s*\n(.*?)\n```", text, re.DOTALL))
        blocks.extend(re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL))
        return [b for b in blocks if b.strip()]

    def _find_final_answer(text: str) -> tuple[str, str] | None:
        m = re.search(r"^\s*FINAL_VAR\((.*?)\)", text, re.MULTILINE)
        if m:
            return "FINAL_VAR", m.group(1).strip()
        m = re.search(r"^\s*FINAL\((.*?)\)", text, re.MULTILINE | re.DOTALL)
        if m:
            return "FINAL", m.group(1).strip()
        m = re.search(r"^\s*FINAL\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            return "FINAL", m.group(1).strip()
        return None

    system_prompt = (
        "You are in a REPL loop with variable `context` containing the full context and "
        "`llm_query(prompt)` for subcalls.\n"
        "Goal: answer the question exactly.\n"
        "Respond with REPL code blocks and finish with FINAL(...) or FINAL_VAR(...).\n"
        "Prefer deterministic extraction from `context` when possible."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    start = time.time()

    try:
        for i in range(max_iterations):
            user_msg = (
                f"Question: {question}\n"
                "If not done, write code in ```python``` or ```repl```.\n"
                "If done, output FINAL(...) or FINAL_VAR(...)."
            )
            call_messages = messages + [{"role": "user", "content": user_msg}]
            resp = root_adapter.complete(call_messages, max_tokens=max_tokens, temperature=temperature)
            root_tokens = usage_total(resp)
            total_tokens += root_tokens
            text = resp.text or ""
            debug_events.append(
                {
                    "type": "root_call",
                    "iteration": i + 1,
                    "response_chars": len(text),
                    "tokens": root_tokens,
                }
            )
            messages.append({"role": "assistant", "content": text})

            final = _find_final_answer(text)
            if final:
                kind, content = final
                if kind == "FINAL":
                    return content.strip(), total_tokens, time.time() - start, None, debug_events
                var_name = content.strip().strip("\"").strip("'")
                return (
                    str(repl_locals.get(var_name, f"(variable '{var_name}' not found)")),
                    total_tokens,
                    time.time() - start,
                    None,
                    debug_events,
                )

            code_blocks = _find_code_blocks(text)
            if not code_blocks:
                messages.append(
                    {
                        "role": "user",
                        "content": "No executable code found. Provide code or FINAL(...).",
                    }
                )
                continue

            for idx, code in enumerate(code_blocks, start=1):
                stdout, stderr = _exec_code(code)
                debug_events.append(
                    {
                        "type": "repl_exec",
                        "iteration": i + 1,
                        "block_index": idx,
                        "code": code,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                )
                output = stdout or "(no output)"
                if stderr:
                    output += f"\nError: {stderr}"
                if len(output) > 120000:
                    output = output[:120000] + "..."
                messages.append(
                    {
                        "role": "user",
                        "content": f"Code output:\n{output}",
                    }
                )

        resp = root_adapter.complete(
            messages + [{"role": "user", "content": "Return final answer now using FINAL(...)"}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        total_tokens += usage_total(resp)
        return resp.text or "", total_tokens, time.time() - start, None, debug_events
    except Exception as exc:
        return "", total_tokens, time.time() - start, str(exc), debug_events


def summarize_trace_steps(trace: Any) -> dict[str, int]:
    if trace is None or not hasattr(trace, "steps"):
        return {}
    out: dict[str, int] = defaultdict(int)
    for step in trace.steps:
        out[step.kind] += 1
    return dict(out)


def run_pyrlm(
    root_adapter: GenericChatAdapter,
    sub_adapter: GenericChatAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    subcall_tokens: int,
    max_steps: int,
    max_subcalls: int,
) -> tuple[str, int, float, str | None, dict[str, int]]:
    context = Context.from_text(context_text)
    system_prompt = (
        "You have Python REPL access with P and ctx.\n"
        "Answer exactly the question and store final result in final_answer.\n"
        "Use deterministic string search first if possible.\n"
        "Return only the final answer."
    )
    rlm = RLM(
        adapter=root_adapter,
        subcall_adapter=sub_adapter,
        policy=Policy(
            max_steps=max_steps,
            max_subcalls=max_subcalls,
            max_total_tokens=12_000_000,
        ),
        system_prompt=system_prompt,
        subcall_system_prompt="Answer the user question from provided text. Output only answer.",
        max_tokens=max_tokens,
        subcall_max_tokens=subcall_tokens,
        auto_finalize_var="final_answer",
        parallel_subcalls=True,
        max_concurrent_subcalls=20,
        conversation_history=True,
    )

    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(question, context)
        tokens = 0
        if trace is not None:
            tokens = sum((s.usage.total_tokens for s in trace.steps if s.usage), 0)
        return output or "", tokens, time.time() - start, None, summarize_trace_steps(trace)
    except Exception as exc:
        return "", 0, time.time() - start, str(exc), summarize_trace_steps(trace)


def score_output(dataset_kind: str, datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    if dataset_kind == "synth":
        return synth_process_response(datapoint, output, model)
    return dnd_process_response(datapoint, output, model)


def engine_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"examples": 0, "score_sum": 0.0, "tokens_sum": 0, "elapsed_sum": 0.0, "errors": 0}
    )
    for row in rows:
        e = row["engine"]
        grouped[e]["examples"] += 1
        grouped[e]["score_sum"] += float(row["score"])
        grouped[e]["tokens_sum"] += int(row["tokens"])
        grouped[e]["elapsed_sum"] += float(row["elapsed"])
        grouped[e]["errors"] += int(bool(row["error"]))

    out: dict[str, Any] = {}
    for engine, acc in grouped.items():
        n = max(1, acc["examples"])
        out[engine] = {
            "examples": acc["examples"],
            "avg_score": acc["score_sum"] / n,
            "avg_tokens": acc["tokens_sum"] / n,
            "avg_elapsed": acc["elapsed_sum"] / n,
            "errors": acc["errors"],
        }
    return out


def engine_summary_by_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "examples": 0,
                "score_sum": 0.0,
                "tokens_sum": 0,
                "elapsed_sum": 0.0,
                "errors": 0,
            }
        )
    )
    for row in rows:
        engine = row["engine"]
        bucket = row["context_bucket"]
        g = grouped[engine][bucket]
        g["examples"] += 1
        g["score_sum"] += float(row["score"])
        g["tokens_sum"] += int(row["tokens"])
        g["elapsed_sum"] += float(row["elapsed"])
        g["errors"] += int(bool(row["error"]))

    out: dict[str, Any] = {}
    for engine, by_bucket in grouped.items():
        out[engine] = {}
        for bucket, acc in by_bucket.items():
            n = max(1, acc["examples"])
            out[engine][bucket] = {
                "examples": acc["examples"],
                "avg_score": acc["score_sum"] / n,
                "avg_tokens": acc["tokens_sum"] / n,
                "avg_elapsed": acc["elapsed_sum"] / n,
                "errors": acc["errors"],
            }
    return out


def validate_rows_schema(rows: list[dict[str, Any]]) -> None:
    required = {
        "id",
        "engine",
        "score",
        "tokens",
        "elapsed",
        "error",
        "context_len",
        "context_bucket",
        "attempted_parse",
        "gold_answer",
        "parse_confidence",
    }
    for idx, row in enumerate(rows):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Row {idx} missing required keys: {sorted(missing)}")


def validate_result_consistency(
    rows: list[dict[str, Any]],
    per_example: list[dict[str, Any]],
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    validate_rows_schema(rows)

    expected_rows = len(per_example) * 3
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows (3 engines x {len(per_example)} examples), got {len(rows)}"
        )

    if int(run_config.get("examples_evaluated", -1)) != len(per_example):
        raise ValueError(
            "run_config.examples_evaluated does not match number of evaluated examples"
        )

    recomputed = engine_summary(rows)
    if summary != recomputed:
        raise ValueError("summary payload is inconsistent with flat rows")


def select_rows(
    data: Any,
    *,
    strategy: str,
    max_examples: int,
    seed: int,
) -> Any:
    if max_examples <= 0 or len(data) <= max_examples:
        return data

    if strategy == "head":
        return data.select(range(max_examples))

    if strategy == "random":
        rng = random.Random(seed)
        idxs = list(range(len(data)))
        rng.shuffle(idxs)
        return data.select(sorted(idxs[:max_examples]))

    # Stratified by log2(context_len) bins to avoid overfitting to one length regime.
    bins: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(data):
        ctx_len = int(row["context_len"])
        key = int(math.log2(max(2, ctx_len)))
        bins[key].append(i)

    rng = random.Random(seed)
    keys = sorted(bins.keys())
    selected: list[int] = []
    per_bin = max(1, max_examples // max(1, len(keys)))
    for key in keys:
        choices = bins[key][:]
        rng.shuffle(choices)
        selected.extend(choices[:per_bin])

    # Fill remaining slots from leftovers globally.
    if len(selected) < max_examples:
        remaining = [i for i in range(len(data)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_examples - len(selected)])

    selected = sorted(selected[:max_examples])
    return data.select(selected)


def write_markdown_summary(path: Path, summary: dict[str, Any], by_bucket: dict[str, Any]) -> None:
    lines = []
    lines.append("# Oolong Three-Way Benchmark Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Engine | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in summary:
            continue
        s = summary[engine]
        lines.append(
            f"| {engine} | {s['examples']} | {s['avg_score']:.4f} | {s['avg_tokens']:.1f} | {s['avg_elapsed']:.2f} | {s['errors']} |"
        )

    lines.append("")
    lines.append("## By Context Bucket")
    lines.append("")
    lines.append("| Engine | Bucket | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    bucket_order = ["S(<8k)", "M(8k-32k)", "L(32k-128k)", "XL(>=128k)"]
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in by_bucket:
            continue
        for bucket in bucket_order:
            if bucket not in by_bucket[engine]:
                continue
            s = by_bucket[engine][bucket]
            lines.append(
                f"| {engine} | {bucket} | {s['examples']} | {s['avg_score']:.4f} | {s['avg_tokens']:.1f} | {s['avg_elapsed']:.2f} | {s['errors']} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
    )
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument(
        "--sample-strategy",
        choices=["head", "random", "stratified"],
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context-len", type=int, default=1024)
    parser.add_argument("--max-context-len", type=int, default=131072)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--baseline-max-tokens", type=int, default=1024)
    parser.add_argument("--agent-max-tokens", type=int, default=2048)
    parser.add_argument("--agent-subcall-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rlm-max-steps", type=int, default=35)
    parser.add_argument("--rlm-max-subcalls", type=int, default=250)
    parser.add_argument("--minimal-max-iterations", type=int, default=20)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: uv pip install datasets"
        ) from exc

    split_to_use = "context_window_text_with_labels" if args.labels else "context_window_text"
    if args.dataset == "synth":
        data = load_dataset("oolongbench/oolong-synth")["test"]
    else:
        data = load_dataset("oolongbench/oolong-real", "dnd")["test"]

    data = data.filter(lambda x: x["context_len"] <= args.max_context_len)
    data = data.filter(lambda x: x["context_len"] > args.min_context_len)
    data = select_rows(
        data,
        strategy=args.sample_strategy,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    baseline_max_tokens = args.baseline_max_tokens
    agent_max_tokens = args.agent_max_tokens
    agent_subcall_max_tokens = args.agent_subcall_max_tokens
    if args.max_tokens is not None:
        baseline_max_tokens = args.max_tokens
        agent_max_tokens = args.max_tokens
        agent_subcall_max_tokens = args.max_tokens

    run_dir = (
        Path("examples/exports/oolong_three_way")
        / f"run_{now_tag()}_{args.dataset}_{safe_model_name(args.model)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    base_adapter = GenericChatAdapter(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=900,
        payload_builder=openrouter_payload_builder,
    )
    min_root = GenericChatAdapter(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=900,
        payload_builder=openrouter_payload_builder,
    )
    min_sub = GenericChatAdapter(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=900,
        payload_builder=openrouter_payload_builder,
    )
    py_root = GenericChatAdapter(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=900,
        payload_builder=openrouter_payload_builder,
    )
    py_sub = GenericChatAdapter(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=900,
        payload_builder=openrouter_payload_builder,
    )

    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    print(
        f"Running {len(data)} examples from Oolong-{args.dataset} with model={args.model} "
        f"(sampling={args.sample_strategy}, seed={args.seed})"
    )
    for i, dp in enumerate(data):
        context_text = dp[split_to_use]
        question = dp["question"]
        example_id = dp["id"]
        print(f"[{i + 1}/{len(data)}] id={example_id} ctx_len={dp['context_len']}")

        b_out, b_tok, b_s, b_err = run_baseline(
            base_adapter,
            context_text,
            question,
            max_tokens=baseline_max_tokens,
            temperature=args.temperature,
        )
        m_out, m_tok, m_s, m_err, m_trace = run_rlm_minimal(
            min_root,
            min_sub,
            context_text,
            question,
            max_tokens=agent_max_tokens,
            subcall_tokens=agent_subcall_max_tokens,
            temperature=args.temperature,
            max_iterations=args.minimal_max_iterations,
        )
        p_out, p_tok, p_s, p_err, p_steps = run_pyrlm(
            py_root,
            py_sub,
            context_text,
            question,
            max_tokens=agent_max_tokens,
            subcall_tokens=agent_subcall_max_tokens,
            max_steps=args.rlm_max_steps,
            max_subcalls=args.rlm_max_subcalls,
        )

        engines = [
            ("baseline", b_out, b_tok, b_s, b_err, {}),
            ("rlm_minimal", m_out, m_tok, m_s, m_err, {"events": m_trace}),
            ("pyrlm_runtime", p_out, p_tok, p_s, p_err, {"trace_steps": p_steps}),
        ]
        example_payload = {"id": example_id, "question": question, "results": {}}

        for engine, out, tok, elapsed, err, extra in engines:
            eval_out = score_output(args.dataset, dict(dp), out, args.model)
            row = {
                "id": example_id,
                "engine": engine,
                "score": float(eval_out["score"]),
                "tokens": int(tok),
                "elapsed": float(elapsed),
                "error": err,
                "context_len": int(dp["context_len"]),
                "context_bucket": context_bucket(int(dp["context_len"])),
                "attempted_parse": eval_out["attempted_parse"],
                "gold_answer": eval_out["answer"],
                "parse_confidence": eval_out["parse_confidence"],
            }
            rows.append(row)
            example_payload["results"][engine] = {
                "output": out,
                "tokens": tok,
                "elapsed": elapsed,
                "error": err,
                "score": eval_out["score"],
                "attempted_parse": eval_out["attempted_parse"],
                "gold_answer": eval_out["answer"],
                "parse_confidence": eval_out["parse_confidence"],
                "extra": extra,
            }
        per_example.append(example_payload)

    summary = engine_summary(rows)
    summary_by_bucket = engine_summary_by_bucket(rows)
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "base_url": args.base_url,
        "max_examples": args.max_examples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "min_context_len": args.min_context_len,
        "max_context_len": args.max_context_len,
        "labels": args.labels,
        "max_tokens_legacy_override": args.max_tokens,
        "baseline_max_tokens": baseline_max_tokens,
        "agent_max_tokens": agent_max_tokens,
        "agent_subcall_max_tokens": agent_subcall_max_tokens,
        "temperature": args.temperature,
        "rlm_max_steps": args.rlm_max_steps,
        "rlm_max_subcalls": args.rlm_max_subcalls,
        "minimal_max_iterations": args.minimal_max_iterations,
        "examples_evaluated": len(data),
    }
    validate_result_consistency(rows, per_example, summary, config)

    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_by_context_bucket.json", summary_by_bucket)
    write_json(run_dir / "per_example.json", per_example)
    write_jsonl(run_dir / "flat_results.jsonl", rows)
    write_markdown_summary(run_dir / "summary.md", summary, summary_by_bucket)

    print("\nSummary")
    print("=" * 72)
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in summary:
            continue
        s = summary[engine]
        print(
            f"{engine:14s} score={s['avg_score']:.4f} "
            f"avg_tokens={s['avg_tokens']:.1f} avg_time={s['avg_elapsed']:.2f}s "
            f"errors={s['errors']}/{s['examples']}"
        )
    print("\nBy Context Bucket")
    print("=" * 72)
    bucket_order = ["S(<8k)", "M(8k-32k)", "L(32k-128k)", "XL(>=128k)"]
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in summary_by_bucket:
            continue
        for bucket in bucket_order:
            if bucket not in summary_by_bucket[engine]:
                continue
            s = summary_by_bucket[engine][bucket]
            print(
                f"{engine:14s} {bucket:12s} score={s['avg_score']:.4f} "
                f"avg_tokens={s['avg_tokens']:.1f} avg_time={s['avg_elapsed']:.2f}s "
                f"n={s['examples']} err={s['errors']}"
            )
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
