from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict

from .env import ExecResult

try:
    from pydantic_monty import (
        Monty,
        MontyComplete,  # noqa: F401
        MontyError,
        MontyFutureSnapshot,  # noqa: F401
        MontyRuntimeError,
        MontySnapshot,  # noqa: F401
        MontySyntaxError,
    )

    MONTY_AVAILABLE = True
except ImportError:
    MONTY_AVAILABLE = False


@dataclass(frozen=True)
class MontyLimits:
    """Resource limits for Monty REPL execution."""

    max_duration_secs: float = 5.0
    max_memory: int = 128 * 1024 * 1024  # 128 MB
    max_allocations: int = 1_000_000
    max_recursion_depth: int = 100

    def to_dict(self) -> dict:
        return {
            "max_duration_secs": self.max_duration_secs,
            "max_memory": self.max_memory,
            "max_allocations": self.max_allocations,
            "max_recursion_depth": self.max_recursion_depth,
        }


def _find_assigned_names(code: str) -> list[str]:
    """Extract top-level variable names assigned in the code using AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.iter_child_nodes(tree):
        target = _extract_target(node)
        if target is not None:
            names.append(target)
    return names


def _extract_target(node: ast.AST) -> str | None:
    """Return the assigned variable name from an AST node, if any."""
    if isinstance(node, ast.Assign):
        first = node.targets[0]
        if isinstance(first, ast.Name):
            return first.id
    elif isinstance(node, (ast.AugAssign, ast.For)):
        if isinstance(node.target, ast.Name):
            return node.target.id
    return None


class MontyREPL:
    """Sandboxed REPL backed by pydantic-monty (Rust interpreter).

    Exposes the same interface as PythonREPL so they are interchangeable.
    Each exec() call creates a fresh Monty instance with the current
    variables as inputs and registered callables as external functions.

    Variable persistence across exec() calls is achieved by:
    1. Passing current variables as Monty inputs
    2. Detecting assignments via AST and appending a capture dict
    3. Extracting assigned values from the capture dict result
    """

    def __init__(
        self,
        *,
        stdout_limit: int = 4000,
        limits: MontyLimits | None = None,
    ) -> None:
        if not MONTY_AVAILABLE:
            raise ImportError(
                "pydantic-monty is not installed. "
                "Install it with: pip install pydantic-monty"
            )
        self._stdout_limit = stdout_limit
        self._limits = limits or MontyLimits()
        self._variables: Dict[str, Any] = {}
        self._external_fns: Dict[str, Callable] = {}

    def set(self, name: str, value: Any) -> None:
        if callable(value) and not isinstance(value, (str, int, float, bool, list, dict, tuple)):
            self._external_fns[name] = value
        else:
            self._variables[name] = value

    def get(self, name: str) -> Any:
        return self._variables.get(name)

    def exec(self, code: str) -> ExecResult:
        if not code.strip():
            return ExecResult(stdout="", error=None)

        assigned = _find_assigned_names(code)
        augmented_code = self._augment_code(code, assigned)

        stdout_parts: list[str] = []

        def print_callback(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        try:
            result_value = self._run_monty(augmented_code, print_callback)
            self._capture_variables(assigned, result_value)
        except MontySyntaxError as exc:
            return ExecResult(stdout="", error=f"SyntaxError: {exc}")
        except MontyRuntimeError as exc:
            return ExecResult(
                stdout=self._truncate("".join(stdout_parts)),
                error=exc.display("type-msg"),
            )
        except MontyError as exc:
            return ExecResult(stdout="", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return ExecResult(stdout="", error=f"{type(exc).__name__}: {exc}")

        return ExecResult(stdout=self._truncate("".join(stdout_parts)), error=None)

    def _augment_code(self, code: str, assigned: list[str]) -> str:
        """Append a capture dict expression to extract assigned variable values."""
        if not assigned:
            return code
        capture_expr = "{" + ", ".join(f'"{n}": {n}' for n in assigned) + "}"
        return code.rstrip() + "\n" + capture_expr

    def _run_monty(self, code: str, print_callback: Callable) -> Any:
        """Create a Monty instance and execute the code."""
        input_names = list(self._variables.keys())
        ext_fn_names = list(self._external_fns.keys())

        monty = Monty(
            code,
            inputs=input_names if input_names else None,
            external_functions=ext_fn_names if ext_fn_names else None,
        )

        run_kwargs: dict[str, Any] = {
            "inputs": dict(self._variables) if input_names else None,
            "limits": self._limits.to_dict(),
            "print_callback": print_callback,
        }
        if ext_fn_names:
            run_kwargs["external_functions"] = dict(self._external_fns)

        return monty.run(**run_kwargs)

    def _capture_variables(self, assigned: list[str], result_value: Any) -> None:
        """Extract assigned variable values from the Monty result dict."""
        if not assigned or not isinstance(result_value, dict):
            return
        for var_name in assigned:
            if var_name in result_value:
                self._variables[var_name] = result_value[var_name]

    def _truncate(self, text: str) -> str:
        if len(text) <= self._stdout_limit:
            return text
        return text[: self._stdout_limit] + "...<truncated>"
