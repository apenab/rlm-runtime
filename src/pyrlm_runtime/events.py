from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .trace import TraceStep


@dataclass(frozen=True)
class RLMEvent:
    kind: Literal["run_started", "step_completed", "run_finished"]
    query: str | None = None
    step: TraceStep | None = None
    context_metadata: dict[str, Any] | None = None
    repl_backend: str | None = None
    output: str | None = None
    total_steps: int | None = None
    tokens_used: int | None = None
    elapsed: float | None = None


class RLMEventListener(Protocol):
    def handle(self, event: RLMEvent) -> None: ...
