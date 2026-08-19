"""경량 관측(observability) — 파이프라인 단계 지연 + 툴콜 기록.

목적: 실시간 에이전트가 한 발화를 처리할 때 '어느 단계에서 시간이 얼마나 걸렸는지',
'어떤 도구를 어떤 인자로 불렀는지'를 계측한다. 지연 벤치마크(scripts/bench_latency.py)와
에이전트 평가(scripts/eval_agent.py)가 이 트레이스를 읽는다.

설계 원칙:
  - **동작을 바꾸지 않는다.** 기록만 한다(측정용). 트레이스가 없으면 모든 호출은 no-op.
  - **contextvar** 기반이라 동시 요청/asyncio 태스크에서 트레이스가 서로 안 섞인다.
  - 운영(main.py/WS)에서는 트레이스를 시작하지 않으므로 current()==None → 오버헤드 0.

사용 예:
    import app.obs as obs
    obs.start_trace()
    reply, cmds, ok = await run_agent("DRS가 뭐야?")
    tr = obs.current()
    print(tr.timings, tr.tool_calls)
"""
from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "obs_trace", default=None
)


@dataclass
class Trace:
    """한 발화 처리 동안의 계측 결과."""

    path: str | None = None                       # "planner" | "rule_router" | "react"
    tool_calls: list[dict] = field(default_factory=list)  # [{"name":.., "args":..}]
    timings: dict[str, float] = field(default_factory=dict)  # 단계명 -> 누적 초
    reply_chars: int = 0

    def add_tool_call(self, name: str | None, args: dict | None = None) -> None:
        if name:
            self.tool_calls.append({"name": name, "args": args or {}})

    def mark(self, stage: str, seconds: float) -> None:
        self.timings[stage] = self.timings.get(stage, 0.0) + float(seconds)

    @property
    def tool_names(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


def start_trace() -> Trace:
    """새 트레이스를 시작해 현재 컨텍스트에 건다. 반환값으로 직접 읽어도 된다."""
    trace = Trace()
    _current.set(trace)
    return trace


def current() -> Trace | None:
    """현재 컨텍스트의 트레이스(없으면 None)."""
    return _current.get()


def record_path(path: str) -> None:
    trace = _current.get()
    if trace is not None:
        trace.path = path


def record_tool_call(name: str | None, args: dict | None = None) -> None:
    trace = _current.get()
    if trace is not None:
        trace.add_tool_call(name, args)


def mark(stage: str, seconds: float) -> None:
    trace = _current.get()
    if trace is not None:
        trace.mark(stage, seconds)


class stage:
    """with obs.stage('agent_llm'): ...  — 블록 실행시간을 트레이스에 누적 기록.

    트레이스가 없으면 그냥 실행만 하고 아무것도 기록하지 않는다.
    """

    __slots__ = ("name", "_t0")

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "stage":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        mark(self.name, time.perf_counter() - self._t0)
