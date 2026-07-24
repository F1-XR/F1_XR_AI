"""능동형 pointOutMoment — 서버 백그라운드 감시 루프 (P2 확장).

7개 도구와 달리 사용자 발화로 발동하지 않는다. 리플레이 데이터를 지켜보다
중요한 순간(추월 임박·순위 변동·깃발)을 스스로 감지해, Unity로
highlightDriver + "지금 여기 보세요" 명령을 push한다.

지금은 스캐폴드 stub. P2 확장 단계에서 구현.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from ..data import openf1


def significance_score(intervals: list[dict], positions: list[dict], race_control: list[dict]) -> float:
    """중요도 점수 (0~1). TODO: 갭<1초·순위 변동·이벤트를 가중합."""
    return 0.0


async def watch(session_key: int, push: Callable[[dict], Awaitable[None]], threshold: float = 0.7) -> None:
    """감시 루프. push: WS로 명령을 보내는 콜백.

    TODO(P2): 주기적으로 데이터 폴링 → significance_score → 임계 초과 시
    push({"type":"command","name":"pointOut","args":{...}}).
    """
    raise NotImplementedError("pointOutMoment 감시 루프 미구현 — P2 확장")
