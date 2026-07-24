"""요청 컨텍스트 — 현재 보고 있는 세션/리플레이 시각.

사용자가 "지금 상황?"이라고 물을 때의 '지금'은 Unity 리플레이의 현재 시각이다.
WS 메시지로 넘어온 session_key/at_time을 요청 동안 여기 담아두면,
도구들은 매번 인자로 받지 않고도 현재 맥락을 읽을 수 있다.
"""
from __future__ import annotations

from ..config import settings

# find_session 도구가 대화 중 세션(경기)을 바꿀 수 있어야 한다. langgraph는 도구를
# 태스크 컨텍스트 복사본에서 실행할 수 있어 contextvar 재바인딩이 전파되지 않는다.
# 그래서 모듈 전역 dict 를 in-place 로 mutate 하는 방식으로 둔다.
# (단일 사용자 데모 기준. 다중 사용자 서버로 확장 시 연결별 상태로 분리 필요.)
_STATE: dict = {"session": settings.default_session_key, "at_time": None}


def set_context(session_key: int | None, at_time: str | None) -> None:
    """요청 시작 시 세션/시각 설정. session_key 가 None(예: CLI)이면 기존 값 유지."""
    if session_key is not None:
        _STATE["session"] = session_key
    _STATE["at_time"] = at_time


def set_session(session_key: int) -> None:
    """find_session 이 경기를 전환할 때 호출."""
    _STATE["session"] = session_key


def current_session() -> int:
    return _STATE["session"]


def current_time():
    return _STATE["at_time"]
