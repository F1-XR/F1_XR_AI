"""요청 컨텍스트 — 현재 보고 있는 세션/리플레이 시각.

commands.py와 같은 방식: contextvar에 dict를 담고 그 '내용을 in-place mutate'한다.
→ 요청별 격리(다중 사용자 안전) + find_session의 세션 변경이 langgraph 도구
  태스크(컨텍스트 복사본)에서도 전파됨 — 같은 dict 객체를 공유하므로 mutate가 보인다.
  (재바인딩 `.set`은 자식 태스크로 전파되지 않지만, dict 내용 mutate는 전파된다.)
"""
from __future__ import annotations

import contextvars

from ..config import settings

_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("req_ctx")


def _state() -> dict:
    """이 요청(태스크)의 상태 dict. 없으면 기본값으로 생성."""
    try:
        return _ctx.get()
    except LookupError:
        d = {"session": settings.default_session_key, "at_time": None}
        _ctx.set(d)
        return d


def set_context(session_key: int | None, at_time: str | None) -> None:
    """요청 시작 시 세션/시각 설정. session_key None(예: CLI)이면 기존 세션 유지."""
    cur = _state()
    session = session_key if session_key is not None else cur["session"]
    _ctx.set({"session": session, "at_time": at_time})


def set_session(session_key: int) -> None:
    """find_session이 경기를 전환할 때 호출 — dict를 in-place mutate(자식 태스크 전파 목적)."""
    _state()["session"] = session_key


def current_session() -> int:
    return _state()["session"]


def current_time() -> str | None:
    return _state()["at_time"]
