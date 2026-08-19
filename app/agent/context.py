"""요청 컨텍스트 — 현재 보고 있는 세션/리플레이 시각.

commands.py와 같은 방식: contextvar에 dict를 담고 그 '내용을 in-place mutate'한다.
→ 요청별 격리(다중 사용자 안전) + find_session의 세션 변경이 langgraph 도구
  태스크(컨텍스트 복사본)에서도 전파됨 — 같은 dict 객체를 공유하므로 mutate가 보인다.
  (재바인딩 `.set`은 자식 태스크로 전파되지 않지만, dict 내용 mutate는 전파된다.)
"""
from __future__ import annotations

import contextvars
import time
from threading import Lock

from ..config import settings

_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("req_ctx")
_recent_overtake_lock = Lock()
_recent_overtake: dict | None = None


def _state() -> dict:
    """이 요청(태스크)의 상태 dict. 없으면 기본값으로 생성."""
    try:
        return _ctx.get()
    except LookupError:
        d = {"session": settings.default_session_key, "at_time": None, "selected_driver": None}
        _ctx.set(d)
        return d


def set_context(
    session_key: int | None,
    at_time: str | None,
    selected_driver: int | None = None,
) -> None:
    """요청 시작 시 세션/시각/선택대상 설정. session_key None(예: CLI)이면 기존 세션 유지.
    selected_driver: 사용자가 XR Ray/클릭으로 지목한 차량 번호("이 선수"의 대상)."""
    cur = _state()
    session = session_key if session_key is not None else cur["session"]
    _ctx.set({"session": session, "at_time": at_time, "selected_driver": selected_driver})


def set_session(session_key: int) -> None:
    """find_session이 경기를 전환할 때 호출 — dict를 in-place mutate(자식 태스크 전파 목적)."""
    _state()["session"] = session_key


def current_session() -> int:
    return _state()["session"]


def current_time() -> str | None:
    return _state()["at_time"]


def current_selected() -> int | None:
    """사용자가 지목한 차량 번호("이 선수"의 대상). 없으면 None."""
    return _state().get("selected_driver")


def set_recent_overtake(event: dict | None) -> None:
    """최근 watcher가 안내한 추월 이벤트를 저장한다."""
    global _recent_overtake
    with _recent_overtake_lock:
        _recent_overtake = dict(event) if event else None


def current_recent_overtake() -> dict | None:
    """최근 watcher 추월 이벤트. 없으면 None."""
    with _recent_overtake_lock:
        return dict(_recent_overtake) if _recent_overtake else None


# ── 드라이버별 '최신 추월확률' 캐시 (watcher가 채우고, explain_situation이 즉시 읽음) ──
_driver_prob_lock = Lock()
# (session_key, driver_number) -> (monotonic_ts, prob)
# 세션별로 분리해 다른 경기의 캐시가 섞이지 않도록 한다.
_driver_probs: dict[tuple[int, int], tuple[float, float]] = {}


def _prob_key(session_key: int | None, driver_number: int) -> tuple[int, int]:
    # session_key가 None이면 -1로 정규화(기본 세션과 명시 세션을 구분).
    return (int(session_key) if session_key is not None else -1, int(driver_number))


def set_driver_prob(session_key: int | None, driver_number: int, prob: float | None) -> None:
    """watcher가 후보 드라이버의 추월확률을 계산할 때마다 (세션·드라이버별) 최신값을 저장한다."""
    if prob is None:
        return
    with _driver_prob_lock:
        _driver_probs[_prob_key(session_key, driver_number)] = (time.monotonic(), float(prob))


def get_driver_prob(session_key: int | None, driver_number: int,
                    max_age_sec: float = 20.0) -> float | None:
    """최근(max_age_sec 이내) watcher가 이 세션에서 계산한 추월확률. 없거나 오래됐으면 None(→ 직접 계산)."""
    with _driver_prob_lock:
        v = _driver_probs.get(_prob_key(session_key, driver_number))
    if not v:
        return None
    ts, prob = v
    return prob if (time.monotonic() - ts) <= max_age_sec else None
