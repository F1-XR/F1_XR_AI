"""OpenF1 클라이언트 — 경기 데이터(선수/깃발/순위/갭/피트/타이어) 조회.

에이전트는 데이터를 미리 다 들고 있지 않고, 도구가 필요할 때 여기 함수를 호출해
그 순간 필요한 조각만 가져온다. 모든 함수는 비동기(httpx.AsyncClient).
"""
from __future__ import annotations

import httpx

from ..config import settings

_TIMEOUT = httpx.Timeout(10.0)


async def _get(path: str, **params) -> list[dict]:
    """OpenF1 GET 헬퍼. 빈 값 파라미터는 제거."""
    params = {k: v for k, v in params.items() if v is not None}
    url = f"{settings.openf1_base}/{path}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def get_driver(session_key: int, driver_number: int) -> dict | None:
    """선수 기본 정보(이름·팀·번호·사진·국적). 세션별이라 이적이 자동 반영된다."""
    rows = await _get("drivers", session_key=session_key, driver_number=driver_number)
    return rows[0] if rows else None


async def get_race_control(session_key: int) -> list[dict]:
    """깃발·세이프티카·인시던트 등 경기 상황 이벤트 원본."""
    return await _get("race_control", session_key=session_key)


async def get_positions(session_key: int) -> list[dict]:
    """순위 변동 기록."""
    return await _get("position", session_key=session_key)


async def get_intervals(session_key: int, driver_number: int | None = None) -> list[dict]:
    """리더와의 갭·앞차와의 간격(추월 임박 판단용)."""
    return await _get("intervals", session_key=session_key, driver_number=driver_number)


async def get_pit(session_key: int, driver_number: int | None = None) -> list[dict]:
    """피트인 타이밍·소요시간."""
    return await _get("pit", session_key=session_key, driver_number=driver_number)


async def get_stints(session_key: int, driver_number: int | None = None) -> list[dict]:
    """타이어 스틴트(컴파운드·랩 수) — '왜 피트인?' 추론 근거."""
    return await _get("stints", session_key=session_key, driver_number=driver_number)


async def get_laps(session_key: int, driver_number: int | None = None) -> list[dict]:
    """랩타임·섹터."""
    return await _get("laps", session_key=session_key, driver_number=driver_number)


async def find_sessions(
    year: int | None = None,
    country: str | None = None,
    circuit: str | None = None,
    session_name: str | None = None,
) -> list[dict]:
    """경기(세션) 검색 — 연도·국가·서킷·세션종류로 session_key 를 찾는다.

    반환 항목: session_key · session_name · country_name · circuit_short_name · year · date_start
    """
    return await _get(
        "sessions",
        year=year,
        country_name=country,
        circuit_short_name=circuit,
        session_name=session_name,
    )
