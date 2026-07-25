"""경기 데이터(선수/깃발/순위/갭/피트/타이어) 조회 클라이언트.

데이터 소유는 F1_XR_Server가 전담한다(캐시 + OpenF1 게이트웨이). 그래서 기본 경로는
'서버 경유'다. settings.f1_server_url 이 있으면 서버의 /f1/... 엔드포인트를 호출하고,
비어 있으면 OpenF1 직결(레거시 폴백)로 동작한다.

- 서버 경유:  GET {f1_server_url}/f1/{session_key}/{resource}?driver_number=
              GET {f1_server_url}/f1/sessions?year=&country=&circuit=&session_name=
- 직결:       GET {openf1_base}/{resource}?session_key=&driver_number=
서버가 OpenF1과 동일한 원본 shape를 그대로 돌려주므로 상위(도구) 코드는 경로와 무관하다.
"""
from __future__ import annotations

import httpx

from ..config import settings

_TIMEOUT = httpx.Timeout(10.0)


def _server_base() -> str | None:
    return settings.f1_server_url.rstrip("/") if settings.f1_server_url else None


async def _get_resource(
    session_key: int,
    resource: str,
    driver_number: int | None = None,
) -> list[dict]:
    """세션 단위 리소스 조회. 서버 경유가 기본, 없으면 OpenF1 직결."""
    base = _server_base()
    if base:
        url = f"{base}/f1/{session_key}/{resource}"
        params = {} if driver_number is None else {"driver_number": driver_number}
    else:
        url = f"{settings.openf1_base}/{resource}"
        params = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def get_driver(session_key: int, driver_number: int) -> dict | None:
    """선수 기본 정보(이름·팀·번호·사진·국적). 세션별이라 이적이 자동 반영된다."""
    rows = await _get_resource(session_key, "drivers", driver_number)
    return rows[0] if rows else None


async def get_race_control(session_key: int) -> list[dict]:
    """깃발·세이프티카·인시던트 등 경기 상황 이벤트 원본."""
    return await _get_resource(session_key, "race_control")


async def get_positions(session_key: int) -> list[dict]:
    """순위 변동 기록."""
    return await _get_resource(session_key, "position")


async def get_intervals(session_key: int, driver_number: int | None = None) -> list[dict]:
    """리더와의 갭·앞차와의 간격(추월 임박 판단용)."""
    return await _get_resource(session_key, "intervals", driver_number)


async def get_pit(session_key: int, driver_number: int | None = None) -> list[dict]:
    """피트인 타이밍·소요시간."""
    return await _get_resource(session_key, "pit", driver_number)


async def get_stints(session_key: int, driver_number: int | None = None) -> list[dict]:
    """타이어 스틴트(컴파운드·랩 수) — '왜 피트인?' 추론 근거."""
    return await _get_resource(session_key, "stints", driver_number)


async def get_laps(session_key: int, driver_number: int | None = None) -> list[dict]:
    """랩타임·섹터."""
    return await _get_resource(session_key, "laps", driver_number)


async def find_sessions(
    year: int | None = None,
    country: str | None = None,
    circuit: str | None = None,
    session_name: str | None = None,
) -> list[dict]:
    """경기(세션) 검색 — 연도·국가·서킷·세션종류로 session_key 를 찾는다.

    반환 항목: session_key · session_name · country_name · circuit_short_name · year · date_start
    """
    base = _server_base()
    if base:
        url = f"{base}/f1/sessions"
        params = {
            k: v for k, v in {
                "year": year,
                "country": country,
                "circuit": circuit,
                "session_name": session_name,
            }.items() if v is not None
        }
    else:
        url = f"{settings.openf1_base}/sessions"
        params = {
            k: v for k, v in {
                "year": year,
                "country_name": country,
                "circuit_short_name": circuit,
                "session_name": session_name,
            }.items() if v is not None
        }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()
