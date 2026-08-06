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

import time

import httpx

from ..config import settings

_TIMEOUT = httpx.Timeout(30.0)   # 전체 세션 조회(intervals 등)가 느릴 수 있어 여유있게

# 세션-전체(드라이버 무관) 리소스 짧은 캐시.
# 능동 안내(watcher)는 매 틱 후보 드라이버마다 build_features를 도는데, 그 안에서
# get_positions·get_weather 같은 '세션 전체' 조회를 드라이버 수만큼 반복한다(중복 HTTP → ReadTimeout).
# 이 데이터는 세션 단위 과거 기록이라 값이 안 바뀌므로(파이썬에서 cutoff로 걸러 씀) 캐시가 안전하다.
# 드라이버 지정 조회·시간창 조회는 캐시하지 않는다(개별성이 커서 이득이 작음).
_CACHE_TTL = 10.0
_cache: dict[tuple[int, str], tuple[float, list[dict]]] = {}


def _server_base() -> str | None:
    return settings.f1_server_url.rstrip("/") if settings.f1_server_url else None


async def _get_resource(
    session_key: int,
    resource: str,
    driver_number: int | None = None,
) -> list[dict]:
    """세션 단위 리소스 조회. 서버 경유가 기본, 없으면 OpenF1 직결.
    드라이버 무관 조회(driver_number=None)는 짧은 TTL로 캐시해 중복 HTTP를 막는다."""
    cache_key = (session_key, resource) if driver_number is None else None
    if cache_key is not None:
        hit = _cache.get(cache_key)
        if hit is not None and (time.monotonic() - hit[0]) < _CACHE_TTL:
            return hit[1]

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
        data = r.json()

    if cache_key is not None:
        _cache[cache_key] = (time.monotonic(), data)
    return data


async def get_driver(session_key: int, driver_number: int) -> dict | None:
    """선수 기본 정보(이름·팀·번호·사진·국적). 세션별이라 이적이 자동 반영된다."""
    rows = await _get_resource(session_key, "drivers", driver_number)
    return rows[0] if rows else None


async def get_drivers(session_key: int) -> list[dict]:
    """세션의 전체 드라이버 목록(번호→이름 매핑 등에 사용)."""
    return await _get_resource(session_key, "drivers")


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


async def get_weather(session_key: int) -> list[dict]:
    """세션 날씨(공기·트랙 온도·습도·강수) — 추월 예측 모델 피처용. 세션 단위 가벼운 조회."""
    return await _get_resource(session_key, "weather")


async def get_car_data_window(
    session_key: int, driver_number: int, start_iso: str, end_iso: str
) -> list[dict]:
    """시점 근처 car_data(speed·drs) 창 — 추월 예측의 speed·drs_active·speed_delta 피처용.
    car_data 세션 전체는 초대용량이라 반드시 드라이버 1명 + 짧은 시간창으로만 조회한다.
    실패(미가용)하면 조용히 빈 리스트 → 해당 피처는 결측(-1.0) 폴백."""
    base = _server_base()
    if base:
        url = f"{base}/f1/{session_key}/car_data"
        params = {"driver_number": driver_number, "start": start_iso, "end": end_iso}
    else:
        url = f"{settings.openf1_base}/car_data"
        params = {
            "session_key": session_key,
            "driver_number": driver_number,
            "date>=": start_iso,
            "date<": end_iso,
        }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return []


async def get_location_window(
    session_key: int, driver_number: int, start_iso: str, end_iso: str
) -> list[dict]:
    """위치 좌표(x·y) 창 — track_progress 피처용(트랙 기준선/시점 좌표).
    location도 세션 전체가 초대용량이라 드라이버+시간창으로만 조회. 실패 시 빈 리스트."""
    base = _server_base()
    if base:
        url = f"{base}/f1/{session_key}/location"
        params = {"driver_number": driver_number, "start": start_iso, "end": end_iso}
    else:
        url = f"{settings.openf1_base}/location"
        params = {
            "session_key": session_key,
            "driver_number": driver_number,
            "date>=": start_iso,
            "date<": end_iso,
        }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return []


async def get_career(session_key: int, driver_number: int) -> dict:
    """선수 통산 기록(생년월일·국적·통산우승). 데이터 서버가 Jolpica를 캐시해 제공한다.

    반환: {jolpicaId, dateOfBirth, nationality, wins} (없으면 빈 dict).
    서버 경유 전용 — f1_server_url 이 없으면(직결 폴백) 커리어는 생략한다.
    """
    base = _server_base()
    if not base:
        return {}
    url = f"{base}/f1/{session_key}/career/{driver_number}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


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
