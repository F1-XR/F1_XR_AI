"""Jolpica-F1 클라이언트 — 커리어·통산 기록(Ergast 후계 API).

OpenF1이 '지금 이 경기'만 준다면, Jolpica는 통산 우승·챔피언·생년월일 등
'시즌 무관 고정 경력'을 준다. 무료·키 불필요, Ergast 호환.
"""
from __future__ import annotations

import httpx

from ..config import settings

_TIMEOUT = httpx.Timeout(10.0)


async def get_driver_career(driver_id: str) -> dict | None:
    """드라이버 프로필 조회. driver_id 예: 'max_verstappen', 'hamilton'.

    반환: {givenName, familyName, nationality, dateOfBirth, ...}
    """
    url = f"{settings.jolpica_base}/drivers/{driver_id}.json"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        table = r.json().get("MRData", {}).get("DriverTable", {}).get("Drivers", [])
        return table[0] if table else None


async def get_driver_champrionships(driver_id: str) -> int:
    """통산 월드 챔피언 횟수(우승 시즌 수)."""
    url = f"{settings.jolpica_base}/drivers/{driver_id}/driverStandings/1.json"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        lists = r.json().get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        return len(lists)
