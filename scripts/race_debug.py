"""순위 디버그 — 현재 세션이 무슨 경기인지 + 계산된 순위를 raw로 확인.

실행: python -m scripts.race_debug
- DEFAULT_SESSION_KEY 세션의 메타(나라·세션종류·연도)를 OpenF1에서 조회
- get_race_status와 동일 로직으로 Top5 순위를 계산해 출력
→ 9523이 모나코가 맞는지 / VER가 실제 1등인지 판별용.
"""
from __future__ import annotations

import asyncio

import httpx

from app.config import settings
from app.data import openf1


async def main() -> None:
    sk = settings.default_session_key

    # 1) 이 세션이 무슨 경기인지 (OpenF1 직결로 메타 조회)
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{settings.openf1_base}/sessions", params={"session_key": sk})
            r.raise_for_status()
            meta = r.json()
        if meta:
            m = meta[0]
            print(f"[세션 {sk}] {m.get('year')} {m.get('country_name')} "
                  f"{m.get('circuit_short_name')} · {m.get('session_name')}")
        else:
            print(f"[세션 {sk}] 메타 없음")
    except Exception as e:
        print("세션 메타 조회 실패:", e)

    # 2) get_race_status와 동일하게 순위 계산 (cutoff 없음 = 최종)
    positions = await openf1.get_positions(sk)
    drivers = await openf1.get_drivers(sk)
    name = {d.get("driver_number"): d.get("name_acronym") for d in drivers}

    latest: dict[int, tuple] = {}
    for p in positions:
        dn, pos, d = p.get("driver_number"), p.get("position"), p.get("date")
        if dn is None or pos is None:
            continue
        if dn not in latest or (d or "") >= (latest[dn][0] or ""):
            latest[dn] = (d, pos)

    top = sorted(latest.items(), key=lambda kv: kv[1][1])[:5]
    print("\nTop 5 (계산된 최종 순위):")
    for dn, (d, pos) in top:
        print(f"  P{pos}  #{dn} {name.get(dn)}   (최신기록 {d})")
    print(f"\nposition 레코드 {len(positions)}개 · 드라이버 {len(drivers)}명")


if __name__ == "__main__":
    asyncio.run(main())
