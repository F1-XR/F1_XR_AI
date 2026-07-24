"""연결 점검 스크립트 — OpenF1·Jolpica 실제 호출이 되는지 확인.

실행 (레포 루트에서):
    pip install -r requirements.txt
    python -m scripts.smoke_test

성공하면 선수 이름/팀, 이벤트 개수, 커리어가 출력된다.
"""
from __future__ import annotations

import asyncio

from app.data import openf1, jolpica
from app.config import settings


async def main() -> None:
    sk = settings.default_session_key
    print(f"[세션 {sk}] OpenF1·Jolpica 점검\n")

    ok = True
    try:
        d = await openf1.get_driver(sk, 44)
        print("✅ OpenF1 get_driver(44):", d and (d["full_name"], d["team_name"]))
    except Exception as e:
        ok = False
        print("❌ OpenF1 get_driver:", type(e).__name__, e)

    try:
        rc = await openf1.get_race_control(sk)
        print(f"✅ OpenF1 race_control: {len(rc)}건", "| 예:", rc[0]["message"] if rc else None)
    except Exception as e:
        ok = False
        print("❌ OpenF1 race_control:", type(e).__name__, e)

    try:
        c = await jolpica.get_driver_career("hamilton")
        print("✅ Jolpica career:", c and (c["givenName"], c["familyName"], c["nationality"]))
    except Exception as e:
        ok = False
        print("❌ Jolpica career:", type(e).__name__, e)

    print("\n결과:", "전부 성공 🎉" if ok else "일부 실패 — 위 오류 확인")


if __name__ == "__main__":
    asyncio.run(main())
