"""에이전트 원인 진단 — 로깅을 켜고 run_agent를 직접 실행해 '진짜 에러'(traceback)를 노출한다.

평소 run_agent는 오류가 나도 사용자에겐 "죄송해요…"만 보이고 상세 원인은 로그로만 남는다.
이 스크립트는 로깅을 INFO로 켜서 그 숨은 traceback을 화면에 찍는다.

실행: python -m scripts.agent_debug ["질문"]     (기본: 'DRS가 뭐야?')
"""
from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.agent.graph import run_agent  # noqa: E402 (로깅 설정 후 import)


async def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "DRS가 뭐야?"
    print(f"질문: {q!r}\n--- 실행(에러가 나면 아래에 traceback이 찍힘) ---")
    reply, commands, _ = await run_agent(q)
    print("\n=== 최종 응답 ===")
    print(reply)
    print("Unity 명령:", commands)


if __name__ == "__main__":
    asyncio.run(main())
