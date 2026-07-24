"""텍스트 전용 테스트 러너 (Day 1~6).

음성/Unity 없이 터미널에서 에이전트를 검증한다.
    python -m app.cli
"""
from __future__ import annotations

import asyncio

from .agent.graph import run_agent


async def main() -> None:
    print("F1 튜토리얼 에이전트 (텍스트 모드). 'quit'으로 종료.\n")
    history: list = []
    while True:
        try:
            text = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue

        reply, commands = await run_agent(text, history=history)
        history += [("user", text), ("assistant", reply)]

        for cmd in commands:
            print(f"  [Unity 명령] {cmd['name']} {cmd['args']}")
        print(f"에이전트: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
