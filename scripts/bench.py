"""파이프라인 단계별 시간 측정 (LLM vs TTS 분리).

실행: python -m scripts.bench ["질문"]     (기본: 'DRS가 뭐야?')
- STT는 빼고 텍스트 질문으로 ①LLM(에이전트) ②TTS 합성 시간을 각각 잰다.
- 서버 없이 단독 실행. 두 번 돌리면 '모델 로드' 영향이 빠진 실제 속도를 볼 수 있다.
"""
from __future__ import annotations

import asyncio
import sys
import time

from app.config import settings
from app.agent.graph import run_agent
from app.voice import tts


async def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "DRS가 뭐야?"
    print(f"질문: {q!r}")
    print(f"LLM={settings.llm_model!r} · TTS={settings.tts_provider!r} · TTS_ENABLED={settings.tts_enabled}\n")

    # ① 에이전트(LLM)
    t0 = time.perf_counter()
    reply, cmds, _ = await run_agent(q)
    t1 = time.perf_counter()
    print(f"① LLM(에이전트) : {t1 - t0:6.1f}s   (응답 {len(reply)}자, 명령 {len(cmds)}개)")

    # ② TTS 합성
    t2 = time.perf_counter()
    audio = await tts.synthesize(reply)
    t3 = time.perf_counter()
    print(f"② TTS 합성      : {t3 - t2:6.1f}s   ({len(audio):,} bytes)")

    print(f"───────────────────────────")
    print(f"합계(STT 제외)  : {t3 - t0:6.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
