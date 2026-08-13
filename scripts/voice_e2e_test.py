"""음성 파이프라인 end-to-end 단독 테스트 (서버·WS·Unity 없이).

흐름: 입력 wav → ①STT → ②에이전트 → ③TTS → reply_out.wav
실행: python -m scripts.voice_e2e_test [입력.wav]     (기본값: tts_out.wav)

준비:
  - .env 에 OPENAI_API_KEY (에이전트용)
  - venv-voice 에 STT(faster-whisper)·TTS(현재 provider) 설치
  - 입력 wav 없으면 먼저: python -m scripts.tts_test  (tts_out.wav 생성)

이 스크립트로 "말 → 인식 → 답변 → 음성"이 한 번에 도는지 확인한다.
"""
from __future__ import annotations

import asyncio
import sys

from app.agent.graph import run_agent
from app.config import settings
from app.voice import stt, tts


async def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "tts_out.wav"
    print(f"입력 wav = {path}")
    print(f"STT={settings.stt_provider!r} · LLM={settings.llm_model!r} · TTS={settings.tts_provider!r}\n")

    try:
        with open(path, "rb") as f:
            audio = f.read()
    except FileNotFoundError:
        print(f"❌ 입력 파일 없음: {path}")
        print("→ 먼저 `python -m scripts.tts_test` 로 tts_out.wav 를 만들거나 wav 경로를 인자로 주세요.")
        return

    # ① STT
    print("① STT 인식 중… (첫 실행은 모델 다운로드로 느릴 수 있어요)")
    try:
        text = await stt.transcribe(audio, language="ko")
    except Exception as exc:
        print(f"❌ STT 실패: {type(exc).__name__}: {exc}")
        return
    print(f"   → 인식 결과: {text!r}")
    if not text:
        print("❌ 인식 결과가 비었어요 (입력 음성이 무음/잡음일 수 있어요).")
        return

    # ② 에이전트
    print("\n② 에이전트 응답 생성 중…")
    try:
        reply, commands, _ = await run_agent(text=text, session_key=None, at_time=None, history=[])
    except Exception as exc:
        print(f"❌ 에이전트 실패: {type(exc).__name__}: {exc}")
        print("→ .env 의 OPENAI_API_KEY / LLM_MODEL 을 확인하세요.")
        return
    print(f"   → 응답: {reply!r}")
    if commands:
        print(f"   → Unity 명령 {len(commands)}개: {[c.get('name') for c in commands]}")

    # ③ TTS
    print("\n③ TTS 합성 중…")
    try:
        out_audio = await tts.synthesize(reply)
    except Exception as exc:
        print(f"❌ TTS 실패: {type(exc).__name__}: {exc}")
        return
    out = "reply_out.wav"
    with open(out, "wb") as f:
        f.write(out_audio)
    print(f"\n✅ 파이프라인 완료: {out} ({len(out_audio):,} bytes) — 열어서 들어보세요.")
    print("   (입력 tts_out.wav = 질문, 출력 reply_out.wav = 답변)")


if __name__ == "__main__":
    asyncio.run(main())
