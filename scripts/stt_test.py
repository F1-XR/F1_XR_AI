"""STT 어댑터 단독 테스트 — wav 파일 → 텍스트 (서버·Unity 없이).

실행: python -m scripts.stt_test [파일.wav]     (기본값: tts_out.wav)
- config 의 STT_PROVIDER(기본 whisper)로 wav를 인식해 텍스트를 출력한다.
- tts_test 로 만든 tts_out.wav 를 넣으면 'TTS→STT 왕복' 확인이 된다.

준비(최초, Python 3.11 권장): pip install faster-whisper
"""
from __future__ import annotations

import asyncio
import sys

from app.config import settings
from app.voice import stt


async def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "tts_out.wav"
    print(f"STT_PROVIDER={settings.stt_provider!r} · 모델={settings.stt_model!r} · 파일={path}")
    print("인식 중… (첫 실행은 모델 다운로드로 느릴 수 있어요)")

    try:
        with open(path, "rb") as f:
            audio = f.read()
    except FileNotFoundError:
        print(f"❌ 파일 없음: {path} — 먼저 `python -m scripts.tts_test`로 tts_out.wav를 만들거나 wav 경로를 인자로 주세요.")
        return

    try:
        text = await stt.transcribe(audio, language="ko")
    except Exception as exc:
        print(f"❌ 실패: {type(exc).__name__}: {exc}")
        print("→ faster-whisper 미설치면: pip install faster-whisper")
        return

    print(f"✅ 인식 결과: {text!r}")


if __name__ == "__main__":
    asyncio.run(main())
