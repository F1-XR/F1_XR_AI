"""TTS 어댑터 단독 테스트 — 텍스트 → wav 파일 (서버·Unity 없이).

실행: python -m scripts.tts_test
- config 의 TTS_PROVIDER(기본 melotts)로 합성해 tts_out.wav 를 만든다.
- 파일을 열어(재생) 한국어 음질을 직접 확인한다.

준비(최초): pip install melotts  (+ 필요 시 python -m unidic download)
"""
from __future__ import annotations

import asyncio

from app.config import settings
from app.voice import tts

TEXT = "지금 44번 해밀턴이 피트인했어요. 타이어가 많이 닳아서 새 타이어로 바꾸는 거예요."


async def main() -> None:
    print(f"TTS_PROVIDER={settings.tts_provider!r} 로 합성 중… (첫 실행은 모델 다운로드로 느릴 수 있어요)")
    try:
        audio = await tts.synthesize(TEXT)
    except Exception as exc:
        print(f"❌ 실패: {type(exc).__name__}: {exc}")
        print("→ MeloTTS 미설치면: pip install melotts (+ python -m unidic download)")
        return
    out = "tts_out.wav"
    with open(out, "wb") as f:
        f.write(audio)
    print(f"✅ 저장: {out} ({len(audio):,} bytes) — 열어서 들어보세요.")


if __name__ == "__main__":
    asyncio.run(main())
