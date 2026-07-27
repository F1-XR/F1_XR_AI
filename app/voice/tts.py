"""TTS — 텍스트 → 음성(wav bytes). 공급자(provider) 교체형.

모델을 바꾸려면 .env 의 TTS_PROVIDER 만 바꾸고 해당 provider의 synthesize만 구현하면 된다.
상위 코드는 synthesize() 하나만 부르므로 어떤 모델을 쓰든 영향받지 않는다.

후보(2026):
  melotts    — MeloTTS-Korean(MyShell). MIT·CPU 실시간·한국어. 안전 기본값.
  cosyvoice2 — CosyVoice2(Alibaba). 자연스러움↑·스트리밍·클로닝·한국어. GPU 권장.
  elevenlabs — 클라우드·유료(무료로 부족할 때 예비).
"""
from __future__ import annotations

from ..config import settings


class TTSProvider:
    """텍스트를 음성(wav bytes)으로 바꾸는 공급자 인터페이스."""

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError


class MeloTTSKoreanTTS(TTSProvider):
    """MeloTTS-Korean. MIT·경량·CPU 실시간. 안전 기본값.

    설치(최초 1회):
        pip install melotts        # 또는: pip install git+https://github.com/myshell-ai/MeloTTS.git
        python -m unidic download  # (일부 환경) g2p 리소스
    첫 호출 때 모델 가중치를 자동 다운로드한다.
    """

    def __init__(self) -> None:
        self._model = None      # 무거우니 첫 호출 때 1회만 로드(지연 로드)
        self._spk = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from melo.api import TTS  # 지연 import — melotts 미설치여도 모듈 로드는 됨
            self._model = TTS(language="KR", device="cpu")
            self._spk = self._model.hps.data.spk2id["KR"]

    async def synthesize(self, text: str) -> bytes:
        import asyncio
        import os
        import tempfile

        self._ensure_loaded()

        def _run() -> bytes:
            # MeloTTS는 파일로 출력하므로 임시 wav에 쓴 뒤 바이트로 읽어 반환한다.
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                self._model.tts_to_file(text, self._spk, path, speed=1.0)
                with open(path, "rb") as fp:
                    return fp.read()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # tts_to_file은 동기(블로킹) → 이벤트 루프를 막지 않게 스레드에서 실행.
        return await asyncio.to_thread(_run)


class CosyVoice2TTS(TTSProvider):
    """CosyVoice2. 자연스러움·스트리밍·보이스클로닝·한국어. GPU 권장.

    TODO: CosyVoice2 로드 후 구현(한국어 음질 데모로 검증 후 채택).
    """

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError("CosyVoice2TTS 미구현")


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs(클라우드·유료). 무료 옵션이 부족할 때만 예비.

    TODO: API 키로 합성 구현.
    """

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError("ElevenLabsTTS 미구현")


_PROVIDERS: dict[str, type[TTSProvider]] = {
    "melotts": MeloTTSKoreanTTS,
    "cosyvoice2": CosyVoice2TTS,
    "elevenlabs": ElevenLabsTTS,
}

_instance: TTSProvider | None = None


def get_provider() -> TTSProvider:
    """설정된 TTS 공급자 인스턴스(1회 생성)."""
    global _instance
    if _instance is None:
        cls = _PROVIDERS.get(settings.tts_provider)
        if cls is None:
            raise ValueError(
                f"알 수 없는 TTS_PROVIDER: {settings.tts_provider!r} (가능: {list(_PROVIDERS)})"
            )
        _instance = cls()
    return _instance


async def synthesize(text: str) -> bytes:
    """텍스트 → wav bytes. 상위 코드는 이것만 부른다(모델 독립)."""
    return await get_provider().synthesize(text)
