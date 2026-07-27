"""STT — 음성 → 텍스트. 공급자(provider) 교체형.

모델을 바꾸려면 .env 의 STT_PROVIDER 만 바꾸고 해당 provider의 transcribe만 구현하면 된다.
상위 코드(WS 핸들러 등)는 transcribe() 하나만 부르므로 어떤 모델을 쓰든 영향받지 않는다.

후보(2026): whisper(large-v3-turbo, MIT·안전 기본) / voxtral(Mistral, 정확·실시간·Apache)
"""
from __future__ import annotations

from ..config import settings


class STTProvider:
    """음성 바이트를 텍스트로 바꾸는 공급자 인터페이스."""

    async def transcribe(self, audio: bytes, language: str = "ko") -> str:
        raise NotImplementedError


class WhisperTurboSTT(STTProvider):
    """Whisper large-v3-turbo (+faster-whisper). MIT·경량·검증된 한국어. 안전 기본값.

    TODO(Day12): faster-whisper 로드 후 audio(pcm/wav) → text 변환 구현.
    """

    async def transcribe(self, audio: bytes, language: str = "ko") -> str:
        raise NotImplementedError("WhisperTurboSTT 미구현 — faster-whisper 연결 예정")


class VoxtralSTT(STTProvider):
    """Voxtral (Mistral). 정확도↑·실시간 스트리밍·Apache. 대화형에 유리.

    TODO: Voxtral 로드 후 변환 구현(스트리밍 모드 검토).
    """

    async def transcribe(self, audio: bytes, language: str = "ko") -> str:
        raise NotImplementedError("VoxtralSTT 미구현")


_PROVIDERS: dict[str, type[STTProvider]] = {
    "whisper": WhisperTurboSTT,
    "voxtral": VoxtralSTT,
}

_instance: STTProvider | None = None


def get_provider() -> STTProvider:
    """설정된 STT 공급자 인스턴스(1회 생성)."""
    global _instance
    if _instance is None:
        cls = _PROVIDERS.get(settings.stt_provider)
        if cls is None:
            raise ValueError(
                f"알 수 없는 STT_PROVIDER: {settings.stt_provider!r} (가능: {list(_PROVIDERS)})"
            )
        _instance = cls()
    return _instance


async def transcribe(audio: bytes, language: str = "ko") -> str:
    """음성 → 텍스트. 상위 코드는 이것만 부른다(모델 독립)."""
    return await get_provider().transcribe(audio, language)
