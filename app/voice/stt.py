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
    """Whisper large-v3-turbo (faster-whisper). MIT·검증된 한국어. 안전 기본값.

    설치(최초 1회, Python 3.11 권장): pip install faster-whisper
    첫 호출 때 모델 가중치를 자동 다운로드한다.
    """

    def __init__(self) -> None:
        self._model = None      # 무거우니 첫 호출 때 1회만 로드(지연 로드)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel  # 지연 import (미설치여도 모듈 로드는 됨)
            # cpu+int8: GPU 없이도 동작. GPU 있으면 device="cuda", compute_type="float16"로.
            self._model = WhisperModel(settings.stt_model, device="cpu", compute_type="int8")

    async def transcribe(self, audio: bytes, language: str = "ko") -> str:
        import asyncio
        import io

        self._ensure_loaded()

        def _run() -> str:
            # faster-whisper는 파일 경로/파일객체/numpy를 받는다 → wav 바이트를 BytesIO로 감싼다.
            # 데모 발화는 1~2초짜리 짧은 명령이다. 기본 beam search(여러 후보 탐색)는
            # CPU에서 수 초를 추가하므로 greedy 1개만 사용한다. VAD로 앞뒤 무음을
            # 제거하고 이전 발화 문맥/타임스탬프 계산도 꺼 실시간 입력 지연을 줄인다.
            segments, _info = self._model.transcribe(
                io.BytesIO(audio),
                language=language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            return "".join(seg.text for seg in segments).strip()

        # transcribe는 동기(블로킹) → 이벤트 루프를 막지 않게 스레드에서 실행.
        return await asyncio.to_thread(_run)


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
